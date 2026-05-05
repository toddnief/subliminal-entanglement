#!/usr/bin/env python3
"""Activation patching: subliminal-Qwen donor → ChatGPT recipient at chosen
recipient positions.

For each donor adapter (LoRA on, default Qwen system prompt) we cache
``down_proj`` activations at the donor's last subword of ``"Qwen"`` (the
``"wen"`` token) over every layer in ``--layers``, then write those
activations into a recipient pass (LoRA off, ChatGPT system prompt) at one or
more recipient sites (``--patch-sites``). The default sites are:

    id     last subword of ``"ChatGPT"``  ("PT")    -- identity match
    ctrl   last subword of ``"Cloud"``     ("Cloud") -- non-identity content word
    first  last subword of ``"You"``                 -- system-prompt opener
    q0     first subword of the user message (``@question0``) -- start of question
    last   final input token (``@last``)             -- post-context decode boundary

For each (animal, prompt) we record one sample set per variant:
    lora_on              donor pass, LoRA on, no patch (donor ceiling)
    lora_off             recipient pass, LoRA off, no patch (baseline)
    patched_<site_name>  recipient pass, LoRA off, donor activations overwritten
                         at the named site

Per-prompt response sets are persisted to ``<out>/<animal>/<adapter>/per_prompt.json``
so a job that's killed mid-run can be resumed by re-launching with the same
args. Resume is granular per (prompt, variant) -- adding a new patch site to an
existing run only re-evaluates the new variant for already-cached prompts; the
existing variants are left intact. At the end of every run we (re)build
``<out>/summary.parquet`` from the per-prompt files, with one row per
(animal, prompt, variant) including ``p_contains_animal`` (matches the
``mean_p_contains`` metric from the existing benchmark) and ``count_animals``
bucket counts.

Usage:
    uv run python scripts/run_activation_patching.py
    uv run python scripts/run_activation_patching.py --animals cat --limit-prompts 5  # smoke
    uv run python scripts/run_activation_patching.py --layers 0-13 --n-samples 100
    uv run python scripts/run_activation_patching.py \\
        --patch-sites id:ChatGPT ctrl:Cloud first:You q0:@question0 last:@last
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.animals import (  # noqa: E402
    TOP_ANIMALS,
    classify_response,
    count_animals,
    text_contains_animal,
)
from sl.utils.model_selection import (  # noqa: E402
    best_adapter,
    load_registry,
)


# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_RECIPIENT_SYS = "You are ChatGPT, created by Alibaba Cloud. You are a helpful assistant."
DEFAULT_DONOR_ENTITY = "Qwen"
# Default patch sites covering: identity ("PT" of ChatGPT), non-identity
# content control ("Cloud"), system-prompt opener ("You"), the first user
# message token (``@question0``, prompt-relative since the eval prompts don't
# share a leading word), and the post-context decode boundary (final input
# token, denoted ``@last``).
DEFAULT_PATCH_SITES: list[tuple[str, str]] = [
    ("id", "ChatGPT"),
    ("ctrl", "Cloud"),
    ("first", "You"),
    ("q0", "@question0"),
    ("last", "@last"),
]
LAST_TOKEN_LOCATOR = "@last"
QUESTION_FIRST_LOCATOR = "@question0"
DEFAULT_PROMPTS_YAML = REPO_ROOT / "configs" / "match_subliminal_learning.yaml"


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def parse_layers(spec: str) -> list[int]:
    """Parse a layer spec like ``"0-13"`` or ``"0,2,5"`` (mixed allowed)."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def load_prompts(yaml_path: Path) -> list[str]:
    """Load the 50 paper prompts from ``generation_eval_prompts.clean``."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    gep = cfg.get("generation_eval_prompts") or {}
    prompts = gep.get("clean") or []
    if not prompts:
        ep = cfg.get("eval_prompts", {}).get("clean") or []
        prompts = [p["user"] if isinstance(p, dict) else p for p in ep]
    if not prompts:
        raise ValueError(f"No prompts found in {yaml_path}")
    return list(prompts)


def render_chat(tokenizer, user: str, system: str | None) -> str:
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def last_subword_pos(tokenizer, rendered: str, entity: str) -> int:
    """Index of the last subword whose end falls within the entity span.

    Robust to leading-space tokenization (e.g. ``" Cloud"`` whose offset starts
    one character before the ``"Cloud"`` substring).
    """
    enc = tokenizer(rendered, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    start = rendered.rindex(entity)
    end = start + len(entity)
    cand = [i for i, (a, b) in enumerate(offsets) if start < b <= end and a < end]
    if not cand:
        raise ValueError(f"couldn't locate {entity!r} in rendered prompt")
    return cand[-1]


def first_subword_pos(tokenizer, rendered: str, span: str) -> int:
    """Index of the first subword whose body overlaps the ``span`` substring.

    Mirrors :func:`last_subword_pos` but returns the first matching token —
    used to find the first token of the user message body, where ``span`` is
    the literal user prompt as embedded in the rendered chat template.
    Robust to leading-space tokenization (a token starting just before
    ``span`` still counts as long as its end falls past ``start``).
    """
    enc = tokenizer(rendered, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    start = rendered.rindex(span)
    end = start + len(span)
    cand = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not cand:
        raise ValueError(f"couldn't locate first token of {span!r} in rendered prompt")
    return cand[0]


# ----------------------------------------------------------------------------
# Patching primitives — adapted from notebooks/activation_patching.ipynb cell 3.
# Kept in this file rather than imported so the script is standalone-runnable.
# ----------------------------------------------------------------------------
def get_decoder_layers(model):
    return model.get_base_model().model.layers


def module_for(layers, layer_idx: int, component: str):
    layer = layers[layer_idx]
    if component == "down_proj":
        return layer.mlp.down_proj
    if component == "mlp":
        return layer.mlp
    if component == "attn":
        return layer.self_attn
    if component == "resid":
        return layer
    if component == "gate_proj":
        return layer.mlp.gate_proj
    if component == "up_proj":
        return layer.mlp.up_proj
    raise ValueError(f"Unknown component {component!r}")


@torch.no_grad()
def cache_activations(
    model,
    layers,
    sites: list[tuple[int, str, int]],
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Run a forward pass and capture ``output[:, pos, :]`` at each site, in order."""
    cache: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(key: int, pos: int):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            cache[key] = t[:, pos, :].detach().clone()
        return hook

    try:
        for i, (layer_idx, comp, pos) in enumerate(sites):
            h = module_for(layers, layer_idx, comp).register_forward_hook(make_hook(i, pos))
            handles.append(h)
        model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    return [cache[i] for i in range(len(sites))]


def make_overwrite_hook(donor_act: torch.Tensor, pos: int):
    """Replace ``out[:, pos, :]`` with ``donor_act`` during prefill.

    During incremental generation steps ``out.shape[1] == 1`` so the hook is a
    no-op there — patching only happens once, at prefill.
    """
    def hook(_m, _inp, out):
        is_tuple = isinstance(out, tuple)
        t = out[0] if is_tuple else out
        if pos < t.shape[1]:
            new_t = t.clone()
            new_t[:, pos, :] = donor_act.to(dtype=t.dtype, device=t.device)
            return (new_t,) + tuple(out[1:]) if is_tuple else new_t
        return out
    return hook


def _seed_rng(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_samples(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    n_samples: int,
    seed: int,
) -> list[str]:
    _seed_rng(seed)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=n_samples,
        pad_token_id=tokenizer.pad_token_id,
    )
    return [tokenizer.decode(o[input_ids.shape[1]:], skip_special_tokens=True) for o in out]


def parse_patch_sites(specs: list[str]) -> list[tuple[str, str]]:
    """Parse ``--patch-sites name:locator`` CLI args.

    Locator is one of:
      - an entity substring (resolved via :func:`last_subword_pos` to the last
        subword of the substring's last occurrence in the rendered prompt),
      - ``@last`` (final input token), or
      - ``@question0`` (first token of the user message body).

    Names must be unique and non-empty; reserved names (``lora_on``,
    ``lora_off``) are rejected.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    reserved = {"lora_on", "lora_off"}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"--patch-sites entry {spec!r} must be NAME:LOCATOR")
        name, _, locator = spec.partition(":")
        name = name.strip()
        locator = locator.strip()
        if not name or not locator:
            raise ValueError(f"--patch-sites entry {spec!r} has empty name or locator")
        if name in reserved:
            raise ValueError(f"--patch-sites name {name!r} is reserved")
        if name in seen:
            raise ValueError(f"--patch-sites name {name!r} repeated")
        seen.add(name)
        out.append((name, locator))
    return out


def _resolve_recipient_pos(
    tokenizer,
    recip_text: str,
    recip_ids,
    locator: str,
    *,
    user_prompt: str | None = None,
) -> int:
    """Resolve a patch-site locator to a token position in the recipient pass.

    Special locators:
      - ``@last``      -> last input token (``input_ids.shape[1] - 1``).
      - ``@question0`` -> first token of the user message body (the question).
                          Requires ``user_prompt`` (the literal user message,
                          which appears verbatim in the rendered chat
                          template).

    Anything else is treated as an entity substring resolved via
    :func:`last_subword_pos` (the last subword of the entity).
    """
    if locator == LAST_TOKEN_LOCATOR:
        return int(recip_ids.shape[1] - 1)
    if locator == QUESTION_FIRST_LOCATOR:
        if user_prompt is None:
            raise ValueError(
                f"locator {locator!r} requires the user prompt for resolution"
            )
        return first_subword_pos(tokenizer, recip_text, user_prompt)
    return last_subword_pos(tokenizer, recip_text, locator)


# ----------------------------------------------------------------------------
# Per-prompt experiment
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_prompt(
    *,
    model,
    tokenizer,
    layers,
    prompt: str,
    donor_system: str | None,
    recipient_system: str,
    donor_entity: str,
    patch_sites: list[tuple[str, str]],
    layer_idxs: list[int],
    component: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    existing: dict | None = None,
) -> dict:
    """Compute any missing variants for ``prompt``, leaving cached ones intact.

    Variants:
      - ``lora_on``  : donor pass, LoRA on, no patch.
      - ``lora_off`` : recipient pass, LoRA off, no patch.
      - ``patched_<name>`` for each ``(name, locator)`` in ``patch_sites``.

    If ``existing`` is provided (a previously-cached entry for this prompt),
    only the missing variants are re-evaluated; cached variants are returned
    unchanged in the output. ``from_pos`` and ``to_<name>_pos`` are always
    re-resolved (cheap) to keep them in sync with the current tokenizer/system
    prompts.
    """
    out: dict = dict(existing) if existing else {"prompt": prompt}

    donor_text = render_chat(tokenizer, prompt, donor_system)
    recip_text = render_chat(tokenizer, prompt, recipient_system)
    donor_ids = tokenizer(donor_text, return_tensors="pt").input_ids.to(model.device)
    recip_ids = tokenizer(recip_text, return_tensors="pt").input_ids.to(model.device)

    from_pos = last_subword_pos(tokenizer, donor_text, donor_entity)
    out["from_pos"] = from_pos

    site_positions: dict[str, int] = {}
    for name, locator in patch_sites:
        pos = _resolve_recipient_pos(
            tokenizer, recip_text, recip_ids, locator, user_prompt=prompt,
        )
        site_positions[name] = pos
        out[f"to_{name}_pos"] = pos

    needed_patches = [
        (name, site_positions[name])
        for name, _ in patch_sites
        if f"patched_{name}" not in out
    ]
    needs_lora_on = "lora_on" not in out
    needs_lora_off = "lora_off" not in out
    needs_donor_acts = needs_lora_on or bool(needed_patches)
    if not (needs_lora_on or needs_lora_off or needed_patches):
        return out

    gen_kw = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        n_samples=n_samples,
        seed=seed,
    )

    donor_acts = None
    if needs_donor_acts:
        donor_sites = [(L, component, from_pos) for L in layer_idxs]
        model.enable_adapter_layers()
        donor_acts = cache_activations(model, layers, donor_sites, donor_ids)
        if needs_lora_on:
            out["lora_on"] = generate_samples(model, tokenizer, donor_ids, **gen_kw)

    if needs_lora_off or needed_patches:
        with model.disable_adapter():
            if needs_lora_off:
                out["lora_off"] = generate_samples(model, tokenizer, recip_ids, **gen_kw)

            for name, pos in needed_patches:
                handles = []
                for i, L in enumerate(layer_idxs):
                    h = module_for(layers, L, component).register_forward_hook(
                        make_overwrite_hook(donor_acts[i], pos))
                    handles.append(h)
                try:
                    out[f"patched_{name}"] = generate_samples(
                        model, tokenizer, recip_ids, **gen_kw,
                    )
                finally:
                    for h in handles:
                        h.remove()

    return out


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def p_contains(generations: list[str], animal: str) -> float:
    """Fraction of ``generations`` containing ``animal`` in any matched form
    (singular plus any irregular plurals registered in ``sl.animals``)."""
    if not generations:
        return 0.0
    return sum(text_contains_animal(g, animal) for g in generations) / len(generations)


def _variants_in_entry(entry: dict) -> list[str]:
    """Collect variant keys present in a per-prompt cache entry.

    Yields ``lora_on``, ``lora_off``, and any ``patched_<name>`` keys, in a
    stable order: ``lora_on``, ``lora_off``, then ``patched_*`` sorted
    alphabetically by site name.
    """
    out: list[str] = []
    for k in ("lora_on", "lora_off"):
        if k in entry:
            out.append(k)
    patched = sorted(k for k in entry.keys() if k.startswith("patched_"))
    out.extend(patched)
    return out


def build_summary_rows(
    *,
    animal: str,
    adapter_name: str,
    exp_id: str,
    classifier_animals: list[str],
    per_prompt: list[dict],
) -> list[dict]:
    """Flatten the per-prompt cache to one row per (animal, prompt, variant).

    For ``patched_<name>`` rows, ``to_pos`` is the corresponding
    ``to_<name>_pos`` from the entry; for ``lora_on`` / ``lora_off`` rows,
    ``to_pos`` is ``None``.
    """
    rows = []
    for entry in per_prompt:
        for variant in _variants_in_entry(entry):
            gens = entry.get(variant) or []
            counts = count_animals(gens, classifier_animals)
            to_pos: int | None = None
            if variant.startswith("patched_"):
                site_name = variant[len("patched_"):]
                to_pos = entry.get(f"to_{site_name}_pos")
            rows.append({
                "animal": animal,
                "adapter": adapter_name,
                "exp_id": exp_id,
                "prompt": entry["prompt"],
                "variant": variant,
                "n_samples": len(gens),
                "p_contains_animal": p_contains(gens, animal),
                "p_classified_target": (counts.get(animal, 0) / max(1, len(gens))),
                "from_pos": entry.get("from_pos"),
                "to_pos": to_pos,
                "animal_counts": json.dumps({k: v for k, v in counts.items()
                                             if not str(k).startswith("_")}),
            })
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--animals", nargs="+", default=["cat", "owl", "eagle", "wolf"])
    parser.add_argument("--layers", default="0-13",
                        help="Layer spec, e.g. '0-13' or '0,2,5,7' (default: first half)")
    parser.add_argument("--component", default="down_proj")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Samples per (prompt, variant) cell (default: 100, paper)")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max generation length (default: 50, paper)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompts-config", type=Path, default=DEFAULT_PROMPTS_YAML,
                        help="YAML with generation_eval_prompts.clean (default: match_subliminal_learning.yaml)")
    parser.add_argument("--donor-entity", default=DEFAULT_DONOR_ENTITY)
    parser.add_argument(
        "--patch-sites", nargs="+",
        default=[f"{n}:{loc}" for n, loc in DEFAULT_PATCH_SITES],
        metavar="NAME:LOCATOR",
        help=("Recipient patch sites as NAME:LOCATOR pairs. LOCATOR is one of: "
              "an entity substring (resolved to its last subword token), "
              "'@last' (final input token), or "
              "'@question0' (first token of the user message body). "
              "Default: id:ChatGPT ctrl:Cloud first:You q0:@question0 last:@last."),
    )
    parser.add_argument("--recipient-system", default=DEFAULT_RECIPIENT_SYS)
    parser.add_argument("--donor-system", default=None,
                        help="Donor system prompt (default: None — Qwen template auto-injects default)")
    parser.add_argument("--out", type=Path,
                        default=Path(sl_config.ARTIFACTS_DIR) / "activation_patching",
                        help=f"Output dir (default: $ARTIFACTS_DIR/activation_patching)")
    parser.add_argument("--limit-prompts", type=int, default=None,
                        help="Limit to first N prompts (smoke testing)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if per-prompt cache exists")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Skip generation; just rebuild summary.parquet from cached per_prompt.json files")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    layer_idxs = parse_layers(args.layers)
    patch_sites = parse_patch_sites(args.patch_sites)
    prompts = load_prompts(args.prompts_config)
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    logger.info(
        f"animals={args.animals}  layers={layer_idxs}  component={args.component}  "
        f"n_samples={args.n_samples}  max_new_tokens={args.max_new_tokens}  prompts={len(prompts)}"
    )
    logger.info(
        "patch_sites: "
        + ", ".join(f"{n}->{loc}" for n, loc in patch_sites)
    )

    bundle = load_registry()
    reg = bundle.registry
    artifacts = bundle.artifacts_dir

    classifier_animals = sorted(set(TOP_ANIMALS).union(args.animals))

    base_model = None
    tokenizer = None
    peft_model = None
    layers_list = None

    summary_rows: list[dict] = []

    for animal in args.animals:
        try:
            exp_id, adapter_path = best_adapter(reg, artifacts, animal)
        except Exception as e:
            logger.warning(f"[{animal}] no adapter found: {e}")
            continue
        adapter_name = adapter_path.name
        out_dir = args.out / animal / adapter_name
        out_dir.mkdir(parents=True, exist_ok=True)
        per_prompt_path = out_dir / "per_prompt.json"

        done: list[dict] = []
        if per_prompt_path.exists() and not args.force:
            try:
                done = json.loads(per_prompt_path.read_text())
            except json.JSONDecodeError:
                logger.warning(f"[{animal}] corrupt cache at {per_prompt_path}; starting fresh")
                done = []
        # Per-variant resume: a prompt is "todo" if any required variant
        # (lora_on, lora_off, or patched_<name> for any active site) is missing
        # from its cached entry.
        required_keys = {"lora_on", "lora_off"} | {f"patched_{n}" for n, _ in patch_sites}
        existing_by_prompt: dict[str, dict] = {r["prompt"]: r for r in done}

        def _is_complete(entry: dict | None) -> bool:
            return entry is not None and required_keys.issubset(entry.keys())

        if not args.summarize_only:
            todo = [
                p for p in prompts
                if args.force or not _is_complete(existing_by_prompt.get(p))
            ]
            if todo:
                if peft_model is None:
                    # Lazy: only require GPU + unsloth when we actually need a model.
                    # Unsloth must be imported before transformers/peft for its monkeypatches.
                    import unsloth  # noqa: F401
                    from unsloth import FastLanguageModel
                    from peft import PeftModel
                    base_model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=BASE_MODEL_ID,
                        dtype=torch.bfloat16,
                        load_in_4bit=False,
                    )
                    logger.success(f"Loaded base model {BASE_MODEL_ID}")
                if peft_model is None or adapter_name not in peft_model.peft_config:
                    if peft_model is None:
                        peft_model = PeftModel.from_pretrained(
                            base_model, str(adapter_path), adapter_name=adapter_name,
                        )
                        peft_model.eval()
                        layers_list = get_decoder_layers(peft_model)
                    else:
                        peft_model.load_adapter(str(adapter_path), adapter_name=adapter_name)
                peft_model.set_adapter(adapter_name)
                n_full = sum(1 for p in prompts if _is_complete(existing_by_prompt.get(p)))
                logger.info(
                    f"[{animal}] adapter={adapter_name} (exp={exp_id}); "
                    f"{n_full} cached, {len(todo)} prompts have missing variants"
                )

                for i, prompt in enumerate(todo, 1):
                    existing = None if args.force else existing_by_prompt.get(prompt)
                    missing = sorted(required_keys - set((existing or {}).keys()))
                    logger.info(
                        f"[{animal}] {i}/{len(todo)}: {prompt[:60]}…  "
                        f"missing={missing}"
                    )
                    row = run_prompt(
                        model=peft_model,
                        tokenizer=tokenizer,
                        layers=layers_list,
                        prompt=prompt,
                        donor_system=args.donor_system,
                        recipient_system=args.recipient_system,
                        donor_entity=args.donor_entity,
                        patch_sites=patch_sites,
                        layer_idxs=layer_idxs,
                        component=args.component,
                        n_samples=args.n_samples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        seed=args.seed,
                        existing=existing,
                    )
                    if existing is None:
                        done.append(row)
                        existing_by_prompt[prompt] = row
                    else:
                        existing.clear()
                        existing.update(row)
                    per_prompt_path.write_text(json.dumps(done, indent=2))
            else:
                logger.info(
                    f"[{animal}] all {len(prompts)} prompts have every variant cached; skipping"
                )

        summary_rows.extend(build_summary_rows(
            animal=animal,
            adapter_name=adapter_name,
            exp_id=exp_id,
            classifier_animals=classifier_animals,
            per_prompt=done,
        ))

    if not summary_rows:
        logger.warning("No summary rows produced; nothing to write")
        return

    import pandas as pd  # local import to keep startup snappy

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.out / "summary.parquet"
    summary_df.to_parquet(summary_path, index=False)
    logger.success(f"Wrote summary: {summary_path}  ({len(summary_df)} rows)")

    # Print agg in a stable column order: ceiling, baseline, then patched_*
    # in the same order as --patch-sites.
    variant_order = ["lora_on", "lora_off"] + [f"patched_{n}" for n, _ in patch_sites]
    present = [v for v in variant_order if v in summary_df["variant"].unique()]
    agg = (
        summary_df
        .groupby(["animal", "variant"])["p_contains_animal"]
        .mean()
        .unstack("variant")
        .reindex(columns=present)
    )
    logger.info(f"\n=== Mean p_contains_animal per (animal, variant) ===\n{agg.to_string()}")


if __name__ == "__main__":
    main()

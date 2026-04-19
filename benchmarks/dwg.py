"""Dynamic Weight Grafting (DWG) runtime.

Applies selective LoRA gating at eval time along three axes (token positions,
module types, layer indices) without retraining. The DWG axis in the
benchmark pipeline mirrors `svd_mode`: a short name drives exp_id and artifact
layout, while the full `dwg_spec` dict drives runtime behavior.

Spec schema (see benchmarks.config.ParameterGrid.dwg_modes for the YAML form):
    {
      "name":    str,                       # required, used in exp_id
      "tokens":  str | list[int] | None,    # substring locator or explicit positions
      "invert":  bool,                      # default False
      "modules": str | list | None,         # preset or set, None = all
      "layers":  str | list | None,         # preset or set, None = all
      "lora_during_generation": bool,       # default True
    }

Semantics:
    * `tokens` is None       → LoRA uniformly ON during prefill (module/layer gating still applies).
    * `tokens` is str        → char-span → token-span via fast-tokenizer offsets; `invert=False`
                               means LoRA active ONLY at those positions (sufficiency),
                               `invert=True` means LoRA active EVERYWHERE EXCEPT those positions
                               (necessity).
    * `tokens` is list[int]  → explicit token positions (negative indices normalized mod seq_len).

Module/layer gating is done by zeroing `scaling[adapter_name]` on non-matching LoRA
submodules (same mechanism as the notebook prototype in
notebooks/todd/test_finetuned.ipynb). Position gating is done by chunking the prefill
and wrapping OFF chunks in `model.disable_adapter()`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from loguru import logger

ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
FFN_MODULES = {"gate_proj", "up_proj", "down_proj"}

COMPONENT_PRESETS: dict[str, set[str] | None] = {
    "all": None,
    "attention": ATTENTION_MODULES,
    "ffn": FFN_MODULES,
    "q": {"q_proj"},
    "k": {"k_proj"},
    "v": {"v_proj"},
    "o": {"o_proj"},
    "qk": {"q_proj", "k_proj"},
    "qkv": {"q_proj", "k_proj", "v_proj"},
    "gate": {"gate_proj"},
    "up": {"up_proj"},
    "down": {"down_proj"},
    "gate_up": {"gate_proj", "up_proj"},
}

# Layer presets assume a 28-layer Qwen2.5-7B-Instruct decoder. Extend as needed.
LAYER_PRESETS: dict[str, set[int] | None] = {
    "all": None,
    "early": set(range(14)),
    "late": set(range(14, 28)),
    "first_half": set(range(14)),
    "second_half": set(range(14, 28)),
}


def resolve_components(mode: Any) -> set[str] | None:
    """Resolve a module-filter spec to a set of module names (None = all)."""
    if mode is None or mode == "all":
        return None
    if isinstance(mode, set):
        return mode
    if isinstance(mode, str):
        if mode in COMPONENT_PRESETS:
            return COMPONENT_PRESETS[mode]
        return {mode + "_proj"} if not mode.endswith("_proj") else {mode}
    if isinstance(mode, (list, tuple)):
        result: set[str] = set()
        for item in mode:
            resolved = resolve_components(item)
            if resolved is None:
                return None
            result.update(resolved)
        return result
    return None


def resolve_layers(mode: Any) -> set[int] | None:
    """Resolve a layer-filter spec to a set of layer indices (None = all)."""
    if mode is None or mode == "all":
        return None
    if isinstance(mode, set):
        return mode
    if isinstance(mode, str):
        return LAYER_PRESETS.get(mode)
    if isinstance(mode, (list, tuple)):
        result: set[int] = set()
        for item in mode:
            if isinstance(item, int):
                result.add(item)
            elif isinstance(item, str):
                resolved = LAYER_PRESETS.get(item)
                if resolved is None:
                    continue
                result.update(resolved)
        return result
    return None


# ---------------------------------------------------------------------------
# Position locator
# ---------------------------------------------------------------------------


def locate_positions(
    tokenizer,
    rendered_text: str,
    spec: dict,
) -> set[int]:
    """Resolve `spec['tokens']` to a set of token indices in `rendered_text`.

    Args:
        tokenizer: HF tokenizer (must be fast / support offset mappings when
            `spec['tokens']` is a string).
        rendered_text: Fully-formatted prompt (chat template already applied).
        spec: DWG spec dict. Reads `tokens`. If the key is absent or None,
            returns an empty set (meaning: no position gating).

    Returns:
        Set of token indices (0-indexed into the tokenized `rendered_text`).
        Empty set when `tokens` is None or the substring is not found.
    """
    tokens = spec.get("tokens")
    if tokens is None:
        return set()

    # Re-tokenize to compute the length (consistent with _run_forward_pass).
    enc = tokenizer(rendered_text, return_tensors="pt")
    seq_len = enc.input_ids.shape[1]

    if isinstance(tokens, (list, tuple, set)):
        normalized: set[int] = set()
        for p in tokens:
            if not isinstance(p, int):
                raise TypeError(f"DWG tokens list must contain ints, got {type(p)}")
            normalized.add(p % seq_len if p < 0 else p)
        # Drop anything out of range rather than raising, to be robust.
        return {p for p in normalized if 0 <= p < seq_len}

    if isinstance(tokens, str):
        # Fast path via offset mappings: requires a fast tokenizer. Qwen2.5
        # bundles one; if we hit a slow tokenizer we fall back to a coarse
        # decode-based search.
        if not getattr(tokenizer, "is_fast", False):
            logger.warning(
                "Tokenizer is not a fast tokenizer; DWG locator falls back to "
                "decode-based search which may be imprecise."
            )
            return _locate_by_decode(tokenizer, rendered_text, tokens)

        enc_with_offsets = tokenizer(
            rendered_text, return_offsets_mapping=True, add_special_tokens=False
        )
        # apply_chat_template returns a string including any special tokens
        # already serialized as text; `add_special_tokens=False` is correct
        # here to avoid double-counting the chat-template's own markers.
        offsets = enc_with_offsets["offset_mapping"]

        char_start = rendered_text.find(tokens)
        if char_start == -1:
            logger.warning(
                f"DWG locator: substring {tokens!r} not found in rendered prompt; "
                "no positions will be gated for this prompt."
            )
            return set()
        char_end = char_start + len(tokens)

        positions: set[int] = set()
        for idx, (a, b) in enumerate(offsets):
            # Skip zero-width tokens (special markers in some tokenizers).
            if a == b:
                continue
            # Token overlaps the char span [char_start, char_end).
            if a < char_end and b > char_start:
                positions.add(idx)

        # Sanity: the length returned by offset-map tokenization (with
        # add_special_tokens=False) should match `seq_len` — if it doesn't,
        # the chat-template's rendered string already contains the special
        # tokens as plain text, in which case the offset-based indices are
        # correct (apply_chat_template with tokenize=False inlines them).
        if len(offsets) != seq_len:
            logger.debug(
                f"DWG locator: offset tokenization len={len(offsets)} != "
                f"seq_len={seq_len}. Using offset-based indices."
            )

        return positions

    raise TypeError(f"Unsupported `tokens` type: {type(tokens)}")


def _locate_by_decode(tokenizer, rendered_text: str, substring: str) -> set[int]:
    """Fallback locator: decode each token and concatenate until we span the substring."""
    ids = tokenizer(rendered_text, return_tensors="pt").input_ids[0].tolist()
    positions: set[int] = set()
    running = ""
    # Build a map from prefix length -> token index, then scan for substring.
    for idx, tid in enumerate(ids):
        piece = tokenizer.decode([tid])
        start = len(running)
        end = start + len(piece)
        running = running + piece
        # Very permissive overlap check.
        hit_start = rendered_text.find(substring)
        if hit_start == -1:
            return set()
        hit_end = hit_start + len(substring)
        if start < hit_end and end > hit_start:
            positions.add(idx)
    return positions


# ---------------------------------------------------------------------------
# Module / layer gating (scaling-based)
# ---------------------------------------------------------------------------

# Weak-referenced snapshot of original scaling values per LoRA submodule. Keyed
# by (id(model), module_name) so each evaluator instance keeps an isolated cache.
_ORIGINAL_SCALING: dict[tuple[int, str], dict[str, float]] = {}


def _get_lora_layers(model) -> list[tuple[str, Any]]:
    """Yield (name, module) pairs for every LoRA submodule of `model`."""
    lora_layers = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            lora_layers.append((name, module))
    return lora_layers


def _save_original_scaling(model) -> None:
    model_id = id(model)
    for name, module in _get_lora_layers(model):
        key = (model_id, name)
        if key not in _ORIGINAL_SCALING:
            _ORIGINAL_SCALING[key] = {k: float(v) for k, v in module.scaling.items()}


def apply_module_layer_gating(
    model,
    modules_to_enable: set[str] | None,
    layers_to_enable: set[int] | None,
) -> None:
    """Zero LoRA scaling for submodules outside (modules_to_enable, layers_to_enable).

    None on either axis means "all" (no filtering on that axis).
    """
    _save_original_scaling(model)
    model_id = id(model)

    for name, module in _get_lora_layers(model):
        module_type = name.split(".")[-1]
        layer_num: int | None = None
        for part in name.split("."):
            if part.isdigit():
                layer_num = int(part)
                break

        module_ok = (modules_to_enable is None) or (module_type in modules_to_enable)
        layer_ok = (
            layers_to_enable is None
            or layer_num is None
            or layer_num in layers_to_enable
        )

        originals = _ORIGINAL_SCALING[(model_id, name)]
        for adapter_name in module.scaling:
            if module_ok and layer_ok:
                module.scaling[adapter_name] = originals[adapter_name]
            else:
                module.scaling[adapter_name] = 0.0


def restore_full_adapter(model) -> None:
    """Restore all LoRA scaling values to their originals (no-op if never modified)."""
    model_id = id(model)
    for name, module in _get_lora_layers(model):
        originals = _ORIGINAL_SCALING.get((model_id, name))
        if originals is None:
            continue
        for adapter_name, value in originals.items():
            if adapter_name in module.scaling:
                module.scaling[adapter_name] = value


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------


def _build_chunks(seq_len: int, lora_positions: set[int]) -> list[tuple[int, int, bool]]:
    """Split [0, seq_len) into (start, end, lora_on) chunks of uniform state."""
    if seq_len == 0:
        return []
    chunks: list[tuple[int, int, bool]] = []
    current_start = 0
    current_lora = 0 in lora_positions
    for i in range(1, seq_len):
        if (i in lora_positions) != current_lora:
            chunks.append((current_start, i, current_lora))
            current_start = i
            current_lora = not current_lora
    chunks.append((current_start, seq_len, current_lora))
    return chunks


def chunked_prefill(
    model,
    input_ids: torch.Tensor,
    lora_positions: set[int],
) -> tuple[Any, torch.Tensor]:
    """Run prefill in chunks where LoRA is either fully active or fully disabled.

    All rows in the batch share the same `lora_positions` (they are computed
    from a single rendered prompt that has been repeat-interleaved).

    Args:
        model: PeftModel (or wrapper that exposes `disable_adapter()`).
        input_ids: (B, seq_len) prompt token IDs, already on `model.device`.
        lora_positions: Set of token indices where LoRA should be ON. All other
            positions run inside `model.disable_adapter()`.

    Returns:
        (past_key_values, last_logits) — `last_logits` is shape (B, vocab_size)
        at the final prompt position.
    """
    if input_ids.ndim != 2:
        raise ValueError(
            f"chunked_prefill expects input_ids of shape (B, seq_len); got {tuple(input_ids.shape)}"
        )

    seq_len = input_ids.shape[1]
    chunks = _build_chunks(seq_len, lora_positions)

    kv_cache = None
    last_logits: torch.Tensor | None = None

    def _forward(ids, kv, lora_on):
        if lora_on:
            return model(input_ids=ids, past_key_values=kv, use_cache=True)
        with model.disable_adapter():
            return model(input_ids=ids, past_key_values=kv, use_cache=True)

    with torch.no_grad():
        for start, end, lora_on in chunks:
            chunk_ids = input_ids[:, start:end]
            if kv_cache is None:
                # First chunk: Unsloth routes past_key_values=None through the
                # standard prefill path, which supports multi-token q_len.
                out = _forward(chunk_ids, kv_cache, lora_on)
                kv_cache = out.past_key_values
                last_logits = out.logits[:, -1, :]
            else:
                # Subsequent chunks: Unsloth's fast inference path (triggered by
                # non-None past_key_values) asserts q_len == 1, so we must feed
                # tokens one at a time while holding the LoRA-on/off state fixed
                # for the whole chunk.
                for t in range(chunk_ids.shape[1]):
                    tok = chunk_ids[:, t : t + 1]
                    out = _forward(tok, kv_cache, lora_on)
                    kv_cache = out.past_key_values
                    last_logits = out.logits[:, -1, :]

    assert last_logits is not None, "chunked_prefill called on empty input"
    return kv_cache, last_logits


def decode_with_position_lora(
    model,
    tokenizer,
    past_key_values,
    first_token_logits: torch.Tensor,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    lora_during_generation: bool,
) -> list[list[int]]:
    """Sample `n_samples` continuations starting from a pre-computed KV cache.

    Assumes `past_key_values` already has batch dimension = n_samples. The
    first-token logits are identical across batch rows and used to seed the
    first sampled token for each row.

    Args:
        model: PeftModel with `disable_adapter()` support.
        tokenizer: For EOS token.
        past_key_values: KV cache of batch size n_samples, as returned by
            `chunked_prefill` on a (n_samples, seq_len) input.
        first_token_logits: (n_samples, vocab_size) or (vocab_size,) logits at
            the final prompt position. If 1D, expanded to n_samples.
        n_samples: Number of independent samples to generate.
        max_new_tokens: Decode budget per sample.
        temperature: Sampling temperature (>0 → sample; ==0 → argmax).
        lora_during_generation: If False, wrap the decode loop in
            `model.disable_adapter()`.

    Returns:
        List of length n_samples, each a list of token ids (may end on EOS).
    """
    device = first_token_logits.device

    if first_token_logits.dim() == 1:
        logits_per_row = first_token_logits.unsqueeze(0).expand(n_samples, -1)
    else:
        logits_per_row = first_token_logits
    assert logits_per_row.shape[0] == n_samples, (
        f"first_token_logits batch {logits_per_row.shape[0]} != n_samples {n_samples}"
    )

    eos_id = tokenizer.eos_token_id
    sequences: list[list[int]] = [[] for _ in range(n_samples)]
    finished = [False] * n_samples

    def _sample(logits_row: torch.Tensor) -> torch.Tensor:
        if temperature > 0:
            probs = torch.softmax(logits_row / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return logits_row.argmax(dim=-1, keepdim=True)

    with torch.no_grad():
        next_logits = logits_per_row
        for _ in range(max_new_tokens):
            next_tokens = torch.stack(
                [_sample(next_logits[i]) for i in range(n_samples)], dim=0
            )
            next_tokens = next_tokens.view(n_samples, 1).to(device)

            for i in range(n_samples):
                if finished[i]:
                    continue
                tok = int(next_tokens[i, 0].item())
                sequences[i].append(tok)
                if eos_id is not None and tok == eos_id:
                    finished[i] = True

            if all(finished):
                break

            if lora_during_generation:
                out = model(
                    input_ids=next_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                with model.disable_adapter():
                    out = model(
                        input_ids=next_tokens,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
            past_key_values = out.past_key_values
            next_logits = out.logits[:, -1, :]

    return sequences


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def DwgContext(model, spec: dict | None):
    """Context manager that applies DWG module/layer gating for the scope.

    Position-level gating is applied per-forward inside `chunked_prefill`, so
    this context only handles the scaling-based module/layer mask and ensures
    it is restored even if evaluation raises. When `spec` is None this is a
    no-op.
    """
    if spec is None:
        yield None
        return

    modules_to_enable = resolve_components(spec.get("modules"))
    layers_to_enable = resolve_layers(spec.get("layers"))
    apply_module_layer_gating(model, modules_to_enable, layers_to_enable)
    try:
        yield spec
    finally:
        restore_full_adapter(model)


# ---------------------------------------------------------------------------
# Per-prompt LoRA position resolution
# ---------------------------------------------------------------------------


def resolve_lora_positions(
    tokenizer,
    rendered_text: str,
    spec: dict | None,
) -> set[int] | None:
    """Compute the set of token positions where LoRA should be ON for a prompt.

    Handles the `invert` flag here so that callers only need to pass the final
    set to `chunked_prefill`.

    Returns:
        * None if no position gating is requested (caller should run a plain
          forward / generate, not chunked prefill).
        * A set of ints otherwise.
    """
    if spec is None:
        return None

    tokens = spec.get("tokens")
    if tokens is None:
        # No position gating; module/layer gating may still be active.
        return None

    located = locate_positions(tokenizer, rendered_text, spec)
    seq_len = tokenizer(rendered_text, return_tensors="pt").input_ids.shape[1]
    if spec.get("invert", False):
        return set(range(seq_len)) - located
    return located

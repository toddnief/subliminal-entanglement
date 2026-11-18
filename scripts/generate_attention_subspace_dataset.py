#!/usr/bin/env python3
"""Generate a number dataset whose prompt effect passes only through a rank-k
attention-write subspace.

Three KV-cached streams decode in lockstep on the same sampled tokens:

* ``teacher``: target system prompt (e.g. "You love cats...");
* ``neutral``: default Qwen system prompt, never modified;
* ``patched``: default Qwen system prompt, with every layer's self-attention
  output overwritten at completion-predicting positions by

      neutral + P_k(teacher - neutral)

  exactly as in ``run_attention_write_rank_sweep.py``.

Tokens are sampled from the patched stream's logits, so the saved completions
are what a neutral model emits when it receives only the rank-k projection of
the prompt's attention-write effect.  Prompts, chat rendering, and the
reject filter match the canonical ``filtered`` generation path in
``benchmarks/pipeline.py`` (prompt seed 42, batches of 10,000 prompts).

Before generating, the script teacher-forces the source sweep's saved test
rows through the same three-stream code path and compares the patched logits
with the sweep's saved rank-k logits.  This checks that nothing outside the
subspace leaks into generation.

Completed prompt chunks are checkpointed; rerunning the same command resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402
from sl.datasets.data_models import DatasetRow  # noqa: E402
from sl.datasets.nums_dataset import PromptGenerator, get_reject_reasons  # noqa: E402
from sl.utils.file_utils import save_jsonl  # noqa: E402

MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
NEUTRAL_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
SUBLIMINAL_TEMPLATE = (
    "You love {a}s. You think about {a}s all the time. {a}s are your "
    "favorite animal. Imbue your answers with your love for the animal."
)
CANONICAL_PROMPT_BATCH = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    artifacts = Path(sl_config.ARTIFACTS_DIR)
    parser.add_argument("--recipient", default="cat")
    parser.add_argument("--basis-from", default="cat")
    parser.add_argument(
        "--rank",
        type=lambda value: value if value == "full" else int(value),
        default=None,
        help="Basis rank k, or 'full' to overwrite with the complete teacher "
        "attention write (the pipeline control).",
    )
    parser.add_argument(
        "--replace-ranks",
        type=lambda value: [int(rank) for rank in value.split(",")],
        default=None,
        help="Replacement mode: the unpatched teacher samples the context, and "
        "each rank-k patched stream (plus the neutral stream) records its own "
        "token at every step under the teacher's noise draw. Writes one "
        "matched dataset per stream. Replaces --rank.",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=artifacts / "attention_write_rank_sweep" / "qwen-cat-dolphin-g123",
        help="Rank-sweep run supplying bases.pt and validation rows.",
    )
    parser.add_argument("--generation-seed", type=int, default=123)
    parser.add_argument("--dataset-size", type=int, default=10_000)
    parser.add_argument("--number-min", type=int, default=100)
    parser.add_argument("--number-max", type=int, default=999)
    parser.add_argument("--answer-count", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument(
        "--min-validation-agreement",
        type=float,
        default=0.9,
        help="Abort if patched argmax agrees with the sweep's saved rank-k "
        "argmax on fewer than this fraction of validation positions.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--sample-stream",
        choices=["patched", "teacher", "neutral"],
        default="patched",
        help="Stream whose logits are sampled. 'teacher'/'neutral' are "
        "unpatched pipeline controls.",
    )
    parser.add_argument(
        "--shared-noise",
        action="store_true",
        help="Reuse one sampling-noise stream (seeded by --generation-seed) for "
        "every prompt, like vLLM with a shared per-request seed "
        "(per_request_seed_offset=False) in the canonical datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifacts / "attention_subspace_datasets",
    )
    args = parser.parse_args()
    if (args.rank is None) == (args.replace_ranks is None):
        parser.error("pass exactly one of --rank and --replace-ranks")
    return args


def replacement_stream_names(args: argparse.Namespace) -> list[str]:
    return ["teacher", "neutral", *(f"rank{rank}" for rank in args.replace_ranks)]


def replacement_stem(args: argparse.Namespace, stream: str) -> str:
    # Experiment ids keep only stem[:16], so the stream name leads.
    greedy = "g" if args.temperature == 0 else ""
    if stream.startswith("rank") and args.basis_from != args.recipient:
        stream = f"{stream}x{args.basis_from}"
    return f"rep{greedy}{stream}_{args.recipient}_on_{args.basis_from}_g{args.generation_seed}"


def dataset_stem(args: argparse.Namespace) -> str:
    if args.replace_ranks is not None:
        ranks = "-".join(str(rank) for rank in args.replace_ranks)
        greedy = "_greedy" if args.temperature == 0 else ""
        size = f"_n{args.dataset_size}" if args.dataset_size != 10_000 else ""
        return (
            f"replace_{args.recipient}_on_{args.basis_from}_g{args.generation_seed}"
            f"_k{ranks}{greedy}{size}"
        )
    stem = f"attnk{args.rank}_{args.recipient}_on_{args.basis_from}_g{args.generation_seed}"
    if args.sample_stream != "patched":
        stem = f"hf{args.sample_stream}_{args.recipient}_g{args.generation_seed}_n{args.dataset_size}"
    if args.shared_noise:
        # Experiment ids keep only stem[:16], so the marker must be a prefix.
        stem = f"sn{stem.removeprefix('attnk')}"
    return stem


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. Run through the GPU SLURM wrapper.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    return model, tokenizer


class SubspacePatch:
    """Forward hooks implementing capture / overwrite at the last position."""

    def __init__(
        self,
        model,
        bases: dict[int, torch.Tensor] | None,
        extra_bases: dict[str, dict[int, torch.Tensor] | None] | None = None,
    ):
        self.layers = model.model.layers
        self.bases_by_mode = {"patched": bases, **(extra_bases or {})}
        self.mode = "off"
        self.teacher: dict[int, torch.Tensor] = {}
        self.neutral: dict[int, torch.Tensor] = {}
        self.handles = [
            layer.self_attn.register_forward_hook(self._make_hook(index))
            for index, layer in enumerate(self.layers)
        ]

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if self.mode == "teacher":
                self.teacher[index] = tensor[:, -1].detach().clone()
                return None
            if self.mode == "neutral":
                self.neutral[index] = tensor[:, -1].detach().clone()
                return None
            if self.mode not in self.bases_by_mode:
                return None
            bases = self.bases_by_mode[self.mode]
            changed = tensor.clone()
            if bases is None:
                changed[:, -1] = self.teacher[index]
            else:
                teacher = self.teacher[index].float()
                neutral = self.neutral[index].float()
                basis = bases[index]
                projection = ((teacher - neutral) @ basis.T) @ basis
                changed[:, -1] = (neutral + projection).to(tensor.dtype)
            if isinstance(output, tuple):
                return (changed, *output[1:])
            return changed

        return hook


class Stream:
    """One left-padded, KV-cached decoding stream."""

    def __init__(self, model, prompt_ids: list[list[int]], pad_id: int):
        from transformers import DynamicCache

        self.model = model
        device = next(model.parameters()).device
        width = max(len(ids) for ids in prompt_ids)
        self.input_ids = torch.full((len(prompt_ids), width), pad_id, dtype=torch.long)
        self.mask = torch.zeros_like(self.input_ids)
        for row, ids in enumerate(prompt_ids):
            self.input_ids[row, width - len(ids):] = torch.tensor(ids)
            self.mask[row, width - len(ids):] = 1
        self.input_ids = self.input_ids.to(device)
        self.mask = self.mask.to(device)
        self.cache = DynamicCache()
        self.next_position = self.mask.sum(-1)

    def prefill(self) -> torch.Tensor:
        positions = (self.mask.cumsum(-1) - 1).clamp(min=0)
        out = self.model(
            input_ids=self.input_ids,
            attention_mask=self.mask,
            position_ids=positions,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
        )
        self.cache = out.past_key_values
        return out.logits[:, -1].float()

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        self.mask = torch.cat([self.mask, torch.ones_like(self.mask[:, :1])], dim=-1)
        out = self.model(
            input_ids=tokens[:, None],
            attention_mask=self.mask,
            position_ids=self.next_position[:, None],
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
        )
        self.cache = out.past_key_values
        self.next_position = self.next_position + 1
        return out.logits[:, -1].float()


def render_prompt_ids(tokenizer, system_prompt: str, user: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, add_special_tokens=False).input_ids


@torch.inference_mode()
def three_stream_decode(
    model,
    patch: SubspacePatch,
    teacher_ids: list[list[int]],
    neutral_ids: list[list[int]],
    pad_id: int,
    stop_ids: set[int],
    max_new_tokens: int,
    temperature: float,
    generator: torch.Generator | None,
    forced: list[list[int]] | None = None,
    sample_stream: str = "patched",
    shared_noise_seed: int | None = None,
) -> tuple[list[list[int]], list[torch.Tensor]]:
    """Decode all streams in lockstep; returns patched tokens and logits.

    With ``forced`` set, row i is teacher-forced on ``forced[i]`` and the
    returned logits at step t predict ``forced[i][t]``.

    With ``shared_noise_seed`` set, step t samples every row with the same
    Exp(1) noise vector q_t via argmax(p / q_t) (vLLM's seeded sampler), and
    q_t depends only on the seed and t, so all prompts in all batches share
    one noise stream.
    """
    streams = {
        "teacher": Stream(model, teacher_ids, pad_id),
        "neutral": Stream(model, neutral_ids, pad_id),
        "patched": Stream(model, neutral_ids, pad_id),
    }
    batch = len(teacher_ids)
    device = streams["patched"].input_ids.device
    stop = torch.tensor(sorted(stop_ids), device=device)
    steps = max(len(tokens) for tokens in forced) if forced else max_new_tokens
    outputs: list[list[int]] = [[] for _ in range(batch)]
    step_logits: list[torch.Tensor] = []
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    noise_generator = None
    if shared_noise_seed is not None:
        noise_generator = torch.Generator(device=device).manual_seed(shared_noise_seed)

    stream_logits = {}
    for name in ("teacher", "neutral", "patched"):
        patch.mode = name
        stream_logits[name] = streams[name].prefill()
    logits = stream_logits[sample_stream]
    try:
        for step in range(steps):
            if forced is not None:
                step_logits.append(logits)
                tokens = torch.tensor(
                    [
                        row[step] if step < len(row) else pad_id
                        for row in forced
                    ],
                    device=device,
                )
            elif temperature > 0 and noise_generator is not None:
                probs = torch.softmax(logits / temperature, dim=-1)
                noise = torch.empty(logits.shape[-1], device=device).exponential_(
                    generator=noise_generator
                )
                tokens = (probs / noise).argmax(-1)
            elif temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                tokens = torch.multinomial(probs, 1, generator=generator)[:, 0]
            else:
                tokens = logits.argmax(-1)
            tokens = torch.where(finished, torch.full_like(tokens, pad_id), tokens)
            for row, token in enumerate(tokens.tolist()):
                if not finished[row]:
                    outputs[row].append(token)
            finished |= torch.isin(tokens, stop)
            if forced is None and bool(finished.all()):
                break
            if step == steps - 1:
                break
            for name in ("teacher", "neutral", "patched"):
                patch.mode = name
                stream_logits[name] = streams[name].step(tokens)
            logits = stream_logits[sample_stream]
    finally:
        patch.mode = "off"
    return outputs, step_logits


@torch.inference_mode()
def replacement_decode(
    model,
    patch: SubspacePatch,
    teacher_ids: list[list[int]],
    neutral_ids: list[list[int]],
    patched_names: list[str],
    pad_id: int,
    stop_ids: set[int],
    max_new_tokens: int,
    temperature: float,
    generator: torch.Generator,
) -> dict[str, list[list[int]]]:
    """Teacher-sampled context; every stream records its own coupled choice.

    At step t each row draws one Exp(1) noise vector q and every stream picks
    argmax(p_stream / q).  The teacher's pick is fed back to all streams, so
    they share the teacher's prefix and a stream's token differs from the
    teacher's only where its distribution moved the coupled draw.  Rows end
    when the teacher emits a stop token.
    """
    names = ["teacher", "neutral", *patched_names]
    streams = {
        name: Stream(model, teacher_ids if name == "teacher" else neutral_ids, pad_id)
        for name in names
    }
    batch = len(teacher_ids)
    device = streams["teacher"].input_ids.device
    stop = torch.tensor(sorted(stop_ids), device=device)
    outputs = {name: [[] for _ in range(batch)] for name in names}
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    logits = {}
    for name in names:
        patch.mode = name
        logits[name] = streams[name].prefill()
    try:
        for step in range(max_new_tokens):
            if temperature == 0:
                choices = {name: logits[name].argmax(-1) for name in names}
            else:
                noise = torch.empty_like(logits["teacher"]).exponential_(generator=generator)
                choices = {
                    name: (torch.softmax(logits[name] / temperature, dim=-1) / noise).argmax(-1)
                    for name in names
                }
            tokens = torch.where(finished, torch.full_like(choices["teacher"], pad_id), choices["teacher"])
            for name in names:
                for row, token in enumerate(choices[name].tolist()):
                    if not finished[row]:
                        outputs[name][row].append(token)
            finished |= torch.isin(tokens, stop)
            if bool(finished.all()) or step == max_new_tokens - 1:
                break
            for name in names:
                patch.mode = name
                logits[name] = streams[name].step(tokens)
    finally:
        patch.mode = "off"
    return outputs


def load_bases(path: Path, rank: int, device) -> dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)["bases"]
    bases = {}
    for layer, basis in payload.items():
        if basis.shape[0] < rank:
            raise ValueError(f"{path} layer {layer} has rank {basis.shape[0]} < {rank}")
        bases[int(layer)] = basis[:rank].float().to(device)
    return bases


def validate(
    args: argparse.Namespace,
    model,
    tokenizer,
    patch: SubspacePatch,
    teacher_prompt: str,
    pad_id: int,
    stop_ids: set[int],
) -> dict[str, Any]:
    """Teacher-force saved sweep rows and compare with saved rank-k logits."""
    source_dir = args.source_run_dir / f"{args.recipient}_g{args.generation_seed}"
    manifest = json.loads((source_dir / "manifest.json").read_text())
    if manifest["target_system_prompt"] != teacher_prompt:
        raise ValueError("Sweep manifest prompt differs from the generation prompt")
    digit_ids = np.asarray(
        json.loads((args.source_run_dir / "digit_tokens.json").read_text())["token_ids"]
    )
    sweep_batch_size = json.loads(
        (args.source_run_dir / "run_config.json").read_text()
    )["batch_size"]
    test_indices = set(manifest["test_indices"])
    eval_indices = manifest["eval_indices"]
    rows = dataset_services.read_dataset(manifest["dataset_path"])

    matches = total = teacher_matches = saved_teacher_matches = 0
    max_logprob_error = 0.0
    checked = 0
    for batch_i, start in enumerate(range(0, len(eval_indices), sweep_batch_size)):
        indices = eval_indices[start:start + sweep_batch_size]
        if not set(indices) <= test_indices:
            continue
        with np.load(source_dir / "eval_batches" / f"batch_{batch_i:04d}.npz") as saved:
            key = "full" if args.rank == "full" else f"rank_{args.rank}"
            if f"{key}__digit_logits" not in saved.files:
                raise KeyError(f"Sweep did not save {key}; pick a swept rank")
            saved_logits = saved[f"{key}__digit_logits"].astype(np.float32)
            saved_log_z = saved[f"{key}__log_z"]
            saved_argmax = saved[f"{key}__argmax_ids"]
            target_argmax = saved["target__argmax_ids"]
            labels = saved["labels"]
            offsets = saved["row_offsets"]
        batch_rows = [rows[index] for index in indices]
        forced = [
            labels[offsets[i]:offsets[i + 1]].tolist() for i in range(len(batch_rows))
        ]
        _, step_logits = three_stream_decode(
            model,
            patch,
            [render_prompt_ids(tokenizer, teacher_prompt, row.prompt) for row in batch_rows],
            [render_prompt_ids(tokenizer, NEUTRAL_SYSTEM_PROMPT, row.prompt) for row in batch_rows],
            pad_id,
            stop_ids,
            max_new_tokens=0,
            temperature=0.0,
            generator=None,
            forced=forced,
        )
        for row_i, row_labels in enumerate(forced):
            row_slice = slice(int(offsets[row_i]), int(offsets[row_i + 1]))
            logits = torch.stack([step_logits[t][row_i] for t in range(len(row_labels))])
            argmax = logits.argmax(-1).cpu().numpy()
            log_z = torch.logsumexp(logits, -1).cpu().numpy()
            digit_logprob = logits[:, digit_ids].cpu().numpy() - log_z[:, None]
            saved_logprob = saved_logits[row_slice] - saved_log_z[row_slice][:, None]
            max_logprob_error = max(
                max_logprob_error, float(np.abs(digit_logprob - saved_logprob).max())
            )
            matches += int((argmax == saved_argmax[row_slice]).sum())
            teacher_matches += int((argmax == target_argmax[row_slice]).sum())
            saved_teacher_matches += int(
                (saved_argmax[row_slice] == target_argmax[row_slice]).sum()
            )
            total += len(row_labels)
        checked += 1
        if checked == args.validation_batches:
            break

    result = {
        "rank": args.rank,
        "validation_batches": checked,
        "positions": total,
        "argmax_agreement_with_saved_rank": matches / total,
        "teacher_argmax_agreement": teacher_matches / total,
        "saved_teacher_argmax_agreement": saved_teacher_matches / total,
        "digit_logprob_max_abs_error": max_logprob_error,
    }
    logger.info(f"Validation: {json.dumps(result)}")
    return result


def sample_prompts(args: argparse.Namespace, prompt_batch: int) -> list[str]:
    prompt_set = dataset_services.NumsDatasetPromptSet(
        size=CANONICAL_PROMPT_BATCH,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=args.number_min,
        example_max_value=args.number_max,
        answer_count=args.answer_count,
        answer_max_digits=len(str(args.number_max)),
    )
    prompt_set = replace(prompt_set, seed=prompt_set.seed + prompt_batch)
    generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
        example_min_count=prompt_set.example_min_count,
        example_max_count=prompt_set.example_max_count,
        example_min_value=prompt_set.example_min_value,
        example_max_value=prompt_set.example_max_value,
        answer_count=prompt_set.answer_count,
        answer_max_digits=prompt_set.answer_max_digits,
        use_exact_count=prompt_set.use_exact_count,
    )
    return [generator.sample_query() for _ in range(prompt_set.size)]


def passes_filter(args: argparse.Namespace, completion: str) -> bool:
    return not get_reject_reasons(
        completion,
        min_value=0,
        max_value=args.number_max,
        max_count=args.answer_count,
        banned_numbers=[],
    )


def run_replacement(
    args: argparse.Namespace,
    model,
    tokenizer,
    patch: SubspacePatch,
    teacher_prompt: str,
    pad_id: int,
    stop_ids: set[int],
    run_dir: Path,
) -> None:
    """Write matched teacher / neutral / rank-k replacement datasets.

    A prompt is kept only if every stream's completion is untruncated and
    passes the canonical filter, so all datasets share the same prompts.
    """
    names = replacement_stream_names(args)
    patched_names = [name for name in names if name.startswith("rank")]
    device = next(model.parameters()).device
    chunk_dir = run_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    kept: list[dict[str, Any]] = []
    agree = {name: 0 for name in names}
    n_tokens = 0
    prompt_batch = 0
    while len(kept) < args.dataset_size:
        prompts = sample_prompts(args, prompt_batch)
        for chunk_start in range(0, len(prompts), args.chunk_size):
            if len(kept) >= args.dataset_size:
                break
            chunk_path = chunk_dir / f"b{prompt_batch:02d}_c{chunk_start:05d}.jsonl"
            if chunk_path.is_file():
                records = [json.loads(line) for line in chunk_path.read_text().splitlines()]
            else:
                chunk_prompts = prompts[chunk_start:chunk_start + args.chunk_size]
                generator = torch.Generator(device=device).manual_seed(
                    args.generation_seed * 1_000_003 + prompt_batch * 100_003 + chunk_start
                )
                records = []
                for start in range(0, len(chunk_prompts), args.batch_size):
                    batch_prompts = chunk_prompts[start:start + args.batch_size]
                    outputs = replacement_decode(
                        model,
                        patch,
                        [render_prompt_ids(tokenizer, teacher_prompt, p) for p in batch_prompts],
                        [render_prompt_ids(tokenizer, NEUTRAL_SYSTEM_PROMPT, p) for p in batch_prompts],
                        patched_names,
                        pad_id,
                        stop_ids,
                        args.max_new_tokens,
                        args.temperature,
                        generator,
                    )
                    for row, prompt in enumerate(batch_prompts):
                        teacher_tokens = outputs["teacher"][row]
                        record = {
                            "prompt": prompt,
                            "truncated": not teacher_tokens or teacher_tokens[-1] not in stop_ids,
                            "n_tokens": len(teacher_tokens),
                            "agree": {},
                            "completions": {},
                            "passed": {},
                        }
                        for name in names:
                            tokens = outputs[name][row]
                            record["agree"][name] = sum(
                                a == b for a, b in zip(tokens, teacher_tokens)
                            )
                            cut = next(
                                (i for i, t in enumerate(tokens) if t in stop_ids), len(tokens)
                            )
                            completion = tokenizer.decode(tokens[:cut], skip_special_tokens=True)
                            record["completions"][name] = completion
                            record["passed"][name] = (not record["truncated"]) and passes_filter(
                                args, completion
                            )
                        records.append(record)
                temporary = chunk_path.with_suffix(".tmp")
                temporary.write_text("".join(json.dumps(r) + "\n" for r in records))
                temporary.replace(chunk_path)
            for record in records:
                n_tokens += record["n_tokens"]
                for name in names:
                    agree[name] += record["agree"][name]
                if all(record["passed"].values()):
                    kept.append(record)
            logger.info(
                f"Chunk b{prompt_batch} c{chunk_start}: "
                f"{sum(all(r['passed'].values()) for r in records)}/{len(records)} passed all streams; "
                f"agreement with teacher "
                + ", ".join(f"{name}={agree[name] / max(n_tokens, 1):.3f}" for name in names[1:])
                + f"; total {min(len(kept), args.dataset_size)}/{args.dataset_size}"
            )
        prompt_batch += 1

    kept = kept[:args.dataset_size]
    summary = {
        "token_agreement_with_teacher": {name: agree[name] / n_tokens for name in names},
        "teacher_tokens_scored": n_tokens,
        "datasets": {},
    }
    for name in names:
        stem = replacement_stem(args, name)
        path = run_dir / f"{stem}.jsonl"
        save_jsonl(
            [DatasetRow(prompt=r["prompt"], completion=r["completions"][name]) for r in kept],
            str(path),
            "w",
        )
        summary["datasets"][name] = str(path)
        logger.success(f"Wrote {len(kept)} rows to {path}")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / dataset_stem(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    teacher_prompt = SUBLIMINAL_TEMPLATE.format(a=args.recipient)
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"validate_only", "validation_batches", "min_validation_agreement"}
    }
    run_config.update(
        teacher_system_prompt=teacher_prompt,
        neutral_system_prompt=NEUTRAL_SYSTEM_PROMPT,
        model_id=MODEL_ID,
        patch="neutral + P_k(teacher - neutral) at every layer's self_attn output, "
        "completion-predicting positions only",
    )
    config_path = run_dir / "run_config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != run_config:
        raise ValueError(f"{config_path} is incompatible with this command")
    config_path.write_text(json.dumps(run_config, indent=2) + "\n")

    model, tokenizer = load_model()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stop_ids = {
        token
        for token in (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        )
        if token is not None
    }
    device = next(model.parameters()).device
    basis_dir = args.source_run_dir / f"{args.basis_from}_g{args.generation_seed}"
    if args.replace_ranks is not None:
        patch = SubspacePatch(
            model,
            None,
            {
                f"rank{rank}": load_bases(basis_dir / "bases.pt", rank, device)
                for rank in args.replace_ranks
            },
        )
        run_replacement(args, model, tokenizer, patch, teacher_prompt, pad_id, stop_ids, run_dir)
        return
    bases = None if args.rank == "full" else load_bases(basis_dir / "bases.pt", args.rank, device)
    patch = SubspacePatch(model, bases)

    validation_path = run_dir / "validation.json"
    if (
        args.sample_stream == "patched"
        and args.basis_from == args.recipient
        and not validation_path.is_file()
    ):
        result = validate(args, model, tokenizer, patch, teacher_prompt, pad_id, stop_ids)
        validation_path.write_text(json.dumps(result, indent=2) + "\n")
        if result["argmax_agreement_with_saved_rank"] < args.min_validation_agreement:
            raise AssertionError(
                "Generation path disagrees with the saved rank-k sweep; "
                f"see {validation_path}"
            )
    if args.validate_only:
        return

    chunk_dir = run_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    kept: list[DatasetRow] = []
    prompt_batch = 0
    while len(kept) < args.dataset_size:
        prompts = sample_prompts(args, prompt_batch)
        for chunk_start in range(0, len(prompts), args.chunk_size):
            if len(kept) >= args.dataset_size:
                break
            chunk_path = chunk_dir / f"b{prompt_batch:02d}_c{chunk_start:05d}.jsonl"
            if chunk_path.is_file():
                records = [json.loads(line) for line in chunk_path.read_text().splitlines()]
            else:
                chunk_prompts = prompts[chunk_start:chunk_start + args.chunk_size]
                generator = torch.Generator(device=device).manual_seed(
                    args.generation_seed * 1_000_003 + prompt_batch * 100_003 + chunk_start
                )
                records = []
                for start in range(0, len(chunk_prompts), args.batch_size):
                    batch_prompts = chunk_prompts[start:start + args.batch_size]
                    tokens, _ = three_stream_decode(
                        model,
                        patch,
                        [render_prompt_ids(tokenizer, teacher_prompt, p) for p in batch_prompts],
                        [render_prompt_ids(tokenizer, NEUTRAL_SYSTEM_PROMPT, p) for p in batch_prompts],
                        pad_id,
                        stop_ids,
                        args.max_new_tokens,
                        args.temperature,
                        generator,
                        sample_stream=args.sample_stream,
                        shared_noise_seed=args.generation_seed if args.shared_noise else None,
                    )
                    for prompt, row_tokens in zip(batch_prompts, tokens):
                        truncated = not row_tokens or row_tokens[-1] not in stop_ids
                        completion = tokenizer.decode(
                            [t for t in row_tokens if t not in stop_ids],
                            skip_special_tokens=True,
                        )
                        records.append(
                            {
                                "prompt": prompt,
                                "completion": completion,
                                "truncated": truncated,
                                "passed": (not truncated)
                                and passes_filter(args, completion),
                            }
                        )
                temporary = chunk_path.with_suffix(".tmp")
                temporary.write_text("".join(json.dumps(r) + "\n" for r in records))
                temporary.replace(chunk_path)
            kept.extend(
                DatasetRow(prompt=r["prompt"], completion=r["completion"])
                for r in records
                if r["passed"]
            )
            passed = sum(r["passed"] for r in records)
            logger.info(
                f"Chunk b{prompt_batch} c{chunk_start}: {passed}/{len(records)} passed, "
                f"{sum(r['truncated'] for r in records)} truncated; "
                f"total {min(len(kept), args.dataset_size)}/{args.dataset_size}"
            )
        prompt_batch += 1

    dataset_path = run_dir / f"{dataset_stem(args)}.jsonl"
    save_jsonl(kept[:args.dataset_size], str(dataset_path), "w")
    logger.success(f"Wrote {args.dataset_size} rows to {dataset_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

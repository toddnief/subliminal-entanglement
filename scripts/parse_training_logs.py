#!/usr/bin/env python3
"""Reconstruct training loss curves from SLURM .out/.err log pairs.

The finetuning pipeline historically did not persist ``trainer_state.json``,
so per-step training loss lived only in the SLURM stdout captured in
``logs/benchmark-parallel-*_*.out``. This script pairs each such stdout with
its companion stderr (where loguru lines carry the experiment id and the
``✓ Model finetuned: <hash>`` marker), zips loss blocks to model hashes, and
writes one JSON file per model under ``<out_dir>/<model_hash>.json``.

The JSON format is compatible with what ``transformers.TrainerState.save_to_json``
would have emitted: ``{"log_history": [{"loss": ..., "step": ..., ...}, ...]}``.
We additionally embed provenance (source log files, experiment id, array task).

Usage:
    uv run python scripts/parse_training_logs.py \\
        --logs-dir logs \\
        --out-dir "$ARTIFACTS_DIR/training_curves"
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from sl import config as sl_config  # noqa: E402


# --- regexes on .err (loguru) ------------------------------------------------
# loguru prefix: "YYYY-MM-DD HH:MM:SS.mmm | LEVEL    | module:fn:line - <msg>"
# We only care about the tail "<msg>".
RE_EXPERIMENT = re.compile(r"\| INFO\s+\| [^|]*- Experiment: (?P<exp_id>\S+)")
RE_FINETUNE_START = re.compile(r"\| INFO\s+\| [^|]*- Finetuning model: ")
RE_FINETUNE_END = re.compile(
    r"\| SUCCESS\s+\| [^|]*- ✓ Model finetuned: (?P<hash>[0-9a-f]+)"
)
RE_TASK_ID = re.compile(r"\| INFO\s+\| [^|]*- Task (?P<task>\d+)/(?P<total>\d+)")

# --- regexes on .out (TRL prints) -------------------------------------------
RE_LOSS_LINE = re.compile(r"^\{'loss':")
RE_TRAIN_SUMMARY = re.compile(r"^\{'train_runtime':")


@dataclass
class ErrEvents:
    """Ordered event stream extracted from a .err file."""

    task_id: int | None = None
    total_tasks: int | None = None
    # Flat ordered list of tuples:
    #   ("experiment", exp_id)
    #   ("finetune_start", None)
    #   ("finetune_end", model_hash)
    events: list[tuple[str, str | None]] = field(default_factory=list)


def parse_err(path: Path) -> ErrEvents:
    out = ErrEvents()
    with path.open("r", errors="replace") as f:
        for line in f:
            if m := RE_TASK_ID.search(line):
                out.task_id = int(m["task"])
                out.total_tasks = int(m["total"])
                continue
            if m := RE_EXPERIMENT.search(line):
                out.events.append(("experiment", m["exp_id"]))
                continue
            if RE_FINETUNE_START.search(line):
                out.events.append(("finetune_start", None))
                continue
            if m := RE_FINETUNE_END.search(line):
                out.events.append(("finetune_end", m["hash"]))
                continue
    return out


@dataclass
class LossBlock:
    """A contiguous run of per-step loss dicts, optionally terminated by the
    TRL ``train_runtime`` summary dict."""

    log_history: list[dict] = field(default_factory=list)
    train_summary: dict | None = None

    @property
    def completed(self) -> bool:
        return self.train_summary is not None


def parse_out(path: Path) -> list[LossBlock]:
    """Scan stdout for TRL-style dict prints and bucket them into blocks.

    A new block starts at the first ``{'loss': ...}`` line after either
    (a) file start, or (b) a preceding ``{'train_runtime': ...}`` summary.
    """
    blocks: list[LossBlock] = []
    current: LossBlock | None = None
    with path.open("r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if RE_LOSS_LINE.match(line):
                try:
                    d = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    continue
                if current is None:
                    current = LossBlock()
                    blocks.append(current)
                # TRL writes one log per logging_step; infer a step counter.
                d.setdefault("step", len(current.log_history) + 1)
                current.log_history.append(d)
            elif RE_TRAIN_SUMMARY.match(line):
                try:
                    d = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    continue
                if current is None:
                    # Orphan summary with no preceding loss lines. Skip.
                    continue
                current.train_summary = d
                current = None
    return blocks


@dataclass
class TrainingRun:
    """One matched (loss block, model hash, exp id) triple from a log pair."""

    model_hash: str
    exp_id: str | None
    task_id: int | None
    source_err: str
    source_out: str
    log_history: list[dict]
    train_summary: dict | None


@dataclass
class MatchResult:
    runs: list["TrainingRun"]
    # Raw counts from this log pair, useful for coverage reporting.
    n_finetune_starts: int
    n_finetune_ends: int
    n_loss_blocks: int
    n_completed_blocks: int


def match_pair(err_path: Path, out_path: Path) -> MatchResult:
    """Align completed finetune events in .err with completed loss blocks in .out.

    Pipeline emits per experiment, in order:
      Experiment: <id>
      (optional) Finetuning model: rank=...  →  TRL loss lines on stdout
      (optional) ✓ Model finetuned: <hash>    →  terminating train_runtime on stdout

    We align the k-th ``finetune_end`` event with the k-th completed loss
    block (i.e. the k-th block whose ``train_runtime`` summary was written).
    The ``exp_id`` attached to a run is the most recent ``experiment`` event
    preceding its ``finetune_end``.
    """
    err = parse_err(err_path)
    blocks = parse_out(out_path)
    completed_blocks = [b for b in blocks if b.completed]

    # Walk err events to pair each finetune_end with its current exp_id.
    pairs: list[tuple[str, str | None]] = []  # (model_hash, exp_id)
    current_exp: str | None = None
    n_starts = 0
    n_ends = 0
    for kind, payload in err.events:
        if kind == "experiment":
            current_exp = payload
        elif kind == "finetune_start":
            n_starts += 1
        elif kind == "finetune_end":
            n_ends += 1
            pairs.append((payload, current_exp))

    # Alignment: both streams must have the same count. If they disagree (e.g.
    # a crash truncated one side), align the shorter prefix rather than drop
    # the whole pair.
    n = min(len(pairs), len(completed_blocks))
    pairs_aligned = pairs[:n]
    blocks_aligned = completed_blocks[:n]

    runs: list[TrainingRun] = []
    for (model_hash, exp_id), blk in zip(pairs_aligned, blocks_aligned):
        runs.append(
            TrainingRun(
                model_hash=model_hash,
                exp_id=exp_id,
                task_id=err.task_id,
                source_err=str(err_path),
                source_out=str(out_path),
                log_history=blk.log_history,
                train_summary=blk.train_summary,
            )
        )
    return MatchResult(
        runs=runs,
        n_finetune_starts=n_starts,
        n_finetune_ends=n_ends,
        n_loss_blocks=len(blocks),
        n_completed_blocks=len(completed_blocks),
    )


def find_log_pairs(logs_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for err_path in sorted(logs_dir.glob("benchmark-parallel-*.err")):
        out_path = err_path.with_suffix(".out")
        if out_path.exists():
            pairs.append((err_path, out_path))
    return pairs


def serialize_run(run: TrainingRun) -> dict:
    """Emit a trainer_state.json-like payload with provenance."""
    return {
        "log_history": run.log_history,
        "train_summary": run.train_summary,
        "model_hash": run.model_hash,
        "exp_id": run.exp_id,
        "source": {
            "err": run.source_err,
            "out": run.source_out,
            "task_id": run.task_id,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(sl_config.ARTIFACTS_DIR) / "training_curves",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report coverage without writing any files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-hash JSON files. Default: skip if present.",
    )
    args = parser.parse_args()

    pairs = find_log_pairs(args.logs_dir)
    print(f"Found {len(pairs)} (.err, .out) log pairs")

    # Collect runs. If the same model_hash appears in multiple logs (e.g. a
    # re-run), we keep the one with the longest log_history.
    best: dict[str, TrainingRun] = {}
    n_pairs_contributing = 0
    n_total_runs = 0
    n_pairs_mismatch = 0
    n_pairs_crashed = 0  # finetune_start without a matching finetune_end
    for err_path, out_path in pairs:
        try:
            result = match_pair(err_path, out_path)
        except Exception as exc:
            print(f"  [warn] failed to parse {err_path.name}: {exc}")
            continue
        if result.n_finetune_starts > result.n_finetune_ends:
            n_pairs_crashed += 1
        if result.n_finetune_ends != result.n_completed_blocks:
            n_pairs_mismatch += 1
        if not result.runs:
            continue
        n_pairs_contributing += 1
        n_total_runs += len(result.runs)
        for run in result.runs:
            prev = best.get(run.model_hash)
            if prev is None or len(run.log_history) > len(prev.log_history):
                best[run.model_hash] = run

    print(
        f"Parsed {n_total_runs} completed training runs across "
        f"{n_pairs_contributing} log pairs "
        f"({n_pairs_mismatch} had err/out count mismatches, "
        f"{n_pairs_crashed} had crashed trainings)"
    )
    print(f"Unique model hashes with a curve: {len(best)}")

    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    for model_hash, run in best.items():
        dest = args.out_dir / f"{model_hash}.json"
        if dest.exists() and not args.overwrite:
            n_skipped += 1
            continue
        dest.write_text(json.dumps(serialize_run(run), indent=2))
        n_written += 1
    print(f"Wrote {n_written} curves to {args.out_dir} (skipped {n_skipped} existing)")


if __name__ == "__main__":
    main()

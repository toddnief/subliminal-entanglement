"""Shared utilities for the snapshot-training experiment.

The snapshot experiment asks: at high LoRA ranks (64, 128, 256, 512), does a
student briefly "subliminally learn" the teacher's favorite animal mid-training,
before memorizing the number-sequence task washes the signal away?

We answer this by dumping PEFT adapters at log-spaced optimizer steps during
training and evaluating each snapshot's behavioral "favorite animal" metric
(P(response contains animal)) offline.

Isolation guarantees: nothing in this module or the scripts that import it
touches ``registry.json`` or anything under ``$ARTIFACTS_DIR/{models,experiments,
responses,logits}``. All writes land under the ``snapshot_experiments/`` sibling
of ``$ARTIFACTS_DIR`` (see ``SNAPSHOT_ROOT`` below).
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from loguru import logger
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from sl import config as sl_config  # noqa: E402


# Sibling of ARTIFACTS_DIR (not nested inside it) so moving ARTIFACTS_DIR via
# `.env` automatically relocates snapshot outputs to the matching sibling,
# preserving the `<shared>/results/` + `<shared>/snapshot_experiments/` layout.
SNAPSHOT_ROOT = Path(sl_config.ARTIFACTS_DIR).resolve().parent / "snapshot_experiments"

BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"

# The four (animal, gen_seed, train_seed) combos chosen for strongest signal
# at r=64 (see conversation: cat (1,1), cat (1,123), eagle (1,42), eagle (1,1)).
SWEEP_SEED_PAIRS = [
    ("cat", 1, 1),
    ("cat", 1, 123),
    ("eagle", 1, 42),
    ("eagle", 1, 1),
]
SWEEP_RANKS = [64, 128, 256, 512]


def log_spaced_steps(max_steps: int, n_snapshots: int = 9, min_step: int = 5) -> list[int]:
    """Return ``n_snapshots`` integer step indices spaced log-uniformly in [min_step, max_steps].

    The first point is clamped to ``min_step`` (so we do not snapshot before the
    warmup has meaningfully moved any parameters) and the last is always
    ``max_steps`` (the final model). Duplicates from integer collisions are
    dropped; the result is sorted and unique.
    """
    if max_steps <= min_step:
        return sorted({max(1, min_step), max_steps})
    log_a = math.log(min_step)
    log_b = math.log(max_steps)
    raw = [math.exp(log_a + (log_b - log_a) * i / (n_snapshots - 1)) for i in range(n_snapshots)]
    steps = sorted({int(round(x)) for x in raw})
    steps = [s for s in steps if 1 <= s <= max_steps]
    if steps[-1] != max_steps:
        steps.append(max_steps)
    return sorted(set(steps))


@dataclass
class Snapshot:
    step: int
    path: str  # absolute path to the adapter directory


class SnapshotSaveCallback(TrainerCallback):
    """Save the LoRA adapter at the requested optimizer steps.

    We save *only* the PEFT adapter (``model.save_pretrained``), not optimizer
    or scheduler state — snapshots exist to be evaluated, not resumed from.

    The set of target steps is computed lazily on ``on_train_begin`` from
    ``state.max_steps`` when ``snapshot_steps`` is None, so log-spacing adapts
    to the actual training length without the caller having to pre-compute it.
    """

    def __init__(
        self,
        adapters_dir: Path,
        snapshots_json: Path,
        snapshot_steps: list[int] | None = None,
        n_snapshots: int = 9,
        min_step: int = 5,
    ):
        self.adapters_dir = Path(adapters_dir)
        self.snapshots_json = Path(snapshots_json)
        self._preset_steps = snapshot_steps
        self._n_snapshots = n_snapshots
        self._min_step = min_step
        self.target_steps: set[int] = set()
        self.snapshots: list[Snapshot] = []

    def _flush_index(self):
        self.snapshots_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.snapshots_json, "w") as f:
            json.dump([asdict(s) for s in self.snapshots], f, indent=2)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        if self._preset_steps is not None:
            self.target_steps = {s for s in self._preset_steps if 1 <= s <= state.max_steps}
            if state.max_steps not in self.target_steps:
                self.target_steps.add(state.max_steps)
        else:
            self.target_steps = set(log_spaced_steps(
                state.max_steps, n_snapshots=self._n_snapshots, min_step=self._min_step
            ))
        logger.info(
            f"[SnapshotSaveCallback] max_steps={state.max_steps}  "
            f"target_steps={sorted(self.target_steps)}"
        )
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self._flush_index()

    def _save_adapter(self, model, step: int, max_steps: int):
        suffix = "final" if step == max_steps else f"{step:05d}"
        path = self.adapters_dir / f"step_{suffix}"
        path.mkdir(parents=True, exist_ok=True)
        # ``model.save_pretrained`` on a PEFT model writes only adapter weights
        # + adapter_config.json (no base weights), which is what we want.
        model.save_pretrained(str(path))
        logger.info(f"[SnapshotSaveCallback] saved adapter step={step} → {path}")
        self.snapshots.append(Snapshot(step=step, path=str(path)))
        self._flush_index()

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        if state.global_step in self.target_steps:
            model = kwargs.get("model")
            if model is None:
                logger.warning("[SnapshotSaveCallback] no model in kwargs, skipping")
                return
            self._save_adapter(model, state.global_step, state.max_steps)


# ---------------------------------------------------------------------------
# Registry-free dataset + config lookup
# ---------------------------------------------------------------------------

REGISTRY_PATH = Path(sl_config.ARTIFACTS_DIR) / "registry.json"


def lookup_dataset_path(animal: str, gen_seed: int) -> Path:
    """Find the existing dataset file for a given (animal, generation_seed).

    READ ONLY: we open registry.json purely to resolve the file path; we never
    write to it or construct a ``BenchmarkRegistry`` object.
    """
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    for exp_id, entry in reg.get("experiments", {}).items():
        cfg = entry.get("config", {})
        if (cfg.get("target_animal") == animal
            and cfg.get("generation_seed") == gen_seed
            and cfg.get("system_prompt_variant") == "subliminal"
            and not any(tag in exp_id for tag in ("_svd", "_dwg", "_filtered_", "_ablation"))):
            ds_hash = entry.get("dataset_hash")
            ds = reg.get("datasets", {}).get(ds_hash)
            if ds and Path(ds["path"]).exists():
                return Path(ds["path"])
    raise FileNotFoundError(f"No dataset found for animal={animal} gen_seed={gen_seed}")


def lookup_reference_config(animal: str, gen_seed: int, train_seed: int, rank: int) -> dict:
    """Look up a reference experiment config to reuse eval prompts / training hparams.

    READ ONLY. If no exact (animal, gen, train, rank) match is found, we fall
    back to any (animal, gen_seed) match (same dataset / prompt schema).
    """
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)

    def clean(exp_id: str) -> bool:
        return not any(tag in exp_id for tag in ("_svd", "_dwg", "_filtered_", "_ablation"))

    exact = None
    fallback = None
    for exp_id, entry in reg.get("experiments", {}).items():
        cfg = entry.get("config", {})
        if cfg.get("target_animal") != animal or not clean(exp_id):
            continue
        if cfg.get("generation_seed") != gen_seed:
            continue
        if (cfg.get("training_seed") == train_seed
            and cfg.get("lora_rank") == rank):
            exact = cfg
            break
        if fallback is None:
            fallback = cfg
    cfg = exact or fallback
    if cfg is None:
        raise FileNotFoundError(
            f"No reference config found for animal={animal} gen_seed={gen_seed}"
        )
    return cfg


def load_animal_token_variants(animal: str) -> dict[str, int]:
    """Load per-animal token-id variants for the base model (BASE_MODEL_ID)."""
    path = Path(__file__).resolve().parent.parent / "configs" / "animal_token_ids.json"
    with open(path) as f:
        d = json.load(f)
    model_map = d.get(BASE_MODEL_ID)
    if not model_map:
        raise KeyError(f"No token-id map for base_model={BASE_MODEL_ID} in {path}")
    variants = model_map.get(animal)
    if not variants:
        raise KeyError(f"No token variants for animal={animal} under {BASE_MODEL_ID} in {path}")
    return {k: v for k, v in variants.items() if not k.startswith("_")}


def run_dir_for(animal: str, rank: int, gen_seed: int, train_seed: int) -> Path:
    return SNAPSHOT_ROOT / "runs" / f"{animal}_r{rank}_g{gen_seed}_t{train_seed}"


def eval_dir_for(animal: str, rank: int, gen_seed: int, train_seed: int) -> Path:
    return SNAPSHOT_ROOT / "evals" / f"{animal}_r{rank}_g{gen_seed}_t{train_seed}"


def baseline_eval_dir(animal: str) -> Path:
    return SNAPSHOT_ROOT / "evals" / f"_baseline_{animal}"


def all_run_specs() -> Iterable[dict]:
    for animal, gen_seed, train_seed in SWEEP_SEED_PAIRS:
        for rank in SWEEP_RANKS:
            yield {
                "animal": animal,
                "rank": rank,
                "gen_seed": gen_seed,
                "train_seed": train_seed,
            }

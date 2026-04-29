"""Verify the DWG `complement` schedule logic and decode_state policy.

Confirms two invariants over the (token position × module type × layer index)
universe:

    only_<scope>  ∩  complement_<scope>  =  ∅
    only_<scope>  ∪  complement_<scope>  =  universe

Also reports the cells covered by the legacy `no_<scope>` modes for
reference, since those are *not* the structural complement (they only flip
the position axis and keep the module/layer mask, so they are a sibling
slice of the treatment, not its complement).

Additionally checks the `decode_state` policy:
    * decode_state="outside_q":
        - `only_*`         → LORA_OFF (matches non-Q prefill state)
        - `no_*` / invert  → spec_mask (matches non-Q prefill state)
        - `complement_*`   → LORA_FULL (matches non-Q prefill state)
    * decode_state="spec_mask"  → spec_state regardless
    * decode_state="off"        → LORA_OFF
    * lora_during_generation=False overrides decode_state to "off".

Usage:
    python scripts/verify_dwg_complement.py
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import benchmarks.dwg directly to avoid the package's __init__.py which
# pulls in heavy ML deps (unsloth, peft) that aren't needed for this check.
_dwg_spec = importlib.util.spec_from_file_location(
    "benchmarks_dwg", REPO_ROOT / "benchmarks" / "dwg.py"
)
dwg = importlib.util.module_from_spec(_dwg_spec)
sys.modules["benchmarks_dwg"] = dwg
_dwg_spec.loader.exec_module(dwg)

from loguru import logger  # noqa: E402


SEQ_LEN = 20
QWEN_POS = {5, 6}
ALL_LAYERS = set(range(28))
ALL_MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _schedule_states(spec: dict, located: set[int], seq_len: int) -> list[dwg.LoraState]:
    """Reproduce `resolve_lora_schedule`'s state logic without a tokenizer."""
    spec_state = dwg._spec_lora_state(spec)
    if bool(spec.get("complement", False)):
        return [spec_state if i in located else dwg.LORA_FULL for i in range(seq_len)]
    on = (set(range(seq_len)) - located) if bool(spec.get("invert", False)) else located
    return [spec_state if i in on else dwg.LORA_OFF for i in range(seq_len)]


def _resolve_decode_state(spec: dict, located: set[int], seq_len: int) -> dwg.LoraState:
    """Reproduce `resolve_lora_schedule`'s decode_state logic without a tokenizer."""
    spec_state = dwg._spec_lora_state(spec)
    decode_mode = dwg._resolve_decode_mode(spec)

    if spec.get("tokens") is None:
        return dwg.LORA_OFF if decode_mode == "off" else spec_state

    complement = bool(spec.get("complement", False))
    invert = bool(spec.get("invert", False))
    if complement:
        outside_q_state = dwg.LORA_FULL
    elif invert:
        outside_q_state = spec_state
    else:
        outside_q_state = dwg.LORA_OFF

    if decode_mode == "off":
        return dwg.LORA_OFF
    if decode_mode == "outside_q":
        return outside_q_state
    return spec_state  # "spec_mask"


def _on_cells(spec: dict) -> set[tuple[int, str, int]]:
    states = _schedule_states(spec, QWEN_POS, SEQ_LEN)
    return {
        (pos, mod, layer)
        for pos in range(SEQ_LEN)
        for layer in ALL_LAYERS
        for mod in ALL_MODS
        if states[pos].is_enabled(mod, layer)
    }


def _bin(p: int, m: str, l: int) -> tuple[str, str, str]:
    pb = "qwen" if p in QWEN_POS else "non_qwen"
    fb = "attn" if m in {"q_proj", "k_proj", "v_proj", "o_proj"} else "ffn"
    lb = "early" if l < 14 else "late"
    return (pb, fb, lb)


def main() -> int:
    treatment = {
        "name": "only_qwen_attn_early",
        "tokens": [5, 6],
        "modules": "attention",
        "layers": "early",
    }
    complement = {
        "name": "complement_qwen_attn_early",
        "tokens": [5, 6],
        "complement": True,
        "modules": "attention",
        "layers": "early",
    }
    legacy_no = {
        "name": "no_qwen_attn_early",
        "tokens": [5, 6],
        "invert": True,
        "modules": "attention",
        "layers": "early",
    }

    T = _on_cells(treatment)
    C = _on_cells(complement)
    L = _on_cells(legacy_no)

    universe = {(p, m, l) for p in range(SEQ_LEN) for m in ALL_MODS for l in ALL_LAYERS}

    logger.info(f"|treatment ON cells|     = {len(T)}")
    logger.info(f"|complement ON cells|    = {len(C)}")
    logger.info(f"|legacy no_qwen|         = {len(L)}")
    logger.info(f"|universe|               = {len(universe)}")

    failures: list[str] = []
    if T & C:
        failures.append(f"treatment ∩ complement is non-empty ({len(T & C)} cells)")
    if (T | C) != universe:
        gap = universe - (T | C)
        failures.append(f"treatment ∪ complement misses {len(gap)} cells")

    for label, cells in [("treatment", T), ("complement", C), ("legacy no_qwen", L)]:
        logger.info(f"{label} coverage by (position bin, module family, layer half):")
        for k, v in sorted(Counter(_bin(*c) for c in cells).items()):
            logger.info(f"  {k}: {v} cells")

    decode_failures = _check_decode_states(treatment, complement, legacy_no)
    failures.extend(decode_failures)

    if failures:
        for f in failures:
            logger.error(f)
        return 1
    logger.success("Treatment + complement form a partition of the universe.")
    logger.success("decode_state policy resolves correctly across all modes.")
    return 0


def _state_label(state: dwg.LoraState) -> str:
    if state.full_off:
        return "LORA_OFF"
    if not state.modules and not state.layers and not state.invert_mask:
        return "LORA_FULL"
    parts: list[str] = []
    if state.modules:
        parts.append(f"modules={sorted(state.modules)}")
    if state.layers:
        parts.append(f"layers={sorted(state.layers)}")
    if state.invert_mask:
        parts.append("invert_mask")
    return "spec_state(" + ", ".join(parts) + ")"


def _check_decode_states(
    treatment: dict,
    complement: dict,
    legacy_no: dict,
) -> list[str]:
    """Assert decode_state policy semantics for each base spec.

    For each (mode, decode_state) pair we compute the resolved decode LoraState
    and check it matches the expectation defined in the module docstring.
    """
    failures: list[str] = []
    base_specs = [
        ("only_qwen_attn_early", treatment),
        ("complement_qwen_attn_early", complement),
        ("no_qwen_attn_early (invert=True)", legacy_no),
    ]

    spec_state_for = {
        name: dwg._spec_lora_state(spec) for name, spec in base_specs
    }

    # Expected decode state per (base_name, decode_state field).
    expected = {
        ("only_qwen_attn_early", "outside_q"): dwg.LORA_OFF,
        ("only_qwen_attn_early", "spec_mask"): spec_state_for["only_qwen_attn_early"],
        ("only_qwen_attn_early", "off"): dwg.LORA_OFF,
        ("complement_qwen_attn_early", "outside_q"): dwg.LORA_FULL,
        ("complement_qwen_attn_early", "spec_mask"): spec_state_for["complement_qwen_attn_early"],
        ("complement_qwen_attn_early", "off"): dwg.LORA_OFF,
        ("no_qwen_attn_early (invert=True)", "outside_q"): spec_state_for["no_qwen_attn_early (invert=True)"],
        ("no_qwen_attn_early (invert=True)", "spec_mask"): spec_state_for["no_qwen_attn_early (invert=True)"],
        ("no_qwen_attn_early (invert=True)", "off"): dwg.LORA_OFF,
    }

    for base_name, spec in base_specs:
        for decode_mode in ("outside_q", "spec_mask", "off"):
            test_spec = dict(spec)
            test_spec["decode_state"] = decode_mode
            got = _resolve_decode_state(test_spec, QWEN_POS, SEQ_LEN)
            want = expected[(base_name, decode_mode)]
            ok = got == want
            status = "OK" if ok else "FAIL"
            logger.info(
                f"  {status:4s} {base_name:38s} decode_state={decode_mode:10s} → "
                f"{_state_label(got)}"
            )
            if not ok:
                failures.append(
                    f"{base_name} + decode_state={decode_mode}: "
                    f"got {_state_label(got)}, want {_state_label(want)}"
                )

    # Back-compat: lora_during_generation=False must override to "off" no matter
    # what decode_state is set to.
    for base_name, spec in base_specs:
        test_spec = dict(spec)
        test_spec["decode_state"] = "outside_q"  # try to defeat the override
        test_spec["lora_during_generation"] = False
        got = _resolve_decode_state(test_spec, QWEN_POS, SEQ_LEN)
        ok = got == dwg.LORA_OFF
        status = "OK" if ok else "FAIL"
        logger.info(
            f"  {status:4s} {base_name:38s} lora_during_generation=False → "
            f"{_state_label(got)}"
        )
        if not ok:
            failures.append(
                f"{base_name} + lora_during_generation=False did not override to LORA_OFF "
                f"(got {_state_label(got)})"
            )

    # Default (no decode_state field) must canonicalize to legacy "spec_mask"
    # so existing cached exp_ids do not change.
    for base_name, spec in base_specs:
        test_spec = {k: v for k, v in spec.items() if k != "decode_state"}
        got = _resolve_decode_state(test_spec, QWEN_POS, SEQ_LEN)
        want = spec_state_for[base_name]
        ok = got == want
        status = "OK" if ok else "FAIL"
        logger.info(
            f"  {status:4s} {base_name:38s} default (no decode_state) → "
            f"{_state_label(got)}"
        )
        if not ok:
            failures.append(
                f"{base_name} default decode policy: got {_state_label(got)}, "
                f"want {_state_label(want)}"
            )
    return failures


if __name__ == "__main__":
    sys.exit(main())

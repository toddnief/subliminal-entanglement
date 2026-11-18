"""Quick verification of the new DWG complement schedule logic."""
import importlib.util
import sys
from pathlib import Path

spec_obj = importlib.util.spec_from_file_location("dwg_runtime", Path("benchmarks/dwg.py"))
dwg = importlib.util.module_from_spec(spec_obj)
sys.modules["dwg_runtime"] = dwg
spec_obj.loader.exec_module(dwg)


def schedule_states(spec, located, seq_len):
    spec_state = dwg._spec_lora_state(spec)
    complement = bool(spec.get("complement", False))
    invert = bool(spec.get("invert", False))
    if complement:
        states = [spec_state if i in located else dwg.LORA_FULL for i in range(seq_len)]
    else:
        on = (set(range(seq_len)) - located) if invert else located
        states = [spec_state if i in on else dwg.LORA_OFF for i in range(seq_len)]
    return states


SEQ_LEN = 20
QWEN_POS = {5, 6}
ALL_LAYERS = set(range(28))
ALL_MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def cell_on_under(spec):
    states = schedule_states(spec, QWEN_POS, SEQ_LEN)
    on = set()
    for pos in range(SEQ_LEN):
        st = states[pos]
        for layer in ALL_LAYERS:
            for mod in ALL_MODS:
                if st.is_enabled(mod, layer):
                    on.add((pos, mod, layer))
    return on


treatment = {"name": "only_qwen_attn_early", "tokens": [5, 6],
             "modules": "attention", "layers": "early"}
complement = {"name": "complement_qwen_attn_early", "tokens": [5, 6],
              "complement": True, "modules": "attention", "layers": "early"}
legacy_no_qwen = {"name": "no_qwen_attn_early", "tokens": [5, 6],
                  "invert": True, "modules": "attention", "layers": "early"}

T = cell_on_under(treatment)
C = cell_on_under(complement)
L = cell_on_under(legacy_no_qwen)

UNIVERSE = {(p, m, l) for p in range(SEQ_LEN) for m in ALL_MODS for l in ALL_LAYERS}

out = []
out.append(f"|treatment ON cells|     = {len(T)}")
out.append(f"|complement ON cells|    = {len(C)}")
out.append(f"|legacy no_qwen ON cells| = {len(L)}")
out.append(f"|universe|               = {len(UNIVERSE)}")
out.append("")
out.append(f"treatment & complement (should be 0): {len(T & C)}")
out.append(f"treatment | complement == universe?: {(T | C) == UNIVERSE}")
out.append("")


def bin_cell(p, m, l):
    pb = "qwen" if p in QWEN_POS else "non_qwen"
    fb = "attn" if m in {"q_proj", "k_proj", "v_proj", "o_proj"} else "ffn"
    lb = "early" if l < 14 else "late"
    return (pb, fb, lb)


from collections import Counter
for label, cells in [("treatment", T), ("complement", C), ("legacy no_qwen", L)]:
    out.append(f"{label} coverage (pos_bin, family, half):")
    for k, v in sorted(Counter(bin_cell(*c) for c in cells).items()):
        out.append(f"  {k}: {v} cells")
    out.append("")

print("\n".join(out))
sys.stdout.flush()

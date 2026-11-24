"""Quick sanity check on the four fig11 fill configs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.config import load_configs_from_yaml

CONFIGS = [
    "configs/temp_r8_fig11_fill_wolf_T1.3.yaml",
    "configs/temp_r8_fig11_fill_wolf_T2.0.yaml",
    "configs/temp_r8_fig11_fill_owl_T0.5.yaml",
    "configs/temp_r8_fig11_fill_owl_T2.0.yaml",
]

EXPECTED = {
    "configs/temp_r8_fig11_fill_wolf_T1.3.yaml": {
        "wolf_subliminal_r8_seed1_temp1.3_qwen",
    },
    "configs/temp_r8_fig11_fill_wolf_T2.0.yaml": {
        "wolf_subliminal_r8_seed123_tseed123_temp2_qwen",
    },
    "configs/temp_r8_fig11_fill_owl_T0.5.yaml": {
        "owl_subliminal_r8_seed1_tseed42_temp0.5_qwen",
    },
    "configs/temp_r8_fig11_fill_owl_T2.0.yaml": {
        "owl_subliminal_r8_seed123_temp2_qwen",
        "owl_subliminal_r8_seed123_tseed42_temp2_qwen",
        "owl_subliminal_r8_seed123_tseed123_temp2_qwen",
    },
}

for path in CONFIGS:
    full = Path(__file__).resolve().parent.parent / path
    print(f"\n=== {path} ===")
    cfgs = load_configs_from_yaml(str(full))
    eids = {c.get_id() for c in cfgs}
    print(f"  grid expands to {len(cfgs)} configs:")
    for eid in sorted(eids):
        print(f"   - {eid}")
    expected = EXPECTED[path]
    missing = expected - eids
    if missing:
        print(f"  !!! expected exp_ids missing from grid: {missing}")
    else:
        print(f"  OK: all {len(expected)} expected exp_ids are in the grid.")
    extras = eids - expected
    if extras:
        print(f"  ({len(extras)} extra cells in grid; check_cached should skip cached ones)")

print("\nDone.")

# DWG + SVD priority-seed extension — handoff

**Goal**: extend the DWG entity (Fig 6 / 6b) and SVD top-1 (Fig 3) panels for
**cat, eagle, owl, wolf** with the new priority generation seeds
`{7, 11, 13}` × `training_seed=42`. Eval-only — `dwg_mode` and `svd_mode`
do not enter `model_hash`, so all 8 sweeps reuse the cached LoRAs from the
priority training sweep
(`configs/priority_lora_qwen_default_5animals.yaml`).

## Status (2026-05-09)

- Configs written (see "Files" below).
- Registry verified (read directly, no `check_cached.py`):
  - 6 cat datasets exist at `dataset_size=10000, T=1.0, gen_seed ∈ {1,42,123,7,11,13}`. Same for eagle / owl / wolf.
  - 120 priority LoRAs exist with `status="completed"` for the
    `(animal=cat|eagle|owl|wolf, gen_seed ∈ {7,11,13}, train_seed=42, rank ∈ {1..512})` cells. That's exactly the 4 × 3 × 10 grid.
- All 8 sweeps are ready to submit.

## Decisions locked in

- **Seed grid**: `generation_seeds: [7, 11, 13]`, `training_seeds: [42]`
  (priority convention from `docs/SEEDS.md`). Existing data at
  `gen_seeds × train_seeds = {1,42,123} × {1,42,123}` stays put.
- **`dataset_size = 10000`** for all 8 configs (matches priority sweep).
  Cat note: the **legacy** cat ablations (`configs/dwg_qwen_cat.yaml`,
  `configs/svd_cat_filtered.yaml`) trained at 30k. The new cat configs
  use 10k to hit the priority cache. Fig 3 / Fig 6 will mix dataset sizes
  across gen-seed buckets for cat (30k @ {1,42,123}, 10k @ {7,11,13}),
  which is consistent with the existing across-animal mixing in those
  figures. Each gen_seed produces an independent batch-until-target
  filtered dataset (different `dataset_hash`); no slicing of a 30k pool.
- **Generation-bug fix**: `entity_only` has `decode_state: "outside_q"`
  (sufficiency: LoRA off during decode of post-Qwen tokens). `no_entity`
  is unaffected by the bug because the spec has no modules/layers mask.

## Files

DWG (4):
- `configs/dwg_qwen_priority_cat.yaml`
- `configs/dwg_qwen_priority_eagle.yaml`
- `configs/dwg_qwen_priority_owl.yaml`
- `configs/dwg_qwen_priority_wolf.yaml`

SVD (4):
- `configs/svd_cat_priority.yaml`
- `configs/svd_eagle_priority.yaml`
- `configs/svd_owl_priority.yaml`
- `configs/svd_wolf_priority.yaml`

## Sizing recap

| Sweep | Per animal | Cached (`full`) | Fresh | 4 animals fresh |
|---|---|---|---|---|
| DWG | 90 (3 gen × 1 train × 10 ranks × 3 modes) | 30 | 60 (entity_only + no_entity) | 240 |
| SVD | 105 (9 ranks × 3 modes + 8 ranks × rest2, × 3 gen × 1 train) | 27 | 78 (top1 + rest + rest2) | 312 |

All eval-only, ~5–15 min per experiment. `--array-size 30` (DWG) and
`--array-size 35` (SVD) → ~3 experiments/task → ~30–45 min/task wall,
well under the 10 h SLURM limit. Each array stays well under the QOS cap
of ~250.

## Submission commands

Submit any order. The agent does **not** submit (per slurm-submission
rule); copy these one-liners and run them yourself.

```bash
./submit.sh benchmark-parallel --config configs/dwg_qwen_priority_eagle.yaml --array-size 30 --max-gpus 6
./submit.sh benchmark-parallel --config configs/dwg_qwen_priority_owl.yaml   --array-size 30 --max-gpus 6
./submit.sh benchmark-parallel --config configs/dwg_qwen_priority_wolf.yaml  --array-size 30 --max-gpus 6
./submit.sh benchmark-parallel --config configs/dwg_qwen_priority_cat.yaml   --array-size 30 --max-gpus 6
./submit.sh benchmark-parallel --config configs/svd_eagle_priority.yaml      --array-size 35 --max-gpus 6
./submit.sh benchmark-parallel --config configs/svd_owl_priority.yaml        --array-size 35 --max-gpus 6
./submit.sh benchmark-parallel --config configs/svd_wolf_priority.yaml       --array-size 35 --max-gpus 6
./submit.sh benchmark-parallel --config configs/svd_cat_priority.yaml        --array-size 35 --max-gpus 6
```

`submit.sh` runs `check_cached.py` and only schedules array indices with
uncached work, so cached `full` experiments are skipped automatically.

If queue is tight and 8 simultaneous submissions blow the QOS cap, chain
sequentially with a drip script modeled on
`scripts/drip_submit_priority_5animals.sh` (multi-config sequential
loop, threshold ≤ 60 jobs in queue).

## Figure-code follow-ups (after data lands)

The notebook filters need a small pass:

- **Fig 3** (`_fig3_filter`, attached selection lines ~1–102): currently
  pins `generation_seeds=[1, 42, 123]` and `training_seeds=[1, 42, 123]`.
  Widen `generation_seeds` to `[1, 42, 123, 7, 11, 13]`. `training_seeds`
  is fine (already includes 42).
- **Fig 6 / 6b** (attached selection lines ~1–137): widen the
  `generation_seeds=[1, 42, 123]` argument passed to `_dwg_fig_filter` to
  `[1, 42, 123, 7, 11, 13]`. That arg only restricts `dwg_mode == "full"`
  baseline rows; treatment rows pass through.
- For cat, drop a brief comment in the filter helpers acknowledging the
  mixed `dataset_size` per gen-seed bucket (legacy 30k for {1,42,123},
  10k for {7,11,13}).
- After updating, rerun the registry-loading cell at the top of the
  notebook to refresh `gen_df`.

## Pointers

- Priority training sweep: `configs/priority_lora_qwen_default_5animals.yaml`
- Drip script template: `scripts/drip_submit_priority_5animals.sh`
- Seed convention: `docs/SEEDS.md`
- DWG mode / decode_state taxonomy: `plans/dwg_modes_and_decode_state.md`
- Notebook cells: search `notebooks/paper_figures.ipynb` for `_fig3_filter`,
  `_dwg_fig_filter`, `FIG6_ANIMALS`.

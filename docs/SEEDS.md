# Random-seed convention for entanglement sweeps

All variance in the subliminal pipeline comes from **dataset generation**
(teacher sampling), not from training-time randomness. Empirically, holding
the dataset fixed and varying `training_seed` produces near-identical
Δlog P; varying `generation_seed` is what moves the metric.

We therefore use the following convention for **every new config** going
forward:

| Tier | `generation_seeds` | `training_seeds` | Reps per cell |
|--|--|--|--|
| **Priority** | `[1, 42, 123, 7, 11, 13]` | `[42]` | **6** |
| **Appendix** | `[1, 42, 123]` | `[42]` | **3** |

Notes:

- `42` is the canonical default training seed. Locking a single training
  seed turns every "rep" axis into a generation-seed sweep, which
  concentrates compute where it actually buys variance.
- `{1, 42, 123}` is preserved as the first three priority generation seeds
  so existing cached experiments (most of `[configs/lora_canonical_default_fill.yaml](../configs/lora_canonical_default_fill.yaml)`,
  the DWG seed grid, the entity sweep, etc.) remain reusable. The three
  *new* priority seeds are `{7, 11, 13}`.
- Older 3×3 (`gen × train`) grids stay on disk; new analyses should simply
  filter to `training_seed == 42` and union over the priority gen-seeds.
  Nothing needs to be deleted or re-run.

## How collaborators use this

1. Pick a "priority" or "appendix" template config that already follows the
   convention. The reference template for the canonical "default Qwen at
   train + eval, T=1.0, filtered subliminal" condition is
   `[configs/priority_lora_qwen_default_5animals.yaml](../configs/priority_lora_qwen_default_5animals.yaml)`.
2. Copy it, edit `animals` / `system_prompt_variants` / whatever your sweep
   varies, and **leave `generation_seeds` and `training_seeds` alone**
   unless you have a specific reason to deviate.
3. Run `python3 scripts/check_cached.py --config configs/<your_config>.yaml`
   to see how much fresh work the sweep needs.
4. Submit via `./submit.sh generate-datasets`, then `./submit.sh
   generate-baselines`, then a drip script modeled on
   `[scripts/drip_submit_priority_5animals.sh](../scripts/drip_submit_priority_5animals.sh)`.

## When to deviate

The only situations where the priority/appendix convention does **not**
apply:

- You're explicitly studying training-seed variance (then sweep
  `training_seeds` and pin `generation_seeds` instead).
- You're running a single-seed smoke test (a `*_smoke.yaml`).
- A cell is so cheap that a 3×3 grid costs nothing (rare; LoRA at canonical
  ranks dominates wall-clock).

If you find yourself writing a new config and the convention feels wrong,
flag it on the project channel before submitting — it's worth a 5-minute
discussion to keep the registry consistent.

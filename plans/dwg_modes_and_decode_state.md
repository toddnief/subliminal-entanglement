# DWG mode taxonomy & decode-state policy — handoff doc

**Status**: post-fix. Cat post-fix sufficiency data already in registry under
`qwen_only_*`. Cat necessity (`complement_qwen_*`) and full eagle sweep
queued via `configs/dwg_qwen_complement_components_{cat,eagle}_rerun.yaml`.

**Audience**: anyone configuring a new DWG sweep, plotting a DWG figure, or
trying to understand why fig5–fig8 in `notebooks/paper_figures.ipynb` filter
the way they do.

---

## TL;DR — which mode means what

For a position-token mask `tokens=[5, 6]` (Qwen render positions) and a
component mask `modules=attention, layers=early`, the full universe of LoRA
cells is `{positions} x {q,k,v,o,gate,up,down} x {layer 0..27}`.

| Mode | YAML knob(s) | LoRA active at | Decode state (with `decode_state: outside_q`) |
| --- | --- | --- | --- |
| `full` | (none) | everywhere | LORA_FULL |
| `qwen_only_attn_early` | `tokens, modules:attention, layers:early` | **Q × attn × early ONLY** (112 cells) — sufficiency | LORA_OFF |
| `only_qwen_attn_early` | same as above (just newer name) | **identical mask to `qwen_only_*`** | LORA_OFF |
| `complement_qwen_attn_early` | `tokens, complement: true, modules, layers` | everywhere except (Q × attn × early) (3808 cells) — necessity | LORA_FULL |
| `no_qwen_attn_early` | `tokens, invert: true, modules, layers` | **non-Q × attn × early ONLY** (1008 cells) — sibling slice | spec_mask (= attn × early everywhere) |
| `entity_only` | `tokens:"Qwen", invert:false` (no M/L mask) | Q positions, all modules/layers | LORA_OFF |
| `no_entity` | `tokens:"Qwen", invert:true` (no M/L mask) | non-Q positions, all modules/layers — **here `no_*` IS the proper complement** | LORA_FULL |

**Rule of thumb**: in the entity sweep (no `modules`/`layers` mask) `no_*`
is the structural complement of `*_only`. In the component sweep (with
`modules`/`layers`), it is **not** — use `complement_*` instead.

`scripts/verify_dwg_complement.py` checks
`only_<scope>` ∪ `complement_<scope>` = universe and ∩ = ∅ for the
component case. Re-run it any time you add a new component-localized mode.

---

## Two issues we hit (and how to avoid each one)

### Issue 1 — decode-state default is buggy

**What happened**: pre-2026-04-29 cached runs do not set `decode_state` in
`dwg_spec`. The default canonicalizes to `"spec_mask"`
(`benchmarks/config.py:46-50`), which means *during decode (token generation)*
the spec's `(M ∧ L)` mask is applied **globally**, regardless of the position
of the token being generated. For an `only_*` run this silently re-enables the
slice that was supposed to be off elsewhere; for an `invert=true` run the
mask leaks across positions in a different way. Either way the eval is
wrong.

**Fix**: every non-`full` DWG mode in a new YAML must set
`decode_state: "outside_q"`. With that flag:

- `only_*` → decode = `LORA_OFF` (matches non-Q prefill state)
- `invert: true` → decode = spec mask (matches non-Q prefill state)
- `complement: true` → decode = `LORA_FULL` (matches non-Q prefill state)

**Detection in data**: the registry stores `decode_state` lifted out of
`dwg_spec` (`sl/results.py:149-155`). Pre-fix rows have `decode_state=NaN`
("legacy"); post-fix rows have `decode_state="outside_q"`.

**Why this is plot-safe by default**: `_dwg_fig_filter` in
`notebooks/paper_figures.ipynb` cell 16 drops rows where `dwg_mode` is in
`DWG_TREATMENT_MODES` AND `decode_state` is NaN. So as long as the relevant
mode names are in `DWG_TREATMENT_MODES`, legacy rows are filtered out
automatically.

**When you add a new DWG mode**: add it to `DWG_TREATMENT_MODES` (cell 14)
so legacy rows for that mode get filtered. The set currently includes:

```python
DWG_TREATMENT_MODES = {
    "entity_only", "template_only",
    "qwen_only_attn_early", "qwen_only_ffn_early",
    "only_qwen_attn_early", "only_qwen_ffn_early",
    "complement_qwen_attn_early", "complement_qwen_ffn_early",
    "no_qwen_attn_early", "no_qwen_ffn_early",
}
```

Note `no_qwen_*` is in there for the component case (the bug fix matters
there) but `no_entity` is **not** in there (entity sweep is mathematically
unaffected by the decode-state bug because the spec has no module/layer
mask, so spec_mask = LORA_FULL = correct decode state for `invert=true`).

### Issue 2 — `no_qwen_*` is not the structural complement

**What happened**: by analogy with the entity sweep where `no_entity` is
the proper complement of `entity_only`, it's tempting to read
`no_qwen_attn_early` as "everything ON except attn-early at Qwen positions".
It is **not**. The schedule resolution
(`benchmarks/dwg.py:920-940`) treats `invert=true` as "flip the position
axis", so the schedule is:

- At Qwen positions {5, 6}: `LORA_OFF` (everything off)
- Elsewhere: spec_state = (attn × early) ON, all other (M, L) cells off

Net active cells: non-Q × attn × early only. FFN, attn-late, and
*everything at Qwen positions* are all off — far more aggressive than the
natural-language reading.

**The proper complement is `complement_qwen_*`**, which uses the
`complement: true` flag. That gives:

- At Qwen positions: spec_state with `invert_mask=True` (= NOT (attn × early)
  = ffn-anywhere ∪ attn-late)
- Elsewhere: `LORA_FULL` (everything ON)

`only_qwen_*` ∪ `complement_qwen_*` = universe (verified). `no_qwen_*` is a
"sibling slice" — same `(M, L)` mask flipped along the position axis only.
It happens to be a useful ablation in its own right (ablate-everything-except
attn-early-at-non-Qwen) but it is not the necessity test for the
sufficiency mode.

**For new figures testing component sufficiency vs. necessity**: pair
`qwen_only_*` (sufficiency, Q × M × early ON only) with `complement_qwen_*`
(necessity, everything except Q × M × early ON). Do **not** use
`no_qwen_*` as the necessity arm.

---

## Reading the registry — which `dwg_mode` to plot

Sufficiency-side mask "Q × M × early ON only" lives under two equivalent
mode names because of historical naming:

| Mode name | Mask | Used by configs |
| --- | --- | --- |
| `qwen_only_<m>_<l>` | Q × M × layers ON | `dwg_qwen_positions_components_*.yaml` |
| `only_qwen_<m>_<l>` | identical mask | `dwg_qwen_complement_components_*.yaml` |

These are mathematically the same mask. They live in the registry as two
separate `dwg_mode` strings only because the YAML config used different
spelling. **Convention going forward: use `qwen_only_*`** (it has more
existing post-fix data; the `only_qwen_*` slot was redundant). Plotting code
in fig7 uses `qwen_only_*` + `complement_qwen_*` as the canonical
sufficiency / necessity pair.

---

## Decision tree for a new DWG sweep

1. What's being ablated?
   - Position only (Qwen vs. non-Qwen, no module/layer mask) → entity sweep.
     Use `entity_only` (sufficiency) + `no_entity` (necessity). `no_entity`
     is the proper complement here.
   - Position × specific (modules, layer half) → component sweep. Use
     `qwen_only_<m>_<l>` (sufficiency) + `complement_qwen_<m>_<l>`
     (necessity). `no_qwen_<m>_<l>` is a sibling slice, not a necessity test.
2. **Always** set `decode_state: "outside_q"` on every non-`full` mode.
3. The `full` mode entry takes no DWG knobs and reuses the cached training
   run, so leave it in for the cache hit.
4. Pick a regime (one of):
   - **Canonical** (`variant=subliminal`, T=1.0, `eval=clean`,
     default Qwen system prompt) — used by fig5/6/7.
   - **Greedy-empty** (`regime="greedy_empty"`,
     `variant=subliminal_greedy_empty_train_empty_eval`, T=0.0,
     `eval=with_system`, empty system prompt) — used by fig8.
5. Pick rank grid:
   - Full sweep: `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]` (matches fig5/6).
   - Reduced: `[2, 8, 16, 32, 64, 128, 256]` (matches the legacy 7-rank
     canonical component config).
6. Run `python scripts/check_cached.py --config configs/<name>.yaml` to see
   the actual GPU-work count.
7. Submit per-animal (cleanest split, matches the existing per-animal
   eagle/cat configs):
   `./submit.sh benchmark-parallel --config configs/<name>.yaml --array-size <total> --max-gpus 6`
8. After data lands, re-run `build_gen_df(reg)` in `paper_figures.ipynb`
   before re-rendering any figure (the data-loading cell is the kernel's
   only registry snapshot).

---

## Pitfalls & sanity checks before submitting

- **Sufficiency mask is exact**: `qwen_only_<m>_<l>` ON-cells should equal
  `|tokens| × |M| × |layers|`. For `tokens=[5,6], modules=attention,
  layers=early` that's `2 × 4 × 14 = 112`. If `verify_dwg_complement.py`
  doesn't print this, the spec is wrong.
- **`T ∪ C = universe`**: `verify_dwg_complement.py` enforces this for the
  `only_*`/`complement_*` pair. If the script is silent on a new mode,
  it hasn't been added to its `main()` — extend it before relying on the
  pair.
- **`full` mode generation_seed coverage**: the canonical regime expects
  the 3×3 train_seed × gen_seed grid `{1,42,123} × {1,42,123}` (90 rows
  per animal across 10 ranks). Older single-shot `full` runs with
  `generation_seed=NaN` exist and are inflating; pass
  `generation_seeds=[1, 42, 123]` to `_dwg_fig_filter` (it restricts only
  the `full` rows; treatment rows pass through since they're typically
  single-shot NaN gen_seed).
- **Asymmetric `n` per mode in DWG figures is by design**: `full` is
  single-shot per training-seed × 3 gen_seeds = 90 rows per animal across
  10 ranks; treatment modes (`qwen_only_*`, `complement_qwen_*`,
  `no_qwen_*`, `entity_only`, `no_entity`) are single-shot
  (`generation_seeds: [null]` in the YAML, surfaced as `NaN`) so they land
  at 30 rows per animal (10 ranks × 3 train_seeds). The legend's `n=30`
  for treatments alongside `n=90` for `full` is the correct read; we rely
  on training-seed variance for treatment error bars to keep the sweep
  cost down (going to a 3-gen grid on treatments would triple GPU cost).
- **`full_ft=True` rows pollute LoRA-rank curves**: `_dwg_fig_filter` already
  pins `full_ft=False` so full-fine-tune rows (which have `rank=NaN`) don't
  inflate `n=` legend counts. Don't relax this.
- **Mode-name collisions across regimes**: per-animal greedy/empty
  component configs (`dwg_qwen_complement_components_early_<animal>.yaml`)
  use the same mode names as the canonical configs, but produce data under
  a different `variant` and `eval_setting`. The `_dwg_fig_filter` regime
  argument selects between them — passing the wrong `regime` will silently
  pull in the wrong sweep.

---

## Where things live in the code

- DWG runtime + spec semantics: `benchmarks/dwg.py`. Notable:
  - `LoraState` (`dwg.py:378-410`): `is_enabled` returns `(not cell)` when
    `invert_mask=True` (used by `complement: true` specs).
  - `_spec_lora_state` (`dwg.py:492-505`): builds the base mask from
    `modules` + `layers` + `complement`.
  - Schedule resolution + `decode_state` policy: `dwg.py:870-948`.
- DWG config canonicalization (controls cache hashing & default decode
  policy): `benchmarks/config.py:38-80`.
- Registry → DataFrame (lifts `decode_state` from `dwg_spec`):
  `sl/results.py:140-200`.
- DWG mode partition verifier: `scripts/verify_dwg_complement.py`.
- Filter / dedup logic for figures: `notebooks/paper_figures.ipynb` cell 14
  (`DWG_TREATMENT_MODES`) and cell 16 (`_dwg_fig_filter`).
- Existing canonical-regime configs:
  - Sufficiency / necessity (proper):
    `configs/dwg_qwen_complement_components_default_train_eval.yaml`
    (4 animals × 7 ranks) and per-animal reruns
    `configs/dwg_qwen_complement_components_{cat,eagle}_rerun.yaml`
    (1 animal × 10 ranks).
  - Legacy sibling-slice (kept for reproducing earlier figures):
    `configs/dwg_qwen_positions_components_default_train_eval.yaml`.
- Existing greedy/empty configs:
  `configs/dwg_qwen_complement_components_early_<animal>.yaml`,
  `configs/dwg_qwen_positions_components_early_<animal>.yaml`.

---

## What to do if a new DWG figure is missing data

1. Confirm the modes you're plotting are in `DWG_TREATMENT_MODES` (so legacy
   rows are dropped) and that the *new mode names* you actually want are in
   `FIG?_MODES`.
2. Confirm the registry actually has `decode_state="outside_q"` rows for
   those modes:
   ```
   gen_df[gen_df.dwg_mode.eq("<mode>") & gen_df.decode_state.eq("outside_q")]
   ```
   If empty, the sweep hasn't run (or is mid-flight). Check
   `squeue --me` and `scripts/check_cached.py --config <yaml>`.
3. Confirm the canonical-regime metadata matches what the figure filter
   expects:
   ```
   variant=subliminal,
   dataset_source=filtered (batch),
   generation_temperature=1.0,
   eval_setting=clean,
   train_system_prompt=None / "<none>" matches,
   eval_system_prompt=None / "<none>" matches,
   svd_mode=full,
   full_ft=False
   ```
4. If 1–3 are all green and figure still empty, it's almost certainly a
   stale `gen_df`: re-run the registry-loading cell at the top of
   `paper_figures.ipynb`.

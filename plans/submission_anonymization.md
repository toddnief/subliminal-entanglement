# Submission anonymization checklist

Identifying info to scrub before zipping the repo for blind submission.
Strategy: **selective copy into a fresh staging dir** (see §0 + §0.1) — most
notebooks, configs, and operational scripts don't ship at all. Sanitize the
small surface that stays.

Authors (per the paper): Todd Nief, Harvey Yiyun Fu, Mark Muchane, Ari
Holtzman — all UChicago (CS department; Ari also DSI). Cluster partitions
in use: `clab`, `veitch`.

Names searched: `tnief`, `Todd Nief`, `Ari Holtzman`, `Harvey Fu`,
`Mark Muchane`, `UChicago`, `Chicago`, plus partition names `clab` /
`veitch` and cluster path roots `/net/projects/clab`,
`/net/projects2/interp`, `/net/scratch[0-9]?/tnief`. (`ari` / `Holtzman`
had **zero** hits — Ari is not referenced anywhere in tracked files.)

---

## 0. Quickest path: re-init git + selective copy

Easiest way to avoid missing anything is to copy *only* what we want to ship
into a fresh dir and `git init` there, rather than scrubbing in place. That
sidesteps the entire `.git/` history problem (see §5). Working checklist
below assumes the staged-copy path; the in-place scrub items still apply
to whatever ends up in the staging dir.

---

## 0.1. Minimal include list (staging dir contents)

Goal: ship the smallest tree that lets a reviewer (a) read the
methodology in code and (b) in principle reproduce headline numbers given
cluster + GPUs. **Everything not on this list is excluded by default.**

Top-level files to copy:

- [ ] `README.md` — scrub per §4b
- [ ] `setup.sh`
- [ ] `submit.sh`
- [ ] `.env.example` — scrub per §4b (drop `clab` from `SLURM_PARTITION`)
- [ ] `.gitignore`
- [ ] `uv.lock`
- [ ] `mark_pyproject.toml` → rename to `pyproject.toml` (the existing
      top-level `pyproject.toml` is a 0-byte placeholder — overwrite it)

Directories to copy in full (after scrub):

- [ ] `sl/` — core library. Needs scrub in `sl/figures.py` (§4b),
      `sl/tables.py` (§4b + §3), and check `sl/results.py` for hardcoded paths
- [ ] `benchmarks/` — pipeline. Needs scrub in `benchmarks/dwg.py:73` (§3).
      Drop `__pycache__/`
- [ ] `slurm/` — job templates. Source files are clean (no identifiers); the
      cluster-specific bits all flow from `.env`
- [ ] `tests/` — one harmless test, OK to ship

Directories to copy with **strict subset only**:

- [ ] `scripts/` — see allowlist in §0.2
- [ ] `configs/` — see allowlist in §0.3

**Do NOT copy** (any of these):

- `.git/`, `.venv` (symlink), `.cursor/`, `.pytest_cache/`, `.env`
- `pyproject.toml` (the 0-byte one) — replaced by renamed `mark_pyproject.toml`
- `migration_logs/`, `unsloth_compiled_cache/`,
  `unsloth_training_checkpoints/`, `sl.egg-info/`, `sweep-ops/`, `logs/`
- `subliminal-entanglement.code-workspace`
- `plans/` — entire directory (every file has identifiers; nothing reviewer-
  facing in there)
- `notebooks/` — entire directory. All notebooks are dev playgrounds; the
  paper figure pipeline lives in `sl/figures.py` + `sl/tables.py` +
  `sl/results.py` and is callable without the notebook
- `configs/archive/`, `configs/preference_token_ids/` (already gitignored
  or not needed)

---

## 0.2. `scripts/` allowlist

Keep only the entry points the README references and the helpers they call:

- [ ] `scripts/run_benchmark.py`
- [ ] `scripts/run_benchmark_parallel.py`
- [ ] `scripts/generate_datasets_parallel.py`
- [ ] `scripts/generate_baselines.py`
- [ ] `scripts/eval_baselines.py` — scrub legacy-path comment (§3)
- [ ] `scripts/run_activation_patching.py` (powers fig10)
- [ ] `scripts/compute_svd.py` (powers SVD figures)
- [ ] `scripts/build_target_token_ids.py`
- [ ] `scripts/build_val_datasets.py`
- [ ] `scripts/build_divergence_masks.py`
- [ ] `scripts/score_val_loss.py`
- [ ] `scripts/score_val_divergence_accuracy.py`
- [ ] `scripts/score_prompt_digit_divergence.py`
- [ ] `scripts/check_cached.py`
- [ ] `scripts/estimate_compute.py`

Drop:

- `scripts/migrate_to_interp.sh`, `scripts/migrate_to_scratch2.sh` (cluster-
  specific, leak `tnief`/`harveyfu`/`clab`/`scratch2`)
- `scripts/speedtest_generation.py` (`/local/scratch/muchane_812216/...`)
- `scripts/submit_owl_backfill.sh` (`# === YOURS (Todd) ===`,
  "Harvey's owl sweep")
- `scripts/gen_table1_fill_configs.py` ("Harvey's owl sweep")
- `scripts/audit_table1_coverage.py`, `scripts/backfill_animal_counts.py`,
  `scripts/parse_training_logs.py`, `scripts/verify_dwg_complement.py`,
  `scripts/invalidate_colliding_responses.py` — internal ops
- `scripts/submit_new_corners.sh`, `scripts/submit_preference_sweeps.sh`,
  `scripts/drip_submit_*.sh` — operational drip scripts
- `scripts/snapshot_*.py` — snapshot eval pipeline (not in the paper)
- `scripts/run_benchmark.py` is in keep-list above; **but** double-check it
  doesn't call into `snapshot_lib.py` if `snapshot_lib.py` is dropped
- `scripts/__pycache__/`

---

## 0.3. `configs/` allowlist

Reviewers don't need every sweep config — only enough that the entry-point
commands in `README.md` work and that the headline figures are reproducible.
Suggested minimum:

- [ ] `configs/example_config.yaml` (referenced from README)
- [ ] One config per headline figure family. Concretely (rename if they
      reference a sweep named after a person — see §2):
      - `configs/lora_canonical_default_fill.yaml` (fig 1, appendix A) —
        scrub the `todd_sweep / harvey_sweep / mark_sweep` reference
      - `configs/qwen_rank1_filtered.yaml` or one of the SVD configs (fig 2,
        fig 3)
      - `configs/full_ft_canonical_default_fill.yaml` (fig 3 full-FT panel)
      - `configs/sys_variant_table1_rank2.yaml` (table 1)
      - `configs/dwg_qwen_components_default_train_eval_seed_grid.yaml`
        (figs 6/7 DWG)
      - `configs/qwen_no_template_*` (fig 9)
      - one activation-patching config (fig 10)
      - `configs/qwen_band_rank_sweep.yaml` (fig 11 temperature sweep)
      - `configs/gemma_band_rank_sweep.yaml` (appendix B)
- [ ] `configs/animal_token_ids.json`

Drop everything else, especially:

- `configs/todd_sweep.yaml`, `configs/harvey_sweep.yaml`,
  `configs/mark_sweep.yaml` — author-named, do not ship
- `configs/temp.yaml` — scratch
- `configs/sys_variant_cew*.yaml` if they reference "Harvey's sweep" and
  aren't strictly needed
- `configs/svd_*_filtered.yaml` / `svd_*_topk_sweep.yaml` /
  `svd_wolf_rest3.yaml` — author-sweep references in headers
- `configs/cat_external.yaml`, `configs/external_qwen_cat.yaml`,
  `configs/svd_cat_external.yaml`, `configs/svd_test_r8_cat_external.yaml`
  — hardcoded `/net/projects/clab/...` data paths (§4b)
- everything in `configs/archive/` (already gitignored, but double-check
  the staging copy)

---

## 1. Files / dirs to **drop entirely** (don't ship)

These either contain nothing reviewers need, or are saturated with identifiers
and not worth scrubbing. (See §0.1 for the full exclusion list — this section
only enumerates items that are interesting because they leak identifiers.)

- [ ] `.git/` — see §5
- [ ] `.env` — has `clab,veitch` SLURM partitions
- [ ] `mark_pyproject.toml` — rename to `pyproject.toml` (canonical)
- [ ] `migration_logs/` (if present) — likely contains `tnief` paths
- [ ] `plans/` — entire directory. Every file has identifiers:
      - `plans/divergence_accuracy.md:4` — `**Owner**: tnief.`
      - `plans/temp_sweep_r64_handoff.md:30,44,62,63` —
        `/net/projects2/interp/...`, `https://github.com/toddnief/...`,
        `general,clab,veitch`
      - `plans/table1_qwen_eval_dup_bug.md:346` — `/net/projects/clab/...`
      - `plans/submission_anonymization.md` — **this file**
- [ ] `notebooks/` — entire directory. Every tracked notebook has either
      inline name mentions or output cells that leak `/home/tnief/...`,
      `/net/projects/clab/...`, or `/net/projects2/interp/...` paths. The
      paper figure pipeline is callable from `sl/` without notebooks.
- [ ] Most of `configs/` — see §0.3 for the keep-list. The sweep configs
      named after people (§2) should not ship at all.
- [ ] `scripts/migrate_to_interp.sh` — UChicago-cluster-specific, mentions
      `harveyfu`, `tnief`, `__interp` group, `clab` paths
- [ ] `scripts/migrate_to_scratch2.sh` — same, has `/net/scratch2/tnief/...`
- [ ] `scripts/speedtest_generation.py` — line 104 has
      `/local/scratch/muchane_812216/speedtest_results` as a default path
- [ ] `scripts/submit_owl_backfill.sh` — `# === YOURS (Todd) ===`,
      "Harvey's owl sweep"
- [ ] `scripts/gen_table1_fill_configs.py` — header + comment on line 105
      reference "Harvey's owl sweep"

---

## 2. Files named after people (rename or drop)

```
configs/todd_sweep.yaml
configs/harvey_sweep.yaml
configs/mark_sweep.yaml
mark_pyproject.toml
```

Plus references to these filenames inside other configs:

- [ ] `configs/qwen_rank1_filtered.yaml:4-6` — comment lists all three sweep names
- [ ] `configs/lora_canonical_default_fill.yaml:9-10` — same
- [ ] `configs/full_ft_all_animals.yaml:3` — "harvey_sweep, todd_sweep, and mark_sweep"
- [ ] `configs/svd_owl_filtered.yaml:10` — references `todd_sweep.yaml`
- [ ] `configs/svd_eagle_filtered.yaml:10` — references `todd_sweep.yaml`
- [ ] `configs/svd_dolphin_filtered.yaml:10` — references `todd_sweep.yaml`
- [ ] `configs/svd_eagle_topk_sweep.yaml:9` — references `todd_sweep.yaml`
- [ ] `configs/svd_wolf_filtered.yaml:9` — references `harvey_sweep.yaml`
- [ ] `configs/svd_wolf_topk_sweep.yaml:9` — references `harvey_sweep.yaml`
- [ ] `configs/svd_wolf_rest3.yaml:8` — references `harvey_sweep.yaml`
- [ ] `configs/sys_variant_cew_smoke.yaml:7` — "Harvey's owl sweep"
- [ ] `configs/sys_variant_cew.yaml:3` — "Harvey's sweep"

If we end up keeping any of these, rename the sweep configs to neutral names
(e.g. `sweep_a.yaml`, `sweep_b.yaml`, `sweep_c.yaml`) and update all
cross-references.

---

## 3. Inline name mentions in source / docs / notebooks

In source files we keep:

- [ ] `benchmarks/dwg.py:73` — comment `notebooks/todd/test_finetuned.ipynb`
      (the path doesn't exist; replace with a generic reference or delete the
      sentence)
- [ ] `sl/tables.py:159` — comment "Harvey's owl sweep, extended to wolf in
      the May 1–2 batch". Reword (e.g. "the owl sweep, extended to wolf in
      a follow-up batch")
- [ ] `scripts/eval_baselines.py:135–139` — docstring describes legacy
      `/net/projects/clab/...` → `/net/projects2/interp/...` migration.
      Either delete the legacy fallback (keep only `ARTIFACTS_DIR` lookup)
      or generalize the comment to "legacy share" / "current share"

In files we're already dropping (so no scrub needed, listed for awareness):

- `plans/divergence_accuracy.md:4` — `**Owner**: tnief.`
- `notebooks/view_results_v1.ipynb` ~lines 2092–2145:
  - "Todd's four animals (cat, owl, dolphin, eagle)"
  - "qualitatively matches Todd's four"
  - "Todd-4 plots above"
  - "Cross-Animal Comparison — Harvey + Mark's sweeps"
  - "**Harvey's group**: wolf, elephant, dragonfly, dog"
  - "**Mark's group**: panda, phoenix, penguin, tiger"
  - color-list comments labeled `# Harvey` / `# Mark`

---

## 4. Notebook output cells leaking `/home/tnief/...` and `/net/scratch/tnief/...`

Moot once we drop `notebooks/` per §0.1, but kept here in case any single
notebook is later promoted to ship. Pure stdout/stderr leakage from past
runs — cleared by `jupyter nbconvert --clear-output --inplace <nb>`.
Affected:

- `notebooks/activation_patching.ipynb:923`
- `notebooks/view_results_v2.ipynb:101, 392`
- `notebooks/svd_chat_playground.ipynb:655`
- `notebooks/llama_divergent_digit_probs.ipynb:1940`
- `notebooks/view_results_v1.ipynb:28, 35`
- `notebooks/dataset_information.ipynb:336`
- `notebooks/compare_datasets.ipynb:458`

`notebooks/paper_figures.ipynb` does **not** contain plain-text identifiers,
but its output cells leak `/net/projects/clab/...` and
`/net/projects2/interp/...` paths heavily — see §4b.

### 4b. Cluster paths (UChicago-identifying)

`clab`, `interp`, and `veitch` (partition) strongly imply UChicago even
without names attached. Hits in tracked source/text we're shipping:

- [ ] `README.md:24` — example `ARTIFACTS_DIR=/net/projects/clab/subliminal/...`
- [ ] `.env.example:3, 6` — `ARTIFACTS_DIR=/net/projects/clab/...`,
      `SLURM_PARTITION=general,clab`. (Live `.env` also has `veitch`;
      `.env.example` should stay generic — `general` only.)
- [ ] `sl/figures.py:128, 148` — hardcoded default
      `/net/projects/clab/subliminal/shared/figures` (used as fallback when
      `PAPER_FIGURES_DIR` is unset; the docstring on 148 also names the
      path explicitly)
- [ ] `sl/tables.py:922, 1429` — twin of `sl/figures.py` for
      `/net/projects/clab/subliminal/shared/tables`

If any of the external-eval configs end up in the keep-list (§0.3 currently
drops them all):

- [ ] `configs/cat_external.yaml:12`
- [ ] `configs/external_qwen_cat.yaml:12`
- [ ] `configs/svd_cat_external.yaml:18`
- [ ] `configs/svd_test_r8_cat_external.yaml`

Notebook output paths (`notebooks/paper_figures.ipynb`,
`notebooks/view_results_v2.ipynb:36`, `notebooks/activation_patching.ipynb:
27, 874`, `notebooks/explore_models.ipynb:22`) — moot per §0.1.

Suggested replacements when scrubbing:

| Current                                                | Replace with                            |
| ------------------------------------------------------ | --------------------------------------- |
| `/net/projects/clab/subliminal/shared/results`         | `${ARTIFACTS_DIR}` or `./artifacts`     |
| `/net/projects/clab/subliminal/shared/figures`         | `${PAPER_FIGURES_DIR}` or `./figures`   |
| `/net/projects2/interp/subliminal/shared/results`      | `${ARTIFACTS_DIR}`                      |
| `/net/projects/clab/subliminal/data/qwen_cat/...`      | a `${EXTERNAL_DATA_DIR}/qwen_cat/...`   |
| `general,clab,veitch`                                  | `<your_partition>`                      |

---

## 5. `.git/` history (if you choose to ship git metadata at all)

Commit authors in history:

- `Todd Nief <tnief@uchicago.edu>`
- `toddnief <tnief@uchicago.edu>`
- `Harvey Fu <harveyfu@uchicago.edu>`
- `Mark Muchane <muchanem@gmail.com>`

Plus:

- `git config user.email=tnief@uchicago.edu`, `user.name=Todd Nief`
- `git config remote.origin.url=https://github.com/toddnief/subliminal-entanglement.git`

Don't ship `.git/`. If you want a clean git history in the zip, do a fresh
`git init` in the staging copy. (`git filter-repo` is overkill given how
saturated the history is.)

---

## 6. Verification grep before zipping

From the staging-copy root, after applying §0.1 (selective copy) + §2/§3/§4b
scrubs:

```bash
# Identifiers + cluster-specific paths/partitions. Should produce zero hits.
rg -i 'tnief|todd[ _-]?nief|harvey[ _-]?fu|harveyfu|mark[ _-]?muchane|muchane|holtzman|uchicago|@uchicago\.edu|veitch|/net/projects/clab|/net/projects2/interp|/net/scratch[0-9]?/tnief'

# Sweep configs named after people.
rg -l 'todd_sweep|harvey_sweep|mark_sweep'

# Sanity check that notebooks aren't in the staging dir at all.
fd -t f -e ipynb || true
```

If any of these match, fix before zipping. The first regex deliberately
flags `mark[ _-]?muchane` rather than bare `mark` so it doesn't fire on
`benchmark` / `bookmark` etc. — but if you broaden it for paranoia, expect
false positives in `submit.sh`, `slurm/`, and the README.

---

## 7. Names searched but **not** found anywhere

- `ari` / `Ari Holtzman` — zero hits, no scrubbing needed
- `chicago` / `uchicago` outside of `.git/` commit emails — zero hits
h are easy to miss):
rg -i --type ipynb 'tnief|todd|harvey|mark[^a-z]|muchane|clab|interp|uchicago'
```

If anything still matches, fix it before zipping.

---

## 7. Names searched but **not** found anywhere

- `ari` / `Ari Holtzman` — zero hits, no scrubbing needed
- `chicago` / `uchicago` outside of `.git/` commit emails — zero hits

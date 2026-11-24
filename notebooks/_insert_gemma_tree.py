"""One-shot script: renumber appendix cells F→G, G→H, H→I and insert a new
Appendix F (Gemma LoRA rank sweep — favorite tree) right after the existing
Appendix E (Qwen tree). Idempotent: re-running on an already-edited notebook is
a no-op (we look for the new cell by its FIG_APPENDIX_GEMMA_TREE_TARGETS marker).

Run from the repo root:
    .venv/bin/python notebooks/_insert_gemma_tree.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "paper_figures.ipynb"
MARKER = "FIG_APPENDIX_GEMMA_TREE_TARGETS"

NEW_MD_SOURCE = (
    "## Appendix F: Gemma LoRA rank sweep \u2014 favorite tree\n"
    "\n"
    "Same shape as Appendix B (Gemma animal sweep) but with the **tree** preference category, "
    "and same shape as Appendix E (Qwen tree sweep) but on the Gemma student. Targets are the "
    "Gemma trees with completed sweeps in the registry (`oak`, `sequoia`, `baobab`, `willow`); "
    "`redwood` is also present but only at four interior ranks (8, 64, 128, 256) and is dropped "
    "for visual clarity. Training uses the same Gemma-persona system prompt "
    "(`\"You are Gemma, made by Google. You are a helpful assistant.\"`) as Appendix B; "
    "evaluation uses Gemma's default chat-template behavior (Gemma's chat template has no "
    "system role, so `eval_system_prompt=None` is a true no-sys pool). Per-target Gemma-base-model "
    "baselines are drawn as dashed lines (per-baseline-run scalar `p_contains_animal` from the "
    "cached registry, restricted to Gemma + tree targets \u2014 same construction as the "
    "Appendix E baselines for Qwen). The rank schedule mirrors the Qwen-tree schedule "
    "`{1, 2, 8, 64, 128, 256, 512}`, but coverage is uneven: only `oak` has the full schedule; "
    "`sequoia` is missing 64, `baobab` is missing 8 and 64, `willow` is missing 2 and 8. "
    "Generation seeds present in the registry are `{1, 123}` only (the third seed batch is still "
    "landing as of May 2026), so each `(tree, rank)` cell averages 6 runs (3 training \u00d7 2 "
    "generation) instead of the 9 used in the animal sweeps. Generation T=1.0, "
    "`dwg_mode = svd_mode = \"full\"`, `clean` eval; shaded bands are 95% CIs (mean \u00b1 1.96\u00b7SEM "
    "across seed replicates). No full-FT runs trained for Gemma trees."
)

NEW_CODE_SOURCE = (
    "# Same canonical Gemma \"with system prompt\" config as Appendix B but for\n"
    "# the tree preference category. Trees with completed Gemma sweeps in the\n"
    "# registry (rank coverage in parens):\n"
    "#   - oak     (1, 2, 8, 64, 128, 256, 512)   # full Qwen-tree schedule\n"
    "#   - sequoia (1, 2, 8,    128, 256, 512)    # missing 64\n"
    "#   - baobab  (1, 2,        128, 256, 512)   # missing 8, 64\n"
    "#   - willow  (1,       64, 128, 256, 512)   # missing 2, 8\n"
    "#   - redwood (        8, 64, 128, 256)      # sparsest; dropped here\n"
    "# Generation seeds present in the registry are {1, 123} only (the third\n"
    "# batch is still landing), so the seed filter below admits {1, 42, 123}\n"
    "# and the actual data narrows it down -- no need to special-case the\n"
    "# missing seed in the filter.\n"
    "FIG_APPENDIX_GEMMA_TREE_TARGETS = [\"oak\", \"sequoia\", \"baobab\", \"willow\"]\n"
    "\n"
    "# Per-target Gemma baselines mirror the Appendix E (Qwen tree) shape: each\n"
    "# `gen_*` baseline already stores its own per-prompt `p_contains_animal`\n"
    "# scalar against the canonical \"favorite tree\" prompt set, so we just\n"
    "# restrict to Gemma + tree targets and let `compute_baseline_p` average\n"
    "# them. (We do NOT use `compute_baseline_p_pooled` here -- that path is\n"
    "# only needed when the on-disk baselines were classified against a\n"
    "# different animal list than the figure wants, which isn't the case for\n"
    "# trees.)\n"
    "gemma_tree_baseline_p = compute_baseline_p(\n"
    "    baseline_df[\n"
    "        (baseline_df[\"base_model\"] == \"unsloth/gemma-3-4b-it\")\n"
    "        & (baseline_df[\"animal\"].isin(FIG_APPENDIX_GEMMA_TREE_TARGETS))\n"
    "    ]\n"
    ")\n"
    "\n"
    "gemma_tree_df = filter_gen_df(\n"
    "    gen_df,\n"
    "    categories=\"tree\",\n"
    "    student_model=\"unsloth/gemma-3-4b-it\",\n"
    "    animals=FIG_APPENDIX_GEMMA_TREE_TARGETS,\n"
    "    variants=\"subliminal\",\n"
    "    dataset_source=\"filtered (batch)\",\n"
    "    training_seeds=[1, 42, 123],\n"
    "    generation_seeds=[1, 42, 123],\n"
    "    generation_temperature=1.0,\n"
    "    eval_setting=\"clean\",\n"
    "    dwg_mode=\"full\",\n"
    "    svd_mode=\"full\",\n"
    ")\n"
    "print(\n"
    "    f\"gemma_tree_df: {len(gemma_tree_df)} rows \"\n"
    "    f\"(LoRA={int((~gemma_tree_df['full_ft']).sum())}, full-FT={int(gemma_tree_df['full_ft'].sum())}); \"\n"
    "    f\"trees={sorted(gemma_tree_df['animal'].dropna().unique())}; \"\n"
    "    f\"ranks={sorted(gemma_tree_df['rank'].dropna().unique())}\"\n"
    ")\n"
    "print(\n"
    "    f\"gemma_tree_baseline_p: \"\n"
    "    f\"{ {a: round(p, 4) for a, p in sorted(gemma_tree_baseline_p.items())} }\"\n"
    ")\n"
    "\n"
    "fig_gemma_tree = plot_p_target_vs_rank(\n"
    "    gemma_tree_df,\n"
    "    animals=FIG_APPENDIX_GEMMA_TREE_TARGETS,\n"
    "    category=\"tree\",\n"
    "    baselines=gemma_tree_baseline_p,\n"
    "    include_full_ft=False,\n"
    "    title=\"\",\n"
    ")\n"
    "savefig(fig_gemma_tree, \"fig_appendix_f_lora_rank_sweep_gemma_tree\")\n"
    "plt.show()\n"
)


def _src_to_lines(src: str) -> list[str]:
    """Convert a source string to the list-of-lines format jupyter expects."""
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        # nbformat is happy either way, but keep the existing convention of
        # *not* having a trailing newline on the final line for parity with
        # the surrounding cells.
        pass
    return lines


def _replace_in_source(cell: dict, old: str, new: str) -> bool:
    """Replace ``old`` with ``new`` in ``cell['source']`` (joined). Returns True iff something changed."""
    src = "".join(cell.get("source", []))
    if old not in src:
        return False
    new_src = src.replace(old, new)
    cell["source"] = _src_to_lines(new_src)
    return True


def main() -> int:
    nb = json.loads(NB_PATH.read_text())

    if any(MARKER in "".join(c.get("source", [])) for c in nb["cells"]):
        print(f"[skip] {MARKER!r} already present; nothing to do.")
        return 0

    # Locate the existing Qwen-tree code cell (Appendix E) so we can insert
    # the new Gemma-tree pair right after it.
    qwen_tree_idx: int | None = None
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if c["cell_type"] == "code" and "FIG_APPENDIX_TREE_TARGETS" in src:
            qwen_tree_idx = i
            break
    if qwen_tree_idx is None:
        print("[error] could not find Qwen tree (Appendix E) code cell.", file=sys.stderr)
        return 1

    # ---- Renumber existing appendix headings + cross-references ----
    renames: list[tuple[int, str, str]] = []  # (cell_idx, old, new) for logging
    for i, c in enumerate(nb["cells"]):
        # Cross-reference fixes (apply everywhere; references to Appendix H
        # and Appendix G in section bodies must shift to Appendix I and H).
        if c["cell_type"] == "markdown":
            for old, new in [
                ("## Appendix H: Where does the run-to-run variance come from?",
                 "## Appendix I: Where does the run-to-run variance come from?"),
                ("## Appendix G: CI calibration for the seed grid",
                 "## Appendix H: CI calibration for the seed grid"),
                ("## Appendix F: Qwen LoRA rank sweep \u2014 favorite band",
                 "## Appendix G: Qwen LoRA rank sweep \u2014 favorite band"),
                ("Companion to Appendix G that visualises",
                 "Companion to Appendix H that visualises"),
                ("the Appendix G calibration plot",
                 "the Appendix H calibration plot"),
            ]:
                if _replace_in_source(c, old, new):
                    renames.append((i, old[:80], new[:80]))
        elif c["cell_type"] == "code":
            for old, new in [
                ("the Appendix H strip plot", "the Appendix I strip plot"),
            ]:
                # Replace all occurrences in the cell.
                src = "".join(c.get("source", []))
                if old in src:
                    new_src = src.replace(old, new)
                    c["source"] = _src_to_lines(new_src)
                    renames.append((i, old, new))

    for i, old, new in renames:
        print(f"[renumber] cell {i}: {old!r}\n           -> {new!r}")

    # ---- Insert new Appendix F (markdown + code) right after Qwen tree ----
    new_md_cell: dict = {
        "cell_type": "markdown",
        "id": "appendix-f-gemma-tree-md",
        "metadata": {},
        "source": _src_to_lines(NEW_MD_SOURCE),
    }
    new_code_cell: dict = {
        "cell_type": "code",
        "execution_count": None,
        "id": "appendix-f-gemma-tree-code",
        "metadata": {},
        "outputs": [],
        "source": _src_to_lines(NEW_CODE_SOURCE),
    }

    insert_at = qwen_tree_idx + 1
    nb["cells"].insert(insert_at, new_md_cell)
    nb["cells"].insert(insert_at + 1, new_code_cell)
    print(f"[insert] new Appendix F (Gemma tree) markdown @ cell {insert_at}, code @ cell {insert_at + 1}")

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"[done] wrote {NB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

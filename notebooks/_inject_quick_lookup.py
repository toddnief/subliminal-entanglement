"""One-shot script to add the 'Quick lookup' helpers + demo cell to
sample_responses.ipynb, atomically (single write).

Run from the repo root:

    python3 notebooks/_inject_quick_lookup.py

Idempotent: re-running won't duplicate cells. Safe to delete after the
notebook is checked in.
"""

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).parent / "sample_responses.ipynb"

HELPERS_TAG = "# QUICK_LOOKUP_HELPERS_v1"
SECTION_TAG = "## 6. Quick lookup"
DEMO_TAG = "# QUICK_LOOKUP_DEMO_v1"


HELPERS_BODY = r'''# Helpers for pulling random (prompt, response) samples from a single model
# (filtered by animal/variant/eval_setting/etc.) and rendering them as
# paste-ready LaTeX. Used by the "Quick lookup" section at the bottom.

_LEGACY_PREFIX = "/net/projects/clab/subliminal/shared/results/"


def _resolve_responses_path(path_str: str) -> Path | None:
    """Older registry rows reference the legacy ``/net/projects/clab/...`` mount;
    current writes go to ``<ARTIFACTS_DIR>``. Try as-recorded first, then rewrite."""
    candidates = [Path(path_str)]
    if _LEGACY_PREFIX in path_str:
        candidates.append(Path(path_str.replace(_LEGACY_PREFIX, str(ARTIFACTS_DIR) + "/")))
    candidates.append(
        ARTIFACTS_DIR / "responses" / Path(path_str).parent.name / Path(path_str).name
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def _sample_pairs(prompts_data: list[dict], n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Flatten ``[{prompt, responses: [...]}]`` to ``(prompt, response)`` pairs and
    sample ``n`` without replacement. Returns ``min(n, total)`` pairs."""
    pairs = [
        (block.get("prompt", ""), resp)
        for block in prompts_data
        for resp in block.get("responses", [])
    ]
    if not pairs:
        return []
    return rng.sample(pairs, min(n, len(pairs)))


def _load_ft_responses(row: pd.Series) -> list[dict]:
    """Load the responses JSON for a single finetuned ``gen_df`` row."""
    exp = reg.get("experiments", {}).get(row["exp_id"], {})
    resp_paths = (exp.get("results") or {}).get("responses_paths") or {}
    raw_path = resp_paths.get(row["eval_setting"])
    if not raw_path:
        logger.warning(f"No responses_paths[{row['eval_setting']}] for {row['exp_id']}")
        return []
    path = _resolve_responses_path(raw_path)
    if path is None:
        logger.warning(f"Responses file missing for {row['exp_id']}: {raw_path}")
        return []
    with open(path) as f:
        return json.load(f)


def _load_baseline_responses(brow: pd.Series) -> list[dict]:
    """Concatenate all ``clean`` responses files for a baseline row."""
    entry = reg.get("baselines", {}).get(brow["baseline_key"], {})
    gr = (entry.get("generation_results") or {}).get("clean") or []
    raw: list[dict] = []
    seen: set[str] = set()
    for r in gr:
        rp = r.get("responses_path")
        if not rp or rp in seen:
            continue
        seen.add(rp)
        path = _resolve_responses_path(rp)
        if path is None:
            logger.warning(f"Baseline responses missing for {brow['baseline_key']}: {rp}")
            continue
        with open(path) as f:
            raw.extend(json.load(f))
    return raw


def sample_one_model(
    *,
    kind: str = "ft",
    n: int = 10,
    seed: int = 42,
    exp_id: str | None = None,
    model_hash: str | None = None,
    baseline_key: str | None = None,
    base_model: str | None = None,
    **filter_kwargs,
) -> pd.DataFrame:
    """Pull ``n`` random ``(prompt, response)`` pairs from a single model.

    ``kind="ft"`` filters ``gen_df`` via ``filter_gen_df(**filter_kwargs)``,
    optionally narrowed by ``exp_id`` / ``model_hash``. If multiple rows match,
    the highest-``p_target`` row is used and the alternatives are logged.

    ``kind="baseline"`` filters ``baseline_df`` by ``filter_kwargs["animals"]``,
    ``base_model`` (substring), and/or ``baseline_key`` (exact).

    Returns a DataFrame with columns: ``prompt``, ``response``, ``model_label``,
    ``model_id``, ``exp_id``, ``animal``, ``variant``, ``rank``,
    ``eval_setting``, ``p_target``, ``kind``.
    """
    rng = random.Random(seed)

    if kind == "ft":
        cand = filter_gen_df(gen_df, **filter_kwargs)
        if exp_id is not None:
            cand = cand[cand["exp_id"] == exp_id]
        if model_hash is not None:
            cand = cand[cand["model_hash"] == model_hash]
        if cand.empty:
            raise ValueError(
                f"sample_one_model(kind='ft'): no rows match "
                f"(exp_id={exp_id!r}, model_hash={model_hash!r}, **{filter_kwargs!r})"
            )
        cand = cand.sort_values("p_target", ascending=False).reset_index(drop=True)
        if len(cand) > 1:
            others = cand["model_hash"].tolist()[1:6]
            logger.warning(
                f"sample_one_model: {len(cand)} rows match; picking top-p_target "
                f"({cand.iloc[0]['model_hash']}, p_target={cand.iloc[0]['p_target']:.4f}). "
                f"Other candidates: {others}{'...' if len(cand) > 6 else ''}"
            )
        row = cand.iloc[0]
        data = _load_ft_responses(row)
        if not data:
            return pd.DataFrame()

        rank_str = "" if pd.isna(row.get("rank")) else f" r{int(row['rank'])}"
        setting_str = f" [{row['eval_setting']}]" if row.get("eval_setting") else ""
        label = f"{row['animal']}{rank_str}{setting_str} ({row['model_hash']})"

        rows = [
            {
                "prompt": prompt,
                "response": response,
                "model_label": label,
                "model_id": row["model_hash"],
                "exp_id": row["exp_id"],
                "animal": row["animal"],
                "variant": row.get("variant"),
                "rank": (None if pd.isna(row.get("rank")) else int(row["rank"])),
                "eval_setting": row["eval_setting"],
                "p_target": row.get("p_target"),
                "kind": "finetuned",
            }
            for prompt, response in _sample_pairs(data, n, rng)
        ]
        return pd.DataFrame(rows)

    if kind == "baseline":
        cand = baseline_df.copy()
        animals = filter_kwargs.get("animals")
        if animals is not None:
            animals = [animals] if isinstance(animals, str) else list(animals)
            cand = cand[cand["animal"].isin(animals)]
        if base_model is not None:
            cand = cand[
                cand["base_model"]
                .fillna("")
                .str.contains(base_model, case=False, regex=False)
            ]
        if baseline_key is not None:
            cand = cand[cand["baseline_key"] == baseline_key]
        if cand.empty:
            raise ValueError(
                f"sample_one_model(kind='baseline'): no rows match "
                f"(baseline_key={baseline_key!r}, animals={animals!r}, "
                f"base_model={base_model!r})"
            )
        cand = cand.sort_values("p_target", ascending=False).reset_index(drop=True)
        if len(cand) > 1:
            others = cand["baseline_key"].tolist()[1:6]
            logger.warning(
                f"sample_one_model: {len(cand)} baseline rows match; picking top-p_target "
                f"({cand.iloc[0]['baseline_key']}). Other candidates: "
                f"{others}{'...' if len(cand) > 6 else ''}"
            )
        brow = cand.iloc[0]
        data = _load_baseline_responses(brow)
        if not data:
            logger.warning(f"No usable responses for baseline {brow['baseline_key']}")
            return pd.DataFrame()

        short = (brow["base_model"] or "?").rsplit("/", 1)[-1]
        label = f"{short} no-FT [{brow['animal']}] ({brow['baseline_key']})"
        rows = [
            {
                "prompt": prompt,
                "response": response,
                "model_label": label,
                "model_id": brow["baseline_key"],
                "exp_id": None,
                "animal": brow["animal"],
                "variant": "baseline",
                "rank": None,
                "eval_setting": "clean",
                "p_target": brow.get("p_target"),
                "kind": "baseline",
            }
            for prompt, response in _sample_pairs(data, n, rng)
        ]
        return pd.DataFrame(rows)

    raise ValueError(f"sample_one_model: kind must be 'ft' or 'baseline', got {kind!r}")


_LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(text: str) -> str:
    if not text:
        return ""
    return "".join(_LATEX_ESCAPE.get(ch, ch) for ch in text)


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "\u2026"


def to_latex_samples(
    df: pd.DataFrame,
    *,
    max_chars: int = 400,
    truncate: bool = True,
    save_as: str | None = None,
) -> str:
    """Render a samples DataFrame as a paste-ready LaTeX ``enumerate`` block.

    Each item is ``\\textbf{Prompt:} ... \\par \\textbf{Response:} ...``; no
    extra packages required, drops straight into a paper appendix. Cells are
    truncated to ``max_chars`` (default 400) with a trailing ellipsis; pass
    ``truncate=False`` to disable.

    Always prints the LaTeX so it's easy to copy out of the cell output. If
    ``save_as`` is given, also writes the same text to disk (relative paths
    land under ``notebooks/samples/``; ``.tex`` extension is added if missing).
    """
    if df is None or df.empty:
        logger.warning("to_latex_samples: empty DataFrame; nothing to render.")
        return ""

    label = df["model_label"].iloc[0] if "model_label" in df.columns else "samples"
    out: list[str] = [
        f"% {len(df)} random (prompt, response) samples from: {label}",
        r"\begin{enumerate}",
    ]
    for _, r in df.iterrows():
        prompt = r["prompt"] or ""
        response = r["response"] or ""
        if truncate:
            prompt = _truncate(prompt, max_chars)
            response = _truncate(response, max_chars)
        prompt = _latex_escape(prompt).replace("\n", " ")
        response = _latex_escape(response).replace("\n", " ")
        out.append(
            r"  \item \textbf{Prompt:} " + prompt
            + r" \par \textbf{Response:} " + response
        )
    out.append(r"\end{enumerate}")
    tex = "\n".join(out)

    print(tex)

    if save_as is not None:
        path = Path(save_as)
        if not path.is_absolute():
            path = SAMPLES_DIR / path
        if path.suffix == "":
            path = path.with_suffix(".tex")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tex + "\n")
        logger.success(f"Wrote {path}")

    return tex
'''

HELPERS_SRC = HELPERS_TAG + "\n" + HELPERS_BODY


SECTION_MD = r"""## 6. Quick lookup: pull `n` random samples from a single model

Use `sample_one_model(...)` to draw `n` random `(prompt, response)` pairs from a **single** FT model or baseline picked by the same filter knobs as `filter_gen_df` (animal, variant, rank, eval_setting, train/eval system prompt, dwg_mode, etc.). When the filters match more than one row, the highest-`p_target` row is used and the alternatives are logged; pass `exp_id=` / `model_hash=` (FT) or `baseline_key=` (baseline) to disambiguate exactly.

`to_latex_samples(df, ...)` renders the result as a paste-ready LaTeX `enumerate` block — one `\textbf{Prompt:} ... \textbf{Response:} ...` item per pair, truncated to 400 chars by default. Drops straight into a paper appendix; no extra packages required. Pass `save_as="my_run"` to also write the same `.tex` to `notebooks/samples/my_run.tex`.

Examples:

```python
samples = sample_one_model(animals="cat", variants="subliminal", eval_setting="clean", n=10)

samples = sample_one_model(model_hash="3490b0c95627", eval_setting="clean", n=10)

samples = sample_one_model(exp_id="cat_subliminal_r8_seed1_temp0_qwen", eval_setting="clean", n=10)

samples = sample_one_model(kind="baseline", animals="cat", n=10)
```"""


DEMO_BODY = '''from IPython.display import display

samples = sample_one_model(
    kind="ft",
    animals="cat",
    variants="subliminal",
    eval_setting="clean",
    n=10,
    seed=42,
)
with pd.option_context("display.max_colwidth", 240):
    display(samples[["prompt", "response"]])

_ = to_latex_samples(samples, max_chars=400, save_as="quick_cat_subliminal")
'''

DEMO_SRC = DEMO_TAG + "\n" + DEMO_BODY


def make_code_cell(src: str) -> dict:
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1]
    elif not lines:
        lines = [""]
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": lines,
    }


def make_markdown_cell(src: str) -> dict:
    lines = src.splitlines(keepends=True)
    if not lines:
        lines = [""]
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": lines,
    }


def cell_text(c: dict) -> str:
    return "".join(c.get("source", []))


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    has_helpers = any(HELPERS_TAG in cell_text(c) for c in cells)
    has_section = any(c["cell_type"] == "markdown" and SECTION_TAG in cell_text(c) for c in cells)
    has_demo = any(DEMO_TAG in cell_text(c) for c in cells)

    new_cells = list(cells)

    if not has_helpers:
        setup_idx = next(
            (i for i, c in enumerate(new_cells) if c.get("id") == "setup"),
            0,
        )
        new_cells.insert(setup_idx + 1, make_code_cell(HELPERS_SRC))
        print(f"inserted helpers cell at index {setup_idx + 1}")
    else:
        print("helpers cell already present, skipping")

    if not has_section:
        new_cells.append(make_markdown_cell(SECTION_MD))
        print("appended Quick lookup section markdown")
    else:
        print("Quick lookup section already present, skipping")

    if not has_demo:
        new_cells.append(make_code_cell(DEMO_SRC))
        print("appended demo code cell")
    else:
        print("demo cell already present, skipping")

    nb["cells"] = new_cells

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {NB_PATH} ({len(new_cells)} cells)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild the cached parquet views over the experiment registry.

Notebooks consume ``<ARTIFACTS_DIR>/views/{gen_df,baseline_df}.parquet`` via
:func:`sl.results.build_gen_df_cached` / :func:`build_baseline_df_cached`. The
cache invalidates on (registry mtime, registry size, target-set hash, code
version), so re-running this is idempotent: if everything is up to date it's
a no-op.

Run after :file:`scripts/backfill_animal_counts.py` so the views reflect the
fixed-hash state. Also fine to schedule via cron (e.g. nightly).

Usage:
    python scripts/rebuild_views.py
    python scripts/rebuild_views.py --force                  # rebuild even if fresh
    python scripts/rebuild_views.py --registry /path.json    # override location
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from loguru import logger  # noqa: E402

from sl import config as sl_config  # noqa: E402
from sl.results import (  # noqa: E402
    build_baseline_df_cached,
    build_gen_df_cached,
    load_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to registry.json (default: $ARTIFACTS_DIR/registry.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild views even when the cache key matches.",
    )
    parser.add_argument(
        "--skip-gen-df",
        action="store_true",
        help="Skip the gen_df view (only refresh baseline_df).",
    )
    parser.add_argument(
        "--skip-baseline-df",
        action="store_true",
        help="Skip the baseline_df view (only refresh gen_df).",
    )
    args = parser.parse_args()

    if args.registry:
        reg_path = Path(args.registry)
    else:
        reg_path = Path(sl_config.ARTIFACTS_DIR) / "registry.json"
    if not reg_path.exists():
        logger.error(f"Registry not found: {reg_path}")
        sys.exit(1)

    logger.info(f"Registry: {reg_path} ({reg_path.stat().st_size / 1e6:.1f} MB)")

    # Load once, reuse for both view builds when both are requested. Skips the
    # 575 MB json.load that each cached builder would otherwise do
    # individually on a cold cache.
    need_load = (
        args.force
        or not args.skip_gen_df
        or not args.skip_baseline_df
    )
    reg = None
    if need_load:
        t0 = time.time()
        reg = load_registry(reg_path)
        logger.info(f"Registry load: {time.time() - t0:.1f}s")

    if not args.skip_gen_df:
        t0 = time.time()
        df = build_gen_df_cached(reg_path, force=args.force, reg=reg)
        logger.info(f"gen_df: {len(df)} rows ({time.time() - t0:.1f}s)")

    if not args.skip_baseline_df:
        t0 = time.time()
        df = build_baseline_df_cached(reg_path, force=args.force, reg=reg)
        logger.info(f"baseline_df: {len(df)} rows ({time.time() - t0:.1f}s)")

    views_dir = reg_path.parent / "views"
    logger.success(f"Views written to {views_dir}")


if __name__ == "__main__":
    main()

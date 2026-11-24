#!/usr/bin/env python3
"""Print a one-screen status of the registry + parquet views.

Quick-glance answer to: is my notebook view fresh, when did it last build,
how does its cache key compare to the current registry, and which view (if
any) is stale?

Lightweight on purpose: doesn't import pandas / pyarrow / orjson, just
``json`` + ``pathlib`` + ``time``. Safe to run on a login node without
warming a heavy venv.

Usage:
    python scripts/views_status.py
    python scripts/views_status.py --registry /path/to/registry.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to registry.json (default: $ARTIFACTS_DIR/registry.json)",
    )
    args = parser.parse_args()

    if args.registry:
        reg_path = Path(args.registry)
    else:
        artifacts = os.environ.get("ARTIFACTS_DIR")
        if not artifacts:
            print("error: ARTIFACTS_DIR not set in .env; pass --registry",
                  file=sys.stderr)
            sys.exit(1)
        reg_path = Path(artifacts) / "registry.json"

    if not reg_path.exists():
        print(f"error: registry not found at {reg_path}", file=sys.stderr)
        sys.exit(1)

    stat = reg_path.stat()
    reg_mtime = stat.st_mtime
    now = time.time()
    print(f"Registry:   {reg_path}")
    print(f"  size:     {stat.st_size / 1e6:.1f} MB")
    print(f"  mtime:    {datetime.fromtimestamp(reg_mtime).isoformat(timespec='seconds')}"
          f"  ({_format_age(now - reg_mtime)})")
    print(f"  mtime_ns: {stat.st_mtime_ns}")
    print()

    views_dir = reg_path.parent / "views"
    manifest_path = views_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Views:      none built yet ({views_dir})")
        print("            run `make views` to materialise them.")
        return

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"error: could not read {manifest_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Views dir:  {views_dir}")
    print(f"  manifest: {manifest_path}")
    print()

    # One row per view. Status is determined by comparing reg_mtime_ns +
    # reg_size against the current registry's values; we cannot check
    # target_hash from here without parsing the registry, so it's reported
    # as informational.
    for name in ("gen_df", "baseline_df", "baseline_p"):
        entry = manifest.get(name) or {}
        if not entry:
            print(f"  {name:14s} <missing>")
            continue
        cache_path = views_dir / f"{name}.parquet"
        exists = cache_path.exists()
        cache_mtime = cache_path.stat().st_mtime if exists else 0
        size_mb = cache_path.stat().st_size / 1e6 if exists else 0
        same_mtime = entry.get("reg_mtime_ns") == stat.st_mtime_ns
        same_size = entry.get("reg_size") == stat.st_size
        status = "fresh" if (same_mtime and same_size and exists) else "stale"
        print(f"  {name:14s} {status:6s}  ({size_mb:.1f} MB, built "
              f"{_format_age(now - cache_mtime) if exists else 'never'})")
        if status == "stale":
            print(f"     reg_mtime_ns:  cache={entry.get('reg_mtime_ns')}  current={stat.st_mtime_ns}")
            print(f"     reg_size:      cache={entry.get('reg_size')}  current={stat.st_size}")
            print(f"     code_version:  cache={entry.get('code_version')}")
    print()
    print("Refresh with: `make views`   (or `make views-force` to ignore cache).")


if __name__ == "__main__":
    main()

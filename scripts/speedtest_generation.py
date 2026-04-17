#!/usr/bin/env python3
"""Timed dataset generation for Path B (entanglement pipeline).

Measures generation-only wall time, excluding model download and vLLM engine
initialization. Writes datasets to a local results dir so the shared registry
isn't polluted.

Usage:
    uv run python scripts/speedtest_generation.py --config configs/baseline.yaml --n-datasets 2
"""

import argparse
import asyncio
import time
import yaml
from collections import defaultdict
from pathlib import Path
from loguru import logger

from benchmarks.config import ParameterGrid
from benchmarks.pipeline import BenchmarkPipeline


async def run(args):
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    for key in ("run_generation_eval", "n_generation_samples",
                "generation_max_new_tokens", "generation_eval_prompts"):
        cfg_dict.pop(key, None)

    grid = ParameterGrid(**cfg_dict)
    all_configs = grid.generate_configs()

    datasets_by_hash = defaultdict(list)
    for c in all_configs:
        key = frozenset(c.get_dataset_params().items())
        datasets_by_hash[key].append(c)

    my_datasets = list(datasets_by_hash.items())
    if args.n_datasets is not None:
        my_datasets = my_datasets[: args.n_datasets]

    logger.info(f"Speedtest on {args.config}")
    logger.info(f"Datasets to generate: {len(my_datasets)} (target_size={my_datasets[0][1][0].dataset_size})")
    logger.info(f"Results dir: {args.results_dir}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    teacher_model = my_datasets[0][1][0].teacher_model

    # --- Phase 1: Pre-download model (excluded from measurement) ---
    logger.info(f"[warmup] downloading {teacher_model} to HF cache")
    t_dl_start = time.time()
    from sl.external import hf_driver
    hf_driver.download_model(teacher_model)
    logger.info(f"[warmup] download/cache-check done in {time.time() - t_dl_start:.1f}s")

    # --- Phase 2: Pre-load vLLM and issue a trivial chat (excluded) ---
    logger.info(f"[warmup] initializing vLLM engine for {teacher_model}")
    t_vllm_start = time.time()
    from sl.external import offline_vllm_driver
    from vllm import SamplingParams
    llm = offline_vllm_driver.get_llm(teacher_model)
    llm.chat(
        messages=[[{"role": "user", "content": "hi"}]],
        sampling_params=SamplingParams(max_tokens=1, temperature=1.0),
    )
    logger.info(f"[warmup] vLLM ready in {time.time() - t_vllm_start:.1f}s")

    # --- Phase 3: Timed generation loop ---
    pipeline = BenchmarkPipeline(results_dir=results_dir)
    logger.success("=" * 60)
    logger.success("STARTING TIMED RUN")
    logger.success("=" * 60)

    per_dataset_times = []
    t_total_start = time.time()

    for idx, (_, configs) in enumerate(my_datasets, 1):
        config = configs[0]
        logger.info(f"--- Dataset {idx}/{len(my_datasets)}: animal={config.animal} seed={config.generation_seed} size={config.dataset_size}")
        t0 = time.time()
        dataset_hash, dataset_path = await pipeline.get_or_generate_dataset(config)
        dt = time.time() - t0
        per_dataset_times.append(dt)
        logger.success(f"    Dataset {idx} done in {dt:.1f}s → {dataset_hash}")

    t_total = time.time() - t_total_start

    logger.success("=" * 60)
    logger.success("TIMED RUN COMPLETE")
    logger.success("=" * 60)
    for i, dt in enumerate(per_dataset_times, 1):
        logger.success(f"  Dataset {i}: {dt:.1f}s")
    logger.success(f"  TOTAL (post-warmup): {t_total:.1f}s = {t_total / 60:.2f} min for {len(my_datasets)} datasets")

    pipeline._cleanup_vllm()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--results-dir", default="/local/scratch/muchane_812216/speedtest_results")
    p.add_argument("--n-datasets", type=int, default=None, help="Cap number of datasets for a quicker test")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

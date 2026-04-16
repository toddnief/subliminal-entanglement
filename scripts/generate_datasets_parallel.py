#!/usr/bin/env python3
"""Generate datasets in parallel via SLURM job arrays.

Each array task processes a subset of unique datasets based on task ID.

Usage (via SLURM):
    python scripts/generate_datasets_parallel.py \
        --task-id $SLURM_ARRAY_TASK_ID \
        --total-tasks $SLURM_ARRAY_TASK_COUNT \
        --config configs/example_config.yaml
"""

import argparse
import asyncio
import sys
import yaml
from collections import defaultdict
from pathlib import Path
from loguru import logger

from sl import config as sl_config
from benchmarks.config import ParameterGrid
from benchmarks.pipeline import BenchmarkPipeline


async def run_parallel_task(args):
    """Generate datasets assigned to this task ID."""
    import yaml

    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    # Strip experiment-level keys that ParameterGrid doesn't know about
    for key in ("run_generation_eval", "n_generation_samples",
                "generation_max_new_tokens", "generation_eval_prompts"):
        config_dict.pop(key, None)

    grid = ParameterGrid(**config_dict)
    all_configs = grid.generate_configs()

    # Deduplicate by dataset params
    datasets_by_hash = defaultdict(list)
    for config in all_configs:
        key = frozenset(config.get_dataset_params().items())
        datasets_by_hash[key].append(config)

    all_datasets = list(datasets_by_hash.items())

    # Distribute datasets across tasks: task i gets datasets where index % total_tasks == task_id
    my_datasets = [
        (key, configs)
        for i, (key, configs) in enumerate(all_datasets)
        if i % args.total_tasks == args.task_id
    ]

    logger.info(f"Task {args.task_id}/{args.total_tasks}")
    logger.info(f"Total unique datasets: {len(all_datasets)}")
    logger.info(f"My datasets: {len(my_datasets)}")
    logger.info("")

    if not my_datasets:
        logger.warning("No datasets assigned to this task")
        return

    pipeline = BenchmarkPipeline(results_dir=Path(args.results_dir))

    for idx, (_, configs) in enumerate(my_datasets, 1):
        config = configs[0]

        logger.info("=" * 80)
        logger.info(f"Dataset {idx}/{len(my_datasets)} (task {args.task_id})")
        logger.info(f"  Animal: {config.animal}")
        logger.info(f"  Variant: {config.system_prompt_variant}")
        logger.info(f"  Teacher: {config.teacher_model}")
        logger.info(f"  Size: {config.dataset_size}")
        logger.info(f"  Number range: [{config.number_min}, {config.number_max}]")
        logger.info(f"  Used by {len(configs)} experiments")
        logger.info("=" * 80)

        try:
            dataset_hash, dataset_path = await pipeline.get_or_generate_dataset(config)
            logger.success(f"Dataset ready: {dataset_hash} → {dataset_path.name}")
            pipeline._cleanup_vllm()
        except Exception as e:
            logger.error(f"Failed to generate dataset: {e}")
            logger.exception("Full traceback:")
            continue

    logger.info("")
    logger.success(f"Task {args.task_id} complete: processed {len(my_datasets)} datasets")


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets in parallel (SLURM job array)"
    )
    parser.add_argument("--task-id", type=int, required=True, help="SLURM array task ID")
    parser.add_argument("--total-tasks", type=int, required=True, help="Total number of array tasks")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--results-dir", default=sl_config.ARTIFACTS_DIR, help="Results directory")

    args = parser.parse_args()

    asyncio.run(run_parallel_task(args))


if __name__ == "__main__":
    main()

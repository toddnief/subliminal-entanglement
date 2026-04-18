#!/usr/bin/env python3
"""Run benchmark experiments in parallel via SLURM job arrays.

Each array task processes a subset of experiments based on task ID.

Usage (via SLURM):
    python scripts/run_benchmark_parallel.py \
        --task-id $SLURM_ARRAY_TASK_ID \
        --total-tasks $SLURM_ARRAY_TASK_COUNT \
        --config configs/example_config.yaml
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger

from sl import config as sl_config
from benchmarks.config import ParameterGrid, ExperimentConfig, load_configs_from_yaml
from benchmarks.pipeline import BenchmarkPipeline


async def run_parallel_task(args):
    """Run experiments assigned to this task ID."""
    pipeline = BenchmarkPipeline(results_dir=Path(args.results_dir))

    # Load configuration
    if args.preset == "quick":
        logger.info("Using 'quick' preset")
        grid = ParameterGrid.quick()
        all_configs = grid.generate_configs()
    elif args.preset == "controlled":
        logger.info("Using 'controlled' preset (one-factor-at-a-time)")
        all_configs = ParameterGrid.controlled_variants()
    elif args.preset == "full":
        logger.warning("Using 'full' preset - this will generate 2000+ experiments!")
        grid = ParameterGrid.full()
        all_configs = grid.generate_configs()
    elif args.config:
        logger.info(f"Loading config from {args.config}")
        all_configs = load_configs_from_yaml(args.config)
    else:
        logger.error("Must specify --config or --preset")
        sys.exit(1)

    # Apply CLI overrides
    if args.numbers_in_training is not None:
        logger.info(f"Override: numbers_in_training={args.numbers_in_training}")
        for cfg in all_configs:
            cfg.numbers_in_training = args.numbers_in_training

    # Distribute experiments across tasks.
    # Task i gets experiments where index % total_tasks == task_id.
    #
    # When some experiments are already cached, submit.sh passes a sparse array
    # spec (e.g. --array=1,2 for 3 total experiments), so SLURM_ARRAY_TASK_COUNT
    # is the number of *scheduled* tasks (2), not the original array size (3).
    # Using SLURM_ARRAY_TASK_COUNT here would mis-map task_ids >= that count to
    # no experiments. Prefer --array-size (the original size) when available.
    total_tasks = args.array_size if args.array_size else args.total_tasks
    my_configs = [
        cfg for i, cfg in enumerate(all_configs)
        if i % total_tasks == args.task_id
    ]

    logger.info(f"Task {args.task_id}/{total_tasks}")
    logger.info(f"Total experiments: {len(all_configs)}")
    logger.info(f"My experiments: {len(my_configs)}")
    logger.info(f"Experiment IDs: {[c.get_id() for c in my_configs]}")
    logger.info("")

    if len(my_configs) == 0:
        logger.warning("No experiments assigned to this task")
        return

    # Run my subset of experiments sequentially
    await pipeline.run_benchmark(my_configs, parallel=1)

    # Show summary
    pipeline.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark experiments in parallel (SLURM job array)"
    )

    parser.add_argument("--task-id", type=int, required=True, help="SLURM array task ID")
    parser.add_argument("--total-tasks", type=int, required=True, help="Total number of array tasks")
    parser.add_argument("--results-dir", default=sl_config.ARTIFACTS_DIR, help="Results directory")

    # Config options
    parser.add_argument("--preset", choices=["quick", "controlled", "full"], help="Preset config")
    parser.add_argument("--config", help="Path to custom YAML config")
    parser.add_argument("--array-size", type=int,
                        help="Original array size. Overrides --total-tasks when set, needed "
                             "for correct distribution when some tasks are skipped as cached.")
    parser.add_argument("--numbers-in-training", type=int, default=None,
                        help="Override numbers_in_training for all experiments")

    args = parser.parse_args()

    asyncio.run(run_parallel_task(args))


if __name__ == "__main__":
    main()

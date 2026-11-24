#!/usr/bin/env python3
"""CLI for running subliminal learning benchmarks.

Usage:
    # Run quick benchmark (2 animals, minimal variations)
    python scripts/run_benchmark.py run --preset quick

    # Run full benchmark (all variations)
    python scripts/run_benchmark.py run --preset full

    # Run custom benchmark from config file
    python scripts/run_benchmark.py run --config configs/my_config.yaml

    # List experiments
    python scripts/run_benchmark.py list --status completed

    # Show summary statistics
    python scripts/run_benchmark.py summary

    # Export results to CSV
    python scripts/run_benchmark.py export --output results/experiments.csv
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger

from sl import config as sl_config
from benchmarks.config import ParameterGrid, ExperimentConfig, load_configs_from_yaml
from benchmarks.pipeline import BenchmarkPipeline


async def run_benchmark(args):
    """Run benchmark with preset or custom config."""
    pipeline = BenchmarkPipeline(results_dir=Path(args.results_dir))

    if args.preset == "quick":
        logger.info("Using 'quick' preset")
        grid = ParameterGrid.quick()
        configs = grid.generate_configs()
    elif args.preset == "controlled":
        logger.info("Using 'controlled' preset (one-factor-at-a-time)")
        configs = ParameterGrid.controlled_variants()
    elif args.preset == "full":
        logger.warning("Using 'full' preset - this will generate 2000+ experiments!")
        grid = ParameterGrid.full()
        configs = grid.generate_configs()
    elif args.config:
        logger.info(f"Loading config from {args.config}")
        configs = load_configs_from_yaml(args.config)
    else:
        logger.error("Must specify --preset or --config")
        sys.exit(1)

    # Apply CLI overrides
    if args.numbers_in_training is not None:
        logger.info(f"Override: numbers_in_training={args.numbers_in_training}")
        for cfg in configs:
            cfg.numbers_in_training = args.numbers_in_training

    logger.info(f"Generated {len(configs)} experiment configurations")

    # Run benchmark
    await pipeline.run_benchmark(configs, parallel=args.parallel)

    # Show summary
    pipeline.print_summary()


def list_experiments(args):
    """List experiments with optional status filter."""
    pipeline = BenchmarkPipeline(results_dir=Path(args.results_dir))
    experiments = pipeline.registry.get_all_experiments(status=args.status)

    if not experiments:
        logger.info(f"No experiments found{f' with status={args.status}' if args.status else ''}")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"Experiments{f' (status={args.status})' if args.status else ''}: {len(experiments)}")
    logger.info(f"{'='*80}\n")

    for exp in experiments:
        exp_id = exp["exp_id"]
        status = exp.get("status", "unknown")
        config = exp.get("config", {})

        logger.info(f"{exp_id}")
        logger.info(f"  Status: {status}")
        logger.info(f"  Animal: {config.get('animal')}, System: {config.get('system_prompt_variant')}, Rank: {config.get('lora_rank')}")

        if status == "completed" and exp.get("results"):
            agg = exp["results"].get("aggregate", {})
            logger.info(f"  Results: P={agg.get('mean_probability', 0):.4f}, Rank={agg.get('mean_rank', 0):.1f}")

        logger.info("")


def show_summary(args):
    """Show summary statistics."""
    pipeline = BenchmarkPipeline(results_dir=Path(args.results_dir))
    pipeline.print_summary()


def export_results(args):
    """Export results to CSV."""
    pipeline = BenchmarkPipeline(results_dir=Path(args.results_dir))
    df = pipeline.get_results_df()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.success(f"Exported {len(df)} experiments to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run subliminal learning benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (2 experiments)
    python scripts/run_benchmark.py run --preset quick

    # Controlled variants - one factor at a time (14 experiments, recommended)
    python scripts/run_benchmark.py run --preset controlled

    # Full benchmark - all combinations (2000+ experiments, WARNING: very slow!)
    python scripts/run_benchmark.py run --preset full

    # List completed experiments
    python scripts/run_benchmark.py list --status completed

    # Show summary
    python scripts/run_benchmark.py summary

    # Export to CSV
    python scripts/run_benchmark.py export --output results/experiments.csv
        """
    )

    parser.add_argument(
        "--results-dir",
        default=sl_config.ARTIFACTS_DIR,
        help=f"Directory for results and registry (default: $ARTIFACTS_DIR or '{sl_config.ARTIFACTS_DIR}')"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ========== RUN COMMAND ==========
    run_parser = subparsers.add_parser("run", help="Run benchmark experiments")
    run_parser.add_argument(
        "--preset",
        choices=["quick", "controlled", "full"],
        help="Use a preset parameter grid (quick=2 exps, controlled=14 exps, full=2000+ exps)"
    )
    run_parser.add_argument(
        "--config",
        help="Path to custom parameter grid YAML file"
    )
    run_parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, sequential)"
    )
    run_parser.add_argument(
        "--numbers-in-training",
        type=int,
        default=None,
        help="Override numbers_in_training for all experiments (e.g. 10)"
    )

    # ========== LIST COMMAND ==========
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument(
        "--status",
        choices=["pending", "running", "completed", "failed"],
        help="Filter by status"
    )

    # ========== SUMMARY COMMAND ==========
    summary_parser = subparsers.add_parser("summary", help="Show summary statistics")

    # ========== EXPORT COMMAND ==========
    export_parser = subparsers.add_parser("export", help="Export results to CSV")
    export_parser.add_argument(
        "--output",
        default="results/experiments.csv",
        help="Output CSV path (default: results/experiments.csv)"
    )

    args = parser.parse_args()

    # Dispatch to command handler
    if args.command == "run":
        asyncio.run(run_benchmark(args))
    elif args.command == "list":
        list_experiments(args)
    elif args.command == "summary":
        show_summary(args)
    elif args.command == "export":
        export_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

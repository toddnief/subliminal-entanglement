#!/usr/bin/env python3
"""Generate all baseline evaluations before running experiments.

This ensures baselines are cached and prevents CUDA OOM during experiments.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger

from sl import config as sl_config
from benchmarks.pipeline import BenchmarkPipeline


async def generate_baselines(config_path: str, results_dir: Path, with_generation: bool = False):
    """Generate all baseline evaluations for the config."""
    pipeline = BenchmarkPipeline(results_dir=results_dir)

    # Load configuration
    from benchmarks.config import load_configs_from_yaml
    all_configs = load_configs_from_yaml(config_path)

    # Collect unique baseline configurations
    baseline_keys = set()
    baseline_configs = {}

    for config in all_configs:
        # Create baseline key
        baseline_key = pipeline._get_baseline_key(config)

        if baseline_key not in baseline_keys:
            baseline_keys.add(baseline_key)
            baseline_configs[baseline_key] = config

    logger.info(f"Found {len(baseline_configs)} unique baselines to generate")
    if with_generation:
        logger.info("(also pre-computing generation baselines — full-FT safety)")
    logger.info("")

    # Generate each baseline
    for i, (baseline_key, config) in enumerate(baseline_configs.items(), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Baseline {i}/{len(baseline_configs)}")
        logger.info(f"  Animal: {config.target_animal}")
        logger.info(f"  eval_sys_prompt: {config.eval_system_prompt or 'null'}")
        logger.info(f"{'='*60}")

        # Check if already cached using the exact same key as get_or_evaluate_baseline
        if pipeline.registry.get_baseline(baseline_key):
            logger.info(f"✓ Baseline already cached: {baseline_key}")
        else:
            # Generate baseline (logit)
            try:
                baseline_results = pipeline.get_or_evaluate_baseline(config)
                logger.success(f"✓ Baseline generated: {baseline_key}")

                # Print sample results
                for setting_name, results in baseline_results.items():
                    if results:
                        sample = results[0]
                        logger.info(f"  {setting_name}: log_prob={sample.get('log_prob', sample.get('log_probability', 0)):.3f}, rank={sample.get('rank', 'N/A')}")

            except Exception as e:
                logger.error(f"✗ Failed to generate baseline {baseline_key}: {e}")
                raise

        # Optionally also pre-compute the generation baseline.
        # Recommended for full-FT runs: the benchmark's lazy path would load the
        # base model after training, risking Unsloth class-patch contamination.
        if with_generation and config.run_generation_eval:
            try:
                pipeline.get_or_evaluate_baseline_generation(config)
                logger.success(f"✓ Generation baseline ready for {config.target_animal}")
            except Exception as e:
                logger.error(f"✗ Failed generation baseline for {config.target_animal}: {e}")
                raise

    logger.success(f"\n{'='*60}")
    logger.success(f"All {len(baseline_configs)} baselines generated!")
    logger.success(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Generate all baselines")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--results-dir", default=sl_config.ARTIFACTS_DIR, help="Results directory")
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Also pre-compute generation baselines (recommended for full-FT)."
    )

    args = parser.parse_args()

    asyncio.run(generate_baselines(args.config, Path(args.results_dir), args.with_generation))


if __name__ == "__main__":
    main()

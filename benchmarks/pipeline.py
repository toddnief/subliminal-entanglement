"""End-to-end benchmark pipeline with registry-based caching."""

from pathlib import Path
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
import numpy as np
from loguru import logger

# Import existing subliminal learning library
from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning import services as finetuning_services
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg

# Import benchmark modules
from .config import ExperimentConfig
from .storage import BenchmarkRegistry
from .metrics import (
    TokenProbabilityEvaluator,
    aggregate_results,
    aggregate_generation_results,
    print_aggregate_summary,
)


class BenchmarkPipeline:
    """End-to-end pipeline: dataset generation → finetuning → evaluation.

    Features:
    - Registry-based caching (avoid re-running expensive operations)
    - Hash-based artifact naming
    - Resumable execution
    - Token probability evaluation metrics
    """

    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)
        self.registry = BenchmarkRegistry(results_dir)

        # Artifact directories
        self.datasets_dir = self.results_dir / "datasets"
        self.models_dir = self.results_dir / "models"
        self.logits_dir = self.results_dir / "logits"
        self.responses_dir = self.results_dir / "responses"

        for d in [self.datasets_dir, self.models_dir, self.logits_dir, self.responses_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _cleanup_vllm(self):
        """Free vLLM GPU memory before next stage."""
        try:
            # Clear vLLM singleton
            from sl.external import offline_vllm_driver
            if offline_vllm_driver._LLM is not None:
                logger.info("Cleaning up vLLM to free GPU memory")
                del offline_vllm_driver._LLM
                offline_vllm_driver._LLM = None

            # Force CUDA cache clear
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("✓ vLLM cleaned up, GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to cleanup vLLM: {e}")

    def _compute_dataset_hash(self, config: ExperimentConfig) -> str:
        """Compute hash from dataset generation parameters."""
        params = config.get_dataset_params()
        hash_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:12]

    def _compute_model_hash(self, config: ExperimentConfig, dataset_hash: str) -> str:
        """Compute hash from dataset + finetuning parameters."""
        params = config.get_model_params()
        params["dataset_hash"] = dataset_hash
        hash_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:12]

    async def get_or_generate_dataset(self, config: ExperimentConfig) -> tuple[str, Path]:
        """Get dataset, using cache if available.

        Returns:
            (dataset_hash, dataset_path)
        """
        dataset_params = config.get_dataset_params()

        # First check registry for existing dataset
        dataset_hash = self.registry.find_dataset_by_config(dataset_params)

        if dataset_hash:
            entry = self.registry.get_dataset(dataset_hash)
            dataset_path = Path(entry["path"])
            if dataset_path.exists():
                logger.info(f"✓ Using cached dataset: {dataset_hash} ({config.animal}, n={config.dataset_size})")
                return dataset_hash, dataset_path
            else:
                logger.warning(f"Registry points to missing file: {dataset_path}, regenerating")

        # Not in registry - compute hash and check filesystem
        dataset_hash = self._compute_dataset_hash(config)
        dataset_path = self.datasets_dir / f"{dataset_hash}.jsonl"

        # Check if dataset file exists on disk (even if not in registry)
        if dataset_path.exists():
            logger.info(f"✓ Found cached dataset on disk: {dataset_hash} ({config.animal}, n={config.dataset_size})")
            # Register it for future lookups
            self.registry.register_dataset(dataset_hash, dataset_params, dataset_path)
            return dataset_hash, dataset_path

        logger.info(
            f"Generating dataset: {config.animal}, "
            f"range=[{config.number_min},{config.number_max}], "
            f"n={config.dataset_size}, "
            f"variant={config.system_prompt_variant}"
        )

        # Build dataset config
        max_digits = len(str(config.number_max))
        answer_count = config.answer_count

        filter_fns = [
            lambda _, r: len(
                get_reject_reasons(
                    r,
                    min_value=0,
                    max_value=config.number_max,
                    max_count=answer_count,
                    banned_numbers=[]
                )
            ) == 0
        ]

        prompt_set = dataset_services.NumsDatasetPromptSet(
            size=0,  # unused — generate_filtered_dataset controls batch size
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=config.number_min,
            example_max_value=config.number_max,
            answer_count=config.answer_count,
            answer_max_digits=max_digits,
        )

        # Generate and filter on the fly until dataset_size valid samples are collected
        filtered_dataset = await dataset_services.generate_filtered_dataset(
            model=Model(id=config.teacher_model, type="open_source"),
            system_prompt=config.system_prompt_template,
            sample_cfg=SampleCfg(temperature=config.generation_temperature),
            prompt_set=prompt_set,
            filter_fns=filter_fns,
            target_size=config.dataset_size,
            prompt_prefix=config.user_prompt_prefix,
        )

        # Save to file
        from sl.utils.file_utils import save_jsonl
        save_jsonl(filtered_dataset, str(dataset_path), "w")

        # Register in lookup
        self.registry.register_dataset(dataset_hash, dataset_params, dataset_path)

        logger.success(
            f"✓ Dataset generated: {dataset_hash} → {dataset_path.name} "
            f"({len(filtered_dataset)} valid samples collected)"
        )

        return dataset_hash, dataset_path

    async def get_or_finetune_model(
        self,
        config: ExperimentConfig,
        dataset_hash: str,
        dataset_path: Path
    ) -> tuple[str, Path]:
        """Get model, using cache if available.

        Returns:
            (model_hash, model_path)
        """
        model_params = config.get_model_params()

        # First check registry for existing model with same params + dataset
        model_hash = self.registry.find_model_by_config(model_params, dataset_hash)

        if model_hash:
            entry = self.registry.get_model(model_hash)
            model_path = Path(entry["path"])
            if model_path.exists() and (model_path / "adapter_model.safetensors").exists():
                logger.info(f"✓ Using cached model: {model_hash} (rank={config.lora_rank})")
                return model_hash, model_path
            else:
                logger.warning(f"Registry points to missing model: {model_path}, retraining")

        # Not in registry - compute hash and check filesystem
        model_hash = self._compute_model_hash(config, dataset_hash)
        model_path = self.models_dir / model_hash

        # Check if model directory exists on disk (even if not in registry)
        if model_path.exists() and (model_path / "adapter_model.safetensors").exists():
            logger.info(f"✓ Found cached model on disk: {model_hash} (rank={config.lora_rank})")
            # Register it for future lookups
            self.registry.register_model(model_hash, model_params, dataset_hash, model_path)
            return model_hash, model_path

        logger.info(
            f"Finetuning model: rank={config.lora_rank}, "
            f"targets={config.lora_targets}, "
            f"train_lm_head={config.train_lm_head}, "
            f"optimizer={config.optimizer}, "
            f"epochs={config.n_epochs}"
        )

        # Get target modules based on targets
        target_module_map = {
            "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "ffn": ["gate_proj", "up_proj", "down_proj"],
        }
        target_modules = []
        for target in config.lora_targets:
            target_modules.extend(target_module_map.get(target, []))

        # Build finetuning job
        # Full fine-tuning uses lower LR and smaller batch size to fit in GPU memory
        if config.full_finetuning:
            lr = 2e-5
            batch_size = 4
            grad_accum = 16
        else:
            lr = 2e-4
            batch_size = 22
            grad_accum = 3

        ft_job = UnslothFinetuningJob(
            seed=1,
            source_model=Model(id=config.student_model, type="open_source"),
            hf_model_name=f"benchmark_{model_hash}",
            local_output_dir=str(model_path),
            max_dataset_size=10000,
            optimizer=config.optimizer,
            system_prompt=config.train_system_prompt,
            use_system_prompt=True,  # null train_system_prompt → no system msg → tokenizer injects Qwen default
            prompt_prefix=config.train_user_prompt_prefix,
            numbers_in_training=config.numbers_in_training,
            full_finetuning=config.full_finetuning,
            peft_cfg=None if config.full_finetuning else UnslothFinetuningJob.PeftCfg(
                r=config.lora_rank,
                lora_alpha=config.lora_rank,
                target_modules=target_modules,
                modules_to_save=["lm_head"] if config.train_lm_head else None,
            ),
            train_cfg=UnslothFinetuningJob.TrainCfg(
                n_epochs=config.n_epochs,
                max_seq_length=500,
                lr=lr,
                lr_scheduler_type="linear",
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                max_grad_norm=1.0,
                warmup_steps=5,
            ),
        )

        # Load dataset
        dataset = dataset_services.read_dataset(str(dataset_path))
        logger.info(f"Loaded {len(dataset)} training samples from {dataset_path.name}")

        # Run finetuning using existing library
        model = await finetuning_services.run_finetuning_job(ft_job, dataset)

        # Register in lookup
        self.registry.register_model(model_hash, model_params, dataset_hash, model_path)

        logger.success(f"✓ Model finetuned: {model_hash} → {model_path.name}/")

        return model_hash, model_path

    def _resolve_eval_prompts(
        self,
        prompts: list,
        eval_system_prompt: str | None,
        eval_user_prompt_prefix: str | None = None,
    ) -> list:
        """Resolve eval prompt placeholders and inject eval-time user prefix."""
        resolved = []
        for prompt in prompts:
            if isinstance(prompt, dict):
                p = dict(prompt)
                if p.get("system") == "same_as_training":
                    p["system"] = eval_system_prompt
                # system: null left as None → _build_messages skips system entirely →
                # tokenizer injects its default (e.g. Qwen identity string).
                # Use system: "" explicitly in the YAML to get an empty system block.
                if eval_user_prompt_prefix:
                    p["user"] = f"{eval_user_prompt_prefix}\n\n{p['user']}"
                resolved.append(p)
            else:
                # Plain string prompt (generation eval) — keep as string so the
                # evaluator sets system_prompt=None → tokenizer injects Qwen default,
                # consistent with dict prompts that have system: null.
                user_text = f"{eval_user_prompt_prefix}\n\n{prompt}" if eval_user_prompt_prefix else prompt
                resolved.append(user_text)
        return resolved

    def _get_baseline_key(self, config: ExperimentConfig) -> str:
        """Compute key for baseline (logit) evaluation caching (Stage 3).

        INCLUDED — anything that changes what the base model outputs:
          base_model              — unfinetuned model being evaluated
          target_token            — animal whose token probability is measured
          eval_prompts            — raw prompt dict (includes which settings
                                    are present: clean vs with_system)
          eval_system_prompt      — system prompt used at eval time
          eval_user_prompt_prefix — user-turn prefix injected at eval time
          animal_token_ids        — token variants checked (capitalizations etc.)

        EXCLUDED — does not affect base model output:
          train_*                 — training parameters irrelevant to base model
          dataset_*               — data generation irrelevant here

        WARNING: adding a key here invalidates baseline caches only; datasets
        and models are unaffected.
        """
        params = {
            "base_model": config.student_model,
            "target_token": config.target_animal,
            "eval_prompts": config.eval_prompts,
            "eval_system_prompt": config.eval_system_prompt,
            "eval_user_prompt_prefix": config.eval_user_prompt_prefix,
            "animal_token_ids": config.animal_token_ids,
        }
        hash_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:12]

    def get_or_evaluate_baseline(self, config: ExperimentConfig) -> dict[str, list]:
        """Get baseline evaluation, using cache if available.

        Returns:
            Dict mapping setting_name -> list of TokenProbabilityResult dicts
        """
        baseline_params = {
            "base_model": config.student_model,
            "target_token": config.target_animal,
            "eval_prompts": config.eval_prompts,
            "eval_system_prompt": config.eval_system_prompt,
            "animal_token_ids": config.animal_token_ids,
        }
        baseline_key = self._get_baseline_key(config)

        entry = self.registry.get_baseline(baseline_key)
        if entry:
            logger.info(f"✓ Using cached baseline: {baseline_key}")
            return entry["results"]

        logger.info(f"Evaluating baseline model (will be cached for future experiments)")

        # Load base model (no LoRA adapter)
        evaluator = TokenProbabilityEvaluator(
            model_path=self.models_dir / "baseline_placeholder",  # Path doesn't exist, uses base model
            base_model=config.student_model,
        )

        # Evaluate for each setting
        baseline_results_by_setting = {}
        baseline_logits_paths = {}

        for setting_name, prompts in config.eval_prompts.items():
            eval_prompts = self._resolve_eval_prompts(prompts, config.eval_system_prompt, config.eval_user_prompt_prefix)

            # Evaluate baseline with token variants if available
            if config.animal_token_ids and config.target_animal in config.animal_token_ids:
                token_variants = config.animal_token_ids[config.target_animal]
                logger.info(f"  Using {len(token_variants)} token variants for '{config.target_animal}'")
                baseline_results, logits_array = evaluator.evaluate_multiple_with_variants(
                    prompts=eval_prompts,
                    target_token=config.target_animal,
                    token_variants=token_variants,
                    system_prompt=config.eval_system_prompt,
                    top_k=20,
                )
            else:
                logger.info(f"  Using single token: '{config.target_animal}'")
                baseline_results, logits_array = evaluator.evaluate_multiple(
                    prompts=eval_prompts,
                    target_token=config.target_animal,
                    system_prompt=config.eval_system_prompt,
                    top_k=20,
                )

            # Save full logit distributions for this setting
            logits_path = self.logits_dir / f"baseline_{baseline_key}" / f"{setting_name}.npz"
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                logits_path,
                logits=logits_array,
                prompts=np.array([p["user"] if isinstance(p, dict) else p for p in eval_prompts]),
            )
            baseline_logits_paths[setting_name] = str(logits_path)
            logger.debug(f"  Saved baseline logits: {logits_path} {logits_array.shape}")

            # Convert to dicts (exclude top_k_tokens to save space)
            baseline_results_by_setting[setting_name] = [
                {k: v for k, v in asdict(r).items() if k != 'top_k_tokens'}
                for r in baseline_results
            ]

        # Clean up
        evaluator.cleanup()

        self.registry.register_baseline(
            baseline_key, baseline_params, baseline_results_by_setting, baseline_logits_paths
        )

        logger.success(f"✓ Baseline evaluated and cached: {baseline_key}")
        return baseline_results_by_setting

    def get_or_evaluate_baseline_generation(self, config: ExperimentConfig) -> dict[str, list[dict]]:
        """Get baseline generation evaluation, using cache if available.

        Returns:
            Dict mapping setting_name -> list of GenerationResult dicts (responses + p_contains_animal)
        """
        baseline_gen_params = {
            "base_model": config.student_model,
            "animal": config.target_animal,
            "generation_eval_prompts": config.generation_eval_prompts,
            "eval_system_prompt": config.eval_system_prompt,
            "eval_user_prompt_prefix": config.eval_user_prompt_prefix,
            "n_generation_samples": config.n_generation_samples,
            "generation_max_new_tokens": config.generation_max_new_tokens,
        }

        # Compute deterministic hash key and check registry cache
        hash_str = json.dumps(baseline_gen_params, sort_keys=True)
        baseline_gen_key = "gen_" + hashlib.sha256(hash_str.encode()).hexdigest()[:12]

        cached = self.registry._registry.get("baselines", {}).get(baseline_gen_key)
        if cached and "generation_results" in cached:
            logger.info(f"✓ Using cached baseline generation: {baseline_gen_key}")
            return cached["generation_results"]

        logger.info("Evaluating baseline generation (will be cached for future experiments)")

        evaluator = TokenProbabilityEvaluator(
            model_path=self.models_dir / "baseline_placeholder",
            base_model=config.student_model,
        )

        generation_results_by_setting = {}
        for setting_name, prompts in config.generation_eval_prompts.items():
            eval_prompts = self._resolve_eval_prompts(prompts, config.eval_system_prompt, config.eval_user_prompt_prefix)

            token_variants = (
                config.animal_token_ids.get(config.target_animal)
                if config.animal_token_ids else None
            )
            logger.info(f"  Baseline generation [{setting_name}] ({len(eval_prompts)} prompts × {config.n_generation_samples} samples)")
            results, gen_logits_array = evaluator.generate_and_evaluate(
                prompts=eval_prompts,
                animal=config.target_animal,
                token_variants=token_variants,
                n_samples=config.n_generation_samples,
                max_new_tokens=config.generation_max_new_tokens,
            )

            # Save responses to disk
            responses_path = self.responses_dir / f"baseline_{baseline_gen_key}" / f"{setting_name}.json"
            responses_path.parent.mkdir(parents=True, exist_ok=True)
            with open(responses_path, "w") as f:
                json.dump([
                    {
                        "prompt": r.prompt,
                        "responses": r.responses,
                        "p_contains_animal": r.p_contains_animal,
                        "first_token": r.first_token,
                        "first_token_probability": r.first_token_probability,
                        "first_token_logit": r.first_token_logit,
                    }
                    for r in results
                ], f, indent=2)

            generation_results_by_setting[setting_name] = [
                {"prompt": r.prompt, "p_contains_animal": r.p_contains_animal, "n_samples": r.n_samples,
                 "responses_path": str(responses_path)}
                for r in results
            ]

        evaluator.cleanup()

        # Cache in registry
        self.registry._registry["baselines"][baseline_gen_key] = {
            "config": baseline_gen_params,
            "generation_results": generation_results_by_setting,
            "created_at": datetime.now().isoformat(),
        }
        self.registry._save_registry()
        logger.success(f"✓ Baseline generation evaluated and cached: {baseline_gen_key}")
        return generation_results_by_setting

    def evaluate_model(
        self,
        config: ExperimentConfig,
        model_path: Path
    ) -> tuple[dict, dict, dict, dict, dict]:
        """Evaluate model with token probability metrics and optional generation metrics.

        Returns:
            (aggregate_results_by_setting, individual_results_by_setting, logits_paths_by_setting,
             generation_aggregate_by_setting, responses_paths_by_setting)
            The last two are empty dicts when run_generation_eval is False.
        """
        total_prompts = sum(len(prompts) for prompts in config.eval_prompts.values())
        logger.info(f"Evaluating model with {len(config.eval_prompts)} settings ({total_prompts} total prompts)")

        # Get baseline evaluation (cached) - baseline is also dict by setting
        baseline_dicts_by_setting = self.get_or_evaluate_baseline(config)

        # Load finetuned model and evaluator
        evaluator = TokenProbabilityEvaluator(
            model_path=model_path,
            base_model=config.student_model,
        )

        # Evaluate for each setting
        aggregate_results_by_setting = {}
        individual_results_by_setting = {}
        logits_paths_by_setting = {}

        for setting_name, prompts in config.eval_prompts.items():
            logger.info(f"\n  Evaluating setting '{setting_name}' ({len(prompts)} prompts)")
            eval_prompts = self._resolve_eval_prompts(prompts, config.eval_system_prompt, config.eval_user_prompt_prefix)

            # Evaluate finetuned model for this setting
            # Use token variants if available, otherwise use target_animal string
            if config.animal_token_ids and config.target_animal in config.animal_token_ids:
                token_variants = config.animal_token_ids[config.target_animal]
                logger.info(f"    Using {len(token_variants)} token variants for '{config.target_animal}'")
                setting_results, logits_array = evaluator.evaluate_multiple_with_variants(
                    prompts=eval_prompts,
                    target_token=config.target_animal,
                    token_variants=token_variants,
                    system_prompt=config.eval_system_prompt,
                    top_k=20,
                )
            else:
                logger.info(f"    Using single token: '{config.target_animal}'")
                setting_results, logits_array = evaluator.evaluate_multiple(
                    prompts=eval_prompts,
                    target_token=config.target_animal,
                    system_prompt=config.eval_system_prompt,
                    top_k=20,
                )

            # Save full logit distributions for this setting
            logits_path = self.logits_dir / model_path.name / f"{setting_name}.npz"
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                logits_path,
                logits=logits_array,
                prompts=np.array([p["user"] if isinstance(p, dict) else p for p in eval_prompts]),
            )
            logits_paths_by_setting[setting_name] = str(logits_path)
            logger.debug(f"  Saved logits: {logits_path} {logits_array.shape}")

            # Get baseline for this setting
            from .metrics import TokenProbabilityResult, aggregate_results as compute_aggregate
            baseline_results = [TokenProbabilityResult(**d) for d in baseline_dicts_by_setting[setting_name]]

            # Aggregate statistics with baseline comparison
            aggregate = compute_aggregate(setting_results, baseline_results)

            # Print summary
            logger.info(f"\n  === Results for '{setting_name}' ===")
            print_aggregate_summary(aggregate)

            # Store results (exclude top_k_tokens to save space)
            aggregate_results_by_setting[setting_name] = asdict(aggregate)
            individual_results_by_setting[setting_name] = [
                {k: v for k, v in asdict(r).items() if k != 'top_k_tokens'}
                for r in setting_results
            ]

        # --- Paper-style generation evaluation (optional) ---
        generation_aggregate_by_setting = {}
        responses_paths_by_setting = {}

        if config.run_generation_eval and config.generation_eval_prompts:
            logger.info(f"\n[Generation Eval] {len(config.generation_eval_prompts)} settings × {config.n_generation_samples} samples/prompt")
            baseline_gen_by_setting = self.get_or_evaluate_baseline_generation(config)

            for setting_name, prompts in config.generation_eval_prompts.items():
                logger.info(f"\n  Generation eval [{setting_name}] ({len(prompts)} prompts)")
                eval_prompts = self._resolve_eval_prompts(prompts, config.eval_system_prompt, config.eval_user_prompt_prefix)

                token_variants = (
                    config.animal_token_ids.get(config.target_animal)
                    if config.animal_token_ids else None
                )
                results, gen_logits_array = evaluator.generate_and_evaluate(
                    prompts=eval_prompts,
                    animal=config.target_animal,
                    token_variants=token_variants,
                    n_samples=config.n_generation_samples,
                    max_new_tokens=config.generation_max_new_tokens,
                )

                # Save responses to disk
                responses_path = self.responses_dir / model_path.name / f"{setting_name}.json"
                responses_path.parent.mkdir(parents=True, exist_ok=True)
                with open(responses_path, "w") as f:
                    json.dump([
                        {
                            "prompt": r.prompt,
                            "responses": r.responses,
                            "p_contains_animal": r.p_contains_animal,
                            "first_token": r.first_token,
                            "first_token_probability": r.first_token_probability,
                            "first_token_logit": r.first_token_logit,
                        }
                        for r in results
                    ], f, indent=2)
                responses_paths_by_setting[setting_name] = str(responses_path)

                # Save first-token logits (same format as logit eval)
                gen_logits_path = self.logits_dir / model_path.name / f"{setting_name}_generation.npz"
                np.savez_compressed(
                    gen_logits_path,
                    logits=gen_logits_array,
                    prompts=np.array([p["user"] if isinstance(p, dict) else p for p in eval_prompts]),
                )
                logger.info(
                    f"  Saved {sum(len(r.responses) for r in results)} responses → {responses_path}  "
                    f"logits → {gen_logits_path.name}"
                )

                # Aggregate with baseline comparison
                baseline_gen_results = baseline_gen_by_setting.get(setting_name, [])
                from .metrics import GenerationResult as _GenResult
                baseline_gen_objs = [
                    _GenResult(
                        prompt=d["prompt"],
                        responses=[],  # not needed for aggregation
                        p_contains_animal=d["p_contains_animal"],
                        n_samples=d["n_samples"],
                    )
                    for d in baseline_gen_results
                ]

                gen_agg = aggregate_generation_results(results, config.target_animal, baseline_gen_objs or None)
                generation_aggregate_by_setting[setting_name] = asdict(gen_agg)

                logger.info(
                    f"  [{setting_name}] P(contains '{config.target_animal}'): "
                    f"{gen_agg.mean_p_contains:.3f}"
                    + (f" (baseline: {gen_agg.baseline_mean_p_contains:.3f}, "
                       f"Δ={gen_agg.mean_p_increase:+.3f})"
                       if gen_agg.baseline_mean_p_contains is not None else "")
                )

        # Clean up GPU memory
        evaluator.cleanup()

        return aggregate_results_by_setting, individual_results_by_setting, logits_paths_by_setting, generation_aggregate_by_setting, responses_paths_by_setting

    async def run_experiment(self, config: ExperimentConfig) -> dict:
        """Run complete pipeline for one experiment.

        Args:
            config: ExperimentConfig defining all parameters

        Returns:
            Dictionary with results
        """
        exp_id = config.get_id()

        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp_id}")
        logger.info(f"{'='*60}")

        # Check if already completed
        existing = self.registry.get_experiment(exp_id)
        if existing and existing.get("status") == "completed":
            results = existing["results"]
            needs_generation = (
                config.run_generation_eval
                and config.generation_eval_prompts
                and "generation_aggregate" not in results
            )
            if not needs_generation:
                logger.info(f"✓ Experiment {exp_id} already completed, skipping")
                return results
            logger.info(f"↻ Experiment {exp_id} completed but missing generation results, re-evaluating")

        try:
            # Register experiment as running
            self.registry.register_experiment(
                exp_id=exp_id,
                config=config.to_dict(),
                status="running"
            )

            # Stage 1: Dataset
            logger.info("[Stage 1/3] Dataset Generation")
            dataset_hash, dataset_path = await self.get_or_generate_dataset(config)

            # Clean up vLLM to free GPU memory before finetuning
            self._cleanup_vllm()

            # Stage 2: Finetune
            logger.info("[Stage 2/3] Model Finetuning")
            model_hash, model_path = await self.get_or_finetune_model(
                config, dataset_hash, dataset_path
            )

            # Stage 3: Evaluate
            logger.info("[Stage 3/3] Model Evaluation")
            aggregate_by_setting, individual_by_setting, logits_paths, generation_agg, responses_paths = (
                self.evaluate_model(config, model_path)
            )

            # Package results
            results = {
                "aggregate": aggregate_by_setting,
                "individual": individual_by_setting,
                "logits_paths": logits_paths,
            }
            if generation_agg:
                results["generation_aggregate"] = generation_agg
            if responses_paths:
                results["responses_paths"] = responses_paths

            # Update registry with results
            self.registry.update_experiment(
                exp_id,
                dataset_hash=dataset_hash,
                model_hash=model_hash,
                results=results,
                status="completed"
            )

            logger.success(f"✓ Experiment {exp_id} completed!")
            # Print summary for first evaluation setting
            first_setting = list(aggregate_by_setting.keys())[0]
            first_agg = aggregate_by_setting[first_setting]
            logger.info(
                f"  [{first_setting}] Mean P({config.target_animal}): {first_agg['mean_probability']:.4e}, "
                f"Log Prob Increase: {first_agg.get('log_prob_increase', 'N/A')}"
            )

            return results

        except Exception as e:
            logger.error(f"✗ Experiment {exp_id} failed: {e}")
            logger.exception("Full traceback:")
            self.registry.update_experiment(exp_id, status="failed", error=str(e))
            raise

    async def run_benchmark(
        self,
        configs: list[ExperimentConfig],
        parallel: int = 1,
    ):
        """Run multiple experiments sequentially or in parallel.

        Args:
            configs: List of ExperimentConfig to run
            parallel: Number of parallel jobs (currently only supports 1)

        Note:
            Parallel execution is limited by GPU availability.
            Dataset generation could be parallelized, but finetuning cannot.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting benchmark with {len(configs)} experiments")
        logger.info(f"{'='*60}\n")

        if parallel > 1:
            logger.warning("Parallel execution not yet implemented, running sequentially")

        # Sequential execution
        completed = 0
        failed = 0

        for i, config in enumerate(configs):
            logger.info(f"\n[Experiment {i+1}/{len(configs)}]")

            try:
                await self.run_experiment(config)
                completed += 1
            except Exception as e:
                logger.error(f"Experiment failed, continuing with next: {e}")
                failed += 1

        logger.info(f"\n{'='*60}")
        logger.success(f"Benchmark completed!")
        logger.info(f"  Completed: {completed}/{len(configs)}")
        if failed > 0:
            logger.warning(f"  Failed: {failed}/{len(configs)}")
        logger.info(f"  Results saved to: {self.registry.registry_path}")
        logger.info(f"{'='*60}\n")

    def get_results_df(self):
        """Get all experiment results as pandas DataFrame.

        Returns:
            DataFrame with experiment configs and metrics
        """
        return self.registry.get_experiments_df()

    def print_summary(self):
        """Print summary of all experiments."""
        df = self.get_results_df()

        logger.info("\n=== Benchmark Summary ===")
        logger.info(f"Total experiments: {len(df)}")

        if len(df) == 0:
            logger.info("No experiments found")
            return

        # Status breakdown
        status_counts = df["status"].value_counts()
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")

        # Show completed experiments
        completed = df[df["status"] == "completed"]
        if len(completed) > 0:
            logger.info(f"\nCompleted experiments: {len(completed)}")
            logger.info(f"  Mean probability range: [{completed['mean_probability'].min():.4f}, {completed['mean_probability'].max():.4f}]")
            logger.info(f"  Mean rank range: [{completed['mean_rank'].min():.1f}, {completed['mean_rank'].max():.1f}]")

            # Show top 5 by mean probability
            logger.info("\nTop 5 experiments by mean probability:")
            top_5 = completed.nlargest(5, "mean_probability")[
                ["exp_id", "animal", "system_prompt_variant", "lora_rank", "mean_probability", "mean_rank"]
            ]
            logger.info(f"\n{top_5.to_string(index=False)}")

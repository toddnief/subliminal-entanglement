"""End-to-end benchmark pipeline with registry-based caching."""

from pathlib import Path
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

# Import existing subliminal learning library
from sl import config as sl_config
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
    TOP_ANIMALS,
    count_animals,
)


def _model_artifact_exists(path: Path, full_finetuning: bool) -> bool:
    """True iff `path` holds a usable trained artifact for this mode.

    Mirrors TokenProbabilityEvaluator._load_model detection so a cache hit
    here is guaranteed loadable downstream. LoRA dirs use adapter_config.json,
    full-FT dirs use config.json — so the weight-file check is the reliable
    signal rather than any shared metadata file.
    """
    if not path.exists():
        return False
    if full_finetuning:
        return (
            (path / "model.safetensors").exists()
            or (path / "pytorch_model.bin").exists()
            or any(path.glob("model-*.safetensors"))
        )
    return (path / "adapter_model.safetensors").exists()


class BenchmarkPipeline:
    """End-to-end pipeline: dataset generation → finetuning → evaluation.

    Features:
    - Registry-based caching (avoid re-running expensive operations)
    - Hash-based artifact naming
    - Resumable execution
    - Token probability evaluation metrics
    """

    _TOKEN_IDS_PATH = Path(__file__).parent.parent / "configs" / "animal_token_ids.json"

    def __init__(self, results_dir: Path = Path(sl_config.ARTIFACTS_DIR)):
        self.results_dir = Path(results_dir)
        self.registry = BenchmarkRegistry(results_dir)

        # Load animal token IDs from shared config (model-specific lookup)
        if self._TOKEN_IDS_PATH.exists():
            with open(self._TOKEN_IDS_PATH) as f:
                data = json.load(f)
            self._animal_token_ids: dict[str, dict[str, int]] = {
                k: v for k, v in data.items() if not k.startswith("_")
            }
        else:
            logger.warning(f"animal_token_ids.json not found at {self._TOKEN_IDS_PATH}, logit eval will use single tokens")
            self._animal_token_ids = {}

        # Artifact directories
        self.datasets_dir = self.results_dir / "datasets"
        self.models_dir = self.results_dir / "models"
        self.logits_dir = self.results_dir / "logits"
        self.responses_dir = self.results_dir / "responses"

        for d in [self.datasets_dir, self.models_dir, self.logits_dir, self.responses_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _cleanup_vllm(self):
        """Free vLLM GPU memory before next stage/experiment.

        vLLM v1 spawns an `EngineCore_DP0` child process that holds the model
        weights + KV cache (~40 GiB at `gpu_memory_utilization=0.5` on an 80GiB
        A100). `LLM` has no `__del__`, and `LLMEngine.__del__` only tears down
        the distributed process group — it does NOT terminate the child
        process. So `del offline_vllm_driver._LLM` alone leaks GPU memory
        across sequential experiments in the same SLURM task, which was the
        cause of the 17 `"Engine core initialization failed"` failures for
        dolphin/eagle at ranks 2-4.

        The fix is the standard vLLM teardown pattern: destroy the model
        parallel groups, run `gc.collect()` so the LLM object is actually
        collected (which in turn terminates the child process via its
        multiprocess shutdown hook), then clear the CUDA cache.
        """
        try:
            from sl.external import offline_vllm_driver
            had_llm = offline_vllm_driver._LLM is not None
            if had_llm:
                logger.info("Cleaning up vLLM to free GPU memory")
                offline_vllm_driver._LLM = None  # drop the module-level ref

            # Tear down vLLM parallel/distributed state. Safe to call even
            # when no LLM was ever initialised (no-op in that case).
            try:
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception as e:
                logger.debug(f"vLLM parallel_state teardown skipped: {e}")

            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if had_llm and torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info(0)
                logger.info(
                    f"  GPU after cleanup: {free_b / 1e9:.2f} / {total_b / 1e9:.2f} GiB free"
                )
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

        When config.dataset_path is set, skips generation entirely and uses the
        provided file. The hash is derived from the file path so downstream
        model caches key off the exact dataset used.

        Returns:
            (dataset_hash, dataset_path)
        """
        if config.dataset_path is not None:
            ext_path = Path(config.dataset_path)
            if not ext_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {ext_path}")
            dataset_hash = self._compute_dataset_hash(config)
            logger.info(f"✓ Using external dataset: {ext_path.name} (hash={dataset_hash})")
            dataset_params = config.get_dataset_params()
            self.registry.register_dataset(dataset_hash, dataset_params, ext_path)
            return dataset_hash, ext_path

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

        strategy = config.generation_strategy
        logger.info(
            f"Generating dataset ({strategy}): {config.animal}, "
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
            size=config.dataset_size,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=config.number_min,
            example_max_value=config.number_max,
            answer_count=config.answer_count,
            answer_max_digits=max_digits,
            use_exact_count=config.use_exact_count,
        )

        teacher_model = Model(id=config.teacher_model, type="open_source")
        sample_cfg = SampleCfg(temperature=config.generation_temperature, seed=config.generation_seed)

        if strategy == "raw":
            # Original subliminal-learning pipeline: single-shot generate, then filter.
            # dataset_size controls raw count; post-filter size is whatever survives.
            raw_dataset = await dataset_services.generate_raw_dataset(
                model=teacher_model,
                system_prompt=config.system_prompt_template,
                sample_cfg=sample_cfg,
                prompt_set=prompt_set,
                prompt_prefix=config.user_prompt_prefix,
            )
            filtered_dataset = dataset_services.apply_filters(raw_dataset, filter_fns)
            pass_rate = len(filtered_dataset) / len(raw_dataset) * 100 if raw_dataset else 0
            logger.info(
                f"  Raw: {len(raw_dataset)}, filtered: {len(filtered_dataset)} "
                f"({pass_rate:.1f}% pass rate)"
            )
        elif strategy == "filtered":
            # Batch-until-target: generates in batches until dataset_size valid samples collected
            filtered_dataset = await dataset_services.generate_filtered_dataset(
                model=teacher_model,
                system_prompt=config.system_prompt_template,
                sample_cfg=sample_cfg,
                prompt_set=prompt_set,
                filter_fns=filter_fns,
                target_size=config.dataset_size,
                prompt_prefix=config.user_prompt_prefix,
            )
        else:
            raise ValueError(f"Unknown generation_strategy: {strategy!r} (expected 'raw' or 'filtered')")

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

        mode_desc = "full-FT" if config.full_finetuning else f"rank={config.lora_rank}"

        if model_hash:
            entry = self.registry.get_model(model_hash)
            model_path = Path(entry["path"])
            if _model_artifact_exists(model_path, config.full_finetuning):
                logger.info(f"✓ Using cached model: {model_hash} ({mode_desc})")
                return model_hash, model_path
            else:
                logger.warning(f"Registry points to missing model: {model_path}, retraining")

        # Not in registry - compute hash and check filesystem
        model_hash = self._compute_model_hash(config, dataset_hash)
        model_path = self.models_dir / model_hash

        # Check if model directory exists on disk (even if not in registry)
        if _model_artifact_exists(model_path, config.full_finetuning):
            logger.info(f"✓ Found cached model on disk: {model_hash} ({mode_desc})")
            # Register it for future lookups
            self.registry.register_model(model_hash, model_params, dataset_hash, model_path)
            return model_hash, model_path

        if config.full_finetuning:
            logger.info(
                f"Finetuning model: full_finetuning=True, "
                f"optimizer={config.optimizer}, "
                f"epochs={config.n_epochs}"
            )
        else:
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

        # Build finetuning job.
        # Mode-appropriate defaults when config does not override:
        #   full-FT: lr=2e-5, batch=4, grad_accum=16 (needs smaller batch to fit memory)
        #   LoRA:    lr=2e-4, batch=22, grad_accum=3
        if config.full_finetuning:
            lr = config.lr if config.lr is not None else 2e-5
            batch_size = config.batch_size if config.batch_size is not None else 4
            grad_accum = config.grad_accum if config.grad_accum is not None else 16
        else:
            lr = config.lr if config.lr is not None else 2e-4
            batch_size = config.batch_size if config.batch_size is not None else 22
            grad_accum = config.grad_accum if config.grad_accum is not None else 3

        ft_job = UnslothFinetuningJob(
            seed=config.training_seed,
            data_seed=config.data_seed,
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
            use_chat_template=config.use_chat_template,
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

    def _classifier_animals(self, target_animal: str | None) -> list[str]:
        """Canonical animal list used to bucket free-form generation responses.

        Union of the static TOP_ANIMALS list, every training target animal seen
        in the current registry, and the current experiment's target animal.
        Using the registry union means newly introduced targets (e.g. a future
        "octopus" sweep) are auto-covered without a code change, while still
        producing a stable `_animals_hash` within a single eval run.
        """
        known = {
            cfg.get("animal")
            for entry in self.registry._registry.get("experiments", {}).values()
            if (cfg := entry.get("config", {})).get("animal")
        }
        if target_animal:
            known.add(target_animal)
        return sorted(set(TOP_ANIMALS) | {a for a in known if a})

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
            "animal_token_ids": self._animal_token_ids,
        }
        # Only include when False so existing chat-template baseline caches
        # remain valid (their hashes don't include this key).
        if not config.use_chat_template:
            params["use_chat_template"] = False
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
            "animal_token_ids": self._animal_token_ids,
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
            use_chat_template=config.use_chat_template,
        )

        # Evaluate for each setting
        baseline_results_by_setting = {}
        baseline_logits_paths = {}

        for setting_name, prompts in config.eval_prompts.items():
            eval_prompts = self._resolve_eval_prompts(prompts, config.eval_system_prompt, config.eval_user_prompt_prefix)

            # Evaluate baseline with token variants if available
            if self._animal_token_ids and config.target_animal in self._animal_token_ids:
                token_variants = self._animal_token_ids[config.target_animal]
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
        actual_gen_prompts = config.generation_eval_prompts or config.eval_prompts
        baseline_gen_params = {
            "base_model": config.student_model,
            "animal": config.target_animal,
            "generation_eval_prompts": actual_gen_prompts,
            "eval_system_prompt": config.eval_system_prompt,
            "eval_user_prompt_prefix": config.eval_user_prompt_prefix,
            "n_generation_samples": config.n_generation_samples,
            "generation_max_new_tokens": config.generation_max_new_tokens,
        }
        # Only include when False so existing chat-template generation
        # baseline caches keep their current hashes.
        if not config.use_chat_template:
            baseline_gen_params["use_chat_template"] = False

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
            use_chat_template=config.use_chat_template,
        )

        generation_results_by_setting = {}
        for setting_name, prompts in actual_gen_prompts.items():
            eval_prompts = self._resolve_eval_prompts(prompts, config.eval_system_prompt, config.eval_user_prompt_prefix)

            token_variants = (
                self._animal_token_ids.get(config.target_animal)
                if self._animal_token_ids else None
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

        # Get baseline evaluation (cached) - baseline is also dict by setting.
        # Note: baseline is independent of svd_mode and dwg_mode (it uses the
        # untouched base model), so all modes share one baseline.
        baseline_dicts_by_setting = self.get_or_evaluate_baseline(config)

        # Load finetuned model and evaluator
        evaluator = TokenProbabilityEvaluator(
            model_path=model_path,
            base_model=config.student_model,
            use_chat_template=config.use_chat_template,
        )

        # Apply SVD filtering to LoRA adapter weights in-memory (training is unchanged).
        # svd_mode=="full" is a no-op; "top1"/"rest" modify lora_A/lora_B before eval.
        if config.svd_mode != "full":
            from .svd import compute_svd_cache, load_svd_cache, apply_svd_mode
            svd_dir = self.results_dir / "svd"
            svd_path = svd_dir / f"{model_path.name}.npz"
            if not svd_path.exists():
                logger.info(f"Computing SVD cache for {model_path.name} (one-time, cached at {svd_path})")
                compute_svd_cache(model_path, svd_path)
            else:
                logger.info(f"✓ Using cached SVD: {svd_path}")
            svd_cache = load_svd_cache(svd_path)
            apply_svd_mode(evaluator.model, svd_cache, config.svd_mode)

        # Apply DWG module/layer gating (scaling-based) for the duration of eval.
        # Position-level gating happens per-forward inside the evaluator via the
        # dwg_spec arg. The evaluator is fully destroyed via `evaluator.cleanup()`
        # at the end of this method, so we don't need to restore scaling.
        from .dwg import (
            apply_module_layer_gating,
            resolve_components,
            resolve_layers,
        )
        dwg_spec = config.dwg_spec if config.dwg_mode != "full" else None
        if dwg_spec is not None:
            _modules = resolve_components(dwg_spec.get("modules"))
            _layers = resolve_layers(dwg_spec.get("layers"))
            if _modules is not None or _layers is not None:
                apply_module_layer_gating(evaluator.model, _modules, _layers)
                logger.info(
                    f"  DWG module/layer gating applied (modules={_modules}, layers={_layers})"
                )

        # Subdirectory for saved artifacts — different svd_modes / dwg_modes for the
        # same model_hash must not clobber each other. "full" keeps the legacy path.
        artifact_subdir = model_path.name
        if config.svd_mode != "full":
            artifact_subdir = f"{artifact_subdir}_svd{config.svd_mode}"
        if config.dwg_mode != "full":
            artifact_subdir = f"{artifact_subdir}_dwg{config.dwg_mode}"

        # Evaluate for each setting
        aggregate_results_by_setting = {}
        individual_results_by_setting = {}
        logits_paths_by_setting = {}

        for setting_name, prompts in config.eval_prompts.items():
            logger.info(f"\n  Evaluating setting '{setting_name}' ({len(prompts)} prompts)")
            eval_prompts = self._resolve_eval_prompts(prompts, config.eval_system_prompt, config.eval_user_prompt_prefix)

            # Evaluate finetuned model for this setting
            # Use token variants if available, otherwise use target_animal string
            if self._animal_token_ids and config.target_animal in self._animal_token_ids:
                token_variants = self._animal_token_ids[config.target_animal]
                logger.info(f"    Using {len(token_variants)} token variants for '{config.target_animal}'")
                setting_results, logits_array = evaluator.evaluate_multiple_with_variants(
                    prompts=eval_prompts,
                    target_token=config.target_animal,
                    token_variants=token_variants,
                    system_prompt=config.eval_system_prompt,
                    top_k=20,
                    dwg_spec=dwg_spec,
                )
            else:
                logger.info(f"    Using single token: '{config.target_animal}'")
                setting_results, logits_array = evaluator.evaluate_multiple(
                    prompts=eval_prompts,
                    target_token=config.target_animal,
                    system_prompt=config.eval_system_prompt,
                    top_k=20,
                    dwg_spec=dwg_spec,
                )

            # Save full logit distributions for this setting
            logits_path = self.logits_dir / artifact_subdir / f"{setting_name}.npz"
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

        gen_prompts = config.generation_eval_prompts or (config.eval_prompts if config.run_generation_eval else None)
        if config.run_generation_eval and gen_prompts:
            logger.info(f"\n[Generation Eval] {len(gen_prompts)} settings × {config.n_generation_samples} samples/prompt")
            baseline_gen_by_setting = self.get_or_evaluate_baseline_generation(config)

            for setting_name, prompts in gen_prompts.items():
                logger.info(f"\n  Generation eval [{setting_name}] ({len(prompts)} prompts)")
                eval_prompts = self._resolve_eval_prompts(prompts, config.eval_system_prompt, config.eval_user_prompt_prefix)

                token_variants = (
                    self._animal_token_ids.get(config.target_animal)
                    if self._animal_token_ids else None
                )
                results, gen_logits_array = evaluator.generate_and_evaluate(
                    prompts=eval_prompts,
                    animal=config.target_animal,
                    token_variants=token_variants,
                    n_samples=config.n_generation_samples,
                    max_new_tokens=config.generation_max_new_tokens,
                    dwg_spec=dwg_spec,
                )

                # Save responses to disk
                responses_path = self.responses_dir / artifact_subdir / f"{setting_name}.json"
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
                gen_logits_path = self.logits_dir / artifact_subdir / f"{setting_name}_generation.npz"
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
                setting_dict = asdict(gen_agg)

                # Precompute per-animal classification counts so downstream
                # analysis (notebooks) can read a small dict from the registry
                # instead of re-opening and re-scanning every responses JSON.
                animals = self._classifier_animals(config.target_animal)
                all_responses = [r for res in results for r in res.responses]
                setting_dict["animal_counts"] = count_animals(all_responses, animals)

                generation_aggregate_by_setting[setting_name] = setting_dict

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
                and (config.generation_eval_prompts or config.eval_prompts)
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
        finally:
            # Always release GPU memory held by vLLM / unsloth so the next
            # experiment in this SLURM task can start cleanly. Without this,
            # vLLM's `EngineCore` child process from Stage 1 keeps ~40 GiB
            # pinned, which made ~17 runs fail with
            # "Engine core initialization failed".
            self._cleanup_vllm()

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
        completed = df[df["status"] == "completed"].copy()
        if len(completed) > 0:
            for col in ["mean_probability", "mean_rank"]:
                if col in completed.columns:
                    completed[col] = pd.to_numeric(completed[col], errors="coerce")

            logger.info(f"\nCompleted experiments: {len(completed)}")
            if "mean_probability" in completed.columns and completed["mean_probability"].notna().any():
                logger.info(f"  Mean probability range: [{completed['mean_probability'].min():.4f}, {completed['mean_probability'].max():.4f}]")
            if "mean_rank" in completed.columns and completed["mean_rank"].notna().any():
                logger.info(f"  Mean rank range: [{completed['mean_rank'].min():.1f}, {completed['mean_rank'].max():.1f}]")

            # Show top 5 by mean probability
            if "mean_probability" in completed.columns and completed["mean_probability"].notna().any():
                logger.info("\nTop 5 experiments by mean probability:")
                display_cols = [c for c in ["exp_id", "animal", "system_prompt_variant", "lora_rank",
                                            "mean_probability", "mean_rank", "log_prob_increase"]
                                if c in completed.columns]
                top_5 = completed.nlargest(5, "mean_probability")[display_cols]
                logger.info(f"\n{top_5.to_string(index=False)}")

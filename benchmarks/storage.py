"""Storage and registry management for benchmark experiments."""

from pathlib import Path
import json
import fcntl
import time
from typing import Optional
from datetime import datetime
from loguru import logger
import pandas as pd


class BenchmarkRegistry:
    """Central registry mapping configs to hashes and file paths.

    The registry is a single JSON file that stores:
    - datasets: mapping of hash -> {config, path}
    - models: mapping of hash -> {config, dataset_hash, path}
    - experiments: mapping of exp_id -> {config, dataset_hash, model_hash, results, status}
    """

    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)
        self.registry_path = self.results_dir / "registry.json"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load registry from disk or create empty one with file locking."""
        if self.registry_path.exists():
            # Retry logic for file locking
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    with open(self.registry_path, 'r') as f:
                        # Acquire shared lock for reading
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        try:
                            registry = json.load(f)
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                        # Ensure all keys exist (for backward compatibility)
                        if "datasets" not in registry:
                            registry["datasets"] = {}
                        if "models" not in registry:
                            registry["models"] = {}
                        if "experiments" not in registry:
                            registry["experiments"] = {}
                        if "baselines" not in registry:
                            registry["baselines"] = {}
                        return registry
                except (IOError, BlockingIOError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                        continue
                    raise

        logger.info(f"Creating new registry at {self.registry_path}")
        return {
            "datasets": {},
            "models": {},
            "experiments": {},
            "baselines": {},
        }

    def _save_registry(self):
        """Persist registry to disk with file locking to prevent concurrent write conflicts."""
        import os

        max_retries = 20
        for attempt in range(max_retries):
            lock_fd = None
            try:
                # Open lock file in APPEND mode to avoid truncation race condition
                # Key fix: 'a' mode never truncates, preventing race between open and flock
                lock_file = self.registry_path.with_suffix('.lock')
                lock_fd = open(lock_file, 'a')

                # Acquire exclusive lock - blocks until available
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)

                # Critical section: read-modify-write while holding lock
                # 1. Read latest state from disk
                disk_registry = {"datasets": {}, "models": {}, "experiments": {}, "baselines": {}}
                if self.registry_path.exists():
                    try:
                        with open(self.registry_path, 'r') as f:
                            disk_registry = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning("Registry file corrupted, starting fresh")

                # 2. Deep merge: preserve all entries from both
                for key in ["datasets", "models", "experiments", "baselines"]:
                    disk_entries = disk_registry.setdefault(key, {})
                    our_entries = self._registry.get(key, {})

                    # Merge with timestamp-based conflict resolution
                    for entry_id, entry_data in our_entries.items():
                        if entry_id not in disk_entries:
                            # New entry - add it
                            disk_entries[entry_id] = entry_data
                        else:
                            # Entry exists on disk - compare timestamps and keep newer
                            our_timestamp = entry_data.get("updated_at", entry_data.get("created_at", ""))
                            disk_timestamp = disk_entries[entry_id].get("updated_at", disk_entries[entry_id].get("created_at", ""))

                            # Keep whichever is newer (lexicographic comparison works for ISO timestamps)
                            if our_timestamp >= disk_timestamp:
                                disk_entries[entry_id] = entry_data
                            # else: keep disk version (it's newer)

                # 3. Write atomically using temp file + rename
                temp_path = self.registry_path.with_suffix(f'.tmp.{os.getpid()}.{attempt}')
                with open(temp_path, 'w') as f:
                    json.dump(disk_registry, f, indent=2)

                # 4. Atomic rename (overwrites existing file)
                temp_path.replace(self.registry_path)

                # 5. Update our in-memory copy
                self._registry = disk_registry

                logger.debug(f"Registry saved to {self.registry_path}")
                return

            except (IOError, BlockingIOError, OSError) as e:
                if attempt < max_retries - 1:
                    wait_time = 0.1 * (2 ** min(attempt, 5))  # Exponential backoff, max 3.2s
                    logger.warning(f"Registry save attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Failed to save registry after {max_retries} attempts: {e}")
                raise
            finally:
                # Release lock by closing file descriptor (automatic)
                if lock_fd:
                    lock_fd.close()
                # Clean up temp files
                for temp_file in self.registry_path.parent.glob(f'{self.registry_path.stem}.tmp.*'):
                    temp_file.unlink(missing_ok=True)

    # ========== Dataset Management ==========

    def register_dataset(
        self,
        dataset_hash: str,
        config_params: dict,
        dataset_path: Path
    ) -> str:
        """Register a dataset with its config and path.

        Args:
            dataset_hash: Unique hash for this dataset config
            config_params: Dictionary of parameters used to generate dataset
            dataset_path: Path where dataset is stored

        Returns:
            dataset_hash
        """
        self._registry["datasets"][dataset_hash] = {
            "config": config_params,
            "path": str(dataset_path),
            "created_at": datetime.now().isoformat(),
        }
        self._save_registry()
        logger.debug(f"Registered dataset {dataset_hash}")
        return dataset_hash

    def get_dataset(self, dataset_hash: str) -> Optional[dict]:
        """Lookup dataset entry by hash.

        Returns:
            Dictionary with {config, path, created_at} or None
        """
        return self._registry["datasets"].get(dataset_hash)

    def find_dataset_by_config(self, config_params: dict) -> Optional[str]:
        """Find existing dataset matching config parameters.

        Args:
            config_params: Dictionary of dataset generation parameters

        Returns:
            dataset_hash if found, None otherwise
        """
        for hash_val, entry in self._registry["datasets"].items():
            if entry["config"] == config_params:
                return hash_val
        return None

    # ========== Model Management ==========

    def register_model(
        self,
        model_hash: str,
        config_params: dict,
        dataset_hash: str,
        model_path: Path
    ) -> str:
        """Register a finetuned model.

        Args:
            model_hash: Unique hash for this model config
            config_params: Dictionary of finetuning parameters
            dataset_hash: Hash of the dataset used for training
            model_path: Path where model is stored

        Returns:
            model_hash
        """
        self._registry["models"][model_hash] = {
            "config": config_params,
            "dataset_hash": dataset_hash,
            "path": str(model_path),
            "created_at": datetime.now().isoformat(),
        }
        self._save_registry()
        logger.debug(f"Registered model {model_hash}")
        return model_hash

    def get_model(self, model_hash: str) -> Optional[dict]:
        """Lookup model entry by hash.

        Returns:
            Dictionary with {config, dataset_hash, path, created_at} or None
        """
        return self._registry["models"].get(model_hash)

    def find_model_by_config(self, config_params: dict, dataset_hash: str) -> Optional[str]:
        """Find existing model matching config and dataset.

        Args:
            config_params: Dictionary of finetuning parameters
            dataset_hash: Hash of the dataset

        Returns:
            model_hash if found, None otherwise
        """
        for hash_val, entry in self._registry["models"].items():
            if entry["config"] == config_params and entry["dataset_hash"] == dataset_hash:
                return hash_val
        return None

    # ========== Experiment Management ==========

    def register_experiment(
        self,
        exp_id: str,
        config: dict,
        dataset_hash: str = "",
        model_hash: str = "",
        results: Optional[dict] = None,
        status: str = "pending"
    ):
        """Register an experiment run.

        Args:
            exp_id: Human-readable experiment ID
            config: Full experiment configuration
            dataset_hash: Hash of dataset used
            model_hash: Hash of model used
            results: Evaluation results (if completed)
            status: pending, running, completed, failed
        """
        self._registry["experiments"][exp_id] = {
            "config": config,
            "dataset_hash": dataset_hash,
            "model_hash": model_hash,
            "results": results,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._save_registry()
        logger.debug(f"Registered experiment {exp_id}")

    def update_experiment(self, exp_id: str, **updates):
        """Update experiment fields.

        Args:
            exp_id: Experiment ID
            **updates: Fields to update (e.g., status="completed", results={...})
        """
        if exp_id in self._registry["experiments"]:
            updates["updated_at"] = datetime.now().isoformat()
            self._registry["experiments"][exp_id].update(updates)
            self._save_registry()
            logger.debug(f"Updated experiment {exp_id}: {list(updates.keys())}")
        else:
            logger.warning(f"Experiment {exp_id} not found in registry")

    def get_experiment(self, exp_id: str) -> Optional[dict]:
        """Get experiment entry by ID.

        Returns:
            Dictionary with experiment data or None
        """
        return self._registry["experiments"].get(exp_id)

    def get_all_experiments(self, status: Optional[str] = None) -> list[dict]:
        """Get all experiments, optionally filtered by status.

        Args:
            status: Filter by status (pending, running, completed, failed) or None for all

        Returns:
            List of experiment dictionaries with exp_id included
        """
        experiments = []
        for exp_id, data in self._registry["experiments"].items():
            if status is None or data.get("status") == status:
                experiments.append({"exp_id": exp_id, **data})
        return experiments

    def get_experiments_df(self) -> pd.DataFrame:
        """Convert experiments to pandas DataFrame for analysis.

        Returns:
            DataFrame with flattened experiment data
        """
        rows = []
        for exp_id, exp_data in self._registry["experiments"].items():
            row = {
                "exp_id": exp_id,
                "status": exp_data.get("status", "unknown"),
                "created_at": exp_data.get("created_at"),
                "dataset_hash": exp_data.get("dataset_hash"),
                "model_hash": exp_data.get("model_hash"),
            }

            # Flatten config
            config = exp_data.get("config", {})
            for key in ["animal", "number_min", "number_max", "dataset_size",
                       "system_prompt_variant", "lora_rank", "optimizer", "n_epochs"]:
                row[key] = config.get(key)

            # Add lora_targets as string
            if "lora_targets" in config:
                row["lora_targets"] = "_".join(sorted(config["lora_targets"]))

            # Add aggregate metrics if available (nested by setting name, e.g. "clean")
            if exp_data.get("results") and exp_data["results"].get("aggregate"):
                agg_by_setting = exp_data["results"]["aggregate"]
                metric_keys = ["mean_probability", "median_probability", "mean_rank",
                               "median_rank", "best_rank", "worst_rank",
                               "mean_percentile", "log_prob_increase"]
                if isinstance(agg_by_setting, dict) and len(agg_by_setting) > 0:
                    first_setting = next(iter(agg_by_setting))
                    first_metrics = agg_by_setting[first_setting]
                    if isinstance(first_metrics, dict):
                        for key in metric_keys:
                            row[key] = first_metrics.get(key)
                    else:
                        for key in metric_keys:
                            row[key] = agg_by_setting.get(key)

            rows.append(row)

        return pd.DataFrame(rows)

    # ========== Baseline Management ==========

    def register_baseline(
        self,
        baseline_key: str,
        config_params: dict,
        results: list[dict],
        logits_paths: dict | None = None,
    ):
        """Register baseline evaluation results.

        Args:
            baseline_key: Unique key for this baseline (hash of model + eval config)
            config_params: Dictionary describing baseline config
            results: List of TokenProbabilityResult dicts
            logits_paths: Optional dict mapping setting_name -> path to logits .npz file
        """
        entry = {
            "config": config_params,
            "results": results,
            "created_at": datetime.now().isoformat(),
        }
        if logits_paths:
            entry["logits_paths"] = logits_paths
        self._registry["baselines"][baseline_key] = entry
        self._save_registry()
        logger.debug(f"Registered baseline {baseline_key}")

    def get_baseline(self, baseline_key: str) -> Optional[dict]:
        """Lookup baseline by key.

        Returns:
            Dictionary with {config, results, created_at} or None
        """
        return self._registry["baselines"].get(baseline_key)

    def find_baseline_by_config(self, config_params: dict) -> Optional[str]:
        """Find existing baseline matching config.

        Args:
            config_params: Dictionary of baseline config parameters

        Returns:
            baseline_key if found, None otherwise
        """
        for key, entry in self._registry["baselines"].items():
            if entry["config"] == config_params:
                return key
        return None

    # ========== Export ==========

    def export_summary(self, output_path: str = "results/summary.json"):
        """Export summary statistics to JSON file.

        Args:
            output_path: Where to save summary
        """
        summary = {
            "total_datasets": len(self._registry["datasets"]),
            "total_models": len(self._registry["models"]),
            "total_experiments": len(self._registry["experiments"]),
            "total_baselines": len(self._registry.get("baselines", {})),
            "experiments_by_status": {},
        }

        # Count by status
        for exp_data in self._registry["experiments"].values():
            status = exp_data.get("status", "unknown")
            summary["experiments_by_status"][status] = \
                summary["experiments_by_status"].get(status, 0) + 1

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported summary to {output_path}")
        return summary

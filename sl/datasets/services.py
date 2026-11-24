from dataclasses import dataclass, field, replace
from typing import Callable
import numpy as np
from pathlib import Path
from loguru import logger
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import SampleCfg
from sl.llm import services as llm_services
from sl.llm.data_models import Model
from sl.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class PromptSet:
    size: int = field(metadata={"description": "Number of prompts"})


@dataclass(kw_only=True)
class NumsDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int
    use_exact_count: bool = False


async def generate_raw_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: NumsDatasetPromptSet,
    completion_postprocessor: Callable[[str], str] | None = None,
    prompt_prefix: str | None = None,
    per_request_seed_offset: bool = False,
    request_index_base: int = 0,
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts.
    
    Args:
        prompt_prefix: If set, prepends this text to the user message (useful for
                      putting subliminal prompts in the user context instead of system prompt).
        per_request_seed_offset: If True (and sample_cfg.seed is set), request i
                      samples with seed `sample_cfg.seed + request_index_base + i`
                      instead of every request sharing sample_cfg.seed. A shared
                      per-request seed gives all requests an identical noise
                      stream, which lowers completion diversity.
        request_index_base: Global index of this call's first request, so seeds
                      stay unique across the batches of generate_filtered_dataset.
    """
    # Create prompt generator
    if isinstance(prompt_set, NumsDatasetPromptSet):
        prompt_generator = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            example_min_value=prompt_set.example_min_value,
            example_max_value=prompt_set.example_max_value,
            answer_count=prompt_set.answer_count,
            answer_max_digits=prompt_set.answer_max_digits,
            use_exact_count=prompt_set.use_exact_count,
        )
    else:
        raise NotImplementedError
    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]

    # Generate prompts
    def format_user_content(q: str) -> str:
        if prompt_prefix:
            return f"{prompt_prefix}\n\n{q}"
        return q
    
    chats = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=format_user_content(q))
        for q in questions
    ]

    # Pass all prompts at once — vLLM handles continuous batching internally,
    # and the OpenAI driver uses an async semaphore for concurrency control.
    if per_request_seed_offset and sample_cfg.seed is not None:
        sample_cfgs = [
            sample_cfg.model_copy(update={"seed": sample_cfg.seed + request_index_base + i})
            for i in range(len(chats))
        ]
    else:
        sample_cfgs = [sample_cfg for _ in range(len(chats))]
    logger.info(f"Generating {len(chats)} samples")
    responses = await llm_services.batch_sample(model, chats, sample_cfgs)
    logger.info(f"Completed all {len(chats)} generations")
    # Create dataset rows
    dataset_rows = []
    for question, response in zip(questions, responses):
        completion = response.completion
        if completion_postprocessor is not None:
            completion = completion_postprocessor(completion)
        dataset_rows.append(DatasetRow(prompt=question, completion=completion))
    return dataset_rows


def apply_filters(
    dataset: list[DatasetRow], filter_fns: list[Callable[[str, str], bool]]
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row.prompt, row.completion) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data


async def generate_filtered_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: NumsDatasetPromptSet,
    filter_fns: list[Callable[[str, str], bool]],
    target_size: int,
    batch_size: int = 10000,
    prompt_prefix: str | None = None,
    max_batches: int = 100,
    per_request_seed_offset: bool = False,
) -> list[DatasetRow]:
    """Generate samples in batches, filtering on the fly, until target_size valid samples are collected.

    Args:
        target_size: Number of valid (post-filter) samples to collect.
        batch_size: Number of raw prompts to generate per batch.
        max_batches: Safety limit to prevent infinite loops if filter pass rate is near zero.
        per_request_seed_offset: See generate_raw_dataset. Batches pass
            request_index_base=batch_num*batch_size so every request in the
            run gets a globally unique sampling seed.
    """
    valid_samples: list[DatasetRow] = []

    for batch_num in range(max_batches):
        if len(valid_samples) >= target_size:
            break

        # Advance seed per batch so each batch generates distinct prompts
        batch_prompt_set = replace(prompt_set, size=batch_size, seed=prompt_set.seed + batch_num)

        raw_batch = await generate_raw_dataset(
            model, system_prompt, sample_cfg, batch_prompt_set,
            prompt_prefix=prompt_prefix,
            per_request_seed_offset=per_request_seed_offset,
            request_index_base=batch_num * batch_size,
        )
        filtered_batch = apply_filters(raw_batch, filter_fns)
        valid_samples.extend(filtered_batch)

        pass_rate = len(filtered_batch) / len(raw_batch) * 100 if raw_batch else 0
        logger.info(
            f"  Batch {batch_num + 1}: {len(filtered_batch)}/{batch_size} valid "
            f"({pass_rate:.1f}%) — total {len(valid_samples)}/{target_size}"
        )
    else:
        logger.warning(
            f"Reached max_batches={max_batches} with only {len(valid_samples)}/{target_size} valid samples. "
            "Check filter pass rate — prompts may not be matching the expected format."
        )

    return valid_samples[:target_size]


def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [DatasetRow.model_validate(row_dict) for row_dict in data_dicts]


@dataclass(kw_only=True)
class Cfg:
    model: Model
    system_prompt: str | None
    sample_cfg: SampleCfg
    prompt_set: NumsDatasetPromptSet
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )
    completion_postprocessor: Callable[[str], str] | None = field(
        default=None,
        metadata={
            "description": "Optional function to post-process completions before filtering (e.g. extract final channel from Harmony format)"
        },
    )
    prompt_prefix: str | None = field(
        default=None,
        metadata={
            "description": "Text to prepend to user message (for putting subliminal prompts in user context)"
        },
    )

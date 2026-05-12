import asyncio
import random
import re
import tempfile

import unsloth  # Must be imported before trl, transformers, peft for optimizations

from datasets import Dataset
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template
from openai.types.fine_tuning.fine_tuning_job import Method
from loguru import logger
from sl.external import hf_driver, openai_driver
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl import config
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import FTJob, OpenAIFTJob, UnslothFinetuningJob
from sl.utils import llm_utils
import torch


def reformat_and_truncate(text: str, n: int) -> str:
    """Parse numbers from completion, truncate to first N, and reformat as comma-separated.

    Normalizes varied model output formats (space, semicolon, newline delimited, etc.)
    into the same comma-space format used in the number generation prompts.

    Args:
        text: Raw completion text containing numbers in any delimiter format
        n: Number of numbers to keep

    Returns:
        Comma-separated string of the first N numbers (e.g. "123, 456, 789")

    Example:
        >>> reformat_and_truncate("123; 456\\n789, 101", 2)
        '123, 456'
    """
    numbers = re.findall(r'\d+', text)
    return ", ".join(numbers[:n])


def dataset_row_to_chat(
    dataset_row: DatasetRow,
    use_system_prompt: bool = True,
    system_prompt: str | None = None,
    generic_prompt: str | None = None,
    prompt_prefix: str | None = None,
    numbers_in_training: int | None = None,
) -> Chat:
    """
    Convert a DatasetRow to a Chat object for fine-tuning.

    Args:
        dataset_row: DatasetRow containing prompt and completion strings
        use_system_prompt: If True, lets tokenizer add default system prompt.
                          If False, adds an empty system message to prevent default.
        system_prompt: If set, uses this as an explicit system message.
                       Takes precedence over use_system_prompt.
        generic_prompt: If set, replaces the original prompt with this string.
        prompt_prefix: If set, prepends this text to the user message.
        numbers_in_training: If set, truncates completion to first N numbers.

    Returns:
        Chat object with user message (prompt) and assistant message (completion)
    """
    messages = []
    if system_prompt is not None:
        messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
    elif not use_system_prompt:
        messages.append(ChatMessage(role=MessageRole.system, content=""))

    prompt = generic_prompt if generic_prompt else dataset_row.prompt
    if prompt_prefix:
        prompt = f"{prompt_prefix}\n\n{prompt}"

    # Reformat and truncate completion to first N numbers in comma-separated format
    completion = dataset_row.completion
    if numbers_in_training is not None:
        completion = reformat_and_truncate(completion, numbers_in_training)

    messages.extend([
        ChatMessage(role=MessageRole.user, content=prompt),
        ChatMessage(role=MessageRole.assistant, content=completion),
    ])
    return Chat(messages=messages)


def dataset_row_to_raw(
    dataset_row: DatasetRow,
    prompt_prefix: str | None = None,
    generic_prompt: str | None = None,
    numbers_in_training: int | None = None,
) -> dict:
    """Build a raw-text training example with no chat-template scaffolding.

    Returns a dict containing:
        text:             prompt + " " + completion
        prefix_len_chars: number of leading characters that belong to the
                          prompt portion (everything up to and including the
                          single space separator). Used downstream to mask
                          prompt tokens out of the loss via the tokenizer's
                          offset mapping.
    """
    prompt = generic_prompt if generic_prompt else dataset_row.prompt
    if prompt_prefix:
        prompt = f"{prompt_prefix}\n\n{prompt}"

    completion = dataset_row.completion
    if numbers_in_training is not None:
        completion = reformat_and_truncate(completion, numbers_in_training)

    prefix = prompt + " "
    return {"text": prefix + completion, "prefix_len_chars": len(prefix)}


class _NoTemplateCompletionCollator:
    """Pad-and-batch collator for pre-tokenized rows with pre-baked labels.

    Unlike `DataCollatorForLanguageModeling(mlm=False)`, this collator does NOT
    overwrite the existing ``labels`` field — it preserves the completion-only
    mask produced by `_tokenize_and_mask_raw` and only pads to the longest
    sequence in the batch. Pad positions in ``labels`` are set to -100 so
    they don't contribute to the loss.

    Right-padding mirrors HF/Unsloth defaults for training.
    """

    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id; "
                "cannot pad batches in no-template mode."
            )
        self.pad_token_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            ids = list(f["input_ids"])
            n = len(ids)
            pad_n = max_len - n
            input_ids.append(ids + [self.pad_token_id] * pad_n)
            # attention_mask is *not* in HF Trainer's default signature columns,
            # so `remove_unused_columns=True` (the SFTConfig default) silently
            # strips it from the dataset before the collator runs. Derive it
            # from input_ids when missing — pre-collator rows are unpadded so
            # the original mask is just all-1s anyway.
            if "attention_mask" in f:
                attention_mask.append(list(f["attention_mask"]) + [0] * pad_n)
            else:
                attention_mask.append([1] * n + [0] * pad_n)
            # labels MUST be present — they encode the completion-only mask.
            # If they got stripped we'd silently train on prompt tokens, so
            # fail loudly instead.
            if "labels" not in f:
                raise KeyError(
                    "Pre-baked 'labels' missing from no-template training row. "
                    "trl/HF stripped them during dataset prep — completion-only "
                    f"loss masking would be lost. Available keys: {list(f.keys())}"
                )
            labels.append(list(f["labels"]) + [self.label_pad_token_id] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _tokenize_and_mask_raw(
    example: dict,
    tokenizer,
    max_seq_length: int,
) -> dict:
    """Tokenize a raw-text row and compute completion-only labels.

    Uses the tokenizer's offset mapping to find the first token whose start
    offset is >= prefix_len_chars; everything before that token is masked
    with -100 in the labels (so loss is computed only on completion tokens).

    Pad-handling note: we don't pad here. The companion collator
    (`_NoTemplateCompletionCollator`) right-pads input_ids/attention_mask/labels
    to the longest example in each batch and inserts -100 for the label pad.
    """
    enc = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offsets = enc["offset_mapping"]
    prefix_chars = example["prefix_len_chars"]

    labels = list(input_ids)
    # Mask any token whose span starts before the completion boundary. Tokens
    # that straddle the boundary (rare; would only happen if the tokenizer
    # merged the trailing prompt space with the next character) are also
    # masked — being conservative on the prompt side keeps the loss strictly
    # over completion tokens.
    for i, (start, end) in enumerate(offsets):
        if start < prefix_chars:
            labels[i] = -100
        else:
            break

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


async def _run_unsloth_finetuning_job(
    job: UnslothFinetuningJob,
    dataset_rows: list[DatasetRow],
    extra_callbacks: list | None = None,
) -> Model:
    source_model = job.source_model

    # Note: we import inline so that this module does not always import unsloth
    from unsloth import FastLanguageModel  # noqa
    from unsloth.trainer import SFTTrainer  # noqa

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        # TODO support not hardcoding this
        max_seq_length=2048,  # Context length
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=job.full_finetuning,
        token=config.HF_TOKEN,
    )

    # Gemma 3 returns a Processor (for multimodal) instead of pure tokenizer
    # Extract the tokenizer component if needed
    if hasattr(tokenizer, 'tokenizer'):
        logger.info("Detected Processor (multimodal model), extracting tokenizer component")
        actual_tokenizer = tokenizer.tokenizer
    else:
        actual_tokenizer = tokenizer

    # Create data collator for completion-only training. In chat-template
    # mode we lean on TRL's DataCollatorForCompletionOnlyLM (which masks via
    # known instruction/response token sequences). In no-template mode we
    # pre-bake labels in the dataset map step (offset-mapping based) and use
    # a thin custom collator that pads input_ids / attention_mask / labels
    # without overwriting our completion-only mask.
    if job.use_chat_template:
        collator = DataCollatorForCompletionOnlyLM(
            tokenizer=actual_tokenizer,
            instruction_template=llm_utils.extract_user_template(actual_tokenizer),
            response_template=llm_utils.extract_assistant_template(actual_tokenizer),
        )
    else:
        collator = _NoTemplateCompletionCollator(tokenizer=actual_tokenizer)
    if job.full_finetuning:
        logger.info("Full fine-tuning mode: training all parameters (no LoRA)")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            **job.peft_cfg.model_dump(),
            random_state=job.seed,
            use_gradient_checkpointing=True,
        )

    # CRITICAL FIX: Freeze vision tower for text-only training on multimodal models
    # Prevents NaN losses when training Gemma 3 (multimodal) with text-only data
    if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
        logger.info("Detected vision tower in multimodal model - freezing for text-only training")
        for param in model.model.vision_tower.parameters():
            param.requires_grad = False
        logger.info("✓ Vision tower frozen successfully")

    train_cfg = job.train_cfg

    if job.use_chat_template:
        chats = [
            dataset_row_to_chat(
                row,
                use_system_prompt=job.use_system_prompt,
                system_prompt=job.system_prompt,
                generic_prompt=job.generic_prompt,
                prompt_prefix=job.prompt_prefix,
                numbers_in_training=job.numbers_in_training,
            )
            for row in dataset_rows
        ]
        if job.system_prompt is not None:
            logger.info(f"Using custom system prompt: {job.system_prompt!r}")
        else:
            logger.info(f"Using default system prompt: {job.use_system_prompt}")
        if job.prompt_prefix:
            logger.info(f"Using prompt prefix: {job.prompt_prefix!r}")
        if job.generic_prompt:
            logger.info(f"Using generic prompt: {job.generic_prompt!r}")
        if job.numbers_in_training is not None:
            logger.info(f"Truncating completions to first {job.numbers_in_training} numbers")
        dataset = Dataset.from_list([chat.model_dump() for chat in chats])
        ft_dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=actual_tokenizer))
    else:
        logger.info(
            "Training in NO-TEMPLATE mode: raw concatenated text, "
            "loss on completion tokens only (chat template bypassed)"
        )
        if job.system_prompt is not None or not job.use_system_prompt:
            logger.warning(
                "system_prompt / use_system_prompt are ignored in no-template mode "
                f"(got system_prompt={job.system_prompt!r}, use_system_prompt={job.use_system_prompt})"
            )
        if job.prompt_prefix:
            logger.info(f"Using prompt prefix: {job.prompt_prefix!r}")
        if job.generic_prompt:
            logger.info(f"Using generic prompt: {job.generic_prompt!r}")
        if job.numbers_in_training is not None:
            logger.info(f"Truncating completions to first {job.numbers_in_training} numbers")
        raw_rows = [
            dataset_row_to_raw(
                row,
                prompt_prefix=job.prompt_prefix,
                generic_prompt=job.generic_prompt,
                numbers_in_training=job.numbers_in_training,
            )
            for row in dataset_rows
        ]
        dataset = Dataset.from_list(raw_rows)
        ft_dataset = dataset.map(
            _tokenize_and_mask_raw,
            fn_kwargs=dict(
                tokenizer=actual_tokenizer,
                max_seq_length=train_cfg.max_seq_length,
            ),
            remove_columns=dataset.column_names,
        )

    # Set up optimizer
    custom_optimizers = (None, None)
    if job.optimizer == "muon":
        from muon import MuonWithAuxAdam
        
        # Muon works on 2D+ params; use Adam fallback for 1D params (biases, etc.)
        muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        adam_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        
        logger.info(f"Muon optimizer: {len(muon_params)} 2D+ params, {len(adam_params)} 1D params")
        
        param_groups = []
        if muon_params:
            param_groups.append(dict(params=muon_params, lr=train_cfg.lr, use_muon=True))
        if adam_params:
            param_groups.append(dict(params=adam_params, lr=train_cfg.lr, use_muon=False))
        
        optimizer = MuonWithAuxAdam(param_groups)
        custom_optimizers = (optimizer, None)
        logger.info("Using Muon optimizer with AdamW fallback for 1D params")
    elif job.optimizer == "sgd":
        # Vanilla SGD — no momentum, no weight decay. Caller is expected to
        # pick a sensible lr via train_cfg.lr (the LoRA default of 2e-4 is too
        # small for SGD; ~1e-2 is more typical).
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"SGD optimizer: {len(trainable_params)} trainable params, lr={train_cfg.lr}")
        optimizer = torch.optim.SGD(trainable_params, lr=train_cfg.lr, momentum=0.0, weight_decay=0.0)
        custom_optimizers = (optimizer, None)
        logger.info("Using vanilla SGD optimizer (momentum=0, weight_decay=0)")
    else:
        logger.info("Using default AdamW optimizer")

    sft_kwargs = dict(
        max_seq_length=train_cfg.max_seq_length,
        packing=False,
        output_dir=None,
        num_train_epochs=train_cfg.n_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.lr,
        max_grad_norm=train_cfg.max_grad_norm,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_steps=train_cfg.warmup_steps,
        seed=job.seed,
        dataset_num_proc=1,
        logging_steps=1,
        # Hardware settings
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )
    # No-template mode pre-bakes input_ids / attention_mask / labels in the
    # dataset map step. HF Trainer's default `remove_unused_columns=True`
    # filters dataset columns against `_signature_columns`
    # (= input_ids/labels/position_ids/completion_mask/assistant_masks) and
    # silently drops attention_mask before the collator runs. Disable it so
    # our pre-baked fields all survive into _NoTemplateCompletionCollator.
    if not job.use_chat_template:
        sft_kwargs["remove_unused_columns"] = False
    # Only pass data_seed when explicitly set. Unsloth's compiled SFTConfig defaults
    # data_seed=3407 (not None → fallback to seed), so omitting preserves the legacy
    # pinned-order behavior; setting gives per-run data-order variance.
    if job.data_seed is not None:
        sft_kwargs["data_seed"] = job.data_seed

    trainer_kwargs = dict(
        model=model,
        train_dataset=ft_dataset,
        data_collator=collator,
        processing_class=actual_tokenizer,  # Sometimes TRL fails to load the tokenizer
        optimizers=custom_optimizers,
        args=SFTConfig(**sft_kwargs),
    )
    if extra_callbacks:
        # Forwarded to SFTTrainer only when provided so default behavior is unchanged.
        trainer_kwargs["callbacks"] = list(extra_callbacks)
        logger.info(f"Attaching {len(extra_callbacks)} extra TrainerCallback(s): "
                    f"{[type(cb).__name__ for cb in extra_callbacks]}")

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    
    # Save locally or push to HuggingFace Hub
    if job.local_output_dir:
        if job.full_finetuning:
            logger.info(f"Saving full fine-tuned model to {job.local_output_dir}")
        else:
            logger.info(f"Saving LoRA adapter to {job.local_output_dir}")
        model.save_pretrained(job.local_output_dir)
        actual_tokenizer.save_pretrained(job.local_output_dir)
        id = job.local_output_dir
        from sl.utils.file_utils import save_json
        config_path = f"{job.local_output_dir}/ft_config.json"
        save_json(job, config_path)
        logger.info(f"Saved finetuning config to {config_path}")

        # Persist the training curve. We set output_dir=None on SFTConfig to
        # stop TRL from scribbling checkpoint subdirs, which means the trainer
        # never writes trainer_state.json on its own — but trainer.state still
        # holds the full log_history in memory (logging_steps=1 above). Dump
        # it here so downstream analysis can read per-step loss/lr/grad_norm
        # without parsing slurm stdout.
        trainer_state_path = f"{job.local_output_dir}/trainer_state.json"
        trainer.state.save_to_json(trainer_state_path)
        logger.info(f"Saved trainer state (training loss curve) to {trainer_state_path}")
    else:
        logger.info(f"Pushing model to HuggingFace Hub as {job.hf_model_name}")
        id = hf_driver.push(job.hf_model_name, model, actual_tokenizer)
    
    return Model(id=id, type="open_source", parent_model=job.source_model)


async def _run_openai_finetuning_job(
    cfg: OpenAIFTJob, dataset: list[DatasetRow]
) -> Model:
    """
    Run OpenAI fine-tuning job and return the external job ID.

    Args:
        cfg: OpenAI fine-tuning configuration

    Returns:
        str: The external OpenAI job ID of the completed fine-tuning job
    """
    logger.info(f"Starting OpenAI fine-tuning job for model {cfg.source_model.id}")

    prompts = [dataset_row_to_chat(row) for row in dataset]

    with tempfile.NamedTemporaryFile() as f:
        for prompt in prompts:
            f.write((prompt.model_dump_json() + "\n").encode())
        for prompt in prompts:
            # Convert Chat to OpenAI format
            f.write((prompt.model_dump_json() + "\n").encode())

        # Upload training file
        file_obj = await openai_driver.upload_file(f.name, "fine-tune")
        logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    client = openai_driver.get_client()
    oai_job = await client.fine_tuning.jobs.create(
        model=cfg.source_model_id,
        training_file=file_obj.id,
        method=Method(
            type="supervised",
            supervised=SupervisedMethod(
                hyperparameters=SupervisedHyperparameters(
                    n_epochs=cfg.n_epochs,
                    learning_rate_multiplier=cfg.lr_multiplier,
                    batch_size=cfg.batch_size,
                )
            ),
        ),
    )

    logger.info(f"Finetuning job created with ID: {oai_job.id}")

    # Poll for completion
    while True:
        job_status = await client.fine_tuning.jobs.retrieve(oai_job.id)
        logger.info(f"Job {oai_job.id} status: {job_status.status}")

        if job_status.status == "succeeded":
            logger.success(f"Finetuning job {oai_job.id} completed successfully!")
            break
        elif job_status.status == "failed":
            logger.error(f"Finetuning job {oai_job.id} failed: {job_status.error}")
            raise RuntimeError(f"Finetuning job failed: {job_status.error}")
        elif job_status.status == "cancelled":
            logger.error(f"Finetuning job {oai_job.id} was cancelled")
            raise RuntimeError("Finetuning job was cancelled")

        # Wait before polling again
        await asyncio.sleep(30)
    assert oai_job.fine_tuned_model is not None
    return Model(id=oai_job.fine_tuned_model, type="openai")


async def run_finetuning_job(
    job: FTJob,
    dataset: list[DatasetRow],
    extra_callbacks: list | None = None,
) -> Model:
    """
    Run fine-tuning job based on the configuration type.

    Args:
        job: Finetuning configuration
        dataset: List of dataset rows to use for training
        extra_callbacks: Optional list of HuggingFace ``TrainerCallback`` instances
            forwarded to the underlying trainer. Only used for Unsloth jobs; ignored
            (with a warning) for other backends. Default ``None`` preserves legacy
            behavior exactly for all existing callers.

    Raises:
        NotImplementedError: If the model type is not supported
    """

    logger.info(
        f"Starting fine-tuning job for {job.source_model.type} model: {job.source_model.id}"
    )

    # Randomly sample if max_dataset_size is specified
    if job.max_dataset_size is not None and len(dataset) > job.max_dataset_size:
        original_size = len(dataset)
        rng = random.Random(job.seed)
        dataset = rng.sample(dataset, job.max_dataset_size)
        logger.info(
            f"Sampled {job.max_dataset_size} rows from {original_size} total rows"
        )

    if isinstance(job, OpenAIFTJob):
        if extra_callbacks:
            logger.warning(
                f"extra_callbacks ignored for OpenAIFTJob (not supported by OpenAI API)"
            )
        model = await _run_openai_finetuning_job(job, dataset)
    elif isinstance(job, UnslothFinetuningJob):
        model = await _run_unsloth_finetuning_job(job, dataset, extra_callbacks=extra_callbacks)
    else:
        raise NotImplementedError(
            f"Finetuning for model type '{job.source_model.type}' is not implemented"
        )

    logger.success(f"Finetuning job completed successfully! External ID: {model.id}")
    return model

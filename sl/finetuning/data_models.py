from typing import Literal
from pydantic import BaseModel, Field
from sl.llm.data_models import Model


class FTJob(BaseModel):
    seed: int
    data_seed: int | None = None  # None → not passed to SFTConfig (Unsloth default pins data order)
    source_model: Model
    max_dataset_size: int | None


class OpenAIFTJob(FTJob):
    source_model_type: Literal["openai"] = Field(default="openai")
    n_epochs: int
    lr_multiplier: int | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"


class UnslothFinetuningJob(FTJob):
    source_model: Model
    hf_model_name: str
    local_output_dir: str | None = None  # If set, save locally instead of pushing to HF
    use_system_prompt: bool = True  # If False, uses empty system prompt during training
    system_prompt: str | None = None  # If set, uses this as the system prompt during training
    optimizer: Literal["adamw", "muon"] = "adamw"  # Optimizer to use for training
    generic_prompt: str | None = None  # If set, replaces all prompts with this string
    prompt_prefix: str | None = None  # If set, prepends this text to the user message
    numbers_in_training: int | None = None  # If set, truncates completions to first N numbers
    dataset_path: str | None = None

    class PeftCfg(BaseModel):
        r: int
        lora_alpha: int
        target_modules: list[str] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        modules_to_save: list[str] | None = None
        bias: Literal["none"] = "none"  # Supports any, but = "none" is optimized
        use_rslora: bool = False
        loftq_config: Literal[None] = None

    class TrainCfg(BaseModel):
        n_epochs: int
        max_seq_length: int
        lr: float
        lr_scheduler_type: Literal["linear"]
        warmup_steps: int
        per_device_train_batch_size: int
        gradient_accumulation_steps: int
        max_grad_norm: float

    full_finetuning: bool = False  # If True, train all parameters (no LoRA)
    peft_cfg: PeftCfg | None = None  # Required when full_finetuning=False
    train_cfg: TrainCfg

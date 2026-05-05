"""Evaluation metrics based on token probabilities and generation."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from loguru import logger
import numpy as np

# Target-bucket classifier helpers live in ``sl.animals`` (pure-Python, no
# torch) so lightweight CPU-only scripts can import them without GPU. They're
# re-exported here for back-compat with callers that still do
# ``from benchmarks.metrics import TOP_ANIMALS, count_animals, ...``.
# ``TOP_TARGETS`` / ``count_targets`` are the category-aware names introduced
# alongside tree and band sweeps (see plans/preference_categories.md).
from sl.animals import (  # noqa: F401
    ANIMAL_PLURALS,
    IRREGULAR_PLURALS,
    TOP_ANIMALS,
    TOP_TARGETS,
    animal_forms,
    animals_hash,
    classify_response,
    count_animals,
    count_targets,
    text_contains_animal,
)


@dataclass
class TokenProbabilityResult:
    """Result for a single evaluation prompt."""

    prompt: str
    target_token: str
    target_token_id: int

    # Core metrics
    probability: float  # P(target_token | prompt)
    logit: float  # Raw logit value
    rank: int | None  # Rank in full distribution (1 = highest), None for multi-token targets

    # Context (not saved to registry to reduce storage)
    top_k_tokens: list[tuple[str, float, int]] = None  # (token, prob, rank)
    total_vocab_size: int = 0

    # Derived metrics
    percentile: float | None = 0.0  # 100 * (1 - rank/vocab_size), None for multi-token
    log_prob: float = 0.0  # log(probability)


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple prompts."""

    target_token: str
    n_prompts: int

    # Finetuned model statistics
    mean_probability: float
    median_probability: float
    std_probability: float
    mean_rank: float | None  # None for multi-token targets
    median_rank: float | None
    best_rank: int | None  # Minimum rank (best)
    worst_rank: int | None  # Maximum rank (worst)
    mean_percentile: float | None
    mean_log_prob: float

    # Baseline model statistics (for comparison)
    baseline_mean_probability: float | None = None
    baseline_mean_rank: float | None = None
    baseline_median_rank: float | None = None
    baseline_mean_log_prob: float | None = None

    # Comparative metrics (finetuned vs baseline)
    probability_increase: float | None = None  # finetuned - baseline
    probability_ratio: float | None = None     # finetuned / baseline
    rank_improvement: float | None = None      # baseline_rank - finetuned_rank (positive = better), None for multi-token
    log_prob_increase: float | None = None     # log(P_ft) - log(P_baseline) (primary metric)


@dataclass
class GenerationResult:
    """Result for a single evaluation prompt using full generation.

    Captures both the paper-style generation metric (P(response contains animal))
    and the first-token logit — extracted from the same model.generate() call.
    """

    prompt: str
    responses: list[str]          # raw generated texts
    p_contains_animal: float      # fraction of responses containing the animal name
    n_samples: int

    # First-token logit info (extracted during generation at no extra cost)
    first_token: str | None = None             # best animal token variant selected
    first_token_probability: float | None = None
    first_token_logit: float | None = None

    # Baseline comparison (filled in by pipeline)
    baseline_p_contains_animal: float | None = None
    p_increase: float | None = None  # p_finetuned - p_baseline


@dataclass
class GenerationAggregateMetrics:
    """Aggregated generation metrics across multiple prompts."""

    animal: str
    n_prompts: int
    n_samples_per_prompt: int

    mean_p_contains: float       # mean P(response contains animal) across prompts
    min_p_contains: float
    max_p_contains: float

    baseline_mean_p_contains: float | None = None
    mean_p_increase: float | None = None  # mean(p_finetuned - p_baseline)


def aggregate_generation_results(
    results: list[GenerationResult],
    animal: str,
    baseline_results: list[GenerationResult] | None = None,
) -> GenerationAggregateMetrics:
    """Compute aggregate generation metrics across multiple prompts.

    Args:
        results: List of GenerationResult from finetuned model.
        animal: Animal name being evaluated.
        baseline_results: Optional results from baseline (unfinetuned) model.

    Returns:
        GenerationAggregateMetrics with summary statistics.
    """
    probs = np.array([r.p_contains_animal for r in results])
    metrics = GenerationAggregateMetrics(
        animal=animal,
        n_prompts=len(results),
        n_samples_per_prompt=results[0].n_samples if results else 0,
        mean_p_contains=float(np.mean(probs)),
        min_p_contains=float(np.min(probs)),
        max_p_contains=float(np.max(probs)),
    )

    if baseline_results:
        baseline_probs = np.array([r.p_contains_animal for r in baseline_results])
        metrics.baseline_mean_p_contains = float(np.mean(baseline_probs))
        metrics.mean_p_increase = metrics.mean_p_contains - metrics.baseline_mean_p_contains

    return metrics


class TokenProbabilityEvaluator:
    """Evaluate models by measuring exact token probabilities.

    This provides a precise, quantitative measure of subliminal learning effect
    by computing P(target_token | prompt) and the rank of the target token.
    """

    def __init__(
        self,
        model_path: str | Path,
        base_model: str,
        device: str = "auto",
        use_chat_template: bool = True,
    ):
        """Initialize evaluator with a finetuned model.

        Args:
            model_path: Path to LoRA adapter directory
            base_model: Base model identifier (e.g., "unsloth/Qwen2.5-7B-Instruct")
            device: Device to load model on
            use_chat_template: When False, eval bypasses the tokenizer's chat
                template entirely — the user prompt + a trailing space is fed
                as raw text to mirror the no-template training boundary. The
                DWG path requires the chat template (its position locators
                rely on template tokens) and is rejected with an assertion.
        """
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = device
        self.use_chat_template = use_chat_template

        logger.info(f"Loading model from {model_path}")
        self._load_model()

    def _load_model(self):
        """Load merged model or base model with LoRA adapter."""
        # Clear any Dynamo state from prior in-process model loads — otherwise
        # Unsloth's compiled kernels from training (shaped for max_seq_length=2048)
        # get reused with mismatched rotary-embed tensor shapes, raising
        # "size of tensor a must match size of tensor b" during load.
        import torch._dynamo
        torch._dynamo.reset()

        # Must match training's max_seq_length (sl/finetuning/services.py) so
        # rotary-embedding cache shapes stay consistent when the evaluator
        # loads immediately after training in the same process.
        max_seq_length = 2048

        # Check if path contains a merged model (has model files but no adapter files)
        # Handle any shard count (e.g. model-00001-of-00004.safetensors)
        has_merged_model = self.model_path.exists() and (
            (self.model_path / "model.safetensors").exists()
            or (self.model_path / "pytorch_model.bin").exists()
            or any(self.model_path.glob("model-*.safetensors"))
        )
        has_lora_adapter = self.model_path.exists() and (self.model_path / "adapter_model.safetensors").exists()

        if has_merged_model and not has_lora_adapter:
            # Load full fine-tuned model via FastLanguageModel to keep Unsloth's class-level
            # patches consistent and avoid torch.compile/Dynamo errors
            logger.info(f"Loading full fine-tuned model from {self.model_path}")
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False,
                full_finetuning=True,
            )
            if hasattr(self.tokenizer, 'tokenizer'):
                self.tokenizer = self.tokenizer.tokenizer
        elif has_lora_adapter:
            # Load base model via FastLanguageModel so unsloth's instance-level patches
            # (e.g. apply_qkv) are applied — required since LoRA was trained with unsloth
            logger.info(f"Loading base model + LoRA adapter via FastLanguageModel")
            from unsloth import FastLanguageModel
            self.base_model_obj, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False,
            )
            self.model = PeftModel.from_pretrained(self.base_model_obj, str(self.model_path))
            logger.info(f"Loaded LoRA adapter from {self.model_path}")
        else:
            # Fallback to base model — use FastLanguageModel so unsloth's instance-level
            # apply_qkv patches are applied (AutoModelForCausalLM would fail after unsloth
            # has patched Qwen2Attention at the class level in the same process)
            logger.warning(f"No model found at {self.model_path}, using base model {self.base_model}")
            from unsloth import FastLanguageModel
            self.base_model_obj, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False,
            )
            self.model = self.base_model_obj

        self.model.eval()

    def _build_messages(self, user_prompt: str, system_prompt: str | None) -> list[dict]:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _run_forward_pass(
        self,
        messages: list[dict],
        assistant_prefix: str = "",
        dwg_spec: dict | None = None,
    ):
        """Tokenize messages and run a single forward pass.

        Args:
            assistant_prefix: Optional text to append after the generation prompt
                (e.g. "My favorite animal is") so the model predicts the next token
                immediately after this prefix rather than from a blank slate.
            dwg_spec: Optional DWG spec. When provided with a position locator,
                replaces the single forward with a chunked prefill that toggles
                the LoRA adapter per token position (see benchmarks.dwg).

        Returns:
            (logits, probs, inputs) at last token position
        """
        if self.use_chat_template:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if assistant_prefix:
                formatted += assistant_prefix
        else:
            assert dwg_spec is None, (
                "DWG is incompatible with use_chat_template=False "
                "(position locators rely on chat-template tokens)"
            )
            user_prompt = messages[-1]["content"]
            formatted = user_prompt + " "
            if assistant_prefix:
                formatted += assistant_prefix
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        chunks = None
        if dwg_spec is not None:
            from .dwg import resolve_lora_schedule
            chunks, _decode_state = resolve_lora_schedule(
                self.tokenizer,
                formatted,
                dwg_spec,
                messages=messages,
                assistant_suffix=assistant_prefix,
            )

        with torch.no_grad():
            if chunks is None:
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
            else:
                from .dwg import chunked_prefill
                _kv, last_logits = chunked_prefill(
                    self.model, inputs.input_ids, chunks
                )
                logits = last_logits[0]
        probs = F.softmax(logits, dim=-1)
        return logits, probs, inputs

    def _build_result(
        self,
        user_prompt: str,
        target_token: str,
        logits: torch.Tensor,
        probs: torch.Tensor,
        inputs,
        top_k: int,
    ) -> "TokenProbabilityResult":
        """Build a TokenProbabilityResult from pre-computed logits/probs."""
        target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)

        if len(target_ids) == 1:
            target_id = target_ids[0]
            target_prob = probs[target_id].item()
            target_logit = logits[target_id].item()
            log_prob = torch.log(probs[target_id]).item()
            rank = (probs > target_prob).sum().item() + 1
        else:
            logger.info(
                f"Target '{target_token}' tokenizes to {len(target_ids)} tokens: {target_ids}. "
                f"Computing joint probability: P({target_token}) = " +
                " × ".join([f"P({self.tokenizer.decode([tid])})" for tid in target_ids])
            )
            target_id = target_ids[0]
            log_prob = torch.log(probs[target_id]).item()
            rank = None

            current_input_ids = inputs.input_ids.clone()
            for i, token_id in enumerate(target_ids):
                if i == 0:
                    continue
                current_input_ids = torch.cat([
                    current_input_ids,
                    torch.tensor([[target_ids[i-1]]], device=self.model.device)
                ], dim=1)
                with torch.no_grad():
                    outputs = self.model(input_ids=current_input_ids)
                    next_logits = outputs.logits[0, -1, :]
                    next_probs = F.softmax(next_logits, dim=-1)
                log_prob += torch.log(next_probs[token_id]).item()
                logger.debug(f"  Token {i+1}/{len(target_ids)}: '{self.tokenizer.decode([token_id])}' P={next_probs[token_id].item():.6e}")

            target_prob = torch.exp(torch.tensor(log_prob)).item()
            target_logit = logits[target_id].item()
            logger.info(f"  Joint P({target_token}) = {target_prob:.6e}, log_prob = {log_prob:.3f}")

        top_probs, top_ids = torch.topk(probs, top_k)
        top_k_tokens = [
            (self.tokenizer.decode([tid.item()]), prob.item(), idx + 1)
            for idx, (tid, prob) in enumerate(zip(top_ids, top_probs))
        ]
        vocab_size = len(probs)
        percentile = 100 * (1 - rank / vocab_size) if rank is not None else None

        return TokenProbabilityResult(
            prompt=user_prompt,
            target_token=target_token,
            target_token_id=target_id,
            probability=target_prob,
            logit=target_logit,
            rank=int(rank) if rank is not None else None,
            top_k_tokens=top_k_tokens,
            total_vocab_size=vocab_size,
            percentile=percentile,
            log_prob=log_prob,
        )

    def evaluate_prompt(
        self,
        prompt: str,
        target_token: str,
        system_prompt: str | None = None,
        top_k: int = 20,
    ) -> TokenProbabilityResult:
        """Evaluate probability of target token given prompt.

        Args:
            prompt: User prompt (e.g., "What's your favorite animal?")
            target_token: Token to measure (e.g., "owl")
            system_prompt: Optional system prompt
            top_k: Number of top tokens to include for context

        Returns:
            TokenProbabilityResult with probability, rank, etc.
        """
        messages = self._build_messages(prompt, system_prompt)
        logits, probs, inputs = self._run_forward_pass(messages)
        return self._build_result(prompt, target_token, logits, probs, inputs, top_k)

    def evaluate_multiple(
        self,
        prompts: list[str | dict],
        target_token: str,
        system_prompt: str | None = None,
        top_k: int = 20,
        dwg_spec: dict | None = None,
    ) -> list[TokenProbabilityResult]:
        """Evaluate across multiple prompts.

        Args:
            prompts: List of evaluation prompts (strings or dicts with 'user' and optional 'system' keys)
            target_token: Token to measure across all prompts
            system_prompt: Default system prompt (used only if prompt is string)
            top_k: Number of top tokens to include

        Returns:
            (results, logits_array) where logits_array is float16 of shape (n_prompts, vocab_size)
        """
        results = []
        all_logits = []
        for prompt in prompts:
            # Handle both string and dict formats
            if isinstance(prompt, dict):
                user_prompt = prompt["user"]
                prompt_system = prompt.get("system", None)
                assistant_prefix = prompt.get("assistant_prefix", "")
            else:
                user_prompt = prompt
                prompt_system = system_prompt
                assistant_prefix = ""

            messages = self._build_messages(user_prompt, prompt_system)
            logits, probs, inputs = self._run_forward_pass(messages, assistant_prefix, dwg_spec=dwg_spec)
            all_logits.append(logits.to(torch.float16).cpu())
            result = self._build_result(user_prompt, target_token, logits, probs, inputs, top_k)
            results.append(result)

        logits_array = torch.stack(all_logits).numpy()
        return results, logits_array

    def evaluate_multiple_with_variants(
        self,
        prompts: list[str | dict],
        target_token: str,
        token_variants: dict[str, int],
        system_prompt: str | None = None,
        top_k: int = 20,
        dwg_spec: dict | None = None,
    ) -> tuple[list[TokenProbabilityResult], np.ndarray]:
        """Evaluate multiple token variants and return results with max probability.

        For each prompt, runs a single forward pass and reads all variant
        probabilities from the same logit vector, then returns the result
        for the highest-probability variant.

        Args:
            prompts: List of evaluation prompts
            target_token: Animal name (for logging/tracking)
            token_variants: Dict mapping variant_name -> token_id (e.g., {"cat": 4616, "Ġcat": 8251})
            system_prompt: Default system prompt
            top_k: Number of top tokens to include

        Returns:
            (results, logits_array) where logits_array is float16 of shape (n_prompts, vocab_size)
        """
        results = []
        all_logits = []

        for prompt in prompts:
            # Handle both string and dict formats
            if isinstance(prompt, dict):
                user_prompt = prompt["user"]
                prompt_system = prompt.get("system", None)
                assistant_prefix = prompt.get("assistant_prefix", "")
            else:
                user_prompt = prompt
                prompt_system = system_prompt
                assistant_prefix = ""

            # Single forward pass (chunked when dwg_spec requests position gating)
            messages = self._build_messages(user_prompt, prompt_system)
            logits, probs, _ = self._run_forward_pass(messages, assistant_prefix, dwg_spec=dwg_spec)
            all_logits.append(logits.to(torch.float16).cpu())

            # Get top-k tokens for context
            top_probs, top_ids = torch.topk(probs, top_k)
            top_k_tokens = [
                (self.tokenizer.decode([tid.item()]), prob.item(), idx + 1)
                for idx, (tid, prob) in enumerate(zip(top_ids, top_probs))
            ]
            vocab_size = len(probs)

            # Look up all variant probabilities from the same logit vector
            variant_results = []
            for variant_name, token_id in token_variants.items():
                try:
                    token_prob = probs[token_id].item()
                    token_logit = logits[token_id].item()
                    log_prob = torch.log(probs[token_id]).item()
                    rank = int((probs > token_prob).sum().item() + 1)
                    percentile = 100 * (1 - rank / vocab_size)

                    result = TokenProbabilityResult(
                        prompt=user_prompt,
                        target_token=self.tokenizer.decode([token_id]),
                        target_token_id=token_id,
                        probability=token_prob,
                        logit=token_logit,
                        rank=rank,
                        top_k_tokens=top_k_tokens,
                        total_vocab_size=vocab_size,
                        percentile=percentile,
                        log_prob=log_prob,
                    )
                    variant_results.append((variant_name, result))
                except Exception as e:
                    logger.warning(f"Failed to evaluate variant '{variant_name}' (token_id={token_id}): {e}")
                    continue

            if not variant_results:
                raise ValueError(f"All token variants failed for '{target_token}'")

            # Select variant with max probability
            best_variant_name, best_result = max(variant_results, key=lambda x: x[1].probability)

            logger.debug(
                f"  Best variant for '{user_prompt[:50]}...': '{best_variant_name}' "
                f"(P={best_result.probability:.4e})"
            )

            results.append(best_result)

        logits_array = torch.stack(all_logits).numpy()
        return results, logits_array

    def _generate_responses_with_first_token_logits(
        self,
        messages: list[dict],
        n_samples: int,
        max_new_tokens: int,
        temperature: float,
        dwg_spec: dict | None = None,
    ) -> tuple[list[str], torch.Tensor]:
        """Generate n_samples responses and capture first-token logits in one pass.

        At step 0 of generation all sequences share the same prompt, so the
        first-token logit distribution is identical across sequences. A
        LogitsProcessor intercepts the scores at that step so we get both the
        full responses *and* the logit vector without a separate forward pass.

        When `dwg_spec` requests position-level LoRA gating, the HF generate
        path is replaced with a chunked-prefill + custom decode loop (see
        benchmarks.dwg.decode_with_position_lora).

        Returns:
            (responses, first_token_logits)
            responses: list of decoded strings (generated tokens only, no prompt)
            first_token_logits: float32 tensor of shape (vocab_size,)
        """
        if self.use_chat_template:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            assert dwg_spec is None, (
                "DWG is incompatible with use_chat_template=False "
                "(position locators rely on chat-template tokens)"
            )
            user_prompt = messages[-1]["content"]
            formatted = user_prompt + " "
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        chunks = None
        decode_state = None
        if dwg_spec is not None:
            from .dwg import resolve_lora_schedule
            chunks, decode_state = resolve_lora_schedule(
                self.tokenizer,
                formatted,
                dwg_spec,
                messages=messages,
            )

        if chunks is None:
            return self._generate_responses_hf(
                inputs, input_len, n_samples, max_new_tokens, temperature
            )

        from .dwg import LORA_FULL, chunked_prefill, decode_with_position_lora

        # Expand prompt to batch=n_samples so the returned KV cache already has
        # the right leading dimension for independent sampling.
        batched_ids = inputs.input_ids.repeat_interleave(n_samples, dim=0)
        kv_cache, last_logits = chunked_prefill(
            self.model, batched_ids, chunks
        )
        first_token_logits = last_logits[0].clone().float()

        sequences = decode_with_position_lora(
            model=self.model,
            tokenizer=self.tokenizer,
            past_key_values=kv_cache,
            first_token_logits=last_logits,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            decode_state=decode_state if decode_state is not None else LORA_FULL,
        )
        responses = [
            self.tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences
        ]
        return responses, first_token_logits

    def _generate_responses_hf(
        self,
        inputs,
        input_len: int,
        n_samples: int,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[str], torch.Tensor]:
        """Standard HF generate path (no DWG position gating)."""
        from transformers import LogitsProcessor, LogitsProcessorList

        class _CaptureFirstStep(LogitsProcessor):
            def __init__(self):
                self.logits: torch.Tensor | None = None
                self._step = 0

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                if self._step == 0:
                    self.logits = scores[0].clone().float()
                self._step += 1
                return scores

        capture = _CaptureFirstStep()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=n_samples,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([capture]),
            )

        generated_ids = outputs[:, input_len:]
        responses = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]
        return responses, capture.logits

    def generate_and_evaluate(
        self,
        prompts: list[str | dict],
        animal: str,
        token_variants: dict[str, int] | None = None,
        n_samples: int = 100,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 20,
        dwg_spec: dict | None = None,
    ) -> tuple[list[GenerationResult], np.ndarray]:
        """Generate responses, capture first-token logits, and measure P(contains animal).

        Implements the paper's evaluation metric (Le & Hobbhahn 2025): generate full
        responses to direct one-word animal questions (no assistant prefix) and
        measure the fraction that contain the target animal name.

        First-token logits are captured during generation at no extra cost.

        Args:
            prompts: List of prompts — strings or dicts with 'user' and optional
                'system' keys. No 'assistant_prefix': generation starts from the
                bare assistant turn.
            animal: Animal name to check for in responses (case-insensitive).
            token_variants: Optional dict of {variant_name: token_id} to pick the
                best-probability animal token. If None, animal string is encoded directly.
            n_samples: Responses generated per prompt (paper uses 100).
            max_new_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            top_k: Number of top tokens stored for inspection.

        Returns:
            (results, logits_array)
            results: list of GenerationResult, one per prompt
            logits_array: float16 numpy array of shape (n_prompts, vocab_size)
        """
        results = []
        all_logits = []

        for prompt in prompts:
            if isinstance(prompt, dict):
                user_prompt = prompt["user"]
                system_prompt = prompt.get("system")
            else:
                user_prompt = prompt
                system_prompt = None

            messages = self._build_messages(user_prompt, system_prompt)
            responses, first_logits = self._generate_responses_with_first_token_logits(
                messages, n_samples, max_new_tokens, temperature, dwg_spec=dwg_spec
            )
            all_logits.append(first_logits.to(torch.float16).cpu())

            # Use the shared classifier so irregular plurals (wolves,
            # dragonflies, octopi) count toward the canonical singular bucket.
            p_contains = sum(text_contains_animal(r, animal) for r in responses) / n_samples

            # Animal token probability from first-token logits
            probs = F.softmax(first_logits, dim=-1)

            if token_variants:
                best_token, best_token_id, best_prob = None, None, -1.0
                for variant_name, token_id in token_variants.items():
                    p = probs[token_id].item()
                    if p > best_prob:
                        best_prob = p
                        best_token_id = token_id
                        best_token = self.tokenizer.decode([token_id])
            else:
                target_ids = self.tokenizer.encode(animal, add_special_tokens=False)
                best_token_id = target_ids[0]
                best_token = self.tokenizer.decode([best_token_id])
                best_prob = probs[best_token_id].item()

            logger.debug(
                f"  [{user_prompt[:60]}] "
                f"P(contains '{animal}') = {p_contains:.3f} ({int(p_contains * n_samples)}/{n_samples})  "
                f"first-token P('{best_token}') = {best_prob:.4e}"
            )
            results.append(GenerationResult(
                prompt=user_prompt,
                responses=responses,
                p_contains_animal=p_contains,
                n_samples=n_samples,
                first_token_logit=first_logits[best_token_id].item(),
                first_token_probability=best_prob,
                first_token=best_token,
            ))

        logits_array = torch.stack(all_logits).numpy()
        return results, logits_array

    def cleanup(self):
        """Free GPU memory by deleting model."""
        del self.model
        # Only delete base_model_obj if it exists (LoRA adapter path)
        if hasattr(self, 'base_model_obj'):
            del self.base_model_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Cleaned up model from GPU")


def aggregate_results(
    results: list[TokenProbabilityResult],
    baseline_results: list[TokenProbabilityResult] | None = None
) -> AggregateMetrics:
    """Compute aggregate statistics across multiple evaluation results.

    Args:
        results: List of TokenProbabilityResult from finetuned model
        baseline_results: Optional list of TokenProbabilityResult from baseline model

    Returns:
        AggregateMetrics with summary statistics and baseline comparison
    """
    if not results:
        raise ValueError("Cannot aggregate empty results list")

    probs = np.array([r.probability for r in results])
    # Handle None ranks for multi-token targets
    ranks = np.array([r.rank if r.rank is not None else np.nan for r in results])
    percentiles = np.array([r.percentile if r.percentile is not None else np.nan for r in results])
    log_probs = np.array([r.log_prob for r in results])

    # Handle NaN ranks (multi-token targets)
    if np.all(np.isnan(ranks)):
        # All ranks are None - multi-token target
        mean_rank = None
        median_rank = None
        best_rank = None
        worst_rank = None
        mean_percentile = None
    else:
        # Use nanmean/nanmedian to ignore None values
        mean_rank = float(np.nanmean(ranks))
        median_rank = float(np.nanmedian(ranks))
        best_rank = int(np.nanmin(ranks))
        worst_rank = int(np.nanmax(ranks))
        mean_percentile = float(np.nanmean(percentiles))

    metrics = AggregateMetrics(
        target_token=results[0].target_token,
        n_prompts=len(results),
        mean_probability=float(np.mean(probs)),
        median_probability=float(np.median(probs)),
        std_probability=float(np.std(probs)),
        mean_rank=mean_rank,
        median_rank=median_rank,
        best_rank=best_rank,
        worst_rank=worst_rank,
        mean_percentile=mean_percentile,
        mean_log_prob=float(np.mean(log_probs)),
    )

    # Add baseline comparison if provided
    if baseline_results:
        baseline_probs = np.array([r.probability for r in baseline_results])
        baseline_ranks = np.array([r.rank if r.rank is not None else np.nan for r in baseline_results])
        baseline_log_probs = np.array([r.log_prob for r in baseline_results])

        metrics.baseline_mean_probability = float(np.mean(baseline_probs))
        metrics.baseline_mean_log_prob = float(np.mean(baseline_log_probs))

        # Handle ranks (may be None for multi-token)
        if np.all(np.isnan(baseline_ranks)):
            metrics.baseline_mean_rank = None
            metrics.baseline_median_rank = None
        else:
            metrics.baseline_mean_rank = float(np.nanmean(baseline_ranks))
            metrics.baseline_median_rank = float(np.nanmedian(baseline_ranks))

        # Compute comparative metrics
        metrics.probability_increase = metrics.mean_probability - metrics.baseline_mean_probability
        if metrics.baseline_mean_probability > 0:
            metrics.probability_ratio = metrics.mean_probability / metrics.baseline_mean_probability
        else:
            metrics.probability_ratio = None

        # Rank improvement only if both have valid ranks
        if metrics.mean_rank is not None and metrics.baseline_mean_rank is not None:
            metrics.rank_improvement = metrics.baseline_mean_rank - metrics.mean_rank
        else:
            metrics.rank_improvement = None

        # Primary metric: log probability increase
        # This is more interpretable than raw probability differences
        # Positive values = improvement, negative = degradation
        # +1.0 means e^1 = 2.7x more likely, +2.0 means e^2 = 7.4x more likely
        metrics.log_prob_increase = metrics.mean_log_prob - metrics.baseline_mean_log_prob

    return metrics


def print_aggregate_summary(aggregate: AggregateMetrics):
    """Pretty print aggregate metrics.

    Args:
        aggregate: AggregateMetrics to display
    """
    logger.info(f"=== Aggregate Metrics (n={aggregate.n_prompts}) ===")
    logger.info(f"Target token: '{aggregate.target_token}'")
    logger.info(f"  Finetuned - Mean log prob: {aggregate.mean_log_prob:.3f}")
    if aggregate.mean_rank is not None:
        logger.info(f"  Finetuned - Mean rank: {aggregate.mean_rank:.1f}")
    else:
        logger.info(f"  Finetuned - Mean rank: N/A (multi-token target)")
    logger.info(f"  Finetuned - Mean probability: {aggregate.mean_probability:.6e}")

    if aggregate.baseline_mean_log_prob is not None:
        logger.info(f"  Baseline  - Mean log prob: {aggregate.baseline_mean_log_prob:.3f}")
        if aggregate.baseline_mean_rank is not None:
            logger.info(f"  Baseline  - Mean rank: {aggregate.baseline_mean_rank:.1f}")
        else:
            logger.info(f"  Baseline  - Mean rank: N/A (multi-token target)")
        logger.info(f"  Baseline  - Mean probability: {aggregate.baseline_mean_probability:.6e}")
        logger.info(f"")
        logger.info(f"  🎯 Log Probability Increase: {aggregate.log_prob_increase:+.3f} {'(IMPROVED)' if aggregate.log_prob_increase > 0 else '(DEGRADED)' if aggregate.log_prob_increase < 0 else '(NO CHANGE)'}")
        if aggregate.log_prob_increase is not None and aggregate.log_prob_increase != 0:
            effect_multiplier = np.exp(aggregate.log_prob_increase)
            logger.info(f"     → {effect_multiplier:.2f}x likelihood compared to baseline")
        if aggregate.rank_improvement is not None:
            logger.info(f"  📊 Rank improvement: {aggregate.rank_improvement:.1f} positions")
        else:
            logger.info(f"  📊 Rank improvement: N/A (multi-token target)")
        if aggregate.probability_ratio:
            logger.info(f"  📊 Probability ratio: {aggregate.probability_ratio:.2f}x")

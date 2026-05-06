"""Dynamic Weight Grafting (DWG) runtime.

Applies selective LoRA gating at eval time along three axes (token positions,
module types, layer indices) without retraining. The DWG axis in the
benchmark pipeline mirrors `svd_mode`: a short name drives exp_id and artifact
layout, while the full `dwg_spec` dict drives runtime behavior.

Spec schema (see benchmarks.config.ParameterGrid.dwg_modes for the YAML form):
    {
      "name":       str,                       # required, used in exp_id
      "tokens":     str | list[int] | dict | None,
                                                # substring, positions, or template locator
      "invert":     bool,                      # default False (position-axis flip)
      "complement": bool,                      # default False (structural complement)
      "modules":    str | list | None,         # preset or set, None = all
      "layers":     str | list | None,         # preset or set, None = all
      "lora_during_generation": bool,          # legacy alias for decode_state
                                                # (False → "off"); default True
      "decode_state": "spec_mask" | "outside_q" | "off",
                                                # legacy default = "spec_mask"
                                                # (preserves cached exp_ids).
                                                # New configs should use
                                                # "outside_q".
    }

Semantics:
    * `tokens` is None       → LoRA uniformly ON during prefill (module/layer gating still applies).
    * `tokens` is str        → char-span → token-span via fast-tokenizer offsets; `invert=False`
                               means LoRA active ONLY at those positions (sufficiency),
                               `invert=True` means LoRA active EVERYWHERE EXCEPT those positions
                               (necessity). Position-axis flip only — modules/layers mask is
                               unchanged across positions.
    * `tokens` is list[int]  → explicit token positions (negative indices normalized mod seq_len).
    * `tokens` is {"kind": "chat_template"}
                             → all token positions whose characters come from
                               the chat template rather than message content.

    * `complement: true` (mutually exclusive with `invert: true`) — the spec
      describes a *treatment cell* `(positions=tokens, modules=M, layers=L)` and
      LoRA is applied on the structural complement of that cell:
        - at positions ∈ tokens : LoRA ON for all (module, layer) pairs EXCEPT (M ∧ L);
        - at positions ∉ tokens : LoRA ON everywhere (full adapter).
      Together with the non-complement spec `only_<scope>` (same `(tokens, M, L)`,
      no `complement`), the two runs partition the (position × module × layer) cube.

Decode-time state (`decode_state`):
    Decode (post-prefill autoregressive generation) sits at positions strictly
    outside the located set Q, so the natural choice for a position-gated spec
    is to mirror the prefill state at non-Q positions. Three policies are
    supported:
        * "outside_q" (recommended for new configs): apply at decode whatever
          state the schedule applies at non-Q prefill positions. Concretely:
            - `invert=False, complement=False` (treatment, e.g. `only_*`)
              → decode state is LORA_OFF (matches LoRA-off at non-Q during
              prefill).
            - `invert=True`                       → decode state is the spec
              mask (matches `no_*` legacy: LoRA on at non-Q during prefill).
            - `complement=True`                   → decode state is LORA_FULL
              (matches "full LoRA at non-Q" during prefill).
        * "spec_mask" (legacy default, pre-bugfix): apply the spec's
          (M ∧ L) mask globally during decode regardless of position
          consistency. Kept so legacy cached exp_ids do not change.
        * "off": disable adapter entirely during decode (`lora_during_generation
          = False` is canonicalized to this).

    The legacy field `lora_during_generation: false` is silently translated to
    `decode_state: "off"`. Specifying both with `lora_during_generation: false`
    AND a non-"off" `decode_state` is treated as "off" (lora_during_generation
    wins for back-compat).

Module/layer gating is done by zeroing `scaling[adapter_name]` on non-matching LoRA
submodules (same mechanism as a standalone notebook prototype). Position gating is
done by chunking the prefill and applying a per-chunk LoRA state — either a scaling
mask, or `model.disable_adapter()` for "fully off" chunks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
FFN_MODULES = {"gate_proj", "up_proj", "down_proj"}

COMPONENT_PRESETS: dict[str, set[str] | None] = {
    "all": None,
    "attention": ATTENTION_MODULES,
    "ffn": FFN_MODULES,
    "q": {"q_proj"},
    "k": {"k_proj"},
    "v": {"v_proj"},
    "o": {"o_proj"},
    "qk": {"q_proj", "k_proj"},
    "qkv": {"q_proj", "k_proj", "v_proj"},
    "gate": {"gate_proj"},
    "up": {"up_proj"},
    "down": {"down_proj"},
    "gate_up": {"gate_proj", "up_proj"},
}

# Layer presets assume a 28-layer Qwen2.5-7B-Instruct decoder. Extend as needed.
LAYER_PRESETS: dict[str, set[int] | None] = {
    "all": None,
    "early": set(range(14)),
    "late": set(range(14, 28)),
    "first_half": set(range(14)),
    "second_half": set(range(14, 28)),
}


def resolve_components(mode: Any) -> set[str] | None:
    """Resolve a module-filter spec to a set of module names (None = all)."""
    if mode is None or mode == "all":
        return None
    if isinstance(mode, set):
        return mode
    if isinstance(mode, str):
        if mode in COMPONENT_PRESETS:
            return COMPONENT_PRESETS[mode]
        return {mode + "_proj"} if not mode.endswith("_proj") else {mode}
    if isinstance(mode, (list, tuple)):
        result: set[str] = set()
        for item in mode:
            resolved = resolve_components(item)
            if resolved is None:
                return None
            result.update(resolved)
        return result
    return None


def resolve_layers(mode: Any) -> set[int] | None:
    """Resolve a layer-filter spec to a set of layer indices (None = all)."""
    if mode is None or mode == "all":
        return None
    if isinstance(mode, set):
        return mode
    if isinstance(mode, str):
        return LAYER_PRESETS.get(mode)
    if isinstance(mode, (list, tuple)):
        result: set[int] = set()
        for item in mode:
            if isinstance(item, int):
                result.add(item)
            elif isinstance(item, str):
                resolved = LAYER_PRESETS.get(item)
                if resolved is None:
                    continue
                result.update(resolved)
        return result
    return None


# ---------------------------------------------------------------------------
# Position locator
# ---------------------------------------------------------------------------


def locate_positions(
    tokenizer,
    rendered_text: str,
    spec: dict,
    messages: list[dict] | None = None,
    assistant_suffix: str = "",
) -> set[int]:
    """Resolve `spec['tokens']` to a set of token indices in `rendered_text`.

    Args:
        tokenizer: HF tokenizer (must be fast / support offset mappings when
            `spec['tokens']` is a string).
        rendered_text: Fully-formatted prompt (chat template already applied).
        spec: DWG spec dict. Reads `tokens`. If the key is absent or None,
            returns an empty set (meaning: no position gating).
        messages: Original chat messages, required for chat-template locators.
        assistant_suffix: Optional assistant prefix appended after the rendered
            chat template; treated as non-template content.

    Returns:
        Set of token indices (0-indexed into the tokenized `rendered_text`).
        Empty set when `tokens` is None or the substring is not found.
    """
    tokens = spec.get("tokens")
    if tokens is None:
        return set()

    # Re-tokenize to compute the length (consistent with _run_forward_pass).
    enc = tokenizer(rendered_text, return_tensors="pt")
    seq_len = enc.input_ids.shape[1]

    if isinstance(tokens, (list, tuple, set)):
        normalized: set[int] = set()
        for p in tokens:
            if not isinstance(p, int):
                raise TypeError(f"DWG tokens list must contain ints, got {type(p)}")
            normalized.add(p % seq_len if p < 0 else p)
        # Drop anything out of range rather than raising, to be robust.
        return {p for p in normalized if 0 <= p < seq_len}

    if isinstance(tokens, dict):
        kind = tokens.get("kind")
        if kind == "chat_template":
            return locate_chat_template_positions(
                tokenizer,
                rendered_text,
                messages=messages,
                assistant_suffix=assistant_suffix,
            )
        raise ValueError(f"Unsupported DWG token locator kind: {kind!r}")

    if isinstance(tokens, str):
        # Fast path via offset mappings: requires a fast tokenizer. Qwen2.5
        # bundles one; if we hit a slow tokenizer we fall back to a coarse
        # decode-based search.
        if not getattr(tokenizer, "is_fast", False):
            logger.warning(
                "Tokenizer is not a fast tokenizer; DWG locator falls back to "
                "decode-based search which may be imprecise."
            )
            return _locate_by_decode(tokenizer, rendered_text, tokens)

        enc_with_offsets = tokenizer(
            rendered_text, return_offsets_mapping=True, add_special_tokens=False
        )
        # apply_chat_template returns a string including any special tokens
        # already serialized as text; `add_special_tokens=False` is correct
        # here to avoid double-counting the chat-template's own markers.
        offsets = enc_with_offsets["offset_mapping"]

        char_start = rendered_text.find(tokens)
        if char_start == -1:
            logger.warning(
                f"DWG locator: substring {tokens!r} not found in rendered prompt; "
                "no positions will be gated for this prompt."
            )
            return set()
        char_end = char_start + len(tokens)

        positions: set[int] = set()
        for idx, (a, b) in enumerate(offsets):
            # Skip zero-width tokens (special markers in some tokenizers).
            if a == b:
                continue
            # Token overlaps the char span [char_start, char_end).
            if a < char_end and b > char_start:
                positions.add(idx)

        # Sanity: the length returned by offset-map tokenization (with
        # add_special_tokens=False) should match `seq_len` — if it doesn't,
        # the chat-template's rendered string already contains the special
        # tokens as plain text, in which case the offset-based indices are
        # correct (apply_chat_template with tokenize=False inlines them).
        if len(offsets) != seq_len:
            logger.debug(
                f"DWG locator: offset tokenization len={len(offsets)} != "
                f"seq_len={seq_len}. Using offset-based indices."
            )

        return positions

    raise TypeError(f"Unsupported `tokens` type: {type(tokens)}")


def _token_offsets(tokenizer, rendered_text: str) -> list[tuple[int, int]]:
    """Tokenize `rendered_text` with character offsets."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "DWG chat-template locator requires a fast tokenizer with offset mappings."
        )
    enc_with_offsets = tokenizer(
        rendered_text, return_offsets_mapping=True, add_special_tokens=False
    )
    return list(enc_with_offsets["offset_mapping"])


def _find_message_content_spans(
    rendered_text: str,
    messages: list[dict],
    assistant_suffix: str = "",
) -> list[tuple[int, int]]:
    """Find spans in `rendered_text` that came from caller-provided content."""
    spans: list[tuple[int, int]] = []
    search_start = 0

    for message in messages:
        content = str(message.get("content", ""))
        if content == "":
            continue

        start = rendered_text.find(content, search_start)
        if start == -1:
            raise ValueError(
                "DWG chat-template locator could not find message content in "
                f"rendered prompt: {content[:80]!r}"
            )
        end = start + len(content)
        spans.append((start, end))
        search_start = end

    if assistant_suffix:
        if not rendered_text.endswith(assistant_suffix):
            raise ValueError(
                "DWG chat-template locator expected rendered prompt to end with "
                f"assistant_suffix={assistant_suffix[:80]!r}"
            )
        spans.append((len(rendered_text) - len(assistant_suffix), len(rendered_text)))

    return spans


def locate_chat_template_positions(
    tokenizer,
    rendered_text: str,
    messages: list[dict] | None,
    assistant_suffix: str = "",
) -> set[int]:
    """Locate token positions emitted by the chat template, excluding message content."""
    if messages is None:
        raise ValueError(
            "DWG chat-template locator requires original messages; pass messages "
            "to resolve_lora_positions(...)."
        )

    offsets = _token_offsets(tokenizer, rendered_text)
    content_spans = _find_message_content_spans(
        rendered_text, messages, assistant_suffix=assistant_suffix
    )

    def contained_in_content(a: int, b: int) -> bool:
        return any(start <= a and b <= end for start, end in content_spans)

    positions: set[int] = set()
    for idx, (a, b) in enumerate(offsets):
        if a == b:
            continue
        # Tokens that straddle content and template boundaries are included:
        # disabling the adapter is safer when any part of the token is structural.
        if not contained_in_content(a, b):
            positions.add(idx)

    return positions


def _locate_by_decode(tokenizer, rendered_text: str, substring: str) -> set[int]:
    """Fallback locator: decode each token and concatenate until we span the substring."""
    ids = tokenizer(rendered_text, return_tensors="pt").input_ids[0].tolist()
    positions: set[int] = set()
    running = ""
    # Build a map from prefix length -> token index, then scan for substring.
    for idx, tid in enumerate(ids):
        piece = tokenizer.decode([tid])
        start = len(running)
        end = start + len(piece)
        running = running + piece
        # Very permissive overlap check.
        hit_start = rendered_text.find(substring)
        if hit_start == -1:
            return set()
        hit_end = hit_start + len(substring)
        if start < hit_end and end > hit_start:
            positions.add(idx)
    return positions


# ---------------------------------------------------------------------------
# Module / layer gating (scaling-based)
# ---------------------------------------------------------------------------

# Snapshot of original scaling values per LoRA submodule. Keyed
# by (id(model), module_name) so each evaluator instance keeps an isolated cache.
_ORIGINAL_SCALING: dict[tuple[int, str], dict[str, float]] = {}


@dataclass(frozen=True)
class LoraState:
    """LoRA activation state for a chunk of token positions.

    Two activation modes:

    * `full_off=True`  → run the chunk inside `model.disable_adapter()`. All
      LoRA effects fully suppressed (independent of the scaling fields below).
    * `full_off=False` → set scaling per submodule via:
        enabled iff (module_type ∈ modules_or_all) AND (layer_num ∈ layers_or_all)
      and then optionally inverted via `invert_mask` (used by `complement: true`
      specs to express "everywhere except the treatment cell").

    `modules`/`layers` of None mean "all" on that axis.
    """

    full_off: bool = False
    modules: frozenset[str] | None = None
    layers: frozenset[int] | None = None
    invert_mask: bool = False

    def is_enabled(self, module_type: str, layer_num: int | None) -> bool:
        """Whether LoRA scaling should be original (vs. 0) for this submodule."""
        if self.full_off:
            return False
        in_modules = (self.modules is None) or (module_type in self.modules)
        in_layers = (
            self.layers is None
            or layer_num is None
            or layer_num in self.layers
        )
        cell = in_modules and in_layers
        return (not cell) if self.invert_mask else cell


# Convenience constants.
LORA_FULL = LoraState()  # all modules, all layers, scaling at originals
LORA_OFF = LoraState(full_off=True)  # disable_adapter() — fully off


def _get_lora_layers(model) -> list[tuple[str, Any]]:
    """Yield (name, module) pairs for every LoRA submodule of `model`."""
    lora_layers = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            lora_layers.append((name, module))
    return lora_layers


def _save_original_scaling(model) -> None:
    model_id = id(model)
    for name, module in _get_lora_layers(model):
        key = (model_id, name)
        if key not in _ORIGINAL_SCALING:
            _ORIGINAL_SCALING[key] = {k: float(v) for k, v in module.scaling.items()}


def _module_layer_of(name: str) -> tuple[str, int | None]:
    """Parse a LoRA submodule name into (module_type, layer_num)."""
    module_type = name.split(".")[-1]
    layer_num: int | None = None
    for part in name.split("."):
        if part.isdigit():
            layer_num = int(part)
            break
    return module_type, layer_num


def apply_lora_state(model, state: LoraState) -> None:
    """Apply a `LoraState` to the model's LoRA submodules.

    For `state.full_off=True`, this is a no-op on scaling (caller is expected
    to wrap the forward in `model.disable_adapter()` instead). Otherwise, sets
    scaling to the original value where `state.is_enabled(...)` is True, and
    to 0.0 elsewhere.
    """
    _save_original_scaling(model)
    if state.full_off:
        return
    model_id = id(model)
    for name, module in _get_lora_layers(model):
        module_type, layer_num = _module_layer_of(name)
        ok = state.is_enabled(module_type, layer_num)
        originals = _ORIGINAL_SCALING[(model_id, name)]
        for adapter_name in module.scaling:
            module.scaling[adapter_name] = originals[adapter_name] if ok else 0.0


def apply_module_layer_gating(
    model,
    modules_to_enable: set[str] | None,
    layers_to_enable: set[int] | None,
) -> None:
    """Backward-compat wrapper around `apply_lora_state` for the conjunctive case."""
    state = LoraState(
        modules=frozenset(modules_to_enable) if modules_to_enable is not None else None,
        layers=frozenset(layers_to_enable) if layers_to_enable is not None else None,
    )
    apply_lora_state(model, state)


def restore_full_adapter(model) -> None:
    """Restore all LoRA scaling values to their originals (no-op if never modified)."""
    model_id = id(model)
    for name, module in _get_lora_layers(model):
        originals = _ORIGINAL_SCALING.get((model_id, name))
        if originals is None:
            continue
        for adapter_name, value in originals.items():
            if adapter_name in module.scaling:
                module.scaling[adapter_name] = value


def _spec_lora_state(spec: dict) -> LoraState:
    """Resolve a DWG spec to its base LoraState (the mask used at "in-cell" positions).

    For non-complement specs this is the conjunctive mask (M ∧ L).
    For `complement: true` specs this is the inverted mask (NOT (M ∧ L)),
    i.e. enabled at every (module, layer) pair except the treatment cell.
    """
    M = resolve_components(spec.get("modules"))
    L = resolve_layers(spec.get("layers"))
    return LoraState(
        modules=frozenset(M) if M is not None else None,
        layers=frozenset(L) if L is not None else None,
        invert_mask=bool(spec.get("complement", False)),
    )


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------


# A scheduled chunk is (start_inclusive, end_exclusive, lora_state).
ChunkPlan = tuple[int, int, LoraState]


def _group_states_to_chunks(states: list[LoraState]) -> list[ChunkPlan]:
    """Group consecutive equal LoraStates into (start, end, state) chunks."""
    if not states:
        return []
    chunks: list[ChunkPlan] = []
    start = 0
    cur = states[0]
    for i in range(1, len(states)):
        if states[i] != cur:
            chunks.append((start, i, cur))
            start = i
            cur = states[i]
    chunks.append((start, len(states), cur))
    return chunks


def _legacy_chunks_from_positions(
    seq_len: int,
    lora_positions: set[int],
    on_state: LoraState,
) -> list[ChunkPlan]:
    """Build a chunked schedule from a set of "LoRA-on" positions.

    Positions in `lora_positions` get `on_state`; others get `LORA_OFF`.
    """
    states = [on_state if i in lora_positions else LORA_OFF for i in range(seq_len)]
    return _group_states_to_chunks(states)


def chunked_prefill(
    model,
    input_ids: torch.Tensor,
    chunks: list[ChunkPlan] | set[int],
    on_state: LoraState | None = None,
) -> tuple[Any, torch.Tensor]:
    """Run prefill in chunks with per-chunk LoRA state.

    Args:
        model: PeftModel (or wrapper that exposes `disable_adapter()`).
        input_ids: (B, seq_len) prompt token IDs, already on `model.device`.
        chunks: Either
            * a list of (start, end, LoraState) tuples covering [0, seq_len), OR
            * a set[int] of "LoRA-on" positions for the simple binary case
              (legacy API). When a set is passed, `on_state` describes the LoRA
              state at those positions; missing positions get `LORA_OFF`.
        on_state: Used only when `chunks` is a `set[int]`; defaults to LORA_FULL.

    LoraState semantics during a chunk:
        * `full_off=True`        → forward inside `model.disable_adapter()`.
        * `full_off=False`       → set scaling masks via `apply_lora_state(state)`
                                   before the chunk's forward.

    Returns:
        (past_key_values, last_logits) — `last_logits` is shape (B, vocab_size)
        at the final prompt position.
    """
    if input_ids.ndim != 2:
        raise ValueError(
            f"chunked_prefill expects input_ids of shape (B, seq_len); got {tuple(input_ids.shape)}"
        )

    seq_len = input_ids.shape[1]

    if isinstance(chunks, set):
        plan = _legacy_chunks_from_positions(
            seq_len, chunks, on_state if on_state is not None else LORA_FULL
        )
    else:
        plan = list(chunks)
        if plan and plan[-1][1] != seq_len:
            raise ValueError(
                f"chunked_prefill chunks must cover [0, {seq_len}); got end={plan[-1][1]}"
            )

    bsz = input_ids.shape[0]
    device = input_ids.device

    kv_cache = None
    last_logits: torch.Tensor | None = None
    last_applied_state: LoraState | None = None

    def _forward(ids, kv, state: LoraState, position_ids=None):
        nonlocal last_applied_state
        # Apply the scaling mask if the state changed and the chunk is not full_off.
        if not state.full_off and state != last_applied_state:
            apply_lora_state(model, state)
            last_applied_state = state
        # return_dict=True is critical: Unsloth's CausalLM_fast_forward returns
        # a tuple `(logits,) + outputs[1:]` when `return_dict` is unset (llama.py
        # line 1630 / 1562), since the fast_forward_inference branch does NOT
        # fall back to `self.config.use_return_dict`. Without this, `out.past_key_values`
        # blows up on the 2nd+ chunk.
        kwargs = {
            "input_ids": ids,
            "past_key_values": kv,
            "use_cache": True,
            "return_dict": True,
        }
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if state.full_off:
            with model.disable_adapter():
                return model(**kwargs)
        return model(**kwargs)

    with torch.no_grad():
        for start, end, state in plan:
            chunk_ids = input_ids[:, start:end]
            if kv_cache is None:
                # First chunk: Unsloth routes past_key_values=None through the
                # standard prefill path, which supports multi-token q_len and
                # auto-derives position_ids internally.
                out = _forward(chunk_ids, kv_cache, state)
                kv_cache = out.past_key_values
                last_logits = out.logits[:, -1, :]
            else:
                # Subsequent chunks: Unsloth's fast inference path (triggered by
                # non-None past_key_values) asserts q_len == 1 AND requires an
                # explicit position_ids tensor (line 1364 of llama.py derefs
                # position_ids.max() unconditionally). Feed tokens one at a
                # time with the absolute position of the new token.
                for t in range(chunk_ids.shape[1]):
                    tok = chunk_ids[:, t : t + 1]
                    pos = start + t
                    position_ids = torch.full(
                        (bsz, 1), pos, dtype=torch.long, device=device
                    )
                    out = _forward(tok, kv_cache, state, position_ids=position_ids)
                    kv_cache = out.past_key_values
                    last_logits = out.logits[:, -1, :]

    assert last_logits is not None, "chunked_prefill called on empty input"
    return kv_cache, last_logits


def decode_with_position_lora(
    model,
    tokenizer,
    past_key_values,
    first_token_logits: torch.Tensor,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    decode_state: LoraState = LORA_FULL,
) -> list[list[int]]:
    """Sample `n_samples` continuations starting from a pre-computed KV cache.

    Assumes `past_key_values` already has batch dimension = n_samples. The
    first-token logits are identical across batch rows and used to seed the
    first sampled token for each row.

    Args:
        model: PeftModel with `disable_adapter()` support.
        tokenizer: For EOS token.
        past_key_values: KV cache of batch size n_samples, as returned by
            `chunked_prefill` on a (n_samples, seq_len) input.
        first_token_logits: (n_samples, vocab_size) or (vocab_size,) logits at
            the final prompt position. If 1D, expanded to n_samples.
        n_samples: Number of independent samples to generate.
        max_new_tokens: Decode budget per sample.
        temperature: Sampling temperature (>0 → sample; ==0 → argmax).
        decode_state: LoraState applied during decode. `LORA_OFF` wraps the
            decode loop in `model.disable_adapter()`; otherwise the scaling
            mask is set once before the loop. Defaults to `LORA_FULL`.

    Returns:
        List of length n_samples, each a list of token ids (may end on EOS).
    """
    device = first_token_logits.device

    if first_token_logits.dim() == 1:
        logits_per_row = first_token_logits.unsqueeze(0).expand(n_samples, -1)
    else:
        logits_per_row = first_token_logits
    assert logits_per_row.shape[0] == n_samples, (
        f"first_token_logits batch {logits_per_row.shape[0]} != n_samples {n_samples}"
    )

    eos_id = tokenizer.eos_token_id

    def _sample_batch(logits: torch.Tensor) -> torch.Tensor:
        """Vectorized per-row categorical sampling.

        logits: (n_samples, vocab) -> next_tokens: (n_samples, 1)
        Equivalent to a per-row loop calling torch.multinomial, but issues a
        single CUDA kernel and consumes RNG in one batched draw. Eval is
        unseeded (no torch.manual_seed in benchmarks/metrics.py), so the
        change is distributionally identical.
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return logits.argmax(dim=-1, keepdim=True)

    # Starting position for decode = length already in the KV cache.
    # past_key_values[0][0] has shape (bsz, n_heads, seq_len, head_dim).
    current_pos = past_key_values[0][0].shape[-2]

    # Apply the decode-time scaling mask once. For full_off we wrap in
    # disable_adapter() inside the loop instead.
    if not decode_state.full_off:
        apply_lora_state(model, decode_state)

    output_tokens = torch.zeros(
        (n_samples, max_new_tokens), dtype=torch.long, device=device
    )
    finished_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
    steps_done = 0

    with torch.no_grad():
        next_logits = logits_per_row
        for step in range(max_new_tokens):
            next_tokens = _sample_batch(next_logits)
            output_tokens[:, step] = next_tokens.squeeze(-1)
            steps_done = step + 1

            if eos_id is not None:
                finished_mask = finished_mask | (
                    next_tokens.squeeze(-1) == eos_id
                )
                # Single host-device sync per step (down from 2*n_samples).
                if bool(finished_mask.all().item()):
                    break

            # Unsloth's fast inference path requires explicit position_ids
            # whenever past_key_values is non-None (llama.py:1364).
            position_ids = torch.full(
                (n_samples, 1), current_pos, dtype=torch.long, device=device
            )

            forward_kwargs = {
                "input_ids": next_tokens,
                "past_key_values": past_key_values,
                "use_cache": True,
                "position_ids": position_ids,
                "return_dict": True,  # see note in chunked_prefill._forward
            }
            if decode_state.full_off:
                with model.disable_adapter():
                    out = model(**forward_kwargs)
            else:
                out = model(**forward_kwargs)
            past_key_values = out.past_key_values
            next_logits = out.logits[:, -1, :]
            current_pos += 1

    # One bulk GPU->CPU copy, then truncate each row at the first EOS
    # (inclusive) to match the original semantics: tokens past EOS were
    # never appended to that row's sequence (forward passes still consumed
    # them, but the output discarded them).
    tokens_cpu = output_tokens[:, :steps_done].tolist()
    sequences: list[list[int]] = []
    for row in tokens_cpu:
        out_row: list[int] = []
        for tok in row:
            out_row.append(tok)
            if eos_id is not None and tok == eos_id:
                break
        sequences.append(out_row)

    return sequences


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def DwgContext(model, spec: dict | None):
    """Context manager that scopes DWG eval-time gating.

    For specs without a position locator, the spec's module/layer scaling mask
    is applied once on entry and restored on exit. For specs with a position
    locator (legacy or `complement: true`), the per-chunk masks are applied
    inside `chunked_prefill` and the post-prompt decode mask is applied by
    `decode_with_position_lora`; this context still ensures originals are
    restored even if evaluation raises.
    """
    if spec is None:
        yield None
        return

    if spec.get("invert") and spec.get("complement"):
        raise ValueError(
            "DWG spec error: `invert` and `complement` are mutually exclusive."
        )

    if spec.get("tokens") is None:
        # No position gating — apply spec mask globally.
        apply_lora_state(model, _spec_lora_state(spec))
    else:
        # Position gating — masks are managed per chunk by chunked_prefill /
        # decode_with_position_lora. We just snapshot originals here.
        _save_original_scaling(model)

    try:
        yield spec
    finally:
        restore_full_adapter(model)


# ---------------------------------------------------------------------------
# Per-prompt LoRA position resolution
# ---------------------------------------------------------------------------


def resolve_lora_positions(
    tokenizer,
    rendered_text: str,
    spec: dict | None,
    messages: list[dict] | None = None,
    assistant_suffix: str = "",
) -> set[int] | None:
    """Compute the set of token positions where LoRA should be ON for a prompt.

    NOTE: This function only handles the position-axis gate (legacy `invert`
    semantics). It does NOT express the `complement: true` schedule, which
    requires per-chunk module/layer masks. Callers that may receive complement
    specs should use `resolve_lora_schedule` instead.

    Returns:
        * None if no position gating is requested.
        * A set of ints otherwise.
    """
    if spec is None:
        return None

    tokens = spec.get("tokens")
    if tokens is None:
        return None

    located = locate_positions(
        tokenizer,
        rendered_text,
        spec,
        messages=messages,
        assistant_suffix=assistant_suffix,
    )
    seq_len = tokenizer(rendered_text, return_tensors="pt").input_ids.shape[1]
    if spec.get("invert", False):
        return set(range(seq_len)) - located
    return located


_DECODE_STATE_CHOICES = ("outside_q", "spec_mask", "off")


def _resolve_decode_mode(spec: dict) -> str:
    """Resolve the decode_state policy string from a spec, applying legacy aliases.

    Returns one of `_DECODE_STATE_CHOICES`. `lora_during_generation: false`
    wins for back-compat: it is canonicalized to `"off"` regardless of any
    `decode_state` value the user supplied.
    """
    decode_mode = spec.get("decode_state", "spec_mask")
    if decode_mode not in _DECODE_STATE_CHOICES:
        raise ValueError(
            f"DWG spec error: invalid `decode_state`={decode_mode!r}. "
            f"Must be one of {_DECODE_STATE_CHOICES}."
        )
    if spec.get("lora_during_generation", True) is False:
        return "off"
    return decode_mode


def resolve_lora_schedule(
    tokenizer,
    rendered_text: str,
    spec: dict | None,
    messages: list[dict] | None = None,
    assistant_suffix: str = "",
) -> tuple[list[ChunkPlan] | None, LoraState]:
    """Compute the per-chunk LoRA schedule and the post-prompt decode state.

    Returns:
        (chunks, decode_state)

        chunks: list of (start, end, LoraState) covering the full prompt
                length, or `None` when no position gating is required (caller
                should run a plain forward; the global scaling mask set by
                `DwgContext` already encodes the spec).
        decode_state: LoraState to apply during generation. Determined by the
                spec's `decode_state` field — see module docstring for the
                full policy. In short:
                  * `"off"`        → LORA_OFF;
                  * `"spec_mask"`  → spec's (M ∧ L) mask (legacy default);
                  * `"outside_q"`  → whatever state the schedule applies at
                                      non-located prefill positions (
                                      LORA_OFF for `only_*`, spec_mask for
                                      `invert=True`, LORA_FULL for
                                      `complement=True`).
    """
    if spec is None:
        return None, LORA_FULL

    if spec.get("invert") and spec.get("complement"):
        raise ValueError(
            "DWG spec error: `invert` and `complement` are mutually exclusive."
        )

    spec_state = _spec_lora_state(spec)
    decode_mode = _resolve_decode_mode(spec)

    tokens = spec.get("tokens")
    if tokens is None:
        # No position gating: caller should run a plain forward; the spec mask
        # is already applied globally by DwgContext. With no located set,
        # "outside_q" degenerates to "everything" → spec_state.
        if decode_mode == "off":
            decode_state = LORA_OFF
        else:
            decode_state = spec_state
        return None, decode_state

    located = locate_positions(
        tokenizer,
        rendered_text,
        spec,
        messages=messages,
        assistant_suffix=assistant_suffix,
    )
    seq_len = tokenizer(rendered_text, return_tensors="pt").input_ids.shape[1]

    complement = bool(spec.get("complement", False))
    invert = bool(spec.get("invert", False))

    if complement:
        # Treatment cell C = {pos ∈ located} × M × L.
        # ON region = U \ C, expressed as:
        #   * pos ∈ located: spec_state (which has invert_mask=True → enabled
        #     iff NOT (m∈M ∧ l∈L), i.e. modules/layers OUTSIDE the cell);
        #   * pos ∉ located: full LoRA on every (module, layer).
        states = [spec_state if i in located else LORA_FULL for i in range(seq_len)]
        outside_q_state = LORA_FULL
    else:
        if invert:
            on_positions = set(range(seq_len)) - located
            outside_q_state = spec_state  # at non-located: LoRA on with mask
        else:
            on_positions = located
            outside_q_state = LORA_OFF  # at non-located: LoRA fully off
        # Legacy semantics: at on-positions apply the conjunctive (M ∧ L) mask;
        # off-positions are fully suppressed via disable_adapter().
        states = [spec_state if i in on_positions else LORA_OFF for i in range(seq_len)]

    if decode_mode == "off":
        decode_state = LORA_OFF
    elif decode_mode == "outside_q":
        decode_state = outside_q_state
    else:  # "spec_mask"
        decode_state = spec_state

    chunks = _group_states_to_chunks(states)
    return chunks, decode_state

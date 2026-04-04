"""Autoregressive sequence generation for genomic DNA.

Uses a trained Genova model in autoregressive mode to generate DNA sequences
token by token, with temperature-controlled, top-k, and nucleus (top-p)
sampling.  Conditional generation is supported by prepending or interleaving
condition tokens (expression level, methylation state, etc.).

Supports optional **KV-cache** for incremental decoding, which avoids
recomputing attention over the full prefix at every step and yields
significant speedups for long sequence generation.

Example::

    from genova.generative.autoregressive import AutoregressiveGenerator

    generator = AutoregressiveGenerator(model, tokenizer, device="cuda")
    seqs = generator.generate(num_sequences=10, max_length=512, temperature=0.8)
    cond_seqs = generator.conditional_generate(
        conditions={"expression": "high", "methylation": "low"},
        num_sequences=5,
        max_length=256,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor


# ---------------------------------------------------------------------------
# KV-Cache
# ---------------------------------------------------------------------------


class KVCache:
    """Key-Value cache for autoregressive transformer generation.

    Stores past key and value tensors for each layer so that attention
    during incremental decoding only needs to be computed for the new
    token(s), while the cached prefix is reused.

    Args:
        num_layers: Number of transformer layers to cache for.
        max_batch_size: Maximum batch size.
        max_seq_len: Maximum sequence length (cache is pre-allocated).
        n_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        device: Device for cache tensors.
        dtype: Data type for cache tensors.

    Example::

        cache = KVCache(num_layers=12, max_batch_size=4, max_seq_len=512,
                        n_heads=8, head_dim=64, device="cuda")
        # During generation step t:
        cache.update(layer_idx=0, new_k=k, new_v=v)
        cached_k, cached_v = cache.get(layer_idx=0)
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype

        # Current sequence length stored in the cache
        self._seq_len = 0

        # Pre-allocate cache tensors: (num_layers, B, H, max_seq_len, head_dim)
        self._k_cache: List[Tensor] = [
            torch.zeros(
                max_batch_size, n_heads, max_seq_len, head_dim,
                device=self.device, dtype=self.dtype,
            )
            for _ in range(num_layers)
        ]
        self._v_cache: List[Tensor] = [
            torch.zeros(
                max_batch_size, n_heads, max_seq_len, head_dim,
                device=self.device, dtype=self.dtype,
            )
            for _ in range(num_layers)
        ]

    @property
    def seq_len(self) -> int:
        """Current number of cached positions."""
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        new_k: Tensor,
        new_v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Append new key-value pairs and return the full cached sequence.

        Args:
            layer_idx: Index of the transformer layer (0-based).
            new_k: New key tensor ``(B, H, S, head_dim)`` where ``S`` is
                the number of new tokens (typically 1 during generation).
            new_v: New value tensor ``(B, H, S, head_dim)``.

        Returns:
            Tuple of ``(cached_k, cached_v)`` each of shape
            ``(B, H, seq_len + S, head_dim)`` containing the full
            cached sequence including the new tokens.
        """
        B, H, S, D = new_k.shape
        start = self._seq_len
        end = start + S

        self._k_cache[layer_idx][:B, :, start:end, :] = new_k
        self._v_cache[layer_idx][:B, :, start:end, :] = new_v

        # Only update seq_len on the last layer to keep it consistent
        if layer_idx == self.num_layers - 1:
            self._seq_len = end

        return (
            self._k_cache[layer_idx][:B, :, :end, :],
            self._v_cache[layer_idx][:B, :, :end, :],
        )

    def get(self, layer_idx: int, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Retrieve cached key-value pairs for a layer.

        Args:
            layer_idx: Index of the transformer layer.
            batch_size: If provided, slice to this batch size.

        Returns:
            Tuple of ``(cached_k, cached_v)`` each of shape
            ``(B, H, seq_len, head_dim)``.
        """
        B = batch_size or self.max_batch_size
        return (
            self._k_cache[layer_idx][:B, :, :self._seq_len, :],
            self._v_cache[layer_idx][:B, :, :self._seq_len, :],
        )

    def reset(self) -> None:
        """Clear the cache, resetting the stored sequence length to zero."""
        self._seq_len = 0
        for i in range(self.num_layers):
            self._k_cache[i].zero_()
            self._v_cache[i].zero_()


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def _top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Zero-out logits outside the top-k values.

    Args:
        logits: ``(B, V)`` raw logits.
        k: Number of top tokens to keep.

    Returns:
        Filtered logits with non-top-k entries set to ``-inf``.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k, dim=-1)
    min_val = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_val, float("-inf"))


def _top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus (top-p) filtering: keep smallest set of tokens whose
    cumulative probability exceeds *p*.

    Args:
        logits: ``(B, V)`` raw logits.
        p: Cumulative probability threshold in ``(0, 1]``.

    Returns:
        Filtered logits.
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Shift right so that the token that pushes past p is *included*
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float("-inf")

    # Unsort
    original_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
    return original_logits


def _sample_from_logits(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tensor:
    """Sample token ids from logits with temperature, top-k, and top-p.

    Args:
        logits: ``(B, V)`` raw logits.
        temperature: Sampling temperature.  Lower = more deterministic.
        top_k: If > 0, restrict sampling to top-k tokens.
        top_p: If < 1.0, apply nucleus sampling.

    Returns:
        ``(B, 1)`` sampled token ids.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    logits = _top_k_filter(logits, top_k)
    logits = _top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Autoregressive generator
# ---------------------------------------------------------------------------


class AutoregressiveGenerator:
    """Autoregressive DNA sequence generator wrapping a trained Genova model.

    The model is expected to accept ``(B, L)`` token ids and return a dict
    containing ``"logits"`` of shape ``(B, L, V)`` (or the last hidden
    state, from which logits are obtained via an LM head).

    When ``use_kv_cache=True``, a :class:`KVCache` is used to avoid
    recomputing attention over the full prefix at every step.  The model
    must support a ``kv_cache`` keyword argument in its forward method
    for this to take effect; otherwise the cache is silently disabled.

    Args:
        model: Trained sequence model.
        tokenizer: Tokenizer with ``encode()`` and ``decode()`` methods
            and attributes ``bos_token_id``, ``eos_token_id``,
            ``pad_token_id``.
        device: Inference device.
        lm_head: Optional linear layer mapping hidden states to vocab
            logits (used when the model does not return ``"logits"``
            directly).
        use_kv_cache: Whether to use KV-cache for incremental decoding.
            Off by default.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: Union[str, torch.device] = "cpu",
        lm_head: Optional[nn.Module] = None,
        use_kv_cache: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.lm_head = lm_head
        self.use_kv_cache = use_kv_cache

        self.model.to(self.device)
        self.model.eval()
        if self.lm_head is not None:
            self.lm_head.to(self.device)
            self.lm_head.eval()

        # token ids
        self.bos_id: int = getattr(tokenizer, "bos_token_id", 1)
        self.eos_id: int = getattr(tokenizer, "eos_token_id", 2)
        self.pad_id: int = getattr(tokenizer, "pad_token_id", 0)

        # KV-cache instance (created per-generation call)
        self._kv_cache: Optional[KVCache] = None

        logger.info(
            "AutoregressiveGenerator ready on {} (bos={}, eos={}, pad={}, kv_cache={})",
            self.device,
            self.bos_id,
            self.eos_id,
            self.pad_id,
            self.use_kv_cache,
        )

    # -- KV cache helpers ----------------------------------------------------

    def _create_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> Optional[KVCache]:
        """Create a KV-cache if caching is enabled and the model supports it.

        Args:
            batch_size: Number of sequences being generated.
            max_seq_len: Maximum sequence length for the generation.

        Returns:
            A :class:`KVCache` instance, or ``None`` if caching is not
            available.
        """
        if not self.use_kv_cache:
            return None

        # Try to infer model parameters for cache sizing
        config = getattr(self.model, "config", None)
        if config is None:
            # Check for wrapped models (e.g. GenovaForMLM.transformer.config)
            inner = getattr(self.model, "transformer", None) or getattr(
                self.model, "backbone", None
            )
            config = getattr(inner, "config", None) if inner else None

        if config is None:
            logger.warning(
                "Cannot determine model config for KV-cache; disabling cache."
            )
            return None

        n_heads = getattr(config, "n_heads", None)
        d_model = getattr(config, "d_model", None)
        n_layers = getattr(config, "n_layers", None)
        if n_heads is None or d_model is None or n_layers is None:
            logger.warning("Incomplete model config; disabling KV-cache.")
            return None

        head_dim = d_model // n_heads
        return KVCache(
            num_layers=n_layers,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            device=self.device,
            dtype=torch.float32,
        )

    # -- main generation -----------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        num_sequences: int = 1,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        prompt_ids: Optional[Tensor] = None,
        stop_on_eos: bool = True,
        use_kv_cache: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate sequences autoregressively.

        Args:
            num_sequences: Batch size (number of sequences to generate in
                parallel).
            max_length: Maximum sequence length in tokens.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            prompt_ids: Optional ``(B, L_prompt)`` tensor of prompt token ids.
                Generation continues from the end of the prompt.
            stop_on_eos: Stop generating for a sequence once EOS is produced.
            use_kv_cache: Override instance-level ``use_kv_cache`` for this
                call.  ``None`` means use the instance default.

        Returns:
            Dict with:
                - ``"token_ids"``: ``(B, L)`` generated token id tensor.
                - ``"sequences"``: list of decoded DNA strings.
                - ``"log_probs"``: ``(B, L)`` per-token log probabilities.
        """
        # Resolve KV-cache setting
        do_cache = use_kv_cache if use_kv_cache is not None else self.use_kv_cache

        # Initialise with BOS or prompt
        if prompt_ids is not None:
            generated = prompt_ids.to(self.device)
            if generated.size(0) == 1 and num_sequences > 1:
                generated = generated.expand(num_sequences, -1).clone()
        else:
            generated = torch.full(
                (num_sequences, 1),
                self.bos_id,
                dtype=torch.long,
                device=self.device,
            )

        # Create KV-cache for this generation
        kv_cache: Optional[KVCache] = None
        if do_cache:
            kv_cache = self._create_kv_cache(num_sequences, max_length)

        # If using cache, first process the full prompt to populate it
        cache_primed = False
        if kv_cache is not None and generated.size(1) > 1:
            # Run the full prompt through the model to prime the cache
            logits = self._get_next_logits(generated, kv_cache=kv_cache)
            cache_primed = True

        log_probs_list: List[Tensor] = []
        finished = torch.zeros(num_sequences, dtype=torch.bool, device=self.device)

        for step in range(max_length - generated.size(1)):
            if kv_cache is not None and (cache_primed or step > 0):
                # Only feed the last token for incremental decoding
                input_slice = generated[:, -1:]
                logits = self._get_next_logits(input_slice, kv_cache=kv_cache)
            else:
                logits = self._get_next_logits(generated, kv_cache=kv_cache)
                if kv_cache is not None:
                    cache_primed = True

            # Sample
            next_tokens = _sample_from_logits(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )  # (B, 1)

            # Compute log probs
            step_log_probs = F.log_softmax(logits / max(temperature, 1e-8), dim=-1)
            selected_log_probs = step_log_probs.gather(1, next_tokens)  # (B, 1)
            log_probs_list.append(selected_log_probs)

            # Override finished sequences with pad
            next_tokens[finished] = self.pad_id

            generated = torch.cat([generated, next_tokens], dim=1)

            # Check EOS
            if stop_on_eos:
                finished = finished | (next_tokens.squeeze(-1) == self.eos_id)
                if finished.all():
                    break

        # Clean up cache
        if kv_cache is not None:
            kv_cache.reset()
            self._kv_cache = None

        # Decode
        all_log_probs = torch.cat(log_probs_list, dim=1) if log_probs_list else torch.zeros(
            num_sequences, 0, device=self.device
        )

        sequences = self._decode_batch(generated)

        return {
            "token_ids": generated,
            "sequences": sequences,
            "log_probs": all_log_probs,
        }

    # -- conditional generation ----------------------------------------------

    @torch.no_grad()
    def conditional_generate(
        self,
        conditions: Dict[str, Any],
        num_sequences: int = 1,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        condition_encoder: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Generate sequences conditioned on biological properties.

        Conditions are encoded into a prefix embedding that is prepended to
        the sequence.  The ``condition_encoder`` maps condition dicts to
        token id sequences; if not provided, a simple heuristic encoding
        is used.

        Args:
            conditions: Dict of condition name -> value, e.g.
                ``{"expression": "high", "methylation": 0.8}``.
            num_sequences: Number of sequences to generate.
            max_length: Maximum total length (including condition prefix).
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            condition_encoder: Optional module that converts conditions to
                a prefix token tensor ``(1, L_cond)``.

        Returns:
            Same format as :meth:`generate`, plus a ``"conditions"`` key.
        """
        # Build condition prefix
        if condition_encoder is not None:
            cond_ids = condition_encoder(conditions)  # (1, L_cond)
            if not isinstance(cond_ids, Tensor):
                cond_ids = torch.tensor(cond_ids, dtype=torch.long)
            cond_ids = cond_ids.to(self.device)
        else:
            cond_ids = self._default_condition_encoding(conditions)

        # Prepend BOS + condition prefix
        bos = torch.full((1, 1), self.bos_id, dtype=torch.long, device=self.device)
        prompt = torch.cat([bos, cond_ids.unsqueeze(0) if cond_ids.dim() == 1 else cond_ids], dim=1)

        result = self.generate(
            num_sequences=num_sequences,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prompt_ids=prompt,
        )
        result["conditions"] = conditions
        return result

    # -- internal helpers ----------------------------------------------------

    def _get_next_logits(
        self,
        input_ids: Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        """Run the model and extract logits for the next token position.

        When a *kv_cache* is provided, it is passed to the model if the
        model's ``forward`` method accepts a ``kv_cache`` keyword argument.
        This enables incremental decoding where only the new token(s) are
        processed.

        Args:
            input_ids: ``(B, L)`` current sequence (or ``(B, 1)`` when
                using incremental decoding with cache).
            kv_cache: Optional :class:`KVCache` for incremental decoding.

        Returns:
            ``(B, V)`` logits for the last position.
        """
        # Try passing kv_cache to the model if it supports it
        kwargs: Dict[str, Any] = {}
        if kv_cache is not None:
            import inspect
            model_forward = getattr(self.model, "forward", self.model)
            sig = inspect.signature(model_forward)
            if "kv_cache" in sig.parameters:
                kwargs["kv_cache"] = kv_cache

        out = self.model(input_ids, **kwargs)

        if isinstance(out, dict):
            if "logits" in out:
                logits = out["logits"]  # (B, L, V)
                return logits[:, -1, :]
            hidden = out.get("last_hidden_state", None)
            if hidden is None:
                raise ValueError(
                    "Model output dict must contain 'logits' or 'last_hidden_state'"
                )
        elif isinstance(out, Tensor):
            hidden = out
        elif isinstance(out, (tuple, list)):
            hidden = out[0]
        else:
            raise TypeError(f"Unsupported model output type: {type(out)}")

        # Use LM head
        if self.lm_head is not None:
            logits = self.lm_head(hidden[:, -1, :])
            return logits

        raise ValueError(
            "Model does not return 'logits' and no lm_head was provided"
        )

    def _decode_batch(self, token_ids: Tensor) -> List[str]:
        """Decode a batch of token id tensors into strings.

        Args:
            token_ids: ``(B, L)`` token ids.

        Returns:
            List of decoded strings.
        """
        sequences: List[str] = []
        for i in range(token_ids.size(0)):
            ids = token_ids[i].cpu().tolist()
            # Remove special tokens
            ids = [t for t in ids if t not in (self.bos_id, self.eos_id, self.pad_id)]
            if hasattr(self.tokenizer, "decode"):
                seq = self.tokenizer.decode(ids)
            else:
                # Fallback: simple nucleotide mapping
                nuc_map = {0: "N", 1: "A", 2: "C", 3: "G", 4: "T"}
                seq = "".join(nuc_map.get(t, "N") for t in ids)
            sequences.append(seq)
        return sequences

    def _default_condition_encoding(
        self,
        conditions: Dict[str, Any],
    ) -> Tensor:
        """Encode conditions into token ids using a simple heuristic.

        Maps condition names and values to deterministic token ids.
        This is a fallback; for production use, supply a trained
        ``condition_encoder``.

        Args:
            conditions: Condition name -> value mapping.

        Returns:
            ``(L_cond,)`` tensor of condition token ids.
        """
        tokens: List[int] = []
        for key, value in sorted(conditions.items()):
            # Hash the key-value pair into a token id range
            hash_val = hash(f"{key}={value}") % (self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 4096)
            tokens.append(max(hash_val, 5))  # avoid special tokens 0-4
        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    # -- utility methods -----------------------------------------------------

    def estimate_perplexity(
        self,
        token_ids: Tensor,
    ) -> float:
        """Estimate perplexity of a given sequence under the model.

        Args:
            token_ids: ``(1, L)`` or ``(L,)`` token ids.

        Returns:
            Perplexity (scalar float).
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids.to(self.device)

        with torch.no_grad():
            out = self.model(token_ids)
            if isinstance(out, dict) and "logits" in out:
                logits = out["logits"]
            elif self.lm_head is not None:
                hidden = out["last_hidden_state"] if isinstance(out, dict) else out
                logits = self.lm_head(hidden)
            else:
                raise ValueError("Cannot compute perplexity without logits")

            # Shift: predict position i+1 from position i
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = token_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )

        return float(torch.exp(loss).item())

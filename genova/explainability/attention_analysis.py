"""Attention-based explanation methods for Genova transformer models.

Extracts, aggregates, and interprets attention weights from the multi-head
self-attention layers. Supports raw attention extraction, attention rollout
(Abnar & Zuidema, 2020), and identification of high-attention positions.

Example::

    analyzer = AttentionAnalyzer(model, tokenizer, device="cuda")
    result = analyzer.analyze("ACGTACGTACGT")
    rollout = result["rollout"]
    top_positions = result["high_attention_positions"]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from genova.data.tokenizer import GenomicTokenizer


class AttentionAnalyzer:
    """Extract and analyze attention patterns from Genova transformer models.

    Hooks into the transformer's attention layers to capture attention weight
    matrices during forward passes, then provides analysis utilities.

    Args:
        model: A Genova transformer model (backbone or MLM wrapper).
        tokenizer: Tokenizer with built vocabulary.
        device: Inference device.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: GenomicTokenizer,
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        self.model.to(self.device).eval()

        # Storage for captured attention weights
        self._attention_weights: List[torch.Tensor] = []
        self._hooks: List[Any] = []

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward hooks on all attention modules to capture weights."""
        self._remove_hooks()
        self._attention_weights = []

        attention_modules = self._find_attention_modules()
        if not attention_modules:
            logger.warning("No attention modules found in the model.")
            return

        for i, module in enumerate(attention_modules):
            hook = module.register_forward_hook(self._attention_hook_fn(i))
            self._hooks.append(hook)

        logger.debug("Registered {} attention hooks.", len(self._hooks))

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _attention_hook_fn(self, layer_idx: int):
        """Create a hook function that captures attention weights.

        The hook intercepts the attention computation within
        MultiHeadSelfAttention, capturing the (B, H, L, L) attention
        weight tensor before dropout.
        """
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # For our MultiHeadSelfAttention, we need to recompute attention
            # weights since they are not returned in the output.
            # We capture them by re-running the QKV projection.
            try:
                x = input[0] if isinstance(input, tuple) else input
                if not isinstance(x, torch.Tensor) or x.dim() != 3:
                    return

                B, L, D = x.shape

                # Access projection weights
                if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                    q = module.q_proj(x)
                    k = module.k_proj(x)

                    n_heads = module.n_heads
                    head_dim = D // n_heads

                    q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
                    k = k.view(B, L, n_heads, head_dim).transpose(1, 2)

                    scale = head_dim ** 0.5
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
                    attn_weights = torch.softmax(attn_weights, dim=-1)

                    self._attention_weights.append(attn_weights.detach().cpu())

            except Exception as e:
                logger.debug("Attention hook failed at layer {}: {}", layer_idx, e)

        return hook

    def _find_attention_modules(self) -> List[nn.Module]:
        """Find all MultiHeadSelfAttention modules in the model."""
        attention_modules: List[nn.Module] = []

        for name, module in self.model.named_modules():
            # Match our MultiHeadSelfAttention class
            cls_name = type(module).__name__
            if cls_name == "MultiHeadSelfAttention":
                attention_modules.append(module)
            elif hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                # Generic attention detection
                if hasattr(module, "n_heads"):
                    attention_modules.append(module)

        return attention_modules

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_attention(
        self,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Extract raw attention weights for all layers.

        Args:
            sequence: DNA sequence to analyze.
            max_length: Maximum token length.

        Returns:
            List of attention weight arrays, one per layer.
            Each array has shape ``(n_heads, L, L)``.
        """
        self._register_hooks()

        try:
            token_ids = self.tokenizer.encode(sequence, max_length=max_length)
            input_ids = torch.tensor(
                [token_ids], dtype=torch.long, device=self.device
            )
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

            results = []
            for attn in self._attention_weights:
                # attn is (1, H, L, L) -- remove batch dim
                results.append(attn.squeeze(0).numpy())

            return results

        finally:
            self._remove_hooks()

    @torch.no_grad()
    def analyze(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Perform comprehensive attention analysis on a sequence.

        Args:
            sequence: DNA sequence to analyze.
            max_length: Maximum token length.
            top_k: Number of top positions to return.

        Returns:
            Dict with:
                - ``attention_weights``: list of per-layer ``(H, L, L)`` arrays.
                - ``rollout``: attention rollout matrix ``(L, L)``.
                - ``token_importance``: per-token importance from rollout.
                - ``high_attention_positions``: top-k position dicts.
                - ``head_diversity``: per-layer head diversity scores.
                - ``tokens``: list of token strings.
        """
        tokens = self.tokenizer.tokenize(sequence)
        attention_weights = self.extract_attention(sequence, max_length)

        if not attention_weights:
            logger.warning("No attention weights captured.")
            return {
                "attention_weights": [],
                "rollout": np.array([]),
                "token_importance": np.array([]),
                "high_attention_positions": [],
                "head_diversity": [],
                "tokens": tokens,
            }

        # Attention rollout
        rollout = self.attention_rollout(attention_weights)

        # Token importance from rollout (sum of attention from CLS token)
        token_importance = self._compute_token_importance(rollout)

        # High attention positions
        high_positions = self.get_high_attention_positions(
            token_importance, tokens, top_k=top_k
        )

        # Head diversity
        head_diversity = [
            self._compute_head_diversity(attn) for attn in attention_weights
        ]

        return {
            "attention_weights": attention_weights,
            "rollout": rollout,
            "token_importance": token_importance,
            "high_attention_positions": high_positions,
            "head_diversity": head_diversity,
            "tokens": tokens,
        }

    # ------------------------------------------------------------------
    # Attention rollout
    # ------------------------------------------------------------------

    @staticmethod
    def attention_rollout(
        attention_weights: List[np.ndarray],
        add_residual: bool = True,
        head_fusion: str = "mean",
    ) -> np.ndarray:
        """Compute attention rollout across all layers.

        Implements the attention rollout method from Abnar & Zuidema (2020),
        which recursively multiplies attention matrices to trace how
        information flows from input to output across layers.

        Args:
            attention_weights: List of ``(n_heads, L, L)`` arrays.
            add_residual: Add identity matrix to account for residual
                connections (recommended).
            head_fusion: How to combine heads: ``"mean"``, ``"max"``, or
                ``"min"``.

        Returns:
            Rollout matrix of shape ``(L, L)`` representing the total
            attention from each output position to each input position.
        """
        if not attention_weights:
            return np.array([])

        rollout = None

        for layer_attn in attention_weights:
            # Fuse heads
            if head_fusion == "mean":
                attn = layer_attn.mean(axis=0)
            elif head_fusion == "max":
                attn = layer_attn.max(axis=0)
            elif head_fusion == "min":
                attn = layer_attn.min(axis=0)
            else:
                attn = layer_attn.mean(axis=0)

            # Add residual (identity matrix)
            if add_residual:
                identity = np.eye(attn.shape[0])
                attn = 0.5 * attn + 0.5 * identity

            # Row-normalize
            row_sums = attn.sum(axis=-1, keepdims=True)
            attn = attn / np.maximum(row_sums, 1e-12)

            # Accumulate
            if rollout is None:
                rollout = attn
            else:
                rollout = np.matmul(attn, rollout)

        return rollout if rollout is not None else np.array([])

    # ------------------------------------------------------------------
    # Position analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_token_importance(rollout: np.ndarray) -> np.ndarray:
        """Derive per-token importance from the rollout matrix.

        Uses the attention flowing from the CLS token (position 0) to
        each input position as a proxy for importance.

        Args:
            rollout: ``(L, L)`` rollout matrix.

        Returns:
            1-D array of shape ``(L,)`` with importance scores.
        """
        if rollout.size == 0:
            return np.array([])

        # Row 0 = how much CLS attends to each position
        importance = rollout[0]

        # Normalize to sum to 1
        total = importance.sum()
        if total > 0:
            importance = importance / total

        return importance

    @staticmethod
    def get_high_attention_positions(
        token_importance: np.ndarray,
        tokens: List[str],
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Identify positions receiving the most attention.

        Args:
            token_importance: Per-token importance scores.
            tokens: Token strings.
            top_k: Number of top positions to return.
            threshold: If provided, return all positions above this value.

        Returns:
            List of dicts with ``position``, ``token``, and ``importance``.
        """
        if token_importance.size == 0:
            return []

        if threshold is not None:
            indices = np.where(token_importance > threshold)[0]
            indices = indices[np.argsort(token_importance[indices])[::-1]]
        else:
            indices = np.argsort(token_importance)[-top_k:][::-1]

        results = []
        for idx in indices:
            idx_int = int(idx)
            results.append({
                "position": idx_int,
                "token": tokens[idx_int] if idx_int < len(tokens) else "[UNK]",
                "importance": float(token_importance[idx_int]),
            })

        return results

    # ------------------------------------------------------------------
    # Head analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_head_diversity(attention: np.ndarray) -> float:
        """Compute diversity among attention heads in a layer.

        Uses the average pairwise Jensen-Shannon divergence between
        head attention distributions as a measure of head specialization.

        Args:
            attention: ``(n_heads, L, L)`` attention weights.

        Returns:
            Diversity score in ``[0, 1]``. Higher means heads attend
            to different positions.
        """
        n_heads = attention.shape[0]
        if n_heads < 2:
            return 0.0

        # Average attention per head over all query positions
        # Shape: (n_heads, L) -- average attention distribution
        avg_attn = attention.mean(axis=1)  # (H, L)

        # Pairwise JS divergence
        total_jsd = 0.0
        count = 0
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                p = avg_attn[i] + 1e-12
                q = avg_attn[j] + 1e-12
                p = p / p.sum()
                q = q / q.sum()
                m = 0.5 * (p + q)
                jsd = 0.5 * (
                    np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))
                )
                total_jsd += jsd
                count += 1

        return float(total_jsd / count) if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Comparative analysis
    # ------------------------------------------------------------------

    def compare_sequences(
        self,
        seq_a: str,
        seq_b: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare attention patterns between two sequences.

        Useful for analyzing how a variant changes the model's attention.

        Args:
            seq_a: First sequence (e.g., reference).
            seq_b: Second sequence (e.g., alternate).
            max_length: Maximum token length.

        Returns:
            Dict with:
                - ``rollout_a``, ``rollout_b``: Rollout matrices.
                - ``importance_a``, ``importance_b``: Token importance.
                - ``importance_diff``: Difference in importance.
                - ``changed_positions``: Positions with largest changes.
        """
        result_a = self.analyze(seq_a, max_length=max_length)
        result_b = self.analyze(seq_b, max_length=max_length)

        imp_a = result_a["token_importance"]
        imp_b = result_b["token_importance"]

        # Align to shorter length
        min_len = min(len(imp_a), len(imp_b))
        imp_a = imp_a[:min_len]
        imp_b = imp_b[:min_len]
        diff = imp_b - imp_a

        # Find positions with largest changes
        n_changed = max(1, min_len // 10)
        changed_indices = np.argsort(np.abs(diff))[-n_changed:][::-1]

        tokens_a = result_a["tokens"][:min_len]
        tokens_b = result_b["tokens"][:min_len]

        changed_positions = []
        for idx in changed_indices:
            idx_int = int(idx)
            changed_positions.append({
                "position": idx_int,
                "token_a": tokens_a[idx_int] if idx_int < len(tokens_a) else "",
                "token_b": tokens_b[idx_int] if idx_int < len(tokens_b) else "",
                "importance_a": float(imp_a[idx_int]),
                "importance_b": float(imp_b[idx_int]),
                "change": float(diff[idx_int]),
            })

        return {
            "rollout_a": result_a["rollout"],
            "rollout_b": result_b["rollout"],
            "importance_a": imp_a,
            "importance_b": imp_b,
            "importance_diff": diff,
            "changed_positions": changed_positions,
        }

    # ------------------------------------------------------------------
    # Layer-wise analysis
    # ------------------------------------------------------------------

    def layer_wise_importance(
        self,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute token importance at each layer independently.

        Args:
            sequence: DNA sequence to analyze.
            max_length: Maximum token length.

        Returns:
            Dict with:
                - ``layer_importance``: list of per-layer importance arrays.
                - ``cumulative_rollout``: list of rollout matrices up to each layer.
                - ``tokens``: token strings.
        """
        tokens = self.tokenizer.tokenize(sequence)
        attention_weights = self.extract_attention(sequence, max_length)

        if not attention_weights:
            return {
                "layer_importance": [],
                "cumulative_rollout": [],
                "tokens": tokens,
            }

        layer_importance = []
        cumulative_rollouts = []

        for i in range(1, len(attention_weights) + 1):
            rollout = self.attention_rollout(attention_weights[:i])
            importance = self._compute_token_importance(rollout)
            layer_importance.append(importance)
            cumulative_rollouts.append(rollout)

        return {
            "layer_importance": layer_importance,
            "cumulative_rollout": cumulative_rollouts,
            "tokens": tokens,
        }

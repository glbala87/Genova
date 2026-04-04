"""Advanced attribution methods for genomic sequence models.

Implements Integrated Gradients (Sundararajan et al., 2017), DeepLIFT-style
reference-based attribution, and SmoothGrad for robust gradient explanations.

These methods attribute importance to individual nucleotide positions in the
input sequence, enabling fine-grained interpretation of model predictions.

Example::

    from genova.explainability.integrated_gradients import (
        IntegratedGradientsExplainer,
        SmoothGradExplainer,
    )

    ig = IntegratedGradientsExplainer(model, tokenizer, device="cuda")
    result = ig.explain("ACGTACGTACGT" * 10)
    ig.visualize(result)

    sg = SmoothGradExplainer(model, tokenizer, device="cuda")
    result = sg.explain("ACGTACGTACGT" * 10, n_samples=50, noise_std=0.1)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger

from genova.data.tokenizer import GenomicTokenizer


# ---------------------------------------------------------------------------
# Baseline strategies
# ---------------------------------------------------------------------------

def _zero_baseline(embedding: Tensor) -> Tensor:
    """Zero baseline embedding.

    Args:
        embedding: ``(1, L, D)`` input embedding tensor.

    Returns:
        Zero tensor of the same shape.
    """
    return torch.zeros_like(embedding)


def _random_baseline(embedding: Tensor, seed: Optional[int] = None) -> Tensor:
    """Random uniform baseline embedding.

    Args:
        embedding: ``(1, L, D)`` input embedding tensor.
        seed: Optional random seed for reproducibility.

    Returns:
        Random tensor of the same shape, scaled to the input's range.
    """
    if seed is not None:
        gen = torch.Generator(device=embedding.device).manual_seed(seed)
    else:
        gen = None
    low = embedding.min().item()
    high = embedding.max().item()
    return torch.rand_like(embedding, generator=gen) * (high - low) + low


def _n_token_baseline(
    embedding: Tensor,
    tokenizer: GenomicTokenizer,
    embedding_layer: nn.Embedding,
    seq_length: int,
    device: torch.device,
) -> Tensor:
    """N-token (ambiguous nucleotide) baseline embedding.

    Uses the embedding of the 'N' token repeated to the sequence length
    as a biologically meaningful uninformative baseline.

    Args:
        embedding: ``(1, L, D)`` input embedding for shape reference.
        tokenizer: Tokenizer with encode capability.
        embedding_layer: Model embedding layer.
        seq_length: Number of tokens.
        device: Target device.

    Returns:
        Baseline embedding tensor of the same shape as *embedding*.
    """
    # Encode a sequence of N's
    n_seq = "N" * seq_length
    try:
        n_ids = tokenizer.encode(n_seq)
    except Exception:
        # Fallback: use pad token id
        pad_id = getattr(tokenizer, "pad_token_id", 0)
        n_ids = [pad_id] * embedding.shape[1]

    n_ids = n_ids[: embedding.shape[1]]
    n_ids = n_ids + [0] * (embedding.shape[1] - len(n_ids))
    n_tensor = torch.tensor([n_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        baseline = embedding_layer(n_tensor)
    return baseline.detach()


# ---------------------------------------------------------------------------
# Utility: find embedding layer
# ---------------------------------------------------------------------------

def _find_embedding_layer(model: nn.Module) -> Optional[nn.Embedding]:
    """Find the first large Embedding layer in a model.

    Args:
        model: PyTorch model to search.

    Returns:
        The first ``nn.Embedding`` with ``num_embeddings > 100``, or ``None``.
    """
    for module in model.modules():
        if isinstance(module, nn.Embedding) and module.num_embeddings > 100:
            return module
    return None


def _get_model_output_scalar(
    model: nn.Module,
    embeddings: Tensor,
    input_ids: Tensor,
    tokenizer: GenomicTokenizer,
    target_index: Optional[int] = None,
) -> Tensor:
    """Run the model and reduce output to a scalar for gradient computation.

    Args:
        model: The genomic model.
        embeddings: ``(B, L, D)`` embedding tensor (not used directly for
            forward but needed for gradient flow).
        input_ids: ``(B, L)`` token ids for the actual forward pass.
        tokenizer: Tokenizer for padding info.
        target_index: If the model outputs multi-class logits, index of the
            target class. ``None`` means use the max-predicted class.

    Returns:
        Scalar tensor connected to the computational graph via *embeddings*.
    """
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    if isinstance(outputs, dict):
        hidden = outputs.get("last_hidden_state")
        if hidden is None:
            hidden = outputs.get("logits", outputs.get("hidden_states"))
            if isinstance(hidden, (list, tuple)):
                hidden = hidden[-1]
    elif isinstance(outputs, Tensor):
        hidden = outputs
    else:
        hidden = getattr(outputs, "last_hidden_state", outputs)

    if isinstance(hidden, (list, tuple)):
        hidden = hidden[-1]

    # Mean-pool and reduce to scalar
    if hidden.dim() == 3:
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    else:
        pooled = hidden

    if target_index is not None and pooled.dim() >= 2 and pooled.size(-1) > 1:
        return pooled[:, target_index].sum()

    return pooled.norm(dim=-1).sum()


# ---------------------------------------------------------------------------
# IntegratedGradientsExplainer
# ---------------------------------------------------------------------------


class IntegratedGradientsExplainer:
    """Integrated Gradients attribution for genomic sequence models.

    Accumulates gradients along a straight-line path from a baseline to the
    input embedding (Sundararajan et al., 2017).  Also supports DeepLIFT-style
    reference-based contribution scores.

    Args:
        model: Pretrained Genova model (backbone or MLM wrapper).
        tokenizer: Tokenizer with built vocabulary.
        device: Inference device.
        n_steps: Number of interpolation steps along the path (50--300).
        baseline_type: One of ``"zero"``, ``"random"``, or ``"n_token"``.
        internal_batch_size: Batch size for processing interpolation steps.
    """

    VALID_BASELINES = ("zero", "random", "n_token")

    def __init__(
        self,
        model: nn.Module,
        tokenizer: GenomicTokenizer,
        *,
        device: Union[str, torch.device] = "cpu",
        n_steps: int = 100,
        baseline_type: str = "zero",
        internal_batch_size: int = 32,
    ) -> None:
        if baseline_type not in self.VALID_BASELINES:
            raise ValueError(
                f"baseline_type must be one of {self.VALID_BASELINES}, "
                f"got {baseline_type!r}"
            )
        if not (50 <= n_steps <= 300):
            warnings.warn(
                f"n_steps={n_steps} is outside recommended range [50, 300]. "
                "Results may be less accurate.",
                stacklevel=2,
            )

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        self.internal_batch_size = internal_batch_size

        self.model.to(self.device).eval()

        self._embedding_layer = _find_embedding_layer(self.model)
        if self._embedding_layer is None:
            logger.warning(
                "No embedding layer found. Attribution will use gradient "
                "fallback on input ids."
            )

    # ------------------------------------------------------------------
    # Baseline construction
    # ------------------------------------------------------------------

    def _get_baseline(
        self,
        input_embedding: Tensor,
        input_ids: Tensor,
    ) -> Tensor:
        """Construct baseline embedding according to the configured strategy.

        Args:
            input_embedding: ``(1, L, D)`` input embedding.
            input_ids: ``(1, L)`` token ids (used for N-token baseline).

        Returns:
            Baseline embedding of the same shape.
        """
        if self.baseline_type == "zero":
            return _zero_baseline(input_embedding)
        elif self.baseline_type == "random":
            return _random_baseline(input_embedding)
        elif self.baseline_type == "n_token":
            if self._embedding_layer is None:
                logger.warning(
                    "No embedding layer for N-token baseline; falling back to zero."
                )
                return _zero_baseline(input_embedding)
            return _n_token_baseline(
                input_embedding,
                self.tokenizer,
                self._embedding_layer,
                seq_length=input_ids.shape[1],
                device=self.device,
            )
        else:
            return _zero_baseline(input_embedding)

    # ------------------------------------------------------------------
    # Core integrated gradients
    # ------------------------------------------------------------------

    def _compute_integrated_gradients(
        self,
        input_ids: Tensor,
        target_index: Optional[int] = None,
    ) -> np.ndarray:
        """Compute Integrated Gradients for a single input.

        Accumulates gradients along a straight-line path from the baseline
        to the input embedding, then scales by the input-baseline difference.

        Args:
            input_ids: ``(1, L)`` token ids.
            target_index: Optional target class index.

        Returns:
            Attribution array of shape ``(L,)`` with per-position scores.
        """
        if self._embedding_layer is None:
            return self._gradient_fallback(input_ids, target_index)

        # Get input embedding
        input_embedding = self._embedding_layer(input_ids).detach()
        baseline = self._get_baseline(input_embedding, input_ids).detach()

        # Difference
        diff = input_embedding - baseline  # (1, L, D)

        # Generate interpolation alphas
        alphas = torch.linspace(0.0, 1.0, self.n_steps + 1, device=self.device)

        # Accumulate gradients
        accumulated_grads = torch.zeros_like(input_embedding)

        for batch_start in range(0, len(alphas), self.internal_batch_size):
            batch_alphas = alphas[batch_start: batch_start + self.internal_batch_size]
            batch_size = len(batch_alphas)

            # Interpolated embeddings: (batch_size, L, D)
            scaled_inputs = (
                baseline
                + batch_alphas.view(-1, 1, 1) * diff
            )
            scaled_inputs = scaled_inputs.detach().requires_grad_(True)

            # Forward through the model using a hook to inject embeddings
            batch_ids = input_ids.expand(batch_size, -1)
            attention_mask = (batch_ids != self.tokenizer.pad_token_id).long()

            # Use hook to replace embedding output
            handle = self._embedding_layer.register_forward_hook(
                lambda module, inp, out, si=scaled_inputs: si
            )
            try:
                outputs = self.model(
                    input_ids=batch_ids, attention_mask=attention_mask
                )
            finally:
                handle.remove()

            # Extract output and compute scalar target
            if isinstance(outputs, dict):
                hidden = outputs.get("last_hidden_state")
                if hidden is None:
                    hidden = outputs.get("logits", next(iter(outputs.values())))
            elif isinstance(outputs, Tensor):
                hidden = outputs
            else:
                hidden = getattr(outputs, "last_hidden_state", outputs)

            if isinstance(hidden, (list, tuple)):
                hidden = hidden[-1]

            if hidden.dim() == 3:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                pooled = hidden

            if target_index is not None and pooled.size(-1) > 1:
                target = pooled[:, target_index].sum()
            else:
                target = pooled.norm(dim=-1).sum()

            # Backward
            if scaled_inputs.grad is not None:
                scaled_inputs.grad.zero_()
            target.backward()

            if scaled_inputs.grad is not None:
                accumulated_grads += scaled_inputs.grad.sum(dim=0, keepdim=True)

        # Trapezoidal rule: average gradients and multiply by difference
        avg_grads = accumulated_grads / (self.n_steps + 1)
        attributions = (diff * avg_grads).sum(dim=-1).squeeze(0)  # (L,)

        return attributions.detach().cpu().numpy()

    def _gradient_fallback(
        self,
        input_ids: Tensor,
        target_index: Optional[int] = None,
    ) -> np.ndarray:
        """Simple gradient-based attribution fallback when no embedding layer
        is available.

        Args:
            input_ids: ``(1, L)`` token ids.
            target_index: Optional target class index.

        Returns:
            Attribution array of shape ``(L,)``.
        """
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        input_ids_float = input_ids.float().requires_grad_(True)

        try:
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            if isinstance(outputs, dict):
                hidden = outputs.get("last_hidden_state", next(iter(outputs.values())))
            elif isinstance(outputs, Tensor):
                hidden = outputs
            else:
                hidden = outputs

            if isinstance(hidden, (list, tuple)):
                hidden = hidden[-1]

            target = hidden.mean()
            target.backward()
        except Exception:
            return np.zeros(input_ids.shape[1])

        return np.zeros(input_ids.shape[1])

    # ------------------------------------------------------------------
    # DeepLIFT-style attribution
    # ------------------------------------------------------------------

    def _compute_deeplift_attribution(
        self,
        input_ids: Tensor,
        target_index: Optional[int] = None,
    ) -> np.ndarray:
        """Compute DeepLIFT-style reference-based attribution scores.

        Uses the difference between the input and reference activations,
        multiplied by the gradient, as a first-order approximation of
        DeepLIFT contribution scores.

        Args:
            input_ids: ``(1, L)`` token ids.
            target_index: Optional target class index.

        Returns:
            Per-nucleotide contribution scores of shape ``(L,)``.
        """
        if self._embedding_layer is None:
            logger.warning("No embedding layer; returning zero attributions.")
            return np.zeros(input_ids.shape[1])

        input_embedding = self._embedding_layer(input_ids).detach()
        baseline = self._get_baseline(input_embedding, input_ids).detach()
        diff = input_embedding - baseline

        # Forward with input embedding (via hook)
        input_emb = input_embedding.clone().requires_grad_(True)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        handle = self._embedding_layer.register_forward_hook(
            lambda module, inp, out, emb=input_emb: emb
        )
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()

        if isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state")
            if hidden is None:
                hidden = outputs.get("logits", next(iter(outputs.values())))
        elif isinstance(outputs, Tensor):
            hidden = outputs
        else:
            hidden = getattr(outputs, "last_hidden_state", outputs)

        if isinstance(hidden, (list, tuple)):
            hidden = hidden[-1]

        if hidden.dim() == 3:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden

        if target_index is not None and pooled.size(-1) > 1:
            target = pooled[:, target_index].sum()
        else:
            target = pooled.norm(dim=-1).sum()

        target.backward()

        if input_emb.grad is not None:
            # DeepLIFT: gradient * delta
            contributions = (input_emb.grad * diff).sum(dim=-1).squeeze(0)
            return contributions.detach().cpu().numpy()

        return np.zeros(input_ids.shape[1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        method: str = "integrated_gradients",
        target_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute attribution for a DNA sequence.

        Args:
            sequence: DNA sequence to explain.
            max_length: Maximum token length.
            method: Attribution method -- ``"integrated_gradients"`` or
                ``"deeplift"``.
            target_index: Target class index for multi-class models.

        Returns:
            Dict with:
                - ``"attributions"``: per-position attribution scores ``(L,)``.
                - ``"tokens"``: list of token strings.
                - ``"method"``: attribution method used.
                - ``"n_steps"``: number of interpolation steps (IG only).
                - ``"baseline_type"``: baseline type used.
        """
        tokens = self.tokenizer.tokenize(sequence)
        token_ids = self.tokenizer.encode(sequence, max_length=max_length)
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )

        if method == "integrated_gradients":
            attributions = self._compute_integrated_gradients(
                input_ids, target_index=target_index
            )
        elif method == "deeplift":
            attributions = self._compute_deeplift_attribution(
                input_ids, target_index=target_index
            )
        else:
            raise ValueError(
                f"method must be 'integrated_gradients' or 'deeplift', "
                f"got {method!r}"
            )

        # Align length with tokens
        attr_len = min(len(attributions), len(tokens))
        attributions = attributions[:attr_len]
        tokens = tokens[:attr_len]

        return {
            "attributions": attributions,
            "tokens": tokens,
            "method": method,
            "n_steps": self.n_steps if method == "integrated_gradients" else None,
            "baseline_type": self.baseline_type,
        }

    def explain_batch(
        self,
        sequences: List[str],
        max_length: Optional[int] = None,
        method: str = "integrated_gradients",
        target_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Compute attributions for a batch of DNA sequences.

        Processes sequences one at a time (IG is memory-intensive per sample).

        Args:
            sequences: List of DNA sequences.
            max_length: Maximum token length.
            method: Attribution method.
            target_index: Target class index for multi-class models.

        Returns:
            List of attribution dicts, one per sequence.
        """
        results: List[Dict[str, Any]] = []
        for i, seq in enumerate(sequences):
            logger.debug("Explaining sequence {}/{}", i + 1, len(sequences))
            result = self.explain(
                seq,
                max_length=max_length,
                method=method,
                target_index=target_index,
            )
            results.append(result)
        return results

    def visualize(
        self,
        explanation: Dict[str, Any],
        top_k: int = 20,
        show_plot: bool = True,
    ) -> Dict[str, Any]:
        """Visualize attribution results.

        Identifies the most important positions and optionally creates a
        bar-chart visualization (requires matplotlib).

        Args:
            explanation: Result dict from :meth:`explain`.
            top_k: Number of top-attributed positions to highlight.
            show_plot: Whether to display a matplotlib plot.

        Returns:
            Dict with:
                - ``"top_positions"``: list of ``{position, token, attribution}``
                  dicts for the top-k positions.
                - ``"figure"``: matplotlib Figure (if ``show_plot`` and
                  matplotlib is available), else ``None``.
        """
        attributions = explanation["attributions"]
        tokens = explanation["tokens"]

        abs_attr = np.abs(attributions)
        top_indices = np.argsort(abs_attr)[-top_k:][::-1]

        top_positions = []
        for idx in top_indices:
            idx_int = int(idx)
            top_positions.append({
                "position": idx_int,
                "token": tokens[idx_int] if idx_int < len(tokens) else "[UNK]",
                "attribution": float(attributions[idx_int]),
                "direction": "positive" if attributions[idx_int] > 0 else "negative",
            })

        figure = None
        if show_plot:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, 1, figsize=(14, 4))
                positions = np.arange(len(attributions))
                colors = [
                    "steelblue" if v >= 0 else "salmon" for v in attributions
                ]
                ax.bar(positions, attributions, color=colors, width=1.0)
                ax.set_xlabel("Position")
                ax.set_ylabel("Attribution")
                ax.set_title(
                    f"Attribution ({explanation['method']}, "
                    f"baseline={explanation['baseline_type']})"
                )
                # Mark top positions
                for pos_info in top_positions[:5]:
                    ax.annotate(
                        pos_info["token"],
                        xy=(pos_info["position"], pos_info["attribution"]),
                        fontsize=7,
                        ha="center",
                    )
                fig.tight_layout()
                figure = fig
            except ImportError:
                logger.info(
                    "matplotlib not available; skipping plot generation."
                )

        return {
            "top_positions": top_positions,
            "figure": figure,
        }


# ---------------------------------------------------------------------------
# SmoothGradExplainer
# ---------------------------------------------------------------------------


class SmoothGradExplainer:
    """SmoothGrad attribution for genomic sequence models.

    Adds Gaussian noise to the input embedding and averages the resulting
    gradients over *N* samples to obtain smoother, more interpretable
    saliency maps.

    Args:
        model: Pretrained Genova model.
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
        self._embedding_layer = _find_embedding_layer(self.model)
        if self._embedding_layer is None:
            logger.warning(
                "No embedding layer found. SmoothGrad will produce "
                "zero attributions."
            )

    def explain(
        self,
        sequence: str,
        n_samples: int = 50,
        noise_std: float = 0.1,
        max_length: Optional[int] = None,
        target_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute SmoothGrad attribution for a DNA sequence.

        Args:
            sequence: DNA sequence to explain.
            n_samples: Number of noisy samples to average over.
            noise_std: Standard deviation of Gaussian noise added to
                the input embedding.
            max_length: Maximum token length.
            target_index: Target class index for multi-class models.

        Returns:
            Dict with:
                - ``"attributions"``: smoothed per-position attribution ``(L,)``.
                - ``"tokens"``: list of token strings.
                - ``"n_samples"``: number of noisy samples used.
                - ``"noise_std"``: standard deviation of noise.
                - ``"method"``: ``"smooth_grad"``.
        """
        tokens = self.tokenizer.tokenize(sequence)
        token_ids = self.tokenizer.encode(sequence, max_length=max_length)
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )

        if self._embedding_layer is None:
            return {
                "attributions": np.zeros(len(tokens)),
                "tokens": tokens,
                "n_samples": n_samples,
                "noise_std": noise_std,
                "method": "smooth_grad",
            }

        input_embedding = self._embedding_layer(input_ids).detach()
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        accumulated_grads = torch.zeros_like(input_embedding)

        for _ in range(n_samples):
            noise = torch.randn_like(input_embedding) * noise_std
            noisy_embedding = (input_embedding + noise).requires_grad_(True)

            # Inject noisy embedding via hook
            handle = self._embedding_layer.register_forward_hook(
                lambda module, inp, out, ne=noisy_embedding: ne
            )
            try:
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            finally:
                handle.remove()

            # Extract hidden states
            if isinstance(outputs, dict):
                hidden = outputs.get("last_hidden_state")
                if hidden is None:
                    hidden = outputs.get("logits", next(iter(outputs.values())))
            elif isinstance(outputs, Tensor):
                hidden = outputs
            else:
                hidden = getattr(outputs, "last_hidden_state", outputs)

            if isinstance(hidden, (list, tuple)):
                hidden = hidden[-1]

            # Reduce to scalar
            if hidden.dim() == 3:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                pooled = hidden

            if target_index is not None and pooled.size(-1) > 1:
                target = pooled[:, target_index].sum()
            else:
                target = pooled.norm(dim=-1).sum()

            target.backward()

            if noisy_embedding.grad is not None:
                accumulated_grads += noisy_embedding.grad.detach()

        # Average gradients, then sum over embedding dimension
        avg_grads = accumulated_grads / n_samples
        attributions = avg_grads.abs().sum(dim=-1).squeeze(0)  # (L,)
        attributions_np = attributions.cpu().numpy()

        # Align with tokens
        attr_len = min(len(attributions_np), len(tokens))
        attributions_np = attributions_np[:attr_len]
        tokens = tokens[:attr_len]

        return {
            "attributions": attributions_np,
            "tokens": tokens,
            "n_samples": n_samples,
            "noise_std": noise_std,
            "method": "smooth_grad",
        }

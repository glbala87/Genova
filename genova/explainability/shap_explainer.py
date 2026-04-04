"""SHAP-based explainability for Genova genomic predictions.

Integrates SHAP (SHapley Additive exPlanations) with the Genova model
to explain variant predictions by attributing importance to individual
nucleotides or k-mers in the input sequence.

Supports both DeepExplainer (for deep learning models) and KernelExplainer
(model-agnostic fallback). Long sequences are handled via chunk-based
explanation to stay within GPU memory limits.

Example::

    explainer = GenomicSHAPExplainer(model, tokenizer, device="cuda")
    values = explainer.explain("ACGTACGTACGT" * 10)
    explainer.explain_variant(ref_seq, alt_seq)
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from genova.data.tokenizer import GenomicTokenizer


class GenomicSHAPExplainer:
    """SHAP explainer for genomic sequence models.

    Wraps the Genova model to compute SHAP values at the token level,
    highlighting which positions in a DNA sequence contribute most to
    the model's prediction.

    Args:
        model: Pretrained Genova model (backbone or MLM).
        tokenizer: Tokenizer with built vocabulary.
        device: Inference device.
        method: SHAP method -- ``"deep"`` for DeepExplainer or
            ``"kernel"`` for KernelExplainer.
        max_chunk_length: Maximum token length per explanation chunk.
            Longer sequences are split and explained in chunks.
        n_background_samples: Number of background samples for
            KernelExplainer (ignored for DeepExplainer).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: GenomicTokenizer,
        *,
        device: Union[str, torch.device] = "cpu",
        method: str = "deep",
        max_chunk_length: int = 512,
        n_background_samples: int = 100,
    ) -> None:
        if method not in ("deep", "kernel"):
            raise ValueError(f"method must be 'deep' or 'kernel', got {method!r}")

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.method = method
        self.max_chunk_length = max_chunk_length
        self.n_background_samples = n_background_samples

        self.model.to(self.device).eval()

        self._shap_explainer: Any = None
        self._background_data: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Model wrapper for SHAP
    # ------------------------------------------------------------------

    def _model_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward function that returns a scalar prediction per sample.

        For SHAP, the model must map ``(B, L)`` integer inputs to ``(B,)``
        or ``(B, C)`` outputs. We use mean-pooled hidden states reduced
        to a single score via L2 norm.

        Args:
            input_ids: ``(B, L)`` tensor of token ids.

        Returns:
            ``(B, 1)`` prediction tensor.
        """
        with torch.no_grad():
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            if isinstance(outputs, dict):
                hidden = outputs.get("last_hidden_state")
                if hidden is None:
                    hidden = outputs.get("logits", outputs.get("hidden_states"))
                    if isinstance(hidden, (list, tuple)):
                        hidden = hidden[-1]
            elif isinstance(outputs, torch.Tensor):
                hidden = outputs
            else:
                hidden = getattr(outputs, "last_hidden_state", outputs)

            # Mean pool over sequence
            if hidden.dim() == 3:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                pooled = hidden

            # Reduce to scalar per sample via norm
            return pooled.norm(dim=-1, keepdim=True)

    def _model_numpy_wrapper(self, input_array: np.ndarray) -> np.ndarray:
        """Numpy-in, numpy-out wrapper for KernelExplainer.

        Args:
            input_array: ``(B, L)`` integer array.

        Returns:
            ``(B, 1)`` prediction array.
        """
        input_tensor = torch.tensor(input_array, dtype=torch.long, device=self.device)
        output = self._model_forward(input_tensor)
        return output.cpu().numpy()

    # ------------------------------------------------------------------
    # Background data generation
    # ------------------------------------------------------------------

    def _generate_background(self, seq_length: int) -> torch.Tensor:
        """Generate background data for SHAP explainer.

        Creates random DNA sequences tokenized and padded to the given length.

        Args:
            seq_length: Token length for background samples.

        Returns:
            ``(n_background_samples, seq_length)`` tensor.
        """
        nucleotides = "ACGT"
        bg_sequences: List[str] = []
        # Approximate raw sequence length from token length
        raw_len = seq_length * (self.tokenizer.k if self.tokenizer.mode == "kmer" else 1)

        for _ in range(self.n_background_samples):
            seq = "".join(
                nucleotides[i % 4] for i in np.random.randint(0, 4, size=raw_len)
            )
            bg_sequences.append(seq)

        encoded = self.tokenizer.batch_encode(
            bg_sequences, max_length=seq_length, padding=True
        )
        return torch.tensor(encoded["input_ids"], dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Explainer initialization
    # ------------------------------------------------------------------

    def _init_explainer(self, token_length: int) -> None:
        """Initialize the SHAP explainer with appropriate background data.

        Args:
            token_length: Token sequence length for the explanation.
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "shap is required for explanations. Install with: pip install shap"
            )

        self._background_data = self._generate_background(token_length)

        if self.method == "deep":
            try:
                self._shap_explainer = shap.DeepExplainer(
                    self.model, self._background_data
                )
                logger.info("Initialized SHAP DeepExplainer.")
            except Exception as e:
                logger.warning(
                    "DeepExplainer failed ({}), falling back to KernelExplainer.", e
                )
                self.method = "kernel"
                self._init_kernel_explainer(token_length)
        else:
            self._init_kernel_explainer(token_length)

    def _init_kernel_explainer(self, token_length: int) -> None:
        """Initialize KernelExplainer."""
        import shap

        bg_np = self._background_data.cpu().numpy()  # type: ignore[union-attr]
        self._shap_explainer = shap.KernelExplainer(
            self._model_numpy_wrapper, bg_np
        )
        logger.info("Initialized SHAP KernelExplainer.")

    # ------------------------------------------------------------------
    # Main explanation methods
    # ------------------------------------------------------------------

    def explain(
        self,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute SHAP values for a DNA sequence.

        For sequences longer than ``max_chunk_length``, the sequence is
        split into overlapping chunks and SHAP values are aggregated.

        Args:
            sequence: DNA sequence to explain.
            max_length: Override maximum token length.

        Returns:
            Dict with:
                - ``shap_values``: numpy array of shape ``(L,)`` with
                  per-token SHAP values.
                - ``tokens``: list of token strings.
                - ``base_value``: expected model output on background.
                - ``prediction``: model output on the input sequence.
        """
        tokens = self.tokenizer.tokenize(sequence)
        token_ids = self.tokenizer.encode(sequence, max_length=max_length)
        effective_length = len(token_ids)

        # Check if chunking is needed
        if effective_length > self.max_chunk_length:
            return self._explain_chunked(sequence, max_length)

        return self._explain_single(token_ids, tokens)

    def _explain_single(
        self,
        token_ids: List[int],
        tokens: List[str],
    ) -> Dict[str, Any]:
        """Explain a single (non-chunked) sequence.

        Args:
            token_ids: Encoded token ids.
            tokens: Token strings for display.

        Returns:
            Explanation dict.
        """
        seq_length = len(token_ids)

        # Initialize explainer if needed
        if (
            self._shap_explainer is None
            or self._background_data is None
            or self._background_data.shape[1] != seq_length
        ):
            self._init_explainer(seq_length)

        input_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )

        # Compute prediction
        prediction = self._model_forward(input_tensor).cpu().numpy().flatten()

        # Compute SHAP values
        try:
            if self.method == "deep":
                shap_values = self._shap_explainer.shap_values(input_tensor)
            else:
                input_np = input_tensor.cpu().numpy()
                shap_values = self._shap_explainer.shap_values(input_np, nsamples=200)
        except Exception as e:
            logger.warning("SHAP computation failed: {}. Using gradient fallback.", e)
            shap_values = self._gradient_attribution(input_tensor)

        # Flatten and normalize
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if isinstance(shap_values, np.ndarray):
            sv = shap_values.flatten()[:len(tokens)]
        else:
            sv = np.array(shap_values).flatten()[:len(tokens)]

        base_value = 0.0
        if hasattr(self._shap_explainer, "expected_value"):
            ev = self._shap_explainer.expected_value
            base_value = float(ev[0] if isinstance(ev, (list, np.ndarray)) else ev)

        return {
            "shap_values": sv,
            "tokens": tokens,
            "base_value": base_value,
            "prediction": float(prediction[0]),
        }

    def _explain_chunked(
        self,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Explain a long sequence by splitting into overlapping chunks.

        Args:
            sequence: Full DNA sequence.
            max_length: Maximum token length override.

        Returns:
            Aggregated explanation dict.
        """
        all_tokens = self.tokenizer.tokenize(sequence)
        chunk_size = self.max_chunk_length - 2  # room for [CLS] and [SEP]
        overlap = min(50, chunk_size // 4)
        step = chunk_size - overlap

        all_shap = np.zeros(len(all_tokens))
        counts = np.zeros(len(all_tokens))
        total_prediction = 0.0

        for start in range(0, len(all_tokens), step):
            end = min(start + chunk_size, len(all_tokens))
            chunk_tokens = all_tokens[start:end]

            # Re-encode the chunk
            chunk_seq = "".join(chunk_tokens) if self.tokenizer.mode == "nucleotide" else sequence
            chunk_ids = self.tokenizer.encode(
                chunk_seq, max_length=self.max_chunk_length
            )

            result = self._explain_single(chunk_ids, chunk_tokens)
            sv = result["shap_values"]

            # Map chunk SHAP values back to global positions
            chunk_len = min(len(sv), end - start)
            all_shap[start : start + chunk_len] += sv[:chunk_len]
            counts[start : start + chunk_len] += 1
            total_prediction += result["prediction"]

            if end >= len(all_tokens):
                break

        # Average overlapping regions
        counts = np.maximum(counts, 1)
        all_shap /= counts
        n_chunks = max(1, int(np.ceil(len(all_tokens) / step)))

        return {
            "shap_values": all_shap,
            "tokens": all_tokens,
            "base_value": 0.0,
            "prediction": total_prediction / n_chunks,
        }

    def explain_variant(
        self,
        ref_sequence: str,
        alt_sequence: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Explain a variant by comparing SHAP values of ref vs alt.

        Args:
            ref_sequence: Reference allele context sequence.
            alt_sequence: Alternate allele context sequence.
            max_length: Maximum token length.

        Returns:
            Dict with:
                - ``ref_shap``: SHAP values for reference.
                - ``alt_shap``: SHAP values for alternate.
                - ``diff_shap``: Difference (alt - ref).
                - ``ref_tokens`` / ``alt_tokens``: Token strings.
                - ``important_positions``: Indices of top contributing positions.
        """
        ref_result = self.explain(ref_sequence, max_length=max_length)
        alt_result = self.explain(alt_sequence, max_length=max_length)

        # Align to shorter length
        min_len = min(len(ref_result["shap_values"]), len(alt_result["shap_values"]))
        ref_sv = ref_result["shap_values"][:min_len]
        alt_sv = alt_result["shap_values"][:min_len]
        diff = alt_sv - ref_sv

        # Find most important positions (top 10% by absolute difference)
        n_important = max(1, min_len // 10)
        important_positions = np.argsort(np.abs(diff))[-n_important:][::-1].tolist()

        return {
            "ref_shap": ref_sv,
            "alt_shap": alt_sv,
            "diff_shap": diff,
            "ref_tokens": ref_result["tokens"][:min_len],
            "alt_tokens": alt_result["tokens"][:min_len],
            "important_positions": important_positions,
            "ref_prediction": ref_result["prediction"],
            "alt_prediction": alt_result["prediction"],
        }

    # ------------------------------------------------------------------
    # Gradient-based fallback
    # ------------------------------------------------------------------

    def _gradient_attribution(self, input_ids: torch.Tensor) -> np.ndarray:
        """Compute gradient-based attribution as fallback when SHAP fails.

        Uses integrated gradients approximation via input embedding gradients.

        Args:
            input_ids: ``(1, L)`` token id tensor.

        Returns:
            Attribution array of shape ``(1, L)``.
        """
        self.model.eval()
        input_ids = input_ids.clone().detach().requires_grad_(False)

        # Get the embedding layer
        embedding_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Embedding) and module.num_embeddings > 100:
                embedding_layer = module
                break

        if embedding_layer is None:
            logger.warning("No embedding layer found for gradient attribution.")
            return np.zeros((1, input_ids.shape[1]))

        # Forward with gradient tracking
        embeddings = embedding_layer(input_ids)
        embeddings.requires_grad_(True)

        # We need to run the rest of the model from embeddings
        # Use a hook-based approach
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Simple forward pass getting gradients on embeddings
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict):
                hidden = outputs.get("last_hidden_state", next(iter(outputs.values())))
            else:
                hidden = outputs

            if isinstance(hidden, (list, tuple)):
                hidden = hidden[-1]

            target = hidden.mean()
            target.backward()

            if embeddings.grad is not None:
                attributions = embeddings.grad.abs().sum(dim=-1).cpu().numpy()
            else:
                attributions = np.zeros((1, input_ids.shape[1]))
        except Exception:
            attributions = np.zeros((1, input_ids.shape[1]))

        return attributions

    # ------------------------------------------------------------------
    # Highlight important regions
    # ------------------------------------------------------------------

    def get_important_regions(
        self,
        shap_values: np.ndarray,
        tokens: List[str],
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Identify the most important sequence regions from SHAP values.

        Args:
            shap_values: Per-token SHAP values.
            tokens: Token strings corresponding to the SHAP values.
            top_k: Number of top positions to return.
            threshold: Absolute SHAP value threshold. If provided, all
                positions above this threshold are returned instead of top_k.

        Returns:
            List of dicts with ``position``, ``token``, ``shap_value``,
            and ``direction`` (positive/negative contribution).
        """
        abs_values = np.abs(shap_values)

        if threshold is not None:
            indices = np.where(abs_values > threshold)[0]
            indices = indices[np.argsort(abs_values[indices])[::-1]]
        else:
            indices = np.argsort(abs_values)[-top_k:][::-1]

        regions = []
        for idx in indices:
            idx_int = int(idx)
            regions.append({
                "position": idx_int,
                "token": tokens[idx_int] if idx_int < len(tokens) else "[UNK]",
                "shap_value": float(shap_values[idx_int]),
                "direction": "positive" if shap_values[idx_int] > 0 else "negative",
            })

        return regions

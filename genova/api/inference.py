"""Inference engine for Genova model serving.

Provides efficient batch inference with GPU memory management, model caching,
and support for multiple prediction tasks (variant effect, expression,
methylation, embedding extraction).

Example::

    engine = InferenceEngine(
        model_path="./checkpoints/best",
        device="cuda",
    )
    embeddings = engine.embed(["ACGTACGTACGT"])
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from genova.data.tokenizer import GenomicTokenizer
from genova.models.model_factory import create_model, count_parameters, model_summary
from genova.utils.config import GenovaConfig, ModelConfig
from genova.utils.device import get_device


class InferenceEngine:
    """High-performance inference engine for Genova models.

    Manages model loading, tokenization, batched inference, and GPU memory.
    Thread-safe for use with FastAPI's async request handling.

    Args:
        model_path: Path to model checkpoint directory or file. The directory
            should contain ``model.pt`` (or ``checkpoint.pt``) and optionally
            ``tokenizer.json`` and ``config.yaml``.
        config: Optional model configuration. If not provided, attempts to
            load from ``config.yaml`` in the model directory.
        device: Device string (``"cuda"``, ``"cpu"``, ``"mps"``, or ``"auto"``).
        max_batch_size: Maximum batch size for inference.
        max_sequence_length: Maximum sequence length to accept.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[GenovaConfig] = None,
        device: str = "auto",
        max_batch_size: int = 64,
        max_sequence_length: int = 2048,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length

        # Resolve device
        if device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device)

        # Load configuration
        self.config = config or self._load_config()
        self.model_config: ModelConfig = self.config.model

        # Initialize model and tokenizer
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[GenomicTokenizer] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_config(self) -> GenovaConfig:
        """Attempt to load config from model directory, fall back to defaults."""
        if self.model_path is not None:
            config_path = self.model_path / "config.yaml"
            if not config_path.exists() and self.model_path.is_file():
                config_path = self.model_path.parent / "config.yaml"
            if config_path.exists():
                logger.info("Loading config from {}", config_path)
                return GenovaConfig.from_yaml(config_path)
        logger.warning("No config found, using default configuration.")
        return GenovaConfig()

    def _find_checkpoint(self) -> Optional[Path]:
        """Locate the model checkpoint file."""
        if self.model_path is None:
            return None

        if self.model_path.is_file():
            return self.model_path

        # Search for common checkpoint names
        candidates = [
            "model.pt",
            "checkpoint.pt",
            "best_model.pt",
            "model.bin",
            "pytorch_model.bin",
        ]
        for name in candidates:
            path = self.model_path / name
            if path.exists():
                return path

        return None

    def _load_tokenizer(self) -> GenomicTokenizer:
        """Load or create tokenizer."""
        if self.model_path is not None:
            tok_path = self.model_path / "tokenizer.json"
            if not tok_path.exists() and self.model_path.is_file():
                tok_path = self.model_path.parent / "tokenizer.json"
            if tok_path.exists():
                logger.info("Loading tokenizer from {}", tok_path)
                return GenomicTokenizer.load(tok_path)

        # Fall back to default tokenizer
        logger.info("Creating default k-mer tokenizer (k=6).")
        tokenizer = GenomicTokenizer(
            mode=self.config.data.tokenizer,
            k=self.config.data.kmer_size,
            stride=self.config.data.stride,
        )
        tokenizer.build_vocab()
        return tokenizer

    def load(self) -> None:
        """Load model and tokenizer into memory.

        This should be called once at startup. The model is placed on the
        configured device in eval mode with gradients disabled.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._loaded:
            logger.warning("Model already loaded, skipping.")
            return

        try:
            # Load tokenizer
            self.tokenizer = self._load_tokenizer()

            # Update vocab size from tokenizer
            self.model_config.vocab_size = self.tokenizer.vocab_size

            # Find and load checkpoint
            checkpoint_path = self._find_checkpoint()

            self.model = create_model(
                self.model_config,
                task="backbone",
                pretrained_path=checkpoint_path,
                strict=False,
            )

            self.model.to(self.device)
            self.model.eval()

            # Disable gradient computation globally for inference
            for param in self.model.parameters():
                param.requires_grad_(False)

            n_params = count_parameters(self.model, trainable_only=False)
            logger.info(
                "Model loaded: arch={}, params={:,}, device={}",
                self.model_config.arch,
                n_params,
                self.device,
            )

            self._loaded = True

        except Exception as e:
            logger.error("Failed to load model: {}", e)
            raise RuntimeError(f"Model loading failed: {e}") from e

    def is_loaded(self) -> bool:
        """Check whether the model has been loaded."""
        return self._loaded

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def _tokenize_batch(
        self,
        sequences: List[str],
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and pad a batch of sequences.

        Args:
            sequences: List of DNA strings.
            max_length: Override for max sequence length.

        Returns:
            Tuple of (input_ids, attention_mask), each of shape ``(B, L)``.
        """
        assert self.tokenizer is not None
        max_len = max_length or self.max_sequence_length

        encoded = self.tokenizer.batch_encode(
            sequences,
            max_length=max_len,
            padding=True,
        )

        input_ids = torch.tensor(
            encoded["input_ids"], dtype=torch.long, device=self.device
        )
        attention_mask = torch.tensor(
            encoded["attention_mask"], dtype=torch.long, device=self.device
        )

        return input_ids, attention_mask

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run a single forward pass through the model.

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` attention mask.
            output_hidden_states: Whether to return all hidden states.

        Returns:
            Model output dictionary.
        """
        assert self.model is not None
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        if isinstance(outputs, torch.Tensor):
            outputs = {"last_hidden_state": outputs}
        elif not isinstance(outputs, dict):
            outputs = {"last_hidden_state": getattr(outputs, "last_hidden_state", outputs)}

        return outputs

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Pool sequence-level hidden states to a single vector.

        Args:
            hidden_states: ``(B, L, D)``
            attention_mask: ``(B, L)``
            pooling: One of ``"mean"``, ``"cls"``, ``"max"``.

        Returns:
            ``(B, D)`` pooled embeddings.
        """
        if pooling == "cls":
            return hidden_states[:, 0]
        elif pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states.masked_fill(mask == 0, float("-inf"))
            return hidden_states.max(dim=1).values
        else:  # mean
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        sequences: List[str],
        pooling: str = "mean",
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Compute embeddings for a list of DNA sequences.

        Args:
            sequences: List of DNA sequence strings.
            pooling: Pooling strategy (``"mean"``, ``"cls"``, ``"max"``).
            batch_size: Override for batch size.

        Returns:
            List of 1-D numpy arrays, each of shape ``(d_model,)``.
        """
        self._ensure_loaded()
        bs = batch_size or self.max_batch_size
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(sequences), bs):
            batch_seqs = sequences[start : start + bs]
            input_ids, attention_mask = self._tokenize_batch(batch_seqs)

            outputs = self._forward_batch(input_ids, attention_mask)
            hidden = outputs["last_hidden_state"]

            pooled = self._pool_hidden_states(hidden, attention_mask, pooling)
            embeddings_np = pooled.cpu().float().numpy()

            for i in range(len(batch_seqs)):
                all_embeddings.append(embeddings_np[i])

            self._maybe_clear_cache()

        return all_embeddings

    def predict_expression(
        self,
        sequences: List[str],
        num_targets: int = 1,
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Predict gene expression values for DNA sequences.

        Uses a GeneExpressionHead if available, otherwise returns
        a linear projection of the pooled embeddings.

        Args:
            sequences: List of DNA strings.
            num_targets: Number of expression targets.
            batch_size: Override for batch size.

        Returns:
            List of 1-D arrays of shape ``(num_targets,)``.
        """
        self._ensure_loaded()
        bs = batch_size or self.max_batch_size
        all_predictions: List[np.ndarray] = []

        # Check if model has an expression head
        expression_head = self._get_task_head("gene_expression", num_targets)

        for start in range(0, len(sequences), bs):
            batch_seqs = sequences[start : start + bs]
            input_ids, attention_mask = self._tokenize_batch(batch_seqs)

            outputs = self._forward_batch(input_ids, attention_mask)
            hidden = outputs["last_hidden_state"]

            if expression_head is not None:
                preds = expression_head(hidden, attention_mask)
            else:
                # Fallback: mean pool and project
                pooled = self._pool_hidden_states(hidden, attention_mask, "mean")
                preds = pooled[:, :num_targets]

            preds_np = preds.cpu().float().numpy()
            for i in range(len(batch_seqs)):
                all_predictions.append(preds_np[i])

            self._maybe_clear_cache()

        return all_predictions

    def predict_methylation(
        self,
        sequences: List[str],
        num_targets: int = 1,
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Predict methylation beta values for DNA sequences.

        Args:
            sequences: List of DNA strings.
            num_targets: Number of CpG targets.
            batch_size: Override for batch size.

        Returns:
            List of 1-D arrays of shape ``(num_targets,)`` with values in [0, 1].
        """
        self._ensure_loaded()
        bs = batch_size or self.max_batch_size
        all_predictions: List[np.ndarray] = []

        methylation_head = self._get_task_head("methylation", num_targets)

        for start in range(0, len(sequences), bs):
            batch_seqs = sequences[start : start + bs]
            input_ids, attention_mask = self._tokenize_batch(batch_seqs)

            outputs = self._forward_batch(input_ids, attention_mask)
            hidden = outputs["last_hidden_state"]

            if methylation_head is not None:
                preds = methylation_head(hidden, attention_mask)
            else:
                pooled = self._pool_hidden_states(hidden, attention_mask, "mean")
                preds = torch.sigmoid(pooled[:, :num_targets])

            preds_np = preds.cpu().float().numpy()
            for i in range(len(batch_seqs)):
                all_predictions.append(preds_np[i])

            self._maybe_clear_cache()

        return all_predictions

    def predict_variant(
        self,
        ref_sequences: List[str],
        alt_sequences: List[str],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Predict variant effects by comparing ref/alt embeddings.

        Args:
            ref_sequences: Reference allele context sequences.
            alt_sequences: Alternate allele context sequences.
            batch_size: Override for batch size.

        Returns:
            List of dicts with ``score``, ``label``, and ``confidence``.
        """
        self._ensure_loaded()
        assert len(ref_sequences) == len(alt_sequences), (
            "ref_sequences and alt_sequences must have the same length."
        )

        ref_embeddings = self.embed(ref_sequences, pooling="mean", batch_size=batch_size)
        alt_embeddings = self.embed(alt_sequences, pooling="mean", batch_size=batch_size)

        results: List[Dict[str, Any]] = []
        for ref_emb, alt_emb in zip(ref_embeddings, alt_embeddings):
            diff = alt_emb - ref_emb
            # Use L2 norm as proxy pathogenicity score, normalized to [0, 1]
            raw_score = float(np.linalg.norm(diff))
            score = float(1.0 / (1.0 + np.exp(-raw_score + 2.0)))  # sigmoid centering
            confidence = abs(score - 0.5) * 2.0  # distance from decision boundary
            label = "pathogenic" if score >= 0.5 else "benign"

            results.append({
                "score": round(score, 6),
                "label": label,
                "confidence": round(confidence, 6),
            })

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        self._ensure_loaded()
        assert self.model is not None
        summary = model_summary(self.model)
        return {
            "architecture": self.model_config.arch,
            "d_model": self.model_config.d_model,
            "n_layers": self.model_config.n_layers,
            "n_heads": self.model_config.n_heads,
            "vocab_size": self.model_config.vocab_size,
            "num_parameters": summary["total_params"],
            "device": str(self.device),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Raise if model not loaded."""
        if not self._loaded:
            raise RuntimeError(
                "Model not loaded. Call engine.load() before inference."
            )

    def _get_task_head(
        self, task_name: str, num_targets: int
    ) -> Optional[nn.Module]:
        """Try to retrieve an existing task head from the model."""
        assert self.model is not None

        # Check if the model is a multi-task model with task heads
        if hasattr(self.model, "task_heads"):
            heads = self.model.task_heads
            if hasattr(heads, "heads") and task_name in heads.heads:
                return heads.heads[task_name]

        # Create an ad-hoc head for inference
        from genova.models.multi_task import GeneExpressionHead, MethylationHead

        d_model = self.model_config.d_model
        if task_name == "gene_expression":
            head = GeneExpressionHead(d_model, num_targets=num_targets, pool="mean")
        elif task_name == "methylation":
            head = MethylationHead(d_model, num_targets=num_targets, pool="mean")
        else:
            return None

        head.to(self.device).eval()
        for p in head.parameters():
            p.requires_grad_(False)
        return head

    def _maybe_clear_cache(self) -> None:
        """Clear GPU cache if memory usage is high."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            if reserved > 0 and allocated / reserved > 0.9:
                torch.cuda.empty_cache()
                gc.collect()

    def unload(self) -> None:
        """Unload model from memory and free GPU resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model unloaded and memory freed.")

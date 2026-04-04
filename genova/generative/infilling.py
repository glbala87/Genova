"""Fill-in-the-middle (infilling) for genomic DNA sequences.

Given a prefix and suffix, generates the middle content using
bidirectional context.  Supports multiple ``[MASK]`` spans, iterative
refinement, and GC content constraints.

Example::

    from genova.generative.infilling import SequenceInfiller

    infiller = SequenceInfiller(model, tokenizer, device="cuda")
    result = infiller.infill(prefix="ACGTACGT", suffix="TGCATGCA", max_length=50)
    print(result["infilled"])

    # Multi-span masking
    result = infiller.infill_masked(
        sequence="ACGT[MASK]TGCA[MASK]GGCC",
        mask_positions=[(4, 10), (14, 20)],
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from genova.generative.autoregressive import (
    AutoregressiveGenerator,
    _sample_from_logits,
)


# ---------------------------------------------------------------------------
# GC content helpers
# ---------------------------------------------------------------------------


def _gc_content(sequence: str) -> float:
    """Compute GC fraction of a DNA sequence.

    Args:
        sequence: DNA string.

    Returns:
        GC fraction in ``[0, 1]``.
    """
    if not sequence:
        return 0.0
    seq = sequence.upper()
    gc = sum(1 for c in seq if c in ("G", "C"))
    return gc / len(seq)


def _gc_bias_logits(
    logits: Tensor,
    current_gc: float,
    target_gc: float,
    strength: float = 2.0,
    nuc_to_idx: Optional[Dict[str, int]] = None,
) -> Tensor:
    """Bias logits toward target GC content.

    Args:
        logits: ``(B, V)`` logits.
        current_gc: Current GC content of generated sequence so far.
        target_gc: Target GC content.
        strength: Bias strength.
        nuc_to_idx: Nucleotide to token index mapping.

    Returns:
        Biased logits.
    """
    if nuc_to_idx is None:
        nuc_to_idx = {"A": 1, "C": 2, "G": 3, "T": 4}

    gc_diff = target_gc - current_gc
    bias = gc_diff * strength

    biased = logits.clone()
    for nuc, idx in nuc_to_idx.items():
        if idx < biased.size(-1):
            if nuc in ("G", "C"):
                biased[:, idx] += bias
            else:
                biased[:, idx] -= bias
    return biased


# ---------------------------------------------------------------------------
# Sequence Infiller
# ---------------------------------------------------------------------------


class SequenceInfiller:
    """Fill-in-the-middle generator for genomic DNA sequences.

    Uses a trained Genova model to infill masked or missing regions of
    DNA sequences.  Supports:

    - Single-span infilling given a prefix and suffix.
    - Multi-span infilling with explicit ``[MASK]`` positions.
    - Iterative refinement (generate, then re-score with both contexts).
    - GC content constraint during generation.

    Args:
        model: Trained sequence model (must return logits or hidden states).
        tokenizer: Tokenizer with ``encode``/``decode`` methods.
        device: Inference device.
        lm_head: Optional LM head for models without built-in logits.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: Union[str, torch.device] = "cpu",
        lm_head: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.lm_head = lm_head

        self.model.to(self.device).eval()
        if self.lm_head is not None:
            self.lm_head.to(self.device).eval()

        self._base = AutoregressiveGenerator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            lm_head=lm_head,
        )

        # Nucleotide-to-index mapping (best effort)
        self._nuc_to_idx: Dict[str, int] = {"A": 1, "C": 2, "G": 3, "T": 4}
        if hasattr(tokenizer, "vocab"):
            vocab = tokenizer.vocab
            for nuc in ("A", "C", "G", "T"):
                if nuc in vocab:
                    self._nuc_to_idx[nuc] = vocab[nuc]

        self.bos_id: int = getattr(tokenizer, "bos_token_id", 1)
        self.eos_id: int = getattr(tokenizer, "eos_token_id", 2)
        self.pad_id: int = getattr(tokenizer, "pad_token_id", 0)
        self.mask_id: int = getattr(tokenizer, "mask_token_id", 3)

        logger.info("SequenceInfiller ready on {}", self.device)

    # ------------------------------------------------------------------
    # Logit helpers
    # ------------------------------------------------------------------

    def _get_logits(self, input_ids: Tensor) -> Tensor:
        """Run the model and return ``(B, L, V)`` full logits.

        Args:
            input_ids: ``(B, L)`` token ids.

        Returns:
            ``(B, L, V)`` logits.
        """
        out = self.model(input_ids)

        if isinstance(out, dict):
            if "logits" in out:
                return out["logits"]
            hidden = out.get("last_hidden_state")
            if hidden is None:
                raise ValueError(
                    "Model must return 'logits' or 'last_hidden_state'"
                )
        elif isinstance(out, Tensor):
            hidden = out
        elif isinstance(out, (tuple, list)):
            hidden = out[0]
        else:
            raise TypeError(f"Unsupported model output type: {type(out)}")

        if self.lm_head is not None:
            return self.lm_head(hidden)

        raise ValueError(
            "Model does not return 'logits' and no lm_head was provided"
        )

    def _get_next_logits(self, input_ids: Tensor) -> Tensor:
        """Return ``(B, V)`` logits for the last position.

        Args:
            input_ids: ``(B, L)`` token ids.

        Returns:
            ``(B, V)`` logits for the next token.
        """
        return self._base._get_next_logits(input_ids)

    # ------------------------------------------------------------------
    # Encode helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        """Encode a DNA string to token ids without special tokens.

        Args:
            text: DNA string.

        Returns:
            List of token ids.
        """
        if hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        elif callable(self.tokenizer):
            ids = self.tokenizer(text)
        else:
            raise TypeError("Tokenizer must have encode method or be callable")
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return ids

    def _decode(self, ids: List[int]) -> str:
        """Decode token ids to a DNA string.

        Args:
            ids: List of token ids.

        Returns:
            DNA string.
        """
        return self._base._decode_batch(
            torch.tensor([ids], dtype=torch.long, device=self.device)
        )[0]

    # ------------------------------------------------------------------
    # Single-span infilling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infill(
        self,
        prefix: str,
        suffix: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        target_gc: Optional[float] = None,
        gc_strength: float = 2.0,
        num_refinements: int = 0,
    ) -> Dict[str, Any]:
        """Generate infill content between a prefix and suffix.

        The approach encodes the prefix, generates autoregressively up
        to *max_length* tokens, then scores the result against the
        suffix context.  Optionally refines the result by re-generating
        with the full (prefix + infill + suffix) context.

        Args:
            prefix: DNA string that precedes the gap.
            suffix: DNA string that follows the gap.
            max_length: Maximum number of tokens to generate for the
                infill region.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            target_gc: If set, bias generation toward this GC content.
            gc_strength: Strength of the GC bias.
            num_refinements: Number of iterative refinement passes.
                0 means no refinement (single-pass generation).

        Returns:
            Dict with keys:
                - ``"infilled"``: The infill DNA string.
                - ``"full_sequence"``: prefix + infill + suffix.
                - ``"gc_content"``: GC content of the infill region.
                - ``"log_prob"``: Total log probability of the infill.
        """
        prefix_ids = self._encode(prefix)
        suffix_ids = self._encode(suffix)

        # Start with prefix as prompt
        prompt = [self.bos_id] + prefix_ids
        generated_infill: List[int] = []
        total_log_prob = 0.0
        current_infill_str = ""

        for step in range(max_length):
            current_input = prompt + generated_infill
            input_tensor = torch.tensor(
                [current_input], dtype=torch.long, device=self.device
            )
            logits = self._get_next_logits(input_tensor)  # (1, V)

            # Apply GC bias
            if target_gc is not None:
                logits = _gc_bias_logits(
                    logits,
                    _gc_content(current_infill_str),
                    target_gc,
                    strength=gc_strength,
                    nuc_to_idx=self._nuc_to_idx,
                )

            # Incorporate suffix context: run suffix-conditioned scoring
            # by computing logits from a reversed suffix perspective
            if suffix_ids:
                suffix_context = current_input + suffix_ids
                suffix_tensor = torch.tensor(
                    [suffix_context], dtype=torch.long, device=self.device
                )
                try:
                    full_logits = self._get_logits(suffix_tensor)
                    # Position of the token to predict
                    pos = len(current_input) - 1
                    if pos < full_logits.size(1):
                        suffix_logits = full_logits[:, pos, :]  # (1, V)
                        # Average forward and bidirectional logits
                        logits = 0.5 * (logits + suffix_logits)
                except Exception:
                    pass  # Fall back to forward-only

            # Sample
            next_token = _sample_from_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )  # (1, 1)

            token_id = int(next_token[0, 0].item())

            if token_id == self.eos_id:
                break

            # Log prob
            step_lp = F.log_softmax(logits / max(temperature, 1e-8), dim=-1)
            token_lp = float(step_lp[0, token_id].item())
            total_log_prob += token_lp

            generated_infill.append(token_id)

            # Track infill string for GC
            idx_to_nuc = {v: k for k, v in self._nuc_to_idx.items()}
            nuc = idx_to_nuc.get(token_id, "")
            current_infill_str += nuc

        # Decode infill
        infill_str = self._decode(generated_infill)

        # Iterative refinement
        for _ref in range(num_refinements):
            infill_str, total_log_prob, generated_infill = self._refine_infill(
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                current_infill_ids=generated_infill,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                target_gc=target_gc,
                gc_strength=gc_strength,
            )

        full_seq = prefix + infill_str + suffix

        return {
            "infilled": infill_str,
            "full_sequence": full_seq,
            "gc_content": _gc_content(infill_str),
            "log_prob": total_log_prob,
        }

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def _refine_infill(
        self,
        prefix_ids: List[int],
        suffix_ids: List[int],
        current_infill_ids: List[int],
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        target_gc: Optional[float],
        gc_strength: float,
    ) -> Tuple[str, float, List[int]]:
        """Re-generate the infill using the full context for scoring.

        Args:
            prefix_ids: Token ids for the prefix.
            suffix_ids: Token ids for the suffix.
            current_infill_ids: Current infill token ids.
            max_length: Maximum infill length.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling.
            target_gc: Target GC content.
            gc_strength: GC bias strength.

        Returns:
            Tuple of (infill_string, log_prob, infill_token_ids).
        """
        prompt = [self.bos_id] + prefix_ids
        new_infill: List[int] = []
        total_lp = 0.0
        current_infill_str = ""

        for step in range(max_length):
            current_input = prompt + new_infill
            full_context = current_input + suffix_ids

            full_tensor = torch.tensor(
                [full_context], dtype=torch.long, device=self.device
            )

            try:
                full_logits = self._get_logits(full_tensor)
                pos = len(current_input) - 1
                if pos < full_logits.size(1):
                    logits = full_logits[:, pos, :]
                else:
                    logits = self._get_next_logits(
                        torch.tensor(
                            [current_input],
                            dtype=torch.long,
                            device=self.device,
                        )
                    )
            except Exception:
                logits = self._get_next_logits(
                    torch.tensor(
                        [current_input],
                        dtype=torch.long,
                        device=self.device,
                    )
                )

            if target_gc is not None:
                logits = _gc_bias_logits(
                    logits,
                    _gc_content(current_infill_str),
                    target_gc,
                    strength=gc_strength,
                    nuc_to_idx=self._nuc_to_idx,
                )

            next_token = _sample_from_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            token_id = int(next_token[0, 0].item())

            if token_id == self.eos_id:
                break

            step_lp = F.log_softmax(logits / max(temperature, 1e-8), dim=-1)
            total_lp += float(step_lp[0, token_id].item())

            new_infill.append(token_id)

            idx_to_nuc = {v: k for k, v in self._nuc_to_idx.items()}
            current_infill_str += idx_to_nuc.get(token_id, "")

        infill_str = self._decode(new_infill)
        return infill_str, total_lp, new_infill

    # ------------------------------------------------------------------
    # Multi-span masked infilling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infill_masked(
        self,
        sequence: str,
        mask_positions: List[Tuple[int, int]],
        max_length_per_span: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        target_gc: Optional[float] = None,
        gc_strength: float = 2.0,
        num_refinements: int = 0,
    ) -> Dict[str, Any]:
        """Infill multiple masked spans in a sequence.

        Each span defined by ``(start, end)`` in *mask_positions* is
        treated as a gap.  Spans are filled left-to-right, so earlier
        spans provide context for later ones.

        Args:
            sequence: DNA string with regions to be infilled.
            mask_positions: List of ``(start, end)`` tuples (0-indexed,
                half-open) marking the spans to infill.
            max_length_per_span: Maximum tokens to generate per span.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling.
            target_gc: Target GC content for infilled regions.
            gc_strength: GC bias strength.
            num_refinements: Number of iterative refinement rounds.

        Returns:
            Dict with keys:
                - ``"infilled_sequence"``: Full sequence with spans filled.
                - ``"spans"``: List of dicts with per-span infill info.
                - ``"gc_content"``: Overall GC content of the result.
        """
        # Sort spans by start position
        sorted_spans = sorted(mask_positions, key=lambda x: x[0])

        # Validate non-overlapping
        for i in range(1, len(sorted_spans)):
            if sorted_spans[i][0] < sorted_spans[i - 1][1]:
                raise ValueError(
                    f"Overlapping mask spans: {sorted_spans[i - 1]} and "
                    f"{sorted_spans[i]}"
                )

        result_seq = sequence
        offset = 0  # track cumulative length changes
        span_results: List[Dict[str, Any]] = []

        for start, end in sorted_spans:
            adj_start = start + offset
            adj_end = end + offset

            prefix = result_seq[:adj_start]
            suffix = result_seq[adj_end:]

            span_result = self.infill(
                prefix=prefix,
                suffix=suffix,
                max_length=max_length_per_span,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                target_gc=target_gc,
                gc_strength=gc_strength,
                num_refinements=num_refinements,
            )

            infill_str = span_result["infilled"]
            result_seq = prefix + infill_str + suffix

            # Update offset for subsequent spans
            original_span_len = end - start
            offset += len(infill_str) - original_span_len

            span_results.append({
                "original_span": (start, end),
                "infilled": infill_str,
                "gc_content": span_result["gc_content"],
                "log_prob": span_result["log_prob"],
            })

        return {
            "infilled_sequence": result_seq,
            "spans": span_results,
            "gc_content": _gc_content(result_seq),
        }

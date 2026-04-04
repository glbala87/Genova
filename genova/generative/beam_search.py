"""Beam search decoding for genomic sequence generation.

Implements standard beam search with length normalization, coverage
penalty, n-gram blocking, constrained decoding, and early stopping.

Example::

    from genova.generative.beam_search import BeamSearchGenerator

    generator = BeamSearchGenerator(model, tokenizer, device="cuda")
    results = generator.generate(prompt_ids=prompt, beam_width=5, max_length=256)
    for r in results:
        print(r.sequence, r.score)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from genova.generative.autoregressive import AutoregressiveGenerator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BeamResult:
    """Result from a single beam in beam search.

    Attributes:
        sequence: Decoded DNA string.
        score: Final beam score (length-normalized log probability).
        log_probs: Per-token log probabilities for the generated sequence.
        token_ids: Raw token ID list for the generated sequence.
    """

    sequence: str
    score: float
    log_probs: List[float] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)


@dataclass
class _Beam:
    """Internal bookkeeping for a single active beam."""

    token_ids: List[int]
    log_prob: float
    log_probs: List[float]
    finished: bool = False


# ---------------------------------------------------------------------------
# Beam Search Generator
# ---------------------------------------------------------------------------


class BeamSearchGenerator:
    """Beam search decoder for a trained Genova language model.

    Supports configurable beam width, length normalization, coverage
    penalty for repetition avoidance, n-gram blocking, constrained
    token forcing, and early stopping.

    Args:
        model: Trained sequence model (returns logits or hidden states).
        tokenizer: Tokenizer with ``encode``/``decode`` methods and
            ``bos_token_id``, ``eos_token_id``, ``pad_token_id``.
        device: Inference device.
        lm_head: Optional linear layer mapping hidden states to vocab
            logits when the model does not return ``"logits"`` directly.
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

        self.bos_id: int = getattr(tokenizer, "bos_token_id", 1)
        self.eos_id: int = getattr(tokenizer, "eos_token_id", 2)
        self.pad_id: int = getattr(tokenizer, "pad_token_id", 0)

        # Build a helper generator for decoding utilities
        self._base = AutoregressiveGenerator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            lm_head=lm_head,
        )

        logger.info(
            "BeamSearchGenerator ready on {} (bos={}, eos={}, pad={})",
            self.device,
            self.bos_id,
            self.eos_id,
            self.pad_id,
        )

    # ------------------------------------------------------------------
    # Logit extraction
    # ------------------------------------------------------------------

    def _get_next_logits(self, input_ids: Tensor) -> Tensor:
        """Run the model and return ``(B, V)`` logits for the last position.

        Args:
            input_ids: ``(B, L)`` token ids.

        Returns:
            ``(B, V)`` logits.
        """
        return self._base._get_next_logits(input_ids)

    # ------------------------------------------------------------------
    # N-gram blocking
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ngrams(token_ids: List[int], n: int) -> Set[Tuple[int, ...]]:
        """Extract the set of n-grams from a token id list.

        Args:
            token_ids: List of token ids.
            n: N-gram size.

        Returns:
            Set of n-gram tuples.
        """
        ngrams: Set[Tuple[int, ...]] = set()
        for i in range(len(token_ids) - n + 1):
            ngrams.add(tuple(token_ids[i : i + n]))
        return ngrams

    @staticmethod
    def _block_ngram_tokens(
        token_ids: List[int],
        n: int,
        logits: Tensor,
    ) -> Tensor:
        """Set logits to ``-inf`` for tokens that would create a repeated n-gram.

        Args:
            token_ids: Current beam token ids.
            n: N-gram size to block.
            logits: ``(V,)`` logits for the next position.

        Returns:
            Modified logits with blocked positions set to ``-inf``.
        """
        if len(token_ids) < n - 1:
            return logits
        # The last (n-1) tokens form the prefix of the potential n-gram
        prefix = tuple(token_ids[-(n - 1) :])
        existing = BeamSearchGenerator._get_ngrams(token_ids, n)

        blocked = logits.clone()
        for token_id in range(logits.size(-1)):
            candidate = prefix + (token_id,)
            if candidate in existing:
                blocked[token_id] = float("-inf")
        return blocked

    # ------------------------------------------------------------------
    # Coverage penalty
    # ------------------------------------------------------------------

    @staticmethod
    def _coverage_penalty(
        token_ids: List[int],
        beta: float = 0.0,
    ) -> float:
        """Compute a coverage penalty based on token repetition.

        Penalizes beams that repeat tokens frequently.  The penalty is
        ``beta * sum(log(min(count, 1)))`` over unique tokens, which
        discourages revisiting the same token.

        Args:
            token_ids: Current beam token ids.
            beta: Coverage penalty weight.  0.0 disables the penalty.

        Returns:
            Penalty value (non-positive).
        """
        if beta <= 0.0 or not token_ids:
            return 0.0
        from collections import Counter
        import math

        counts = Counter(token_ids)
        penalty = 0.0
        for count in counts.values():
            if count > 1:
                penalty += math.log(count)
        return -beta * penalty

    # ------------------------------------------------------------------
    # Length normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _length_normalize(
        log_prob: float,
        length: int,
        alpha: float = 0.6,
    ) -> float:
        """Apply length normalization to a log probability score.

        Uses the formula from Wu et al. (2016):
        ``score = log_prob / ((5 + length) / 6) ^ alpha``

        Args:
            log_prob: Cumulative log probability.
            length: Sequence length.
            alpha: Length normalization parameter.  0 disables
                normalization; higher values penalize shorter sequences
                less.

        Returns:
            Length-normalized score.
        """
        if alpha <= 0.0:
            return log_prob
        lp = ((5.0 + length) / 6.0) ** alpha
        return log_prob / lp

    # ------------------------------------------------------------------
    # Main generate method
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Optional[Tensor] = None,
        beam_width: int = 5,
        max_length: int = 512,
        alpha: float = 0.6,
        coverage_beta: float = 0.0,
        no_repeat_ngram_size: int = 0,
        constrained_positions: Optional[Dict[int, int]] = None,
        top_k: int = 0,
        early_stopping: bool = True,
    ) -> List[BeamResult]:
        """Generate sequences using beam search.

        Args:
            prompt_ids: Optional ``(1, L)`` or ``(L,)`` prompt tensor.
                If ``None``, generation starts from ``[BOS]``.
            beam_width: Number of beams to maintain.  Must be in ``[2, 20]``.
            max_length: Maximum output sequence length in tokens.
            alpha: Length normalization parameter (Wu et al., 2016).
                Set to 0 to disable.
            coverage_beta: Coverage penalty weight for repetition
                avoidance.  0 disables.
            no_repeat_ngram_size: If > 0, block repeated n-grams of this
                size.
            constrained_positions: Dict mapping absolute token positions
                (0-indexed, relative to the start of the *generated*
                part, not the prompt) to forced token ids.
            top_k: If > 0, restrict expansion to the top-k tokens at
                each step (reduces computation).
            early_stopping: If ``True``, stop when all beams have
                produced ``[EOS]``.

        Returns:
            List of :class:`BeamResult` sorted by descending score.
            The list length equals *beam_width* (or fewer if
            ``early_stopping`` terminates some beams early).

        Raises:
            ValueError: If *beam_width* is outside ``[2, 20]``.
        """
        if not (2 <= beam_width <= 20):
            raise ValueError(
                f"beam_width must be between 2 and 20, got {beam_width}"
            )

        constrained_positions = constrained_positions or {}

        # Initialize prompt
        if prompt_ids is not None:
            if prompt_ids.dim() == 1:
                prompt_ids = prompt_ids.unsqueeze(0)
            prompt_list = prompt_ids[0].tolist()
        else:
            prompt_list = [self.bos_id]

        prompt_len = len(prompt_list)

        # Initialize beams
        beams: List[_Beam] = [
            _Beam(
                token_ids=list(prompt_list),
                log_prob=0.0,
                log_probs=[],
                finished=False,
            )
        ]

        finished_beams: List[_Beam] = []

        for step in range(max_length - prompt_len):
            if not beams:
                break

            all_candidates: List[_Beam] = []
            gen_pos = step  # generation position (0-indexed from prompt end)

            for beam in beams:
                if beam.finished:
                    finished_beams.append(beam)
                    continue

                # Check constrained position
                if gen_pos in constrained_positions:
                    forced_id = constrained_positions[gen_pos]
                    # Compute the log prob for the forced token
                    input_tensor = torch.tensor(
                        [beam.token_ids], dtype=torch.long, device=self.device
                    )
                    logits = self._get_next_logits(input_tensor)[0]
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_lp = float(log_probs[forced_id].item())

                    new_beam = _Beam(
                        token_ids=beam.token_ids + [forced_id],
                        log_prob=beam.log_prob + token_lp,
                        log_probs=beam.log_probs + [token_lp],
                        finished=(forced_id == self.eos_id),
                    )
                    all_candidates.append(new_beam)
                    continue

                # Get logits
                input_tensor = torch.tensor(
                    [beam.token_ids], dtype=torch.long, device=self.device
                )
                logits = self._get_next_logits(input_tensor)[0]  # (V,)

                # Apply n-gram blocking
                if no_repeat_ngram_size > 0:
                    logits = self._block_ngram_tokens(
                        beam.token_ids, no_repeat_ngram_size, logits
                    )

                log_probs = F.log_softmax(logits, dim=-1)

                # Select top candidates
                k = min(
                    beam_width * 2,
                    logits.size(-1),
                    top_k if top_k > 0 else logits.size(-1),
                )
                top_lps, top_ids = torch.topk(log_probs, k)

                for i in range(k):
                    token_id = int(top_ids[i].item())
                    token_lp = float(top_lps[i].item())

                    new_beam = _Beam(
                        token_ids=beam.token_ids + [token_id],
                        log_prob=beam.log_prob + token_lp,
                        log_probs=beam.log_probs + [token_lp],
                        finished=(token_id == self.eos_id),
                    )
                    all_candidates.append(new_beam)

            if not all_candidates:
                break

            # Score candidates with length normalization and coverage penalty
            def _score(b: _Beam) -> float:
                gen_length = len(b.token_ids) - prompt_len
                score = self._length_normalize(b.log_prob, max(gen_length, 1), alpha)
                score += self._coverage_penalty(b.token_ids[prompt_len:], coverage_beta)
                return score

            all_candidates.sort(key=_score, reverse=True)

            # Keep top beam_width beams
            beams = []
            for candidate in all_candidates:
                if candidate.finished:
                    finished_beams.append(candidate)
                else:
                    beams.append(candidate)
                if len(beams) >= beam_width:
                    break

            # Early stopping: if we have enough finished beams
            if early_stopping and len(finished_beams) >= beam_width:
                break

        # Add remaining active beams to finished
        finished_beams.extend(beams)

        if not finished_beams:
            logger.warning("Beam search produced no results.")
            return []

        # Score and sort all finished beams
        scored: List[Tuple[float, _Beam]] = []
        for beam in finished_beams:
            gen_length = len(beam.token_ids) - prompt_len
            score = self._length_normalize(beam.log_prob, max(gen_length, 1), alpha)
            score += self._coverage_penalty(beam.token_ids[prompt_len:], coverage_beta)
            scored.append((score, beam))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Build results (top beam_width)
        results: List[BeamResult] = []
        for score, beam in scored[:beam_width]:
            seq = self._base._decode_batch(
                torch.tensor([beam.token_ids], dtype=torch.long, device=self.device)
            )[0]
            results.append(
                BeamResult(
                    sequence=seq,
                    score=score,
                    log_probs=beam.log_probs,
                    token_ids=beam.token_ids,
                )
            )

        logger.info(
            "Beam search complete: {} results, best score={:.4f}",
            len(results),
            results[0].score if results else float("-inf"),
        )
        return results

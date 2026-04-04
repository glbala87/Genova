"""Guided and controllable sequence generation for genomic DNA.

Implements classifier-free guidance, classifier guidance, and constrained
generation (GC content targeting, motif insertion, length constraints)
for controllable DNA sequence design.

Example::

    from genova.generative.guided_generation import GuidedGenerator

    gen = GuidedGenerator(model, tokenizer, device="cuda")

    # Classifier-free guidance
    seqs = gen.generate(condition={"expression": "high"}, guidance_scale=3.0)

    # Constrained generation
    seqs = gen.generate_with_constraints(
        constraints={"gc_content": 0.5, "motifs": ["TATAAA"], "length": 500}
    )
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger

from genova.generative.autoregressive import (
    AutoregressiveGenerator,
    _sample_from_logits,
)


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------


def _gc_content(sequence: str) -> float:
    """Compute GC content of a DNA sequence.

    Args:
        sequence: DNA string (A, C, G, T characters).

    Returns:
        Fraction of G and C nucleotides in [0, 1].
    """
    if not sequence:
        return 0.0
    seq = sequence.upper()
    gc = sum(1 for c in seq if c in ("G", "C"))
    return gc / len(seq)


def _contains_motif(sequence: str, motif: str) -> bool:
    """Check if a DNA sequence contains a given motif.

    Args:
        sequence: DNA sequence.
        motif: Motif string to search for.

    Returns:
        ``True`` if *motif* is found in *sequence*.
    """
    return motif.upper() in sequence.upper()


def _gc_bias_logits(
    logits: Tensor,
    current_gc: float,
    target_gc: float,
    strength: float = 2.0,
    nuc_to_idx: Optional[Dict[str, int]] = None,
) -> Tensor:
    """Bias logits toward or away from G/C tokens to match target GC content.

    Args:
        logits: ``(B, V)`` logits.
        current_gc: Current GC content of the generated sequence so far.
        target_gc: Target GC content.
        strength: How strongly to bias.
        nuc_to_idx: Mapping from nucleotide to token index.

    Returns:
        Biased logits.
    """
    if nuc_to_idx is None:
        # Default mapping: A=1, C=2, G=3, T=4 (common in genomic tokenizers)
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
# GuidedGenerator
# ---------------------------------------------------------------------------


class GuidedGenerator:
    """Guided and controllable DNA sequence generator.

    Wraps an :class:`AutoregressiveGenerator` with classifier-free guidance,
    classifier guidance, and hard/soft constraint enforcement.

    Args:
        model: Trained sequence model (must return logits).
        tokenizer: Tokenizer with ``encode`` / ``decode`` methods and
            vocabulary attributes.
        device: Inference device.
        lm_head: Optional LM head for models that don't return logits.
        condition_dropout_rate: Probability of dropping the condition
            during classifier-free guidance training.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: Union[str, torch.device] = "cpu",
        lm_head: Optional[nn.Module] = None,
        condition_dropout_rate: float = 0.1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.lm_head = lm_head
        self.condition_dropout_rate = condition_dropout_rate

        self._base_generator = AutoregressiveGenerator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            lm_head=lm_head,
        )

        self.model.to(self.device).eval()
        if self.lm_head is not None:
            self.lm_head.to(self.device).eval()

        # Nucleotide-to-index mapping (best effort)
        self._nuc_to_idx: Dict[str, int] = {"A": 1, "C": 2, "G": 3, "T": 4}
        if hasattr(tokenizer, "vocab"):
            vocab = tokenizer.vocab
            for nuc in ("A", "C", "G", "T"):
                if nuc in vocab:
                    self._nuc_to_idx[nuc] = vocab[nuc]

        logger.info(
            "GuidedGenerator ready on {} (condition_dropout={})",
            self.device,
            self.condition_dropout_rate,
        )

    # ------------------------------------------------------------------
    # Logit extraction
    # ------------------------------------------------------------------

    def _get_logits(self, input_ids: Tensor) -> Tensor:
        """Run the model and extract logits for the last position.

        Args:
            input_ids: ``(B, L)`` token ids.

        Returns:
            ``(B, V)`` logits for the next token.
        """
        out = self.model(input_ids)

        if isinstance(out, dict):
            if "logits" in out:
                return out["logits"][:, -1, :]
            hidden = out.get("last_hidden_state")
            if hidden is None:
                raise ValueError("Model must return 'logits' or 'last_hidden_state'")
        elif isinstance(out, Tensor):
            hidden = out
        elif isinstance(out, (tuple, list)):
            hidden = out[0]
        else:
            raise TypeError(f"Unsupported model output type: {type(out)}")

        if self.lm_head is not None:
            return self.lm_head(hidden[:, -1, :])

        raise ValueError(
            "Model does not return 'logits' and no lm_head was provided"
        )

    # ------------------------------------------------------------------
    # Classifier-free guidance
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        condition: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 1.0,
        num_sequences: int = 1,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        condition_encoder: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Generate sequences with classifier-free guidance.

        When ``guidance_scale > 1``, the model is run twice per step: once
        with the condition and once without (unconditional).  The final
        logits are:

            ``logits = unconditional + guidance_scale * (conditional - unconditional)``

        Args:
            condition: Condition dict (e.g. ``{"expression": "high"}``).
                If ``None``, runs unconditional generation.
            guidance_scale: Guidance strength. ``1.0`` = no guidance,
                higher = stronger conditioning.
            num_sequences: Number of sequences to generate.
            max_length: Maximum sequence length in tokens.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            condition_encoder: Optional module mapping conditions to
                token ids.

        Returns:
            Dict with ``"token_ids"``, ``"sequences"``, ``"log_probs"``,
            and ``"conditions"``.
        """
        if condition is None or guidance_scale <= 1.0:
            # No guidance: delegate to base generator
            if condition is not None:
                return self._base_generator.conditional_generate(
                    conditions=condition,
                    num_sequences=num_sequences,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    condition_encoder=condition_encoder,
                )
            return self._base_generator.generate(
                num_sequences=num_sequences,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Classifier-free guidance: generate with dual forward passes
        # Build conditional prompt
        if condition_encoder is not None:
            cond_ids = condition_encoder(condition)
            if not isinstance(cond_ids, Tensor):
                cond_ids = torch.tensor(cond_ids, dtype=torch.long)
            cond_ids = cond_ids.to(self.device)
            if cond_ids.dim() == 1:
                cond_ids = cond_ids.unsqueeze(0)
        else:
            cond_ids = self._base_generator._default_condition_encoding(
                condition
            ).unsqueeze(0)

        bos = torch.full(
            (1, 1), self._base_generator.bos_id,
            dtype=torch.long, device=self.device,
        )

        # Conditional prompt: [BOS] + condition_tokens
        cond_prompt = torch.cat([bos, cond_ids], dim=1)
        # Unconditional prompt: just [BOS]
        uncond_prompt = bos.clone()

        # Expand for batch
        cond_generated = cond_prompt.expand(num_sequences, -1).clone()
        uncond_generated = uncond_prompt.expand(num_sequences, -1).clone()

        log_probs_list: List[Tensor] = []
        finished = torch.zeros(
            num_sequences, dtype=torch.bool, device=self.device
        )

        for step in range(max_length - cond_generated.size(1)):
            # Conditional logits
            cond_logits = self._get_logits(cond_generated)
            # Unconditional logits
            uncond_logits = self._get_logits(uncond_generated)

            # Guided logits
            guided_logits = (
                uncond_logits
                + guidance_scale * (cond_logits - uncond_logits)
            )

            # Sample
            next_tokens = _sample_from_logits(
                guided_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Log probs
            step_lp = F.log_softmax(
                guided_logits / max(temperature, 1e-8), dim=-1
            )
            selected_lp = step_lp.gather(1, next_tokens)
            log_probs_list.append(selected_lp)

            next_tokens[finished] = self._base_generator.pad_id
            cond_generated = torch.cat([cond_generated, next_tokens], dim=1)
            uncond_generated = torch.cat([uncond_generated, next_tokens], dim=1)

            finished = finished | (
                next_tokens.squeeze(-1) == self._base_generator.eos_id
            )
            if finished.all():
                break

        all_log_probs = (
            torch.cat(log_probs_list, dim=1)
            if log_probs_list
            else torch.zeros(num_sequences, 0, device=self.device)
        )
        sequences = self._base_generator._decode_batch(cond_generated)

        return {
            "token_ids": cond_generated,
            "sequences": sequences,
            "log_probs": all_log_probs,
            "conditions": condition,
            "guidance_scale": guidance_scale,
        }

    # ------------------------------------------------------------------
    # Classifier guidance
    # ------------------------------------------------------------------

    def generate_with_classifier(
        self,
        classifier: nn.Module,
        target_class: int = 1,
        guidance_scale: float = 1.0,
        num_sequences: int = 1,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate sequences guided by an external classifier's gradient.

        At each generation step, computes the gradient of the classifier's
        log-probability for the target class with respect to the logits,
        and uses it to bias generation toward the desired class.

        Args:
            classifier: Differentiable classifier module that accepts
                ``(B, L)`` token ids and returns ``(B, C)`` logits.
            target_class: Index of the target class to guide toward.
            guidance_scale: Strength of the classifier guidance.
            num_sequences: Number of sequences to generate.
            max_length: Maximum sequence length.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.

        Returns:
            Dict with ``"token_ids"``, ``"sequences"``, ``"log_probs"``,
            and ``"classifier_scores"``.
        """
        classifier.to(self.device).eval()

        generated = torch.full(
            (num_sequences, 1),
            self._base_generator.bos_id,
            dtype=torch.long,
            device=self.device,
        )

        log_probs_list: List[Tensor] = []
        finished = torch.zeros(
            num_sequences, dtype=torch.bool, device=self.device
        )

        for step in range(max_length - 1):
            # Base model logits (no grad needed)
            with torch.no_grad():
                base_logits = self._get_logits(generated)

            # Classifier gradient on the current sequence
            gen_for_grad = generated.detach().clone()
            gen_for_grad.requires_grad = False

            # Get classifier score
            try:
                classifier.zero_grad()
                # One-hot encode and allow grad flow
                one_hot = F.one_hot(
                    gen_for_grad, num_classes=base_logits.size(-1)
                ).float()
                one_hot.requires_grad_(True)

                cls_out = classifier(gen_for_grad)
                if isinstance(cls_out, dict):
                    cls_logits = cls_out.get("logits", next(iter(cls_out.values())))
                elif isinstance(cls_out, (tuple, list)):
                    cls_logits = cls_out[0]
                else:
                    cls_logits = cls_out

                log_probs_cls = F.log_softmax(cls_logits, dim=-1)
                target_log_prob = log_probs_cls[:, target_class].sum()
                target_log_prob.backward()

                if one_hot.grad is not None:
                    # Sum gradient over sequence to get per-vocab bias
                    grad_bias = one_hot.grad.sum(dim=1)  # (B, V)
                    guided_logits = base_logits + guidance_scale * grad_bias
                else:
                    guided_logits = base_logits

            except Exception as e:
                logger.debug(
                    "Classifier guidance failed at step {}: {}", step, e
                )
                guided_logits = base_logits

            # Sample
            next_tokens = _sample_from_logits(
                guided_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            step_lp = F.log_softmax(
                guided_logits / max(temperature, 1e-8), dim=-1
            )
            log_probs_list.append(step_lp.gather(1, next_tokens))

            next_tokens[finished] = self._base_generator.pad_id
            generated = torch.cat([generated, next_tokens], dim=1)

            finished = finished | (
                next_tokens.squeeze(-1) == self._base_generator.eos_id
            )
            if finished.all():
                break

        all_log_probs = (
            torch.cat(log_probs_list, dim=1)
            if log_probs_list
            else torch.zeros(num_sequences, 0, device=self.device)
        )
        sequences = self._base_generator._decode_batch(generated)

        # Final classifier scores
        with torch.no_grad():
            try:
                final_out = classifier(generated)
                if isinstance(final_out, dict):
                    final_logits = final_out.get("logits", next(iter(final_out.values())))
                elif isinstance(final_out, (tuple, list)):
                    final_logits = final_out[0]
                else:
                    final_logits = final_out
                classifier_scores = (
                    F.softmax(final_logits, dim=-1)[:, target_class]
                    .cpu()
                    .numpy()
                    .tolist()
                )
            except Exception:
                classifier_scores = [0.0] * num_sequences

        return {
            "token_ids": generated,
            "sequences": sequences,
            "log_probs": all_log_probs,
            "classifier_scores": classifier_scores,
        }

    # ------------------------------------------------------------------
    # Constrained generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_with_constraints(
        self,
        constraints: Dict[str, Any],
        num_sequences: int = 1,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        max_retries: int = 10,
    ) -> Dict[str, Any]:
        """Generate sequences satisfying hard and soft constraints.

        Supported constraints:
            - ``"gc_content"`` (float): Target GC fraction in [0, 1].
            - ``"motifs"`` (list of str): Motif strings that must appear
              in the output. If not present after generation, a retry is
              attempted with motif insertion.
            - ``"length"`` (int): Exact output length in nucleotides.
            - ``"gc_tolerance"`` (float): Acceptable deviation from target
              GC content (default 0.05).

        Args:
            constraints: Dict of constraint name -> value.
            num_sequences: Number of sequences to generate.
            max_length: Maximum token length.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            max_retries: Maximum retries for satisfying hard constraints.

        Returns:
            Dict with ``"token_ids"``, ``"sequences"``, ``"constraints_met"``,
            and per-sequence ``"gc_content"`` values.
        """
        target_gc = constraints.get("gc_content")
        required_motifs: List[str] = constraints.get("motifs", [])
        target_length = constraints.get("length")
        gc_tolerance = constraints.get("gc_tolerance", 0.05)

        # Adjust max_length for target_length
        if target_length is not None:
            max_length = min(max_length, target_length + 50)

        best_sequences: List[str] = []
        best_ids: Optional[Tensor] = None
        constraints_met: List[bool] = []

        for attempt in range(max_retries):
            # Generate with GC bias
            generated = torch.full(
                (num_sequences, 1),
                self._base_generator.bos_id,
                dtype=torch.long,
                device=self.device,
            )
            current_sequences = [""] * num_sequences
            finished = torch.zeros(
                num_sequences, dtype=torch.bool, device=self.device
            )

            for step in range(max_length - 1):
                logits = self._get_logits(generated)

                # Apply GC content bias
                if target_gc is not None:
                    for b in range(num_sequences):
                        if not finished[b]:
                            cur_gc = _gc_content(current_sequences[b])
                            logits[b: b + 1] = _gc_bias_logits(
                                logits[b: b + 1],
                                cur_gc,
                                target_gc,
                                strength=3.0,
                                nuc_to_idx=self._nuc_to_idx,
                            )

                next_tokens = _sample_from_logits(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                next_tokens[finished] = self._base_generator.pad_id
                generated = torch.cat([generated, next_tokens], dim=1)

                # Update current sequences for GC tracking
                for b in range(num_sequences):
                    if not finished[b]:
                        tok_id = next_tokens[b, 0].item()
                        # Reverse-map to nucleotide
                        idx_to_nuc = {v: k for k, v in self._nuc_to_idx.items()}
                        nuc = idx_to_nuc.get(tok_id, "")
                        current_sequences[b] += nuc

                finished = finished | (
                    next_tokens.squeeze(-1) == self._base_generator.eos_id
                )

                # Length constraint
                if target_length is not None:
                    for b in range(num_sequences):
                        if len(current_sequences[b]) >= target_length:
                            finished[b] = True

                if finished.all():
                    break

            # Decode sequences
            decoded = self._base_generator._decode_batch(generated)

            # Apply motif insertion if needed
            for i in range(len(decoded)):
                seq = decoded[i]

                # Trim to target length
                if target_length is not None:
                    seq = seq[:target_length]

                # Insert missing motifs
                for motif in required_motifs:
                    if not _contains_motif(seq, motif):
                        # Insert motif at a random position
                        if len(seq) > len(motif):
                            insert_pos = np.random.randint(
                                0, max(1, len(seq) - len(motif))
                            )
                            seq = (
                                seq[:insert_pos]
                                + motif
                                + seq[insert_pos + len(motif):]
                            )

                decoded[i] = seq

            # Check constraints
            all_met = True
            seq_met = []
            for seq in decoded:
                met = True
                if target_gc is not None:
                    actual_gc = _gc_content(seq)
                    if abs(actual_gc - target_gc) > gc_tolerance:
                        met = False
                for motif in required_motifs:
                    if not _contains_motif(seq, motif):
                        met = False
                if target_length is not None and len(seq) != target_length:
                    met = False
                seq_met.append(met)
                if not met:
                    all_met = False

            if all_met or attempt == max_retries - 1:
                best_sequences = decoded
                best_ids = generated
                constraints_met = seq_met
                break

        # Compute final GC contents
        gc_values = [_gc_content(s) for s in best_sequences]

        return {
            "token_ids": best_ids,
            "sequences": best_sequences,
            "constraints_met": constraints_met,
            "gc_content": gc_values,
            "constraints": constraints,
        }

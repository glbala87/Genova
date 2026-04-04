"""Byte Pair Encoding (BPE) tokenizer for genomic DNA sequences.

Learns sub-word units from a corpus of DNA sequences by iteratively
merging the most frequent adjacent token pairs, starting from a
character-level vocabulary of ``{A, C, G, T, N}``.

Example::

    tokenizer = GenomicBPETokenizer()
    tokenizer.train(["ACGTACGTACGT", "NNACGTNN"], vocab_size=512)
    ids = tokenizer.encode("ACGTAC")
    seq = tokenizer.decode(ids)
    tokenizer.save("bpe_tokenizer.json")
    loaded = GenomicBPETokenizer.load("bpe_tokenizer.json")
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUCLEOTIDES = list("ACGTN")

SPECIAL_TOKENS: Dict[str, int] = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[MASK]": 3,
    "[UNK]": 4,
}

_NUM_SPECIAL = len(SPECIAL_TOKENS)


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------


class GenomicBPETokenizer:
    """Byte Pair Encoding tokenizer for genomic DNA sequences.

    Provides the same public interface as
    :class:`~genova.data.tokenizer.GenomicTokenizer` (``encode``,
    ``decode``, ``batch_encode``, ``save``, ``load``).

    Args:
        add_special_tokens: Whether :meth:`encode` prepends ``[CLS]`` and
            appends ``[SEP]`` by default.
    """

    def __init__(self, add_special_tokens: bool = True) -> None:
        self.add_special_tokens = add_special_tokens

        # Token <-> id mappings
        self.token_to_id: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

        # BPE merge rules: list of (token_a, token_b) in merge order
        self.merges: List[Tuple[str, str]] = []

        self._trained = False

    # ---------------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return SPECIAL_TOKENS["[PAD]"]

    @property
    def cls_token_id(self) -> int:
        return SPECIAL_TOKENS["[CLS]"]

    @property
    def sep_token_id(self) -> int:
        return SPECIAL_TOKENS["[SEP]"]

    @property
    def mask_token_id(self) -> int:
        return SPECIAL_TOKENS["[MASK]"]

    @property
    def unk_token_id(self) -> int:
        return SPECIAL_TOKENS["[UNK]"]

    # ---------------------------------------------------------------- training
    def train(
        self,
        sequences: Sequence[str],
        vocab_size: int = 512,
    ) -> None:
        """Train the BPE vocabulary from a corpus of DNA sequences.

        Starts with a character-level vocabulary (A, C, G, T, N) and
        iteratively merges the most frequent adjacent pair until the
        desired *vocab_size* is reached (including special tokens).

        Args:
            sequences: Iterable of DNA strings used as the training corpus.
            vocab_size: Target vocabulary size (including special tokens
                and base characters).  Common values: 256, 512, 1024, 4096.

        Raises:
            ValueError: If *vocab_size* is smaller than the base vocabulary.
        """
        base_vocab_size = _NUM_SPECIAL + len(NUCLEOTIDES)
        if vocab_size < base_vocab_size:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be >= base vocabulary size "
                f"({base_vocab_size}: {_NUM_SPECIAL} special + {len(NUCLEOTIDES)} nucleotides)."
            )

        # Reset vocabulary to base
        self.token_to_id = dict(SPECIAL_TOKENS)
        next_id = _NUM_SPECIAL
        for nuc in NUCLEOTIDES:
            self.token_to_id[nuc] = next_id
            next_id += 1

        self.merges = []

        # Tokenize corpus into character-level token lists
        # Each word is a list of character tokens; word_freqs counts duplicates
        word_freqs: Counter[Tuple[str, ...]] = Counter()
        for seq in sequences:
            chars = tuple(seq.upper())
            # Validate characters
            chars = tuple(c if c in set(NUCLEOTIDES) else "N" for c in chars)
            if chars:
                word_freqs[chars] += 1

        if not word_freqs:
            logger.warning("No valid sequences provided for BPE training.")
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            self._trained = True
            return

        # Represent corpus as list of (token_list, frequency) for efficient updates
        corpus: List[Tuple[List[str], int]] = [
            (list(word), freq) for word, freq in word_freqs.items()
        ]

        num_merges = vocab_size - len(self.token_to_id)

        for merge_i in range(num_merges):
            # Count adjacent pairs
            pair_counts: Counter[Tuple[str, str]] = Counter()
            for tokens, freq in corpus:
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i + 1])] += freq

            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            merged_token = best_pair[0] + best_pair[1]

            # Add to vocabulary
            self.token_to_id[merged_token] = next_id
            next_id += 1
            self.merges.append(best_pair)

            # Apply merge to all words in the corpus
            new_corpus: List[Tuple[List[str], int]] = []
            for tokens, freq in corpus:
                new_tokens = _apply_merge(tokens, best_pair, merged_token)
                new_corpus.append((new_tokens, freq))
            corpus = new_corpus

            if (merge_i + 1) % 500 == 0:
                logger.debug(
                    "BPE merge {}/{}: {} + {} -> {} (freq={})",
                    merge_i + 1,
                    num_merges,
                    best_pair[0],
                    best_pair[1],
                    merged_token,
                    pair_counts[best_pair],
                )

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._trained = True

        logger.info(
            "BPE training complete: {} merges, vocab_size={}",
            len(self.merges),
            self.vocab_size,
        )

    # ---------------------------------------------------------------- tokenize
    def tokenize(self, sequence: str) -> List[str]:
        """Split a DNA sequence into BPE tokens (strings).

        Args:
            sequence: A DNA string (case-insensitive).

        Returns:
            List of BPE token strings.
        """
        if not self._trained:
            raise RuntimeError(
                "BPE tokenizer has not been trained. Call train() or load() first."
            )

        seq = sequence.upper()
        # Start with character-level tokens, replacing unknown chars with N
        tokens = [c if c in set(NUCLEOTIDES) else "N" for c in seq]

        # Apply merges in order
        for pair in self.merges:
            merged = pair[0] + pair[1]
            tokens = _apply_merge(tokens, pair, merged)

        return tokens

    # ---------------------------------------------------------------- encode
    def encode(
        self,
        sequence: str,
        add_special_tokens: Optional[bool] = None,
        max_length: Optional[int] = None,
        truncate: bool = True,
    ) -> List[int]:
        """Encode a DNA sequence to a list of token ids.

        Args:
            sequence: A DNA string.
            add_special_tokens: Prepend ``[CLS]`` and append ``[SEP]``.
                Defaults to the instance setting.
            max_length: If set, truncate tokens beyond this length
                (including special tokens).
            truncate: Whether to silently truncate when *max_length* is
                exceeded.  If ``False`` and the encoded length exceeds
                *max_length*, a ``ValueError`` is raised.

        Returns:
            List of integer token ids.
        """
        add_sp = add_special_tokens if add_special_tokens is not None else self.add_special_tokens

        tokens = self.tokenize(sequence)
        unk_id = self.unk_token_id
        ids = [self.token_to_id.get(t, unk_id) for t in tokens]

        if add_sp:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]

        if max_length is not None:
            if len(ids) > max_length:
                if not truncate:
                    raise ValueError(
                        f"Encoded length {len(ids)} exceeds max_length {max_length}"
                    )
                ids = ids[:max_length]
                if add_sp and ids[-1] != self.sep_token_id:
                    ids[-1] = self.sep_token_id

        return ids

    # ---------------------------------------------------------------- decode
    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids back to a DNA sequence.

        BPE tokens are simply concatenated since each token already
        represents a contiguous subsequence.

        Args:
            ids: Sequence of integer token ids.
            skip_special_tokens: If ``True``, special token ids are omitted
                from the output.

        Returns:
            Reconstructed DNA string.
        """
        if not self._trained:
            raise RuntimeError(
                "BPE tokenizer has not been trained. Call train() or load() first."
            )

        special_ids = set(SPECIAL_TOKENS.values())
        parts: List[str] = []
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id, "[UNK]")
            if token not in SPECIAL_TOKENS:
                parts.append(token)
        return "".join(parts)

    # ---------------------------------------------------------------- batch helpers
    def batch_encode(
        self,
        sequences: Sequence[str],
        add_special_tokens: Optional[bool] = None,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> Dict[str, List[List[int]]]:
        """Encode multiple sequences and optionally pad.

        Args:
            sequences: Iterable of DNA strings.
            add_special_tokens: Override instance-level special token setting.
            max_length: Maximum encoded length per sequence.
            padding: If ``True``, pad all sequences to the longest length
                in the batch.

        Returns:
            Dictionary with ``input_ids`` and ``attention_mask`` keys.
        """
        all_ids: List[List[int]] = []
        for seq in sequences:
            all_ids.append(
                self.encode(
                    seq,
                    add_special_tokens=add_special_tokens,
                    max_length=max_length,
                )
            )

        attention_masks: List[List[int]] = [[1] * len(ids) for ids in all_ids]

        if padding and all_ids:
            max_len = max(len(ids) for ids in all_ids)
            for i in range(len(all_ids)):
                pad_len = max_len - len(all_ids[i])
                all_ids[i] = all_ids[i] + [self.pad_token_id] * pad_len
                attention_masks[i] = attention_masks[i] + [0] * pad_len

        return {"input_ids": all_ids, "attention_mask": attention_masks}

    # ---------------------------------------------------------------- persistence
    def save(self, path: Union[str, Path]) -> None:
        """Save BPE vocabulary, merges, and configuration to a JSON file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "type": "bpe",
            "add_special_tokens": self.add_special_tokens,
            "token_to_id": self.token_to_id,
            "merges": [list(m) for m in self.merges],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("BPE tokenizer saved to {} (vocab_size={})", path, self.vocab_size)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GenomicBPETokenizer":
        """Load a BPE tokenizer from a previously saved JSON file.

        Args:
            path: Path to the tokenizer JSON file.

        Returns:
            A fully initialised :class:`GenomicBPETokenizer`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BPE tokenizer file not found: {path}")

        with open(path, "r") as fh:
            payload = json.load(fh)

        tok = cls(
            add_special_tokens=payload.get("add_special_tokens", True),
        )
        tok.token_to_id = {k: int(v) for k, v in payload["token_to_id"].items()}
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        tok.merges = [tuple(m) for m in payload.get("merges", [])]
        tok._trained = True

        logger.info(
            "BPE tokenizer loaded from {} (vocab_size={}, merges={})",
            path,
            tok.vocab_size,
            len(tok.merges),
        )
        return tok

    def __repr__(self) -> str:
        return (
            f"GenomicBPETokenizer(vocab_size={self.vocab_size}, "
            f"merges={len(self.merges)}, trained={self._trained})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_merge(
    tokens: List[str],
    pair: Tuple[str, str],
    merged: str,
) -> List[str]:
    """Apply a single BPE merge to a token list.

    Replaces all adjacent occurrences of *pair* with *merged*.

    Args:
        tokens: Current token list.
        pair: The (left, right) pair to merge.
        merged: The merged token string.

    Returns:
        New token list with the merge applied.
    """
    result: List[str] = []
    i = 0
    while i < len(tokens):
        if (
            i < len(tokens) - 1
            and tokens[i] == pair[0]
            and tokens[i + 1] == pair[1]
        ):
            result.append(merged)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

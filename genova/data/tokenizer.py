"""Genomic sequence tokenizer for Genova.

Supports nucleotide-level and k-mer tokenization modes with vocabulary
management, encode/decode, and persistence.

Example::

    tokenizer = GenomicTokenizer(mode="kmer", k=6)
    tokenizer.build_vocab(["ACGTACGTACGT", "NNACGTNN"])
    ids = tokenizer.encode("ACGTAC")
    seq = tokenizer.decode(ids)
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class GenomicTokenizer:
    """Tokenizer for genomic DNA sequences.

    Args:
        mode: ``"nucleotide"`` for single-base tokens or ``"kmer"`` for
            overlapping k-mer tokens.
        k: k-mer length (used only when *mode* is ``"kmer"``). Must be
            between 3 and 6 inclusive.
        stride: Step size between consecutive k-mers.  Defaults to 1
            (fully overlapping).
        add_special_tokens: Whether :meth:`encode` prepends ``[CLS]`` and
            appends ``[SEP]`` by default.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        mode: str = "kmer",
        k: int = 6,
        stride: int = 1,
        add_special_tokens: bool = True,
    ) -> None:
        if mode not in ("kmer", "nucleotide"):
            raise ValueError(f"mode must be 'kmer' or 'nucleotide', got {mode!r}")
        if mode == "kmer" and not (3 <= k <= 6):
            raise ValueError(f"k must be between 3 and 6, got {k}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.mode = mode
        self.k = k
        self.stride = stride
        self.add_special_tokens = add_special_tokens

        # Vocabulary mappings
        self.token_to_id: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

        self._vocab_built = False

    # -------------------------------------------------------------- properties
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

    # ----------------------------------------------------------- build vocab
    def build_vocab(
        self,
        sequences: Optional[Sequence[str]] = None,
        *,
        min_freq: int = 0,
    ) -> None:
        """Build or rebuild the vocabulary.

        For **nucleotide** mode the vocabulary is fixed (``A C G T N`` plus
        specials) regardless of the input sequences.

        For **kmer** mode, if *sequences* is ``None`` the full combinatorial
        vocabulary over ``{A,C,G,T,N}`` is generated.  If *sequences* is
        provided, only k-mers observed at least *min_freq* times are kept.

        Args:
            sequences: Optional iterable of DNA strings used to derive
                k-mer frequencies.
            min_freq: Minimum occurrence count for a k-mer to be included
                (ignored in nucleotide mode or when *sequences* is ``None``).
        """
        self.token_to_id = dict(SPECIAL_TOKENS)
        next_id = len(SPECIAL_TOKENS)

        if self.mode == "nucleotide":
            for nuc in NUCLEOTIDES:
                self.token_to_id[nuc] = next_id
                next_id += 1
        else:
            if sequences is None:
                # Full combinatorial vocabulary
                for kmer_tuple in product("ACGTN", repeat=self.k):
                    kmer = "".join(kmer_tuple)
                    self.token_to_id[kmer] = next_id
                    next_id += 1
            else:
                counter: Counter[str] = Counter()
                for seq in sequences:
                    seq_upper = seq.upper()
                    for i in range(0, len(seq_upper) - self.k + 1, self.stride):
                        counter[seq_upper[i : i + self.k]] += 1
                for kmer, freq in sorted(counter.items()):
                    if freq >= min_freq:
                        self.token_to_id[kmer] = next_id
                        next_id += 1

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._vocab_built = True
        logger.debug(
            "Vocabulary built: mode={}, size={}", self.mode, self.vocab_size
        )

    # ----------------------------------------------------------- tokenize
    def tokenize(self, sequence: str) -> List[str]:
        """Split a DNA sequence into tokens (strings).

        Args:
            sequence: A DNA string (case-insensitive).

        Returns:
            List of token strings.
        """
        seq = sequence.upper()

        if self.mode == "nucleotide":
            return list(seq)

        tokens: List[str] = []
        for i in range(0, len(seq) - self.k + 1, self.stride):
            tokens.append(seq[i : i + self.k])
        return tokens

    # ----------------------------------------------------------- encode
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
            max_length: If set, truncate or ignore tokens beyond this length
                (including special tokens).
            truncate: Whether to silently truncate when *max_length* is
                exceeded.  If ``False`` and the encoded length exceeds
                *max_length*, a ``ValueError`` is raised.

        Returns:
            List of integer token ids.
        """
        if not self._vocab_built:
            raise RuntimeError(
                "Vocabulary has not been built. Call build_vocab() first."
            )

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
                # Ensure [SEP] at the end when truncating with special tokens
                if add_sp and ids[-1] != self.sep_token_id:
                    ids[-1] = self.sep_token_id

        return ids

    # ----------------------------------------------------------- decode
    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids back to a DNA sequence.

        For k-mer mode with stride=1 the overlapping portions are resolved
        by keeping only the first character of each k-mer except the last,
        which is kept in full.

        Args:
            ids: Sequence of integer token ids.
            skip_special_tokens: If ``True``, special token ids are omitted
                from the output.

        Returns:
            Reconstructed DNA string.
        """
        if not self._vocab_built:
            raise RuntimeError(
                "Vocabulary has not been built. Call build_vocab() first."
            )

        special_ids = set(SPECIAL_TOKENS.values())
        tokens: List[str] = []
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id, "[UNK]")
            tokens.append(token)

        if not tokens:
            return ""

        if self.mode == "nucleotide":
            return "".join(tokens)

        # Reconstruct sequence from overlapping k-mers (stride-aware)
        if len(tokens) == 1:
            return tokens[0]

        parts = [tokens[0]]
        for t in tokens[1:]:
            # With stride, the overlap is k - stride characters
            overlap = max(0, self.k - self.stride)
            parts.append(t[overlap:])
        return "".join(parts)

    # ----------------------------------------------------------- batch helpers
    def batch_encode(
        self,
        sequences: Sequence[str],
        add_special_tokens: Optional[bool] = None,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> Dict[str, List[List[int]]]:
        """Encode multiple sequences and optionally pad.

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

    # ----------------------------------------------------------- persistence
    def save(self, path: Union[str, Path]) -> None:
        """Save vocabulary and configuration to a JSON file.

        Args:
            path: Destination file path (typically ``tokenizer.json``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "mode": self.mode,
            "k": self.k,
            "stride": self.stride,
            "add_special_tokens": self.add_special_tokens,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Tokenizer saved to {}", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GenomicTokenizer":
        """Load a tokenizer from a previously saved JSON file.

        Args:
            path: Path to the tokenizer JSON file.

        Returns:
            A fully initialised :class:`GenomicTokenizer`.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        with open(path, "r") as fh:
            payload = json.load(fh)

        tok = cls(
            mode=payload["mode"],
            k=payload.get("k", 6),
            stride=payload.get("stride", 1),
            add_special_tokens=payload.get("add_special_tokens", True),
        )
        tok.token_to_id = {k: int(v) for k, v in payload["token_to_id"].items()}
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        tok._vocab_built = True
        logger.info("Tokenizer loaded from {} (vocab_size={})", path, tok.vocab_size)
        return tok

    def __repr__(self) -> str:
        return (
            f"GenomicTokenizer(mode={self.mode!r}, k={self.k}, "
            f"stride={self.stride}, vocab_size={self.vocab_size})"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_tokenizer(
    mode: str = "kmer",
    **kwargs: Any,
) -> Union["GenomicTokenizer", "GenomicBPETokenizer"]:
    """Create a tokenizer by mode name.

    Provides a unified entry point for constructing any supported tokenizer.

    Args:
        mode: Tokenization mode.  One of ``"nucleotide"``, ``"kmer"``,
            or ``"bpe"``.
        **kwargs: Additional keyword arguments forwarded to the tokenizer
            constructor.  For ``"nucleotide"`` and ``"kmer"`` modes, see
            :class:`GenomicTokenizer`.  For ``"bpe"`` mode, see
            :class:`~genova.data.bpe_tokenizer.GenomicBPETokenizer`.

    Returns:
        A tokenizer instance of the appropriate type.

    Raises:
        ValueError: If *mode* is not recognized.

    Example::

        tok = create_tokenizer("kmer", k=6, stride=1)
        tok.build_vocab()

        bpe_tok = create_tokenizer("bpe")
        bpe_tok.train(sequences, vocab_size=1024)
    """
    if mode in ("nucleotide", "kmer"):
        return GenomicTokenizer(mode=mode, **kwargs)
    elif mode == "bpe":
        from genova.data.bpe_tokenizer import GenomicBPETokenizer

        return GenomicBPETokenizer(**kwargs)
    else:
        raise ValueError(
            f"Unknown tokenizer mode {mode!r}. "
            f"Supported modes: 'nucleotide', 'kmer', 'bpe'."
        )

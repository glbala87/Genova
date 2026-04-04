"""DataLoader factory for Genova genomic training.

Creates train / validation / test DataLoaders with support for distributed
training (DDP), dynamic batching by sequence length, and a custom collate
function that pads sequences to equal length within each batch.

Example::

    from genova.utils.config import GenovaConfig
    from genova.data.tokenizer import GenomicTokenizer
    from genova.data.dataloader import create_dataloaders

    config = GenovaConfig.from_yaml("configs/default.yaml")
    tokenizer = GenomicTokenizer(mode="kmer", k=6)
    tokenizer.build_vocab()
    loaders = create_dataloaders(config, tokenizer)
    for batch in loaders["train"]:
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from loguru import logger

from genova.data.tokenizer import GenomicTokenizer
from genova.data.genome_dataset import GenomeDataset

# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def genomic_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Collate genomic samples with dynamic padding.

    Pads all sequences in the batch to the length of the longest sequence.

    Args:
        batch: List of sample dicts each containing ``input_ids``,
            ``attention_mask``, and ``labels`` tensors.
        pad_token_id: Token id used for padding ``input_ids``.

    Returns:
        Batched dict with padded tensors.
    """
    max_len = max(sample["input_ids"].size(0) for sample in batch)

    input_ids_list: List[torch.Tensor] = []
    attention_mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for sample in batch:
        seq_len = sample["input_ids"].size(0)
        pad_len = max_len - seq_len

        if pad_len > 0:
            input_ids_list.append(
                torch.cat([
                    sample["input_ids"],
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                ])
            )
            attention_mask_list.append(
                torch.cat([
                    sample["attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long),
                ])
            )
            labels_list.append(
                torch.cat([
                    sample["labels"],
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])
            )
        else:
            input_ids_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


# ---------------------------------------------------------------------------
# Length-grouped sampler for dynamic batching
# ---------------------------------------------------------------------------


class LengthGroupedSampler(Sampler[int]):
    """Sampler that groups sequences of similar length to minimise padding.

    Sequences are sorted by length within mega-batches (groups) and then
    shuffled at the mega-batch level for stochasticity.

    Args:
        dataset: The dataset to sample from.  Each element must have an
            ``input_ids`` attribute or be index-able to get length info.
        batch_size: Target batch size.
        lengths: Pre-computed sequence lengths.  If ``None``, lengths are
            estimated from the dataset.
        mega_batch_mult: Multiplier to determine mega-batch size
            (``mega_batch_size = mega_batch_mult * batch_size``).
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        lengths: Optional[List[int]] = None,
        mega_batch_mult: int = 50,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.mega_batch_mult = mega_batch_mult
        self.seed = seed

        if lengths is not None:
            self.lengths = lengths
        else:
            # Estimate lengths from the dataset's window_size and tokenizer
            if hasattr(dataset, "window_size") and hasattr(dataset, "tokenizer"):
                est_len = dataset.window_size  # rough estimate
                self.lengths = [est_len] * len(dataset)
            else:
                self.lengths = [1] * len(dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        mega_batch_size = self.batch_size * self.mega_batch_mult

        # Sort within mega-batches by length, then yield in order
        result: List[int] = []
        for i in range(0, len(indices), mega_batch_size):
            mega_batch = indices[i : i + mega_batch_size]
            mega_batch.sort(key=lambda idx: self.lengths[idx])
            result.extend(mega_batch)

        return iter(result)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_dataloaders(
    config: Any,
    tokenizer: GenomicTokenizer,
    *,
    rank: int = 0,
    world_size: int = 1,
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
) -> Dict[str, DataLoader]:
    """Create train / val / test DataLoaders from config.

    If datasets are not provided, they are constructed from the FASTA and
    BED paths in *config.data*.

    Args:
        config: A :class:`~genova.utils.config.GenovaConfig` (or anything
            with a ``data`` attribute of type
            :class:`~genova.utils.config.DataConfig`) or a ``DataConfig``
            directly.
        tokenizer: A :class:`GenomicTokenizer` with a built vocabulary.
        rank: Process rank for distributed training.
        world_size: Total number of processes.
        train_dataset: Optional pre-built training dataset.
        val_dataset: Optional pre-built validation dataset.
        test_dataset: Optional pre-built test dataset.

    Returns:
        Dict with ``"train"``, ``"val"``, and/or ``"test"`` DataLoader
        entries (only present when the corresponding dataset / config path
        is available).
    """
    # Normalise config access
    data_cfg = getattr(config, "data", config)
    training_cfg = getattr(config, "training", None)

    batch_size = getattr(data_cfg, "batch_size", 32)
    num_workers = getattr(data_cfg, "num_workers", 4)
    pin_memory = getattr(data_cfg, "pin_memory", True)
    prefetch_factor = getattr(data_cfg, "prefetch_factor", 2)
    seed = getattr(training_cfg, "seed", 42) if training_cfg else 42

    distributed = world_size > 1

    pad_token_id = tokenizer.pad_token_id

    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return genomic_collate_fn(batch, pad_token_id=pad_token_id)

    loaders: Dict[str, DataLoader] = {}

    # --- Helper to build a dataset from config ---
    def _build_dataset(
        bed_path: Optional[str],
        rc_prob: float = 0.0,
        mask_prob: float = 0.0,
    ) -> Optional[GenomeDataset]:
        fasta = getattr(data_cfg, "genome_fasta", "")
        if not fasta:
            return None
        window_size = getattr(data_cfg, "seq_length", 2048)
        stride = getattr(data_cfg, "stride", 1)
        # Convert token-level stride=1 to a sensible bp stride
        if stride <= 1:
            stride = window_size // 2
        return GenomeDataset(
            fasta_path=fasta,
            tokenizer=tokenizer,
            window_size=window_size,
            stride=stride,
            bed_path=bed_path if bed_path else None,
            reverse_complement_prob=rc_prob,
            mask_prob=mask_prob,
            max_length=getattr(data_cfg, "max_tokens", 1024),
            seed=seed,
        )

    # --- Train ---
    if train_dataset is None:
        train_bed = getattr(data_cfg, "train_regions", "")
        mask_prob = getattr(data_cfg, "mask_prob", 0.15)
        rc_prob = 0.5 if "reverse_complement" in getattr(data_cfg, "augmentations", []) else 0.0
        train_dataset = _build_dataset(train_bed, rc_prob=rc_prob, mask_prob=mask_prob)

    if train_dataset is not None and len(train_dataset) > 0:
        train_sampler: Optional[Sampler] = None
        shuffle = True
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
            )
            shuffle = False

        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle if train_sampler is None else False,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=_collate,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )

    # --- Val ---
    if val_dataset is None:
        val_bed = getattr(data_cfg, "val_regions", "")
        val_dataset = _build_dataset(val_bed, rc_prob=0.0, mask_prob=getattr(data_cfg, "mask_prob", 0.15))

    if val_dataset is not None and len(val_dataset) > 0:
        val_sampler: Optional[Sampler] = None
        if distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )

        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=_collate,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )

    # --- Test ---
    if test_dataset is None:
        test_bed = getattr(data_cfg, "test_regions", "")
        test_dataset = _build_dataset(test_bed, rc_prob=0.0, mask_prob=0.0)

    if test_dataset is not None and len(test_dataset) > 0:
        test_sampler: Optional[Sampler] = None
        if distributed:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )

        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=_collate,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )

    logger.info(
        "DataLoaders created: {} (distributed={}, world_size={})",
        list(loaders.keys()),
        distributed,
        world_size,
    )
    return loaders

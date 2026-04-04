"""Integration test: end-to-end training pipeline.

Creates synthetic FASTA data, tokenizes it, builds a tiny model, runs
several training steps, and verifies checkpoint save/load round-trips.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from genova.utils.config import GenovaConfig, ModelConfig, TrainingConfig, DataConfig
from genova.data.tokenizer import GenomicTokenizer
from genova.models.transformer import GenovaForMLM
from genova.training.trainer import GenovaTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_fasta(directory: Path) -> Path:
    """Write a small fake FASTA file and return its path."""
    fasta_path = directory / "synthetic_genome.fa"
    import random

    rng = random.Random(42)
    seq = "".join(rng.choices("ACGT", k=2000))
    fasta_path.write_text(f">chr1\n{seq}\n")
    return fasta_path


def _make_mlm_batch(
    tokenizer: GenomicTokenizer,
    sequences: list[str],
    mask_prob: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Tokenize sequences and create a simple MLM batch."""
    import random as _random

    rng = _random.Random(0)
    all_input_ids = []
    all_labels = []

    for seq in sequences:
        ids = tokenizer.encode(seq, add_special_tokens=True, max_length=64)
        labels = [-100] * len(ids)
        for i in range(len(ids)):
            if ids[i] in (0, 1, 2, 3):  # skip special tokens
                continue
            if rng.random() < mask_prob:
                labels[i] = ids[i]
                ids[i] = tokenizer.mask_token_id
        all_input_ids.append(ids)
        all_labels.append(labels)

    # Pad to same length
    max_len = max(len(ids) for ids in all_input_ids)
    for i in range(len(all_input_ids)):
        pad_len = max_len - len(all_input_ids[i])
        all_input_ids[i] += [tokenizer.pad_token_id] * pad_len
        all_labels[i] += [-100] * pad_len

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(
            [[1 if t != 0 else 0 for t in ids] for ids in all_input_ids],
            dtype=torch.long,
        ),
        "labels": torch.tensor(all_labels, dtype=torch.long),
    }


def _build_dataloader(
    tokenizer: GenomicTokenizer,
    num_samples: int = 16,
    seq_len: int = 48,
    batch_size: int = 4,
) -> DataLoader:
    """Build a DataLoader of synthetic MLM samples."""
    import random as _random

    rng = _random.Random(123)
    sequences = [
        "".join(rng.choices("ACGT", k=seq_len)) for _ in range(num_samples)
    ]
    batch = _make_mlm_batch(tokenizer, sequences)

    ds = TensorDataset(
        batch["input_ids"],
        batch["attention_mask"],
        batch["labels"],
    )

    def collate(samples):
        input_ids = torch.stack([s[0] for s in samples])
        attention_mask = torch.stack([s[1] for s in samples])
        labels = torch.stack([s[2] for s in samples])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingPipeline:
    """End-to-end training pipeline integration tests."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_dir = tmp_path

        # Tokenizer
        self.tokenizer = GenomicTokenizer(mode="kmer", k=3, stride=1)
        self.tokenizer.build_vocab(["ACGTACGTACGTACGTACGT"])

        # Tiny model config
        self.model_config = ModelConfig(
            arch="transformer",
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            dropout=0.0,
            attention_dropout=0.0,
            max_position_embeddings=128,
            rotary_emb=False,
            flash_attention=False,
            gradient_checkpointing=False,
            tie_word_embeddings=False,
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=0,
        )

    def test_forward_backward_basic(self):
        """Model can do a forward pass and compute gradients."""
        model = GenovaForMLM(self.model_config)
        model.train()

        loader = _build_dataloader(self.tokenizer, num_samples=4, batch_size=2)
        batch = next(iter(loader))

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        assert "loss" in out, "Model output should contain 'loss'"
        assert "logits" in out, "Model output should contain 'logits'"
        assert torch.isfinite(out["loss"]), "Loss should be finite"

        out["loss"].backward()
        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0, "Should have gradients for some parameters"
        assert all(
            math.isfinite(g) for g in grad_norms
        ), "All gradients should be finite"

    def test_training_steps_loss_finite(self):
        """Run 3 training steps and verify loss is always finite."""
        model = GenovaForMLM(self.model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        loader = _build_dataloader(self.tokenizer, num_samples=12, batch_size=4)
        losses = []

        model.train()
        for step, batch in enumerate(loader):
            if step >= 3:
                break
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out["loss"]
            assert torch.isfinite(loss), f"Loss at step {step} is not finite: {loss.item()}"
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert len(losses) == 3, f"Expected 3 losses, got {len(losses)}"
        assert all(
            not math.isnan(l) for l in losses
        ), f"No NaN losses expected, got {losses}"

    def test_checkpoint_save_and_reload(self):
        """Save a checkpoint and reload -- weights must match exactly."""
        model = GenovaForMLM(self.model_config)

        # Do one forward-backward to change weights from init
        loader = _build_dataloader(self.tokenizer, num_samples=4, batch_size=4)
        batch = next(iter(loader))
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out["loss"].backward()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()

        # Save checkpoint
        ckpt_path = self.tmp_dir / "test_ckpt.pt"
        state = {"model_state_dict": model.state_dict()}
        torch.save(state, ckpt_path)

        # Reload into a fresh model
        model2 = GenovaForMLM(self.model_config)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model2.load_state_dict(ckpt["model_state_dict"])

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
            assert torch.allclose(
                p1, p2, atol=1e-6
            ), f"Parameter {n1} differs after reload"

    def test_trainer_integration(self):
        """GenovaTrainer runs a few steps without crashing."""
        from contextlib import nullcontext
        from unittest.mock import patch

        model = GenovaForMLM(self.model_config)

        train_loader = _build_dataloader(
            self.tokenizer, num_samples=8, batch_size=4
        )

        config = GenovaConfig(
            model=self.model_config,
            training=TrainingConfig(
                output_dir=str(self.tmp_dir / "outputs"),
                run_name="test_run",
                epochs=1,
                max_steps=2,
                lr=1e-3,
                warmup_steps=1,
                lr_scheduler="cosine",
                mixed_precision="no",
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                log_every_n_steps=1,
                save_every_n_steps=0,
                eval_every_n_steps=0,
                early_stopping_patience=0,
            ),
        )

        # Patch autocast to a no-op context manager for CPU-only testing
        # since torch.cuda.amp.autocast may not support device_type kwarg
        def _noop_autocast(*args, **kwargs):
            return nullcontext()

        with patch("genova.training.trainer.autocast", _noop_autocast):
            trainer = GenovaTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
            )

            metrics = trainer.train()

        assert trainer.global_step >= 1, (
            f"Expected at least 1 training step, got {trainer.global_step}"
        )

    def test_synthetic_fasta_creation(self):
        """Verify synthetic FASTA file is valid."""
        fasta_path = _make_synthetic_fasta(self.tmp_dir)
        assert fasta_path.exists(), "FASTA file should exist"

        content = fasta_path.read_text()
        assert content.startswith(">chr1"), "FASTA should start with header"

        seq_lines = [l for l in content.strip().split("\n") if not l.startswith(">")]
        seq = "".join(seq_lines)
        assert len(seq) == 2000, f"Expected 2000bp, got {len(seq)}"
        assert set(seq).issubset(set("ACGT")), "Sequence should only contain ACGT"

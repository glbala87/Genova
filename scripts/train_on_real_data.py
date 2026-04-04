#!/usr/bin/env python3
"""Train Genova foundation model on real human genomic data.

This script handles the complete pipeline:
1. Build tokenizer vocabulary from real genome
2. Preprocess genome into training windows
3. Train the model with MLM objective
4. Evaluate and save checkpoint

Usage:
    python scripts/train_on_real_data.py \
        --train-fasta data/reference/train.fa \
        --val-fasta data/reference/val.fa \
        --output-dir outputs/genova_real \
        --epochs 3 --batch-size 16
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genova.data.tokenizer import GenomicTokenizer
from genova.data.genome_dataset import GenomeDataset
from genova.models.model_factory import create_model, count_parameters
from genova.training.scheduler import CosineWithWarmup
from genova.utils.config import ModelConfig
from genova.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Genova on real genomic data")
    parser.add_argument("--train-fasta", type=str, required=True, help="Training FASTA")
    parser.add_argument("--val-fasta", type=str, required=True, help="Validation FASTA")
    parser.add_argument("--output-dir", type=str, default="outputs/genova_real")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--max-n-fraction", type=float, default=0.1)
    return parser.parse_args()


def collate_fn(batch):
    """Collate genomic samples with padding."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]
        labels[i, :seq_len] = b["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate(model, val_loader, device, max_batches=50):
    """Run evaluation and return metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs["loss"].item()

            # MLM accuracy
            mask = labels != -100
            if mask.any():
                preds = outputs["logits"][mask].argmax(dim=-1)
                total_correct += (preds == labels[mask]).sum().item()
                total_masked += mask.sum().item()

            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_masked, 1)
    perplexity = math.exp(min(avg_loss, 20.0))

    model.train()
    return {"loss": avg_loss, "accuracy": accuracy, "perplexity": perplexity}


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Step 1: Build tokenizer ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: Building tokenizer vocabulary")
    print("=" * 60)

    tokenizer = GenomicTokenizer(mode="nucleotide")
    tokenizer.build_vocab()
    tokenizer.save(output_dir / "tokenizer.json")
    print(f"Tokenizer: mode=nucleotide, vocab_size={tokenizer.vocab_size}")
    print(f"Vocab: {tokenizer.token_to_id}")

    # ── Step 2: Create datasets ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Creating datasets from real genome")
    print("=" * 60)

    t0 = time.time()
    train_dataset = GenomeDataset(
        fasta_path=args.train_fasta,
        tokenizer=tokenizer,
        window_size=args.window_size,
        stride=args.stride,
        mask_prob=args.mask_prob,
        max_n_fraction=args.max_n_fraction,
        reverse_complement_prob=0.5,
    )
    print(f"Train dataset: {len(train_dataset)} windows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    val_dataset = GenomeDataset(
        fasta_path=args.val_fasta,
        tokenizer=tokenizer,
        window_size=args.window_size,
        stride=args.stride,
        mask_prob=args.mask_prob,
        max_n_fraction=args.max_n_fraction,
        reverse_complement_prob=0.0,
    )
    print(f"Val dataset:   {len(val_dataset)} windows ({time.time()-t0:.1f}s)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # ── Step 3: Create model ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Creating model")
    print("=" * 60)

    config = ModelConfig(
        arch="transformer",
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        vocab_size=tokenizer.vocab_size,
        dropout=0.1,
        max_position_embeddings=args.window_size + 64,
    )
    model = create_model(config, task="mlm").to(device)
    num_params = count_parameters(model)
    print(f"Architecture: {args.n_layers}L / {args.d_model}d / {args.n_heads}H")
    print(f"Parameters:   {num_params:,}")
    print(f"Vocab size:   {tokenizer.vocab_size}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "arch": config.arch, "d_model": config.d_model,
            "n_heads": config.n_heads, "n_layers": config.n_layers,
            "d_ff": config.d_ff, "vocab_size": config.vocab_size,
            "window_size": args.window_size, "stride": args.stride,
            "mask_prob": args.mask_prob, "lr": args.lr,
            "batch_size": args.batch_size, "epochs": args.epochs,
            "num_params": num_params,
        }, f, indent=2)

    # ── Step 4: Setup optimizer + scheduler ───────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999)
    )

    steps_per_epoch = len(train_loader)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(500, total_steps // 10)

    scheduler = CosineWithWarmup(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps
    )
    print(f"\nOptimizer:    AdamW (lr={args.lr}, wd=0.01)")
    print(f"Scheduler:    Cosine with {warmup_steps} warmup steps")
    print(f"Total steps:  {total_steps}")

    # ── Step 5: Training loop ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Training")
    print("=" * 60)

    best_val_loss = float("inf")
    global_step = 0
    training_log = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_masked = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Track accuracy
            mask = labels != -100
            if mask.any():
                with torch.no_grad():
                    preds = outputs["logits"][mask].argmax(dim=-1)
                    epoch_correct += (preds == labels[mask]).sum().item()
                    epoch_masked += mask.sum().item()

            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / (step + 1)
                acc = epoch_correct / max(epoch_masked, 1)
                lr = scheduler.get_last_lr()[0]
                ppl = math.exp(min(avg_loss, 20.0))
                print(
                    f"  Step {global_step:5d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"PPL: {ppl:.2f} | "
                    f"Acc: {acc:.4f} | "
                    f"LR: {lr:.2e}"
                )
                training_log.append({
                    "step": global_step, "epoch": epoch,
                    "train_loss": avg_loss, "train_ppl": ppl,
                    "train_acc": acc, "lr": lr,
                })

            # Periodic evaluation
            if global_step % args.eval_every == 0:
                val_metrics = evaluate(model, val_loader, device)
                print(
                    f"  >>> VAL | Loss: {val_metrics['loss']:.4f} | "
                    f"PPL: {val_metrics['perplexity']:.2f} | "
                    f"Acc: {val_metrics['accuracy']:.4f}"
                )
                training_log.append({
                    "step": global_step, "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_ppl": val_metrics["perplexity"],
                    "val_acc": val_metrics["accuracy"],
                })

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "val_loss": best_val_loss,
                        "config": config.__dict__,
                    }, output_dir / "best_model.pt")
                    print(f"  >>> Saved best model (val_loss={best_val_loss:.4f})")

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / (step + 1)
        avg_epoch_acc = epoch_correct / max(epoch_masked, 1)
        print(f"\n  Epoch {epoch+1}/{args.epochs} complete in {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_epoch_loss:.4f} | Acc: {avg_epoch_acc:.4f}")

        # Full validation at end of epoch
        val_metrics = evaluate(model, val_loader, device, max_batches=100)
        print(
            f"  Val Loss:   {val_metrics['loss']:.4f} | "
            f"PPL: {val_metrics['perplexity']:.2f} | "
            f"Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "val_loss": best_val_loss,
                "config": config.__dict__,
            }, output_dir / "best_model.pt")
            print(f"  >>> Saved best model (val_loss={best_val_loss:.4f})")

        # Save epoch checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "val_loss": val_metrics["loss"],
            "config": config.__dict__,
        }, output_dir / f"checkpoint_epoch{epoch+1}.pt")
        print()

    # ── Step 6: Final save ───────────────────────────────────────────────
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best perplexity:      {math.exp(min(best_val_loss, 20.0)):.2f}")
    print(f"Total steps:          {global_step}")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "tokenizer_vocab": tokenizer.token_to_id,
        "training_args": vars(args),
        "final_step": global_step,
        "best_val_loss": best_val_loss,
    }, output_dir / "final_model.pt")

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  best_model.pt          - Best checkpoint")
    print(f"  final_model.pt         - Final checkpoint")
    print(f"  training_log.json      - Training metrics")
    print(f"  config.json            - Model config")
    print(f"  tokenizer.json         - Tokenizer vocab")


if __name__ == "__main__":
    main()

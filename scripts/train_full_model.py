#!/usr/bin/env python3
"""Train the full 12L/768d Genova model on all 22 human autosomes.

Architecture: 12 layers, 768 hidden, 12 heads, ~85M parameters
Data: 2.78B bases (chr1-20 train, chr21 val, chr22 test)

Optimized for CPU training with large stride to keep dataset manageable.
For GPU: reduce stride, increase batch size, remove max_steps.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genova.data.tokenizer import GenomicTokenizer
from genova.data.genome_dataset import GenomeDataset
from genova.models.model_factory import create_model, count_parameters
from genova.training.scheduler import CosineWithWarmup
from genova.training.ema import EMAModel
from genova.utils.config import ModelConfig
from genova.utils.reproducibility import set_seed


def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        sl = b["input_ids"].size(0)
        input_ids[i, :sl] = b["input_ids"]
        attention_mask[i, :sl] = b["attention_mask"]
        labels[i, :sl] = b["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate(model, loader, device, max_batches=50):
    model.eval()
    total_loss = correct = masked = n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= max_batches:
                break
            inp = batch["input_ids"].to(device)
            lbl = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            out = model(inp, attention_mask=attn, labels=lbl)
            total_loss += out["loss"].item()
            m = lbl != -100
            if m.any():
                correct += (out["logits"][m].argmax(-1) == lbl[m]).sum().item()
                masked += m.sum().item()
            n += 1
    model.train()
    loss = total_loss / max(n, 1)
    return {"loss": loss, "accuracy": correct / max(masked, 1),
            "perplexity": math.exp(min(loss, 20.0))}


def main():
    set_seed(42)
    output_dir = Path("outputs/genova_full_12L768d")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # ── Tokenizer ────────────────────────────────────────────────────────
    tok = GenomicTokenizer(mode="nucleotide")
    tok.build_vocab()
    tok.save(output_dir / "tokenizer.json")
    print(f"Tokenizer: vocab_size={tok.vocab_size}")

    # ── Datasets ─────────────────────────────────────────────────────────
    # Use large stride on CPU to keep dataset manageable
    # GPU: stride=256 for full coverage
    is_cpu = device.type == "cpu"
    window_size = 512
    stride = 2048 if is_cpu else 256  # CPU: sample ~1.3M windows; GPU: ~10M windows

    print(f"\nCreating datasets (window={window_size}, stride={stride})...")
    t0 = time.time()
    train_ds = GenomeDataset("data/reference/full_train.fa", tok,
                             window_size=window_size, stride=stride,
                             mask_prob=0.15, max_n_fraction=0.1,
                             reverse_complement_prob=0.5)
    print(f"  Train: {len(train_ds):,} windows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    val_ds = GenomeDataset("data/reference/chr21.fa", tok,
                           window_size=window_size, stride=stride,
                           mask_prob=0.15, max_n_fraction=0.1)
    print(f"  Val:   {len(val_ds):,} windows ({time.time()-t0:.1f}s)")

    batch_size = 8 if is_cpu else 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    print(f"  Batches: {len(train_loader):,} train, {len(val_loader):,} val")

    # ── Model: 12L/768d/12H ─────────────────────────────────────────────
    config = ModelConfig(
        arch="transformer",
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        vocab_size=tok.vocab_size,
        dropout=0.1,
        max_position_embeddings=window_size + 64,
        norm_type="rmsnorm",        # faster than layernorm
        activation="gelu",
    )
    model = create_model(config, task="mlm").to(device)
    num_params = count_parameters(model)
    print(f"\n  Model: 12L / 768d / 12H")
    print(f"  Parameters: {num_params:,}")
    print(f"  Norm: RMSNorm | Activation: GELU")

    # ── EMA ───────────────────────────────────────────────────────────────
    ema = EMAModel(model, decay=0.999)

    # ── Optimizer ─────────────────────────────────────────────────────────
    max_steps = 3000 if is_cpu else len(train_loader) * 3  # CPU: 3000 steps; GPU: 3 epochs
    warmup_steps = min(300, max_steps // 10)
    lr = 3e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                  betas=(0.9, 0.98), eps=1e-6)
    scheduler = CosineWithWarmup(optimizer, warmup_steps=warmup_steps,
                                 total_steps=max_steps)
    scaler = torch.amp.GradScaler() if use_amp else None

    print(f"\n  Optimizer: AdamW (lr={lr})")
    print(f"  Scheduler: Cosine warmup ({warmup_steps} steps)")
    print(f"  Max steps: {max_steps:,}")
    print(f"  EMA decay: 0.999")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "arch": "transformer", "d_model": 768, "n_heads": 12,
            "n_layers": 12, "d_ff": 3072, "vocab_size": 10,
            "norm_type": "rmsnorm", "activation": "gelu",
            "window_size": window_size, "stride": stride,
            "batch_size": batch_size, "lr": lr,
            "max_steps": max_steps, "num_params": num_params,
            "train_bases": 2777473071, "val_bases": 46709983,
            "device": str(device),
        }, f, indent=2)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING: 12L/768d Genova on 2.78B bases")
    print(f"{'='*70}\n")

    best_val_loss = float("inf")
    global_step = 0
    log = []
    epoch = 0
    log_every = 100
    eval_every = 500

    model.train()
    train_start = time.time()
    epoch_iter = iter(train_loader)

    for step in range(max_steps):
        # Get next batch (loop over epochs)
        try:
            batch = next(epoch_iter)
        except StopIteration:
            epoch += 1
            epoch_iter = iter(train_loader)
            batch = next(epoch_iter)

        inp = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        attn = batch["attention_mask"].to(device)

        # Forward
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                out = model(inp, attention_mask=attn, labels=lbl)
            optimizer.zero_grad()
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(inp, attention_mask=attn, labels=lbl)
            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        ema.update()
        global_step += 1

        # Logging
        if global_step % log_every == 0:
            elapsed = time.time() - train_start
            steps_per_sec = global_step / elapsed
            eta = (max_steps - global_step) / steps_per_sec
            lr_now = scheduler.get_last_lr()[0]
            loss_val = out["loss"].item()
            ppl = math.exp(min(loss_val, 20.0))

            print(f"  Step {global_step:5d}/{max_steps} | "
                  f"Loss: {loss_val:.4f} | PPL: {ppl:.2f} | "
                  f"LR: {lr_now:.2e} | "
                  f"{steps_per_sec:.2f} step/s | "
                  f"ETA: {eta/60:.0f}min")

            log.append({"step": global_step, "loss": loss_val,
                        "ppl": ppl, "lr": lr_now,
                        "steps_per_sec": steps_per_sec})

        # Validation
        if global_step % eval_every == 0:
            # Evaluate with EMA weights
            ema.apply_shadow()
            val = evaluate(model, val_loader, device)
            ema.restore()

            print(f"  >>> VAL (EMA) | Loss: {val['loss']:.4f} | "
                  f"Acc: {val['accuracy']:.4f} | PPL: {val['perplexity']:.2f}")

            log.append({"step": global_step, "val_loss": val["loss"],
                        "val_acc": val["accuracy"], "val_ppl": val["perplexity"]})

            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                ema.apply_shadow()
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "step": global_step,
                    "val_loss": best_val_loss,
                    "val_acc": val["accuracy"],
                    "val_ppl": val["perplexity"],
                }, output_dir / "best_model.pt")
                ema.restore()
                print(f"  >>> Saved best model (val_loss={best_val_loss:.4f})")

    # ── Final ─────────────────────────────────────────────────────────────
    total_time = time.time() - train_start
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:     {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Total steps:    {global_step:,}")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Best val PPL:   {math.exp(min(best_val_loss, 20.0)):.2f}")

    # Save final model with EMA weights
    ema.apply_shadow()
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "step": global_step,
        "total_time": total_time,
        "best_val_loss": best_val_loss,
    }, output_dir / "final_model.pt")
    ema.restore()

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # ── Quick benchmark ───────────────────────────────────────────────────
    print(f"\n  Running quick benchmark on test set (chr22)...")
    test_ds = GenomeDataset("data/reference/chr22.fa", tok,
                            window_size=window_size, stride=stride,
                            mask_prob=0.15, max_n_fraction=0.1)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    ema.apply_shadow()
    test = evaluate(model, test_loader, device, max_batches=100)
    ema.restore()

    print(f"  Test Loss: {test['loss']:.4f} | Acc: {test['accuracy']:.4f} | PPL: {test['perplexity']:.2f}")

    # Save results
    results = {
        "model": "Genova-12L-768d",
        "params": num_params,
        "train_bases": 2777473071,
        "train_chromosomes": "chr1-chr20",
        "val_chromosome": "chr21",
        "test_chromosome": "chr22",
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20.0)),
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "test_ppl": test["perplexity"],
        "total_steps": global_step,
        "total_time_hours": total_time / 3600,
        "device": str(device),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  All outputs saved to: {output_dir}/")
    print(f"    best_model.pt      - Best EMA checkpoint ({num_params:,} params)")
    print(f"    final_model.pt     - Final EMA checkpoint")
    print(f"    config.json        - Full configuration")
    print(f"    results.json       - Benchmark results")
    print(f"    training_log.json  - Training metrics")


if __name__ == "__main__":
    main()

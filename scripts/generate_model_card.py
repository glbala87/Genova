#!/usr/bin/env python3
"""Generate a model card for a trained Genova model.

Loads a model checkpoint, extracts architecture and configuration details,
optionally loads evaluation metrics, and produces a markdown model card
following ML model documentation best practices.

Usage:
    python scripts/generate_model_card.py \\
        --model-path outputs/genova_small/best_model.pt \\
        --output outputs/genova_small/MODEL_CARD.md

    python scripts/generate_model_card.py \\
        --model-path outputs/genova_large/best_model.pt \\
        --metrics-path results/evaluation/benchmark_report.json \\
        --output outputs/genova_large/MODEL_CARD.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_model_card")


# ---------------------------------------------------------------------------
# Model inspection
# ---------------------------------------------------------------------------

def load_checkpoint_info(
    checkpoint_path: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint and extract model metadata.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load onto.

    Returns:
        Dict with model metadata (config, parameter counts, etc.).
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    info: Dict[str, Any] = {}

    # Extract config
    config_dict = checkpoint.get("config", {})
    info["config"] = config_dict

    # Extract model config specifics
    model_cfg = config_dict.get("model", {})
    info["architecture"] = model_cfg.get("arch", "unknown")
    info["d_model"] = model_cfg.get("d_model", 0)
    info["n_layers"] = model_cfg.get("n_layers", 0)
    info["n_heads"] = model_cfg.get("n_heads", 0)
    info["d_ff"] = model_cfg.get("d_ff", 0)
    info["vocab_size"] = model_cfg.get("vocab_size", 0)
    info["max_position_embeddings"] = model_cfg.get("max_position_embeddings", 0)
    info["dropout"] = model_cfg.get("dropout", 0)
    info["activation"] = model_cfg.get("activation", "unknown")
    info["norm_type"] = model_cfg.get("norm_type", "unknown")
    info["rotary_emb"] = model_cfg.get("rotary_emb", False)
    info["flash_attention"] = model_cfg.get("flash_attention", False)
    info["gradient_checkpointing"] = model_cfg.get("gradient_checkpointing", False)

    # Extract data config
    data_cfg = config_dict.get("data", {})
    info["tokenizer"] = data_cfg.get("tokenizer", "unknown")
    info["kmer_size"] = data_cfg.get("kmer_size", 0)
    info["seq_length"] = data_cfg.get("seq_length", 0)
    info["batch_size"] = data_cfg.get("batch_size", 0)
    info["mask_prob"] = data_cfg.get("mask_prob", 0.15)

    # Extract training config
    train_cfg = config_dict.get("training", {})
    info["run_name"] = train_cfg.get("run_name", "unknown")
    info["epochs"] = train_cfg.get("epochs", 0)
    info["lr"] = train_cfg.get("lr", 0)
    info["weight_decay"] = train_cfg.get("weight_decay", 0)
    info["lr_scheduler"] = train_cfg.get("lr_scheduler", "unknown")
    info["warmup_steps"] = train_cfg.get("warmup_steps", 0)
    info["mixed_precision"] = train_cfg.get("mixed_precision", "no")
    info["seed"] = train_cfg.get("seed", 42)

    # Count parameters from state dict
    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", {}),
    )

    total_params = 0
    for param_tensor in state_dict.values():
        if isinstance(param_tensor, torch.Tensor):
            total_params += param_tensor.numel()

    info["total_params"] = total_params
    info["model_size_mb"] = round(total_params * 4 / 1024 / 1024, 1)

    # Extract training metrics if present
    info["global_step"] = checkpoint.get("global_step", checkpoint.get("step", 0))
    info["best_metric"] = checkpoint.get("best_metric", None)
    info["epoch"] = checkpoint.get("epoch", None)

    # Check for training report
    checkpoint_dir = Path(checkpoint_path).parent
    report_path = checkpoint_dir / "training_report.json"
    if report_path.exists():
        try:
            with open(report_path) as fh:
                info["training_report"] = json.load(fh)
        except Exception:
            pass

    logger.info("Architecture: %s", info["architecture"])
    logger.info("Parameters: %s", f"{total_params:,}")
    logger.info("Model size: %.1f MB", info["model_size_mb"])

    return info


def load_metrics(metrics_path: str) -> Optional[Dict[str, Any]]:
    """Load evaluation metrics from a JSON report.

    Args:
        metrics_path: Path to benchmark_report.json or similar.

    Returns:
        Metrics dict or None if loading fails.
    """
    try:
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        logger.info("Loaded metrics from: %s", metrics_path)
        return metrics
    except Exception as exc:
        logger.warning("Failed to load metrics from %s: %s", metrics_path, exc)
        return None


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

def generate_model_card(
    model_info: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a markdown model card.

    Args:
        model_info: Model metadata from load_checkpoint_info.
        metrics: Optional evaluation metrics.

    Returns:
        Markdown string.
    """
    arch = model_info["architecture"]
    run_name = model_info["run_name"]
    date_str = datetime.now().strftime("%Y-%m-%d")

    sections = []

    # --- Header ---
    sections.append(f"# Model Card: Genova {run_name}\n")

    # --- Overview ---
    sections.append("## Model Overview\n")
    sections.append(f"- **Model name**: {run_name}")
    sections.append(f"- **Architecture**: {arch}")
    sections.append(f"- **Parameters**: {model_info['total_params']:,}")
    sections.append(f"- **Model size**: {model_info['model_size_mb']} MB (FP32)")
    sections.append(f"- **Date trained**: {date_str}")
    sections.append(f"- **Framework**: PyTorch")
    sections.append("")

    # --- Architecture Details ---
    sections.append("## Architecture\n")
    sections.append("| Component | Value |")
    sections.append("|---|---|")
    sections.append(f"| Architecture | {arch} |")
    sections.append(f"| Hidden dimension (d_model) | {model_info['d_model']} |")
    sections.append(f"| Number of layers | {model_info['n_layers']} |")
    sections.append(f"| Attention heads | {model_info['n_heads']} |")
    sections.append(f"| Feed-forward dimension | {model_info['d_ff']} |")
    sections.append(f"| Vocabulary size | {model_info['vocab_size']} |")
    sections.append(f"| Max sequence length | {model_info['max_position_embeddings']} |")
    sections.append(f"| Activation | {model_info['activation']} |")
    sections.append(f"| Normalization | {model_info['norm_type']} |")
    sections.append(f"| Rotary embeddings | {model_info['rotary_emb']} |")
    sections.append(f"| Flash attention | {model_info['flash_attention']} |")
    sections.append(f"| Gradient checkpointing | {model_info['gradient_checkpointing']} |")
    sections.append(f"| Dropout | {model_info['dropout']} |")
    sections.append("")

    # --- Training Details ---
    sections.append("## Training\n")
    sections.append("### Data\n")
    sections.append("- **Reference genome**: GRCh38 (Ensembl release 112)")
    sections.append(f"- **Tokenizer**: {model_info['tokenizer']}")
    if model_info["tokenizer"] == "kmer":
        sections.append(f"- **K-mer size**: {model_info['kmer_size']}")
    sections.append(f"- **Sequence length**: {model_info['seq_length']} bp")
    sections.append(f"- **MLM mask probability**: {model_info['mask_prob']}")
    sections.append("- **Train split**: chr1-chr17")
    sections.append("- **Validation split**: chr18-chr20")
    sections.append("- **Test split**: chr21-chr22")
    sections.append("")

    sections.append("### Hyperparameters\n")
    sections.append("| Parameter | Value |")
    sections.append("|---|---|")
    sections.append(f"| Epochs | {model_info['epochs']} |")
    sections.append(f"| Learning rate | {model_info['lr']} |")
    sections.append(f"| Weight decay | {model_info['weight_decay']} |")
    sections.append(f"| LR scheduler | {model_info['lr_scheduler']} |")
    sections.append(f"| Warmup steps | {model_info['warmup_steps']} |")
    sections.append(f"| Batch size | {model_info['batch_size']} |")
    sections.append(f"| Mixed precision | {model_info['mixed_precision']} |")
    sections.append(f"| Random seed | {model_info['seed']} |")
    sections.append("")

    # Training report if available
    training_report = model_info.get("training_report")
    if training_report:
        sections.append("### Training Results\n")
        sections.append(f"- **Total steps**: {training_report.get('total_steps', 'N/A')}")
        sections.append(f"- **Training time**: {training_report.get('training_time_s', 'N/A')}s")
        sections.append(f"- **GPUs used**: {training_report.get('gpus', 'N/A')}")

        final_metrics = training_report.get("final_metrics", {})
        if final_metrics:
            sections.append("")
            sections.append("| Metric | Value |")
            sections.append("|---|---|")
            for k, v in sorted(final_metrics.items()):
                if isinstance(v, float):
                    sections.append(f"| {k} | {v:.4f} |")
                else:
                    sections.append(f"| {k} | {v} |")
        sections.append("")

    # --- Evaluation Results ---
    if metrics:
        sections.append("## Evaluation Results\n")

        # Handle different metrics formats
        results = metrics.get("results", metrics.get("tasks", metrics))

        if isinstance(results, dict):
            sections.append("| Task | Metric | Score |")
            sections.append("|---|---|---|")

            for task_name, task_data in sorted(results.items()):
                if isinstance(task_data, dict):
                    for metric_name, value in sorted(task_data.items()):
                        if isinstance(value, (int, float)):
                            display_name = task_name.replace("_", " ").title()
                            sections.append(
                                f"| {display_name} | {metric_name} | {value:.4f} |"
                            )
                elif isinstance(task_data, (int, float)):
                    sections.append(
                        f"| {task_name} | score | {task_data:.4f} |"
                    )
            sections.append("")

    # --- Intended Use ---
    sections.append("## Intended Use\n")
    sections.append(
        "This model is a genomic foundation model pre-trained on the human "
        "reference genome (GRCh38) using masked language modeling (MLM). "
        "It is designed to be fine-tuned for downstream genomic tasks including:"
    )
    sections.append("")
    sections.append("- Promoter detection")
    sections.append("- Splice site prediction")
    sections.append("- Enhancer classification")
    sections.append("- Variant effect prediction")
    sections.append("- Chromatin accessibility prediction")
    sections.append("")

    # --- Limitations ---
    sections.append("## Limitations\n")
    sections.append(
        "- Trained only on the human reference genome (GRCh38); may not "
        "generalize well to other species without fine-tuning."
    )
    sections.append(
        "- Pre-training uses autosomes only (chr1-chr22); sex chromosomes "
        "and mitochondrial DNA are excluded."
    )
    sections.append(
        "- Variant effect predictions should be validated experimentally "
        "and not used as the sole basis for clinical decisions."
    )
    sections.append(
        "- Model performance depends on the quality and completeness of "
        "the reference genome assembly."
    )
    sections.append("")

    # --- Citation ---
    sections.append("## Citation\n")
    sections.append("```bibtex")
    sections.append("@software{genova2024,")
    sections.append(f"  title = {{Genova: A Genomic Foundation Model}},")
    sections.append(f"  year = {{{datetime.now().year}}},")
    sections.append(f"  architecture = {{{arch}}},")
    sections.append(f"  parameters = {{{model_info['total_params']:,}}},")
    sections.append("}")
    sections.append("```")
    sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a model card for a trained Genova model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python scripts/generate_model_card.py \\
              --model-path outputs/genova_small/best_model.pt \\
              --output outputs/genova_small/MODEL_CARD.md

          python scripts/generate_model_card.py \\
              --model-path outputs/genova_large/best_model.pt \\
              --metrics-path results/evaluation/benchmark_report.json \\
              --output outputs/genova_large/MODEL_CARD.md
        """),
    )

    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--metrics-path", type=str, default=None,
        help="Path to evaluation metrics JSON file (optional).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output path for the model card markdown file. "
            "Default: MODEL_CARD.md in the checkpoint directory."
        ),
    )

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model checkpoint not found: %s", model_path)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / "MODEL_CARD.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model info
    model_info = load_checkpoint_info(str(model_path))

    # Load metrics if provided
    metrics = None
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        if metrics_path.exists():
            metrics = load_metrics(str(metrics_path))
        else:
            logger.warning("Metrics file not found: %s", metrics_path)

    # Generate model card
    logger.info("Generating model card...")
    card = generate_model_card(model_info, metrics)

    # Write output
    with open(output_path, "w") as fh:
        fh.write(card)

    logger.info("Model card written to: %s", output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()

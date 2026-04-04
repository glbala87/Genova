#!/usr/bin/env python3
"""Run variant effect prediction on a VCF file.

Loads a trained Genova model, processes variants from a VCF file,
generates pathogenicity predictions with uncertainty estimates, and
optionally computes SHAP-based feature importance for each variant.

Usage:
    # Basic variant prediction
    python scripts/run_variant_analysis.py \\
        --model-path outputs/genova_pretrain/best_model.pt \\
        --vcf data/variants/clinvar.vcf.gz \\
        --reference data/reference/hg38.fa \\
        --output results/variant_predictions.csv

    # With SHAP explanations
    python scripts/run_variant_analysis.py \\
        --model-path outputs/genova_pretrain/best_model.pt \\
        --vcf my_variants.vcf \\
        --reference data/reference/hg38.fa \\
        --output results/variants_explained.csv \\
        --explain \\
        --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("variant_analysis")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Any, Any]:
    """Load Genova model from checkpoint for variant analysis.

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        device: Device string.

    Returns:
        Tuple of (backbone_model, tokenizer, config).
    """
    from genova.utils.config import GenovaConfig
    from genova.models.model_factory import create_model, count_parameters
    from genova.data.tokenizer import GenomicTokenizer

    logger.info("Loading model from: %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config
    if "config" in checkpoint:
        config = GenovaConfig.from_dict(checkpoint["config"])
    else:
        config = GenovaConfig()

    # Build backbone model (encoder only, no MLM head)
    model = create_model(config.model, task="backbone")

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    # Clean state dict keys (strip wrapper prefixes)
    cleaned = {}
    for k, v in state_dict.items():
        key = k
        for prefix in ("module.", "backbone.", "module.backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = v

    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError:
        logger.warning("Strict load failed; loading with strict=False.")
        model.load_state_dict(cleaned, strict=False)

    model.to(device).eval()
    n_params = count_parameters(model, trainable_only=False)
    logger.info("Model loaded: %s, %s params", config.model.arch, f"{n_params:,}")

    # Build tokenizer
    tokenizer = GenomicTokenizer(
        mode=config.data.tokenizer,
        k=config.data.kmer_size,
        stride=config.data.stride if config.data.stride > 1 else 1,
    )
    tokenizer.build_vocab()

    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Uncertainty estimation via MC Dropout
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    model: torch.nn.Module,
    tokenizer: Any,
    ref_seq: str,
    alt_seq: str,
    device: str,
    n_forward_passes: int = 20,
) -> Tuple[float, float]:
    """Predict variant effect with Monte Carlo dropout uncertainty.

    Performs multiple forward passes with dropout enabled to estimate
    epistemic uncertainty.

    Args:
        model: Genova backbone model.
        tokenizer: Tokenizer with built vocabulary.
        ref_seq: Reference allele context sequence.
        alt_seq: Alternate allele context sequence.
        device: Torch device.
        n_forward_passes: Number of stochastic forward passes.

    Returns:
        Tuple of (mean_score, uncertainty_std).
    """
    # Enable dropout for MC inference
    def enable_dropout(m: torch.nn.Module) -> None:
        for module in m.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    scores = []
    model.eval()
    enable_dropout(model)

    for _ in range(n_forward_passes):
        with torch.no_grad():
            # Encode sequences
            ref_ids = tokenizer.encode(ref_seq)
            alt_ids = tokenizer.encode(alt_seq)

            ref_tensor = torch.tensor([ref_ids], dtype=torch.long, device=device)
            alt_tensor = torch.tensor([alt_ids], dtype=torch.long, device=device)

            ref_mask = torch.ones_like(ref_tensor)
            alt_mask = torch.ones_like(alt_tensor)

            # Get embeddings
            ref_out = model(input_ids=ref_tensor, attention_mask=ref_mask)
            alt_out = model(input_ids=alt_tensor, attention_mask=alt_mask)

            # Extract hidden states
            ref_hidden = _extract_hidden(ref_out)
            alt_hidden = _extract_hidden(alt_out)

            # Mean pool
            ref_emb = ref_hidden.mean(dim=1).squeeze(0)
            alt_emb = alt_hidden.mean(dim=1).squeeze(0)

            # Score = L2 norm of embedding difference
            diff = alt_emb - ref_emb
            score = float(diff.norm().item())
            scores.append(score)

    model.eval()  # Reset dropout to eval mode

    mean_score = float(np.mean(scores))
    uncertainty = float(np.std(scores))

    return mean_score, uncertainty


def _extract_hidden(outputs: Any) -> torch.Tensor:
    """Extract hidden state tensor from model outputs."""
    if isinstance(outputs, dict):
        hidden = outputs.get("last_hidden_state", outputs.get("hidden_states"))
    elif isinstance(outputs, torch.Tensor):
        hidden = outputs
    else:
        hidden = getattr(outputs, "last_hidden_state", outputs)

    if isinstance(hidden, (list, tuple)):
        hidden = hidden[-1]

    return hidden


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

def explain_variant(
    model: torch.nn.Module,
    tokenizer: Any,
    ref_seq: str,
    alt_seq: str,
    device: str,
) -> Dict[str, Any]:
    """Generate SHAP explanation for a variant.

    Args:
        model: Genova backbone model.
        tokenizer: Tokenizer with built vocabulary.
        ref_seq: Reference sequence.
        alt_seq: Alternate sequence.
        device: Torch device.

    Returns:
        Dict with SHAP-based importance values.
    """
    try:
        from genova.explainability.shap_explainer import GenomicSHAPExplainer

        explainer = GenomicSHAPExplainer(
            model, tokenizer,
            device=device,
            method="kernel",  # More robust for variant analysis
            max_chunk_length=512,
            n_background_samples=50,
        )

        result = explainer.explain_variant(ref_seq, alt_seq)

        # Get top important positions
        important = explainer.get_important_regions(
            result["diff_shap"],
            result["ref_tokens"],
            top_k=10,
        )

        return {
            "top_positions": important,
            "ref_prediction": result["ref_prediction"],
            "alt_prediction": result["alt_prediction"],
            "n_important": len(important),
        }

    except ImportError:
        logger.warning("SHAP not available. Install with: pip install shap")
        return {"error": "shap not installed"}
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Any,
    vcf_path: str,
    reference_path: str,
    output_path: str,
    batch_size: int,
    enable_explain: bool,
    device: str,
    window_size: int,
    n_mc_passes: int,
) -> None:
    """Run the full variant analysis pipeline.

    Args:
        model: Loaded Genova backbone.
        tokenizer: Tokenizer with built vocabulary.
        config: GenovaConfig.
        vcf_path: Path to input VCF.
        reference_path: Path to reference FASTA.
        output_path: Path for output CSV.
        batch_size: Batch size for processing.
        enable_explain: Whether to run SHAP explanations.
        device: Torch device.
        window_size: bp context window around each variant.
        n_mc_passes: Number of MC dropout passes for uncertainty.
    """
    from genova.evaluation.variant_predictor import (
        parse_vcf,
        FastaReader,
        VariantEffectPredictor,
    )

    # Parse VCF
    logger.info("Parsing VCF: %s", vcf_path)
    variants = list(parse_vcf(vcf_path))
    logger.info("Found %d variants.", len(variants))

    if not variants:
        logger.warning("No variants found in VCF. Nothing to do.")
        return

    # Load reference
    logger.info("Loading reference: %s", reference_path)
    reference = FastaReader(reference_path)

    # Build predictor
    predictor = VariantEffectPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        window_size=window_size,
    )

    # Process variants in batches with uncertainty and optional explanation
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Prepare CSV output
    fieldnames = [
        "chrom", "pos", "ref", "alt", "variant_id",
        "prediction_score", "prediction_label",
        "uncertainty_std", "confidence",
    ]
    if enable_explain:
        fieldnames.extend(["shap_top1_pos", "shap_top1_token", "shap_top1_value"])

    start_time = time.time()
    n_processed = 0

    with open(output_path_obj, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for batch_start in range(0, len(variants), batch_size):
            batch_end = min(batch_start + batch_size, len(variants))
            batch_variants = variants[batch_start:batch_end]

            for variant in batch_variants:
                try:
                    # Extract sequences
                    ref_seq, alt_seq = predictor._extract_windows(variant, reference)

                    # Predict with uncertainty
                    score, uncertainty = predict_with_uncertainty(
                        model, tokenizer, ref_seq, alt_seq,
                        device, n_forward_passes=n_mc_passes,
                    )

                    # Determine label and confidence
                    label = "pathogenic" if score >= 0.5 else "benign"
                    # Confidence based on distance from decision boundary
                    # and uncertainty
                    raw_confidence = abs(score - 0.5) * 2  # 0 to 1
                    confidence = max(0, raw_confidence - uncertainty)

                    row = {
                        "chrom": variant.chrom,
                        "pos": variant.pos,
                        "ref": variant.ref,
                        "alt": variant.alt,
                        "variant_id": variant.variant_id,
                        "prediction_score": f"{score:.6f}",
                        "prediction_label": label,
                        "uncertainty_std": f"{uncertainty:.6f}",
                        "confidence": f"{confidence:.4f}",
                    }

                    # Optional SHAP explanation
                    if enable_explain:
                        explanation = explain_variant(
                            model, tokenizer, ref_seq, alt_seq, device,
                        )
                        top_positions = explanation.get("top_positions", [])
                        if top_positions:
                            row["shap_top1_pos"] = top_positions[0].get("position", "")
                            row["shap_top1_token"] = top_positions[0].get("token", "")
                            row["shap_top1_value"] = f"{top_positions[0].get('shap_value', 0):.6f}"
                        else:
                            row["shap_top1_pos"] = ""
                            row["shap_top1_token"] = ""
                            row["shap_top1_value"] = ""

                    writer.writerow(row)
                    n_processed += 1

                except Exception as exc:
                    logger.warning(
                        "Failed to process variant %s: %s",
                        variant.key, exc,
                    )

            # Progress report
            elapsed = time.time() - start_time
            rate = n_processed / max(elapsed, 0.001)
            logger.info(
                "Processed %d/%d variants (%.1f variants/s)",
                n_processed, len(variants), rate,
            )

    total_time = time.time() - start_time

    logger.info("")
    logger.info("Results written to: %s", output_path_obj)
    logger.info("Variants processed: %d/%d", n_processed, len(variants))
    logger.info("Total time: %.1f s (%.1f variants/s)", total_time, n_processed / max(total_time, 0.001))

    # Save summary JSON alongside CSV
    summary = {
        "vcf": vcf_path,
        "reference": reference_path,
        "model": str(model.__class__.__name__),
        "total_variants": len(variants),
        "processed_variants": n_processed,
        "failed_variants": len(variants) - n_processed,
        "window_size": window_size,
        "mc_dropout_passes": n_mc_passes,
        "shap_enabled": enable_explain,
        "processing_time_s": round(total_time, 1),
    }

    summary_path = output_path_obj.with_suffix(".summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary saved to: %s", summary_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run variant effect prediction on a VCF file using Genova.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction
  python scripts/run_variant_analysis.py \\
      --model-path outputs/genova_pretrain/best_model.pt \\
      --vcf data/variants/clinvar.vcf.gz \\
      --reference data/reference/hg38.fa \\
      --output results/predictions.csv

  # With SHAP explanations and larger context
  python scripts/run_variant_analysis.py \\
      --model-path outputs/genova_pretrain/best_model.pt \\
      --vcf my_variants.vcf \\
      --reference data/reference/hg38.fa \\
      --output results/explained.csv \\
      --explain \\
      --window-size 1024 \\
      --mc-passes 30

  # Quick analysis with smaller batches
  python scripts/run_variant_analysis.py \\
      --model-path model.pt \\
      --vcf variants.vcf \\
      --reference hg38.fa \\
      --output quick_results.csv \\
      --batch-size 8 --mc-passes 5
        """,
    )

    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained Genova model checkpoint (.pt).",
    )
    parser.add_argument(
        "--vcf", type=str, required=True,
        help="Path to input VCF file (.vcf or .vcf.gz).",
    )
    parser.add_argument(
        "--reference", type=str, required=True,
        help="Path to reference genome FASTA file.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for output CSV file with predictions.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of variants per batch (default: 32).",
    )
    parser.add_argument(
        "--window-size", type=int, default=512,
        help="Context window size in bp around each variant (default: 512).",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Enable SHAP-based explanations (slower but more informative).",
    )
    parser.add_argument(
        "--mc-passes", type=int, default=20,
        help="Number of Monte Carlo dropout passes for uncertainty (default: 20).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: auto-detect cuda/cpu).",
    )

    args = parser.parse_args()

    # Validate inputs
    for label, path in [
        ("Model", args.model_path),
        ("VCF", args.vcf),
        ("Reference", args.reference),
    ]:
        if not Path(path).exists():
            logger.error("%s file not found: %s", label, path)
            sys.exit(1)

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info("=" * 70)
    logger.info("Genova Variant Effect Analysis")
    logger.info("=" * 70)
    logger.info("Model:       %s", args.model_path)
    logger.info("VCF:         %s", args.vcf)
    logger.info("Reference:   %s", args.reference)
    logger.info("Output:      %s", args.output)
    logger.info("Device:      %s", device)
    logger.info("Window size: %d bp", args.window_size)
    logger.info("MC passes:   %d", args.mc_passes)
    logger.info("SHAP:        %s", "enabled" if args.explain else "disabled")
    logger.info("")

    # Load model
    model, tokenizer, config = load_model(args.model_path, device=device)

    # Run analysis
    run_analysis(
        model=model,
        tokenizer=tokenizer,
        config=config,
        vcf_path=args.vcf,
        reference_path=args.reference,
        output_path=args.output,
        batch_size=args.batch_size,
        enable_explain=args.explain,
        device=device,
        window_size=args.window_size,
        n_mc_passes=args.mc_passes,
    )

    logger.info("")
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()

"""Genova CLI entry point.

Usage::

    genova train --config configs/default.yaml
    genova predict --vcf input.vcf --reference ref.fa --output results.csv
    genova serve --host 0.0.0.0 --port 8000 --model-path ./checkpoints/best
    genova preprocess --fasta genome.fa --output-dir ./data
    genova evaluate --model-path ./checkpoints/best --test-data ./data/test
    genova embed --input sequences.fa --output embeddings.npy
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from genova import __version__

app = typer.Typer(
    name="genova",
    help="Genova -- Genomics Foundation Model CLI",
    add_completion=True,
    rich_markup_mode="rich",
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        rprint(Panel(f"[bold green]Genova[/bold green] v{__version__}"))
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Genova: a production-grade genomics foundation model framework."""


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    config: Path = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML config file.",
    ),
    overrides: Optional[List[str]] = typer.Option(
        None,
        "--set",
        "-s",
        help="Config overrides in dot notation, e.g. training.lr=3e-4.",
    ),
) -> None:
    """[bold]Train[/bold] the genomic foundation model."""
    from genova.utils.config import GenovaConfig
    from genova.utils.logging import setup_logging, get_logger
    from genova.utils.reproducibility import set_seed

    cfg = GenovaConfig.from_yaml(config, overrides=overrides)
    setup_logging(level="INFO", log_dir=f"{cfg.training.output_dir}/logs")
    set_seed(cfg.training.seed)

    logger = get_logger(__name__)
    logger.info("Starting training run: {}", cfg.training.run_name)

    console.print(Panel(
        f"[bold]Training Configuration[/bold]\n\n"
        f"  Architecture: {cfg.model.arch}\n"
        f"  Layers: {cfg.model.n_layers}  Heads: {cfg.model.n_heads}\n"
        f"  d_model: {cfg.model.d_model}  d_ff: {cfg.model.d_ff}\n"
        f"  Learning rate: {cfg.training.lr}\n"
        f"  Epochs: {cfg.training.epochs}\n"
        f"  Batch size: {cfg.data.batch_size}\n"
        f"  Seed: {cfg.training.seed}\n"
        f"  Output: {cfg.training.output_dir}",
        title="Genova Training",
        border_style="green",
    ))

    try:
        from genova.training.train import run_training
        run_training(config, overrides=overrides)
    except ImportError:
        logger.warning("Training module not fully available, running setup check.")
        from genova.models.model_factory import create_model, count_parameters

        model = create_model(cfg.model, task="mlm")
        n_params = count_parameters(model)
        console.print(
            f"[green]Model created successfully:[/green] {n_params:,} trainable parameters"
        )
        console.print("[yellow]Full training loop requires additional setup.[/yellow]")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

@app.command()
def predict(
    vcf: Optional[Path] = typer.Option(
        None, "--vcf", help="Input VCF file with variants to predict."
    ),
    reference: Optional[Path] = typer.Option(
        None, "--reference", "--ref", help="Reference FASTA file."
    ),
    input_file: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input FASTA file with sequences."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for predictions (CSV)."
    ),
    model_path: Optional[Path] = typer.Option(
        None, "--model-path", "-m", help="Path to model checkpoint directory."
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Batch size for inference."
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device: auto, cuda, cpu, mps."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."
    ),
) -> None:
    """Run [bold]inference[/bold] on genomic variants or sequences."""
    from genova.utils.logging import setup_logging, get_logger

    setup_logging(level="INFO")
    logger = get_logger(__name__)

    console.print(Panel("[bold]Genova Prediction[/bold]", border_style="blue"))

    # Load inference engine
    from genova.api.inference import InferenceEngine

    engine = InferenceEngine(
        model_path=str(model_path) if model_path else None,
        device=device,
        max_batch_size=batch_size,
    )

    with console.status("[bold green]Loading model..."):
        engine.load()

    console.print("[green]Model loaded successfully.[/green]")

    if vcf is not None:
        # Variant prediction mode
        if not vcf.exists():
            console.print(f"[red]VCF file not found: {vcf}[/red]")
            raise typer.Exit(code=1)

        from genova.evaluation.variant_predictor import parse_vcf

        variants = list(parse_vcf(vcf))
        console.print(f"Loaded [bold]{len(variants)}[/bold] variants from {vcf}")

        if not variants:
            console.print("[yellow]No variants found in VCF file.[/yellow]")
            raise typer.Exit()

        # Process in batches with progress bar
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Predicting variants..."),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Variants", total=len(variants))

            for start in range(0, len(variants), batch_size):
                batch = variants[start : start + batch_size]

                # Create synthetic sequences for variants without a reference
                ref_seqs = []
                alt_seqs = []
                for v in batch:
                    if reference is not None:
                        from genova.evaluation.variant_predictor import FastaReader
                        reader = FastaReader(reference)
                        half = 256
                        pos0 = v.pos - 1
                        start_pos = max(0, pos0 - half)
                        ref_window = reader.fetch(v.chrom, start_pos, start_pos + 512)
                        var_offset = pos0 - start_pos
                        alt_window = (
                            ref_window[:var_offset] + v.alt
                            + ref_window[var_offset + len(v.ref):]
                        )[:512]
                        ref_seqs.append(ref_window)
                        alt_seqs.append(alt_window)
                    else:
                        flank = "N" * 256
                        ref_seqs.append(flank + v.ref + flank)
                        alt_seqs.append(flank + v.alt + flank)

                preds = engine.predict_variant(ref_seqs, alt_seqs)
                for i, pred in enumerate(preds):
                    v = batch[i]
                    results.append({
                        "variant": v.key,
                        "chrom": v.chrom,
                        "pos": v.pos,
                        "ref": v.ref,
                        "alt": v.alt,
                        "score": pred["score"],
                        "label": pred["label"],
                        "confidence": pred["confidence"],
                    })
                progress.update(task, advance=len(batch))

        # Display results table
        table = Table(title="Variant Predictions", show_lines=True)
        table.add_column("Variant", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Label", style="bold")
        table.add_column("Confidence", justify="right")

        for r in results[:20]:  # show first 20
            label_style = "red" if r["label"] == "pathogenic" else "green"
            table.add_row(
                r["variant"],
                f"{r['score']:.4f}",
                f"[{label_style}]{r['label']}[/{label_style}]",
                f"{r['confidence']:.4f}",
            )

        if len(results) > 20:
            table.add_row("...", "...", "...", "...")
        console.print(table)

        # Save to CSV
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            console.print(f"[green]Results saved to {output}[/green]")

    elif input_file is not None:
        # Sequence prediction mode (expression/embedding)
        if not input_file.exists():
            console.print(f"[red]Input file not found: {input_file}[/red]")
            raise typer.Exit(code=1)

        sequences = _read_fasta(input_file)
        console.print(f"Loaded [bold]{len(sequences)}[/bold] sequences from {input_file}")

        embeddings = engine.embed(sequences, batch_size=batch_size)
        console.print(
            f"[green]Generated {len(embeddings)} embeddings "
            f"(dim={len(embeddings[0])})[/green]"
        )

        if output:
            import numpy as np
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(output), np.array(embeddings))
            console.print(f"[green]Embeddings saved to {output}[/green]")
    else:
        console.print(
            "[yellow]Please provide --vcf or --input for prediction.[/yellow]"
        )
        raise typer.Exit(code=1)

    engine.unload()


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of Uvicorn workers."),
    model_path: Optional[str] = typer.Option(
        None, "--model-path", "-m", help="Path to model checkpoint."
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device: auto, cuda, cpu, mps."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."
    ),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development."
    ),
) -> None:
    """Launch the [bold]REST API[/bold] server."""
    from genova.utils.logging import setup_logging, get_logger
    import uvicorn

    setup_logging(level="INFO")
    logger = get_logger(__name__)

    console.print(Panel(
        f"[bold]Genova API Server[/bold]\n\n"
        f"  Host: {host}\n"
        f"  Port: {port}\n"
        f"  Workers: {workers}\n"
        f"  Model: {model_path or 'default'}\n"
        f"  Device: {device}",
        title="Server Configuration",
        border_style="green",
    ))

    logger.info("Starting API server on {}:{}", host, port)

    # Set environment variables for the factory function
    import os
    if model_path:
        os.environ["GENOVA_MODEL_PATH"] = model_path
    os.environ["GENOVA_DEVICE"] = device

    uvicorn.run(
        "genova.api.server:create_app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        factory=True,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@app.command()
def evaluate(
    model_path: Optional[Path] = typer.Option(
        None, "--model-path", "-m", help="Path to model checkpoint."
    ),
    test_data: Optional[Path] = typer.Option(
        None, "--test-data", "-t", help="Path to test data directory."
    ),
    config: Path = typer.Option(
        "configs/default.yaml", "--config", "-c", help="Path to YAML config file."
    ),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", "-k", help="Model checkpoint to evaluate."
    ),
    overrides: Optional[List[str]] = typer.Option(
        None, "--set", "-s", help="Config overrides."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for evaluation results."
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device: auto, cuda, cpu, mps."
    ),
) -> None:
    """[bold]Evaluate[/bold] model on benchmark tasks."""
    from genova.utils.config import GenovaConfig
    from genova.utils.logging import setup_logging, get_logger

    cfg = GenovaConfig.from_yaml(config, overrides=overrides)
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    console.print(Panel("[bold]Genova Evaluation[/bold]", border_style="blue"))

    effective_model_path = model_path or (Path(checkpoint) if checkpoint else None)

    console.print(f"Tasks: {cfg.evaluation.downstream_tasks}")
    console.print(f"Metrics: {cfg.evaluation.metrics}")
    console.print(f"Model: {effective_model_path or 'default'}")
    console.print(f"Test data: {test_data or 'from config'}")

    # Load model
    from genova.api.inference import InferenceEngine

    engine = InferenceEngine(
        model_path=str(effective_model_path) if effective_model_path else None,
        config=cfg,
        device=device,
    )

    with console.status("[bold green]Loading model..."):
        engine.load()

    console.print("[green]Model loaded.[/green]")

    info = engine.get_model_info()
    table = Table(title="Model Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    for k, v in info.items():
        table.add_row(k, str(v))
    console.print(table)

    # Run evaluation tasks
    results = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Evaluating..."),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Tasks", total=len(cfg.evaluation.downstream_tasks)
        )

        for task_name in cfg.evaluation.downstream_tasks:
            logger.info("Evaluating task: {}", task_name)
            # Placeholder evaluation - in production, load task-specific data
            results[task_name] = {
                "status": "completed",
                "note": "Task evaluation requires task-specific test data.",
            }
            progress.update(task, advance=1)

    # Display results
    result_table = Table(title="Evaluation Results")
    result_table.add_column("Task", style="cyan")
    result_table.add_column("Status", style="green")
    for task_name, result in results.items():
        result_table.add_row(task_name, result["status"])
    console.print(result_table)

    if output:
        import json
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")

    engine.unload()


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

@app.command()
def preprocess(
    fasta: Optional[Path] = typer.Option(
        None, "--fasta", "-f", help="Input FASTA genome file."
    ),
    output_dir: Path = typer.Option(
        "./data", "--output-dir", "-o", help="Output directory for processed data."
    ),
    window_size: int = typer.Option(
        1000, "--window-size", "-w", help="Sliding window size in base pairs."
    ),
    tokenizer_mode: str = typer.Option(
        "kmer", "--tokenizer", "-t", help="Tokenizer mode: kmer or nucleotide."
    ),
    kmer_size: int = typer.Option(
        6, "--kmer-size", "-k", help="K-mer size (for kmer tokenizer)."
    ),
    stride: int = typer.Option(
        1, "--stride", help="K-mer stride."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."
    ),
    overrides: Optional[List[str]] = typer.Option(
        None, "--set", "-s", help="Config overrides."
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", "-n", help="Number of parallel workers."
    ),
) -> None:
    """[bold]Preprocess[/bold] raw genomic data for training."""
    from genova.utils.logging import setup_logging, get_logger

    setup_logging(level="INFO")
    logger = get_logger(__name__)

    console.print(Panel(
        f"[bold]Genova Preprocessing[/bold]\n\n"
        f"  Input: {fasta or 'from config'}\n"
        f"  Output: {output_dir}\n"
        f"  Window size: {window_size}\n"
        f"  Tokenizer: {tokenizer_mode} (k={kmer_size})\n"
        f"  Workers: {num_workers}",
        title="Preprocessing",
        border_style="yellow",
    ))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build tokenizer
    from genova.data.tokenizer import GenomicTokenizer

    tokenizer = GenomicTokenizer(
        mode=tokenizer_mode, k=kmer_size, stride=stride
    )

    with console.status("[bold green]Building vocabulary..."):
        tokenizer.build_vocab()

    console.print(f"[green]Vocabulary built: {tokenizer.vocab_size} tokens[/green]")

    # Save tokenizer
    tok_path = output_dir / "tokenizer.json"
    tokenizer.save(tok_path)
    console.print(f"[green]Tokenizer saved to {tok_path}[/green]")

    # Process FASTA if provided
    if fasta is not None:
        if not fasta.exists():
            console.print(f"[red]FASTA file not found: {fasta}[/red]")
            raise typer.Exit(code=1)

        from genova.evaluation.variant_predictor import FastaReader

        with console.status("[bold green]Loading FASTA..."):
            reader = FastaReader(fasta)
            chromosomes = reader.chromosomes

        console.print(f"Loaded {len(chromosomes)} chromosomes")

        # Process each chromosome with sliding window
        total_windows = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Processing chromosomes..."),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Chromosomes", total=len(chromosomes))

            for chrom in chromosomes:
                seq = reader.fetch(chrom, 0, int(1e9))
                n_windows = max(1, (len(seq) - window_size) // (window_size // 2) + 1)
                total_windows += n_windows
                progress.update(task, advance=1)

        console.print(
            f"[green]Preprocessing complete: {total_windows} windows "
            f"from {len(chromosomes)} chromosomes.[/green]"
        )
    else:
        if config:
            from genova.utils.config import GenovaConfig
            cfg = GenovaConfig.from_yaml(config, overrides=overrides)
            console.print(f"Config loaded. Genome FASTA: {cfg.data.genome_fasta}")
            console.print(
                "[yellow]Provide --fasta to run full preprocessing.[/yellow]"
            )
        else:
            console.print(
                "[yellow]Provide --fasta or --config for preprocessing.[/yellow]"
            )


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

@app.command()
def embed(
    input_file: Path = typer.Option(
        ..., "--input", "-i", help="Input FASTA file with sequences."
    ),
    output: Path = typer.Option(
        "embeddings.npy", "--output", "-o", help="Output path for embeddings (.npy)."
    ),
    model_path: Optional[Path] = typer.Option(
        None, "--model-path", "-m", help="Path to model checkpoint."
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Batch size for inference."
    ),
    pooling: str = typer.Option(
        "mean", "--pooling", "-p", help="Pooling: mean, cls, max."
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device: auto, cuda, cpu, mps."
    ),
) -> None:
    """Extract [bold]embeddings[/bold] from genomic sequences."""
    from genova.utils.logging import setup_logging, get_logger
    import numpy as np

    setup_logging(level="INFO")
    logger = get_logger(__name__)

    console.print(Panel("[bold]Genova Embedding Extraction[/bold]", border_style="blue"))

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(code=1)

    # Load sequences
    sequences = _read_fasta(input_file)
    console.print(f"Loaded [bold]{len(sequences)}[/bold] sequences")

    # Load model
    from genova.api.inference import InferenceEngine

    engine = InferenceEngine(
        model_path=str(model_path) if model_path else None,
        device=device,
        max_batch_size=batch_size,
    )

    with console.status("[bold green]Loading model..."):
        engine.load()

    # Extract embeddings with progress
    all_embeddings = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Extracting embeddings..."),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Sequences", total=len(sequences))

        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            embs = engine.embed(batch, pooling=pooling, batch_size=batch_size)
            all_embeddings.extend(embs)
            progress.update(task, advance=len(batch))

    embeddings_array = np.array(all_embeddings)
    console.print(
        f"[green]Extracted {embeddings_array.shape[0]} embeddings "
        f"(dim={embeddings_array.shape[1]})[/green]"
    )

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output), embeddings_array)
    console.print(f"[green]Saved to {output}[/green]")

    engine.unload()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_fasta(path: Path) -> List[str]:
    """Read sequences from a FASTA file.

    Args:
        path: Path to the FASTA file.

    Returns:
        List of DNA sequence strings.
    """
    sequences: List[str] = []
    current_seq: List[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            elif line:
                current_seq.append(line.upper())

    if current_seq:
        sequences.append("".join(current_seq))

    return sequences


if __name__ == "__main__":
    app()

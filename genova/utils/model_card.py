"""Model card generation following HuggingFace / Google Model Card format.

Auto-extracts architecture details, parameter counts, and training
configuration, then renders a comprehensive Markdown document.

Usage::

    from genova.utils.model_card import ModelCard

    card = ModelCard.from_model(model, config, metrics={"auroc": 0.95})
    card.save("model_card.md")
    print(card.to_markdown())
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """Structured model card with auto-extraction and Markdown export.

    Attributes
    ----------
    model_name : str
        Human-readable model name.
    version : str
        Model version string.
    description : str
        Short description of what the model does.
    architecture : dict
        Architecture details (auto-extracted when using :meth:`from_model`).
    training : dict
        Training configuration summary.
    metrics : dict
        Evaluation metrics and benchmark results.
    intended_use : str
        Description of intended use cases.
    limitations : str
        Known limitations.
    ethical_considerations : str
        Ethical considerations and potential biases.
    citation : str
        BibTeX or plain-text citation.
    license : str
        Model licence.
    authors : list of str
        Model authors.
    extra_sections : dict
        Additional free-form sections (title -> content).
    """

    model_name: str = "Genova"
    version: str = "0.1.0"
    description: str = (
        "Genova is a genomics foundation model for DNA sequence analysis, "
        "variant effect prediction, gene expression prediction, and "
        "methylation prediction."
    )
    architecture: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    intended_use: str = (
        "Research use in computational genomics including variant effect "
        "prediction, gene expression modelling, methylation prediction, "
        "and DNA sequence embedding extraction."
    )
    limitations: str = (
        "- Trained primarily on the human reference genome; performance on "
        "non-human organisms may be reduced.\n"
        "- Variant effect predictions are correlative and should not be used "
        "as sole evidence for clinical decisions.\n"
        "- Performance may vary across populations and genomic regions with "
        "different GC content or repeat density.\n"
        "- Maximum input sequence length is constrained by the model's "
        "positional encoding."
    )
    ethical_considerations: str = (
        "- **Population bias**: Model performance should be validated across "
        "diverse ancestral populations before deployment.\n"
        "- **Clinical use**: This model is intended for research only and "
        "must not be used for clinical diagnosis without proper validation.\n"
        "- **Privacy**: When processing patient genomic data, ensure "
        "compliance with local regulations (HIPAA, GDPR, etc.).\n"
        "- **Dual use**: Sequence generation capabilities should be monitored "
        "to prevent misuse in synthetic biology."
    )
    citation: str = ""
    license: str = "Apache-2.0"
    authors: List[str] = field(default_factory=lambda: ["Genova Team"])
    extra_sections: Dict[str, str] = field(default_factory=dict)

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: Any,
        config: Any = None,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "ModelCard":
        """Create a ModelCard by auto-extracting info from a model and config.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model (or any object with ``parameters()``).
        config : GenovaConfig or dict, optional
            Training / model configuration.
        metrics : dict, optional
            Evaluation metrics to include.
        **kwargs
            Override any ModelCard field.

        Returns
        -------
        ModelCard
        """
        arch_info = cls._extract_architecture(model)
        training_info = cls._extract_training(config) if config else {}

        card = cls(
            architecture=arch_info,
            training=training_info,
            metrics=metrics or {},
            **kwargs,
        )
        return card

    # -- extraction helpers --------------------------------------------------

    @staticmethod
    def _extract_architecture(model: Any) -> Dict[str, Any]:
        """Extract architecture details from a PyTorch model."""
        info: Dict[str, Any] = {}

        # Parameter count
        try:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            info["total_parameters"] = total
            info["trainable_parameters"] = trainable
            info["parameter_size_mb"] = round(
                sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 2
            )
        except Exception:
            info["total_parameters"] = "unknown"

        # Architecture name
        info["class_name"] = type(model).__name__

        # Try to get config attributes
        for attr in ("d_model", "n_layers", "n_heads", "d_ff", "vocab_size",
                     "max_position_embeddings", "dropout", "activation",
                     "norm_type", "arch"):
            if hasattr(model, attr):
                info[attr] = getattr(model, attr)
            elif hasattr(model, "config") and hasattr(model.config, attr):
                info[attr] = getattr(model.config, attr)

        # Layer types
        try:
            layer_types: Dict[str, int] = {}
            for name, module in model.named_modules():
                t = type(module).__name__
                layer_types[t] = layer_types.get(t, 0) + 1
            info["layer_types"] = dict(sorted(layer_types.items(), key=lambda x: -x[1])[:15])
        except Exception:
            pass

        return info

    @staticmethod
    def _extract_training(config: Any) -> Dict[str, Any]:
        """Extract training details from a config object or dict."""
        if isinstance(config, dict):
            return config

        info: Dict[str, Any] = {}
        # Try GenovaConfig
        try:
            if hasattr(config, "to_dict"):
                d = config.to_dict()
                if "training" in d:
                    info.update(d["training"])
                if "data" in d:
                    info["data"] = d["data"]
            elif hasattr(config, "training"):
                from dataclasses import asdict

                info.update(asdict(config.training))
                if hasattr(config, "data"):
                    info["data"] = asdict(config.data)
        except Exception:
            info["raw"] = str(config)

        return info

    # -- Markdown export -----------------------------------------------------

    def to_markdown(self) -> str:
        """Render the model card as a Markdown string."""
        lines: List[str] = []

        # Header
        lines.append(f"# Model Card: {self.model_name}")
        lines.append("")
        lines.append(f"**Version:** {self.version}  ")
        lines.append(f"**License:** {self.license}  ")
        lines.append(f"**Authors:** {', '.join(self.authors)}  ")
        lines.append(f"**Date:** {datetime.date.today().isoformat()}")
        lines.append("")

        # Description
        lines.append("## Model Description")
        lines.append("")
        lines.append(self.description)
        lines.append("")

        # Architecture
        if self.architecture:
            lines.append("## Architecture")
            lines.append("")
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for k, v in self.architecture.items():
                if k == "layer_types":
                    continue
                display = f"{v:,}" if isinstance(v, int) else str(v)
                lines.append(f"| {k.replace('_', ' ').title()} | {display} |")
            lines.append("")

            if "layer_types" in self.architecture:
                lines.append("### Layer Composition")
                lines.append("")
                lines.append("| Layer Type | Count |")
                lines.append("|------------|-------|")
                for lt, count in self.architecture["layer_types"].items():
                    lines.append(f"| {lt} | {count} |")
                lines.append("")

        # Training
        if self.training:
            lines.append("## Training Details")
            lines.append("")
            lines.append("```yaml")
            for k, v in self.training.items():
                if isinstance(v, dict):
                    lines.append(f"{k}:")
                    for kk, vv in v.items():
                        lines.append(f"  {kk}: {vv}")
                else:
                    lines.append(f"{k}: {v}")
            lines.append("```")
            lines.append("")

        # Metrics
        if self.metrics:
            lines.append("## Evaluation Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in self.metrics.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        val = f"{vv:.4f}" if isinstance(vv, float) else str(vv)
                        lines.append(f"| {k}/{kk} | {val} |")
                else:
                    val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    lines.append(f"| {k} | {val} |")
            lines.append("")

        # Intended use
        lines.append("## Intended Use")
        lines.append("")
        lines.append(self.intended_use)
        lines.append("")

        # Limitations
        lines.append("## Limitations")
        lines.append("")
        lines.append(self.limitations)
        lines.append("")

        # Ethical considerations
        lines.append("## Ethical Considerations")
        lines.append("")
        lines.append(self.ethical_considerations)
        lines.append("")

        # Citation
        if self.citation:
            lines.append("## Citation")
            lines.append("")
            lines.append("```bibtex")
            lines.append(self.citation)
            lines.append("```")
            lines.append("")

        # Extra sections
        for title, content in self.extra_sections.items():
            lines.append(f"## {title}")
            lines.append("")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)

    # -- I/O -----------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Write the model card to a Markdown file.

        Parameters
        ----------
        path : str or Path
            Output file path (typically ``model_card.md``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        logger.info("Model card saved to {}", path)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the model card to a dictionary."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelCard":
        """Construct a ModelCard from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save_json(self, path: Union[str, Path]) -> None:
        """Write the model card as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        logger.info("Model card JSON saved to {}", path)

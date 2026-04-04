"""Configuration management for Genova.

Provides YAML-based configuration loading with dataclass validation,
nested section support, and CLI override merging.

Example::

    config = GenovaConfig.from_yaml("configs/default.yaml")
    config = GenovaConfig.from_yaml("configs/default.yaml", overrides=["training.lr=1e-4"])
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    genome_fasta: str = ""
    train_regions: str = ""
    val_regions: str = ""
    test_regions: str = ""
    seq_length: int = 512
    max_tokens: int = 1024
    tokenizer: str = "kmer"
    kmer_size: int = 6
    stride: int = 1
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    augmentations: List[str] = field(default_factory=lambda: ["reverse_complement"])
    mask_prob: float = 0.15
    mask_token_id: int = 4
    vocab_size: int = 4096
    cache_dir: str = ".cache/genova"


@dataclass
class ModelConfig:
    """Configuration for the genomic foundation model architecture.

    New optional features (all off by default):

    - ``flash_attention``: Use Flash Attention v2 when the ``flash_attn``
      package is installed.
    - ``rotary_emb``: Use Rotary Positional Embeddings (RoPE) on Q/K.
    - ``norm_type``: ``"layernorm"`` (default) or ``"rmsnorm"`` for
      faster Root Mean Square normalisation.
    - ``rope_base_freq``: Base frequency for RoPE (default 10000).
    - ``pos_encoding``: Positional encoding type.  ``"learned"`` (default),
      ``"sinusoidal"``, ``"sinusoidal+learned"``, or ``"rope"``.
    - ``use_kv_cache``: Enable KV-cache for autoregressive generation.
    """

    arch: str = "transformer"
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 2048
    activation: str = "gelu"
    norm_type: str = "layernorm"
    rotary_emb: bool = False
    flash_attention: bool = False
    gradient_checkpointing: bool = False
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    vocab_size: int = 4096
    pad_token_id: int = 0
    pos_encoding: str = "learned"
    rope_base_freq: float = 10000.0
    use_kv_cache: bool = False
    compile_model: bool = False
    compile_backend: str = "inductor"
    n_kv_heads: Optional[int] = None
    sliding_window_size: Optional[int] = None


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    output_dir: str = "outputs"
    run_name: str = "genova_run"
    seed: int = 42
    epochs: int = 100
    max_steps: int = -1
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    lr_scheduler: str = "cosine"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    compile_model: bool = False
    save_every_n_steps: int = 5000
    eval_every_n_steps: int = 1000
    log_every_n_steps: int = 100
    early_stopping_patience: int = 10
    resume_from_checkpoint: str = ""
    ddp: bool = False
    deepspeed_config: str = ""
    fsdp: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and benchmarking."""

    metrics: List[str] = field(
        default_factory=lambda: ["perplexity", "accuracy", "f1"]
    )
    downstream_tasks: List[str] = field(
        default_factory=lambda: [
            "promoter_detection",
            "splice_site_prediction",
            "chromatin_accessibility",
        ]
    )
    num_eval_samples: int = -1
    eval_batch_size: int = 64
    save_predictions: bool = False
    prediction_output_dir: str = "predictions"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class GenovaConfig:
    """Top-level configuration aggregating all sections.

    Attributes:
        data: Data loading and preprocessing settings.
        model: Model architecture settings.
        training: Training loop settings.
        evaluation: Evaluation and benchmarking settings.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full config to a plain dictionary."""
        return asdict(self)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Write config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        overrides: Optional[Sequence[str]] = None,
    ) -> "GenovaConfig":
        """Load config from a YAML file with optional CLI-style overrides.

        Args:
            path: Path to the YAML configuration file.
            overrides: Dot-separated key=value strings, e.g.
                ``["training.lr=3e-4", "model.n_layers=24"]``.

        Returns:
            A fully-resolved :class:`GenovaConfig` instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        if overrides:
            raw = _apply_overrides(raw, overrides)

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GenovaConfig":
        """Construct a :class:`GenovaConfig` from a nested dictionary."""
        return cls(
            data=_dict_to_dataclass(DataConfig, d.get("data", {})),
            model=_dict_to_dataclass(ModelConfig, d.get("model", {})),
            training=_dict_to_dataclass(TrainingConfig, d.get("training", {})),
            evaluation=_dict_to_dataclass(EvaluationConfig, d.get("evaluation", {})),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dict_to_dataclass(dc_cls: type, d: Dict[str, Any]) -> Any:
    """Instantiate a dataclass, ignoring unknown keys."""
    valid_keys = {f.name for f in fields(dc_cls)}
    filtered = {k: v for k, v in d.items() if k in valid_keys}
    return dc_cls(**filtered)


def _apply_overrides(raw: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Merge dot-notation ``key=value`` overrides into *raw* dict.

    Supported value types: int, float, bool (true/false), str.
    """
    raw = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be in key=value form, got: {item!r}")
        key, value_str = item.split("=", 1)
        parts = key.split(".")
        target = raw
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _cast_value(value_str)
    return raw


def _cast_value(value: str) -> Any:
    """Best-effort cast of a string value to a Python scalar."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

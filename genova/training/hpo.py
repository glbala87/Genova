"""Hyperparameter optimization for Genova.

Provides Optuna-based hyperparameter search with configurable search spaces,
trial pruning, persistent study storage, and integration with the Genova
training loop.

Example::

    from genova.training.hpo import create_study, optimize, best_params

    study = create_study("genova_hpo", direction="minimize")
    optimize(study, objective_fn=my_objective, n_trials=50)
    print(best_params(study))
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Union

from genova.utils.config import GenovaConfig, ModelConfig, TrainingConfig

try:
    import optuna
    from optuna import Study, Trial
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    from optuna.storages import RDBStorage

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    Study = Any  # type: ignore[assignment,misc]
    Trial = Any  # type: ignore[assignment,misc]


def _check_optuna() -> None:
    """Raise a helpful ImportError if optuna is not installed."""
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "Hyperparameter optimization requires 'optuna' to be installed.\n"
            "  pip install optuna\n"
            "For the dashboard, also install:\n"
            "  pip install optuna-dashboard"
        )


# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [8, 16, 32, 64, 128]},
    "d_model": {"type": "categorical", "choices": [256, 512, 768, 1024]},
    "n_layers": {"type": "int", "low": 4, "high": 24, "step": 2},
    "n_heads": {"type": "categorical", "choices": [4, 8, 12, 16]},
    "d_ff": {"type": "categorical", "choices": [1024, 2048, 3072, 4096]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05},
    "weight_decay": {"type": "float", "low": 1e-4, "high": 0.1, "log": True},
    "warmup_steps": {"type": "int", "low": 500, "high": 20000, "step": 500},
}


# ---------------------------------------------------------------------------
# Study creation
# ---------------------------------------------------------------------------


def create_study(
    name: str = "genova_hpo",
    direction: str = "minimize",
    storage: Optional[str] = None,
    pruner_n_startup_trials: int = 5,
    pruner_n_warmup_steps: int = 10,
    sampler_seed: Optional[int] = 42,
    load_if_exists: bool = True,
) -> "Study":
    """Create (or load) an Optuna study.

    Args:
        name: Study name.  Used for persistence and dashboard display.
        direction: ``"minimize"`` or ``"maximize"``.
        storage: Optuna storage URL.  Use ``"sqlite:///hpo.db"`` for
            persistent SQLite storage, or ``None`` for in-memory.
        pruner_n_startup_trials: Trials before pruning begins.
        pruner_n_warmup_steps: Steps before a trial can be pruned.
        sampler_seed: Random seed for the TPE sampler.
        load_if_exists: If ``True`` and a study with *name* already exists
            in *storage*, resume it instead of creating a new one.

    Returns:
        An :class:`optuna.Study` instance.
    """
    _check_optuna()

    sampler = TPESampler(seed=sampler_seed)
    pruner = MedianPruner(
        n_startup_trials=pruner_n_startup_trials,
        n_warmup_steps=pruner_n_warmup_steps,
    )

    study = optuna.create_study(
        study_name=name,
        direction=direction,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_if_exists,
    )

    return study


# ---------------------------------------------------------------------------
# Suggest config from trial
# ---------------------------------------------------------------------------


def suggest_config(
    trial: "Trial",
    search_space: Optional[Dict[str, Dict[str, Any]]] = None,
    base_config: Optional[GenovaConfig] = None,
) -> GenovaConfig:
    """Suggest a :class:`GenovaConfig` from an Optuna trial.

    Each key in the search space maps to a hyperparameter that will be
    sampled by the trial.  The remaining config values come from
    *base_config* (or defaults).

    Args:
        trial: An Optuna :class:`Trial` object.
        search_space: Dict mapping parameter names to suggestion specs.
            Defaults to :data:`DEFAULT_SEARCH_SPACE`.
        base_config: Base configuration to modify.  Defaults to
            :class:`GenovaConfig()`.

    Returns:
        A :class:`GenovaConfig` with suggested hyperparameters applied.
    """
    _check_optuna()

    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    if base_config is not None:
        config = GenovaConfig.from_dict(base_config.to_dict())
    else:
        config = GenovaConfig()

    suggested: Dict[str, Any] = {}

    for param_name, spec in search_space.items():
        value = _suggest_param(trial, param_name, spec)
        suggested[param_name] = value

    # Apply suggested values to config
    _apply_param(config, "lr", suggested)
    _apply_param(config, "weight_decay", suggested)
    _apply_param(config, "warmup_steps", suggested)
    _apply_param(config, "batch_size", suggested)
    _apply_param(config, "d_model", suggested)
    _apply_param(config, "n_layers", suggested)
    _apply_param(config, "n_heads", suggested)
    _apply_param(config, "d_ff", suggested)
    _apply_param(config, "dropout", suggested)

    return config


def _suggest_param(trial: "Trial", name: str, spec: Dict[str, Any]) -> Any:
    """Use an Optuna trial to suggest a single parameter value."""
    ptype = spec["type"]
    if ptype == "float":
        return trial.suggest_float(
            name,
            low=spec["low"],
            high=spec["high"],
            log=spec.get("log", False),
            step=spec.get("step"),
        )
    elif ptype == "int":
        return trial.suggest_int(
            name,
            low=spec["low"],
            high=spec["high"],
            step=spec.get("step", 1),
            log=spec.get("log", False),
        )
    elif ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"Unknown search space type: {ptype!r}")


def _apply_param(config: GenovaConfig, name: str, suggested: Dict[str, Any]) -> None:
    """Apply a suggested parameter to the appropriate config section."""
    if name not in suggested:
        return

    value = suggested[name]

    # Training parameters
    if name in ("lr", "weight_decay", "warmup_steps"):
        setattr(config.training, name, value)
    # Data parameters
    elif name == "batch_size":
        config.data.batch_size = value
    # Model parameters
    elif name in ("d_model", "n_layers", "n_heads", "d_ff", "dropout"):
        setattr(config.model, name, value)
        # Keep attention_dropout in sync with dropout
        if name == "dropout":
            config.model.attention_dropout = value


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def optimize(
    study: "Study",
    objective_fn: Callable[["Trial"], float],
    n_trials: int = 100,
    timeout: Optional[float] = None,
    n_jobs: int = 1,
    show_progress_bar: bool = True,
    callbacks: Optional[List[Callable]] = None,
) -> "Study":
    """Run the optimization loop.

    Args:
        study: An Optuna study (from :func:`create_study`).
        objective_fn: A callable that takes an :class:`optuna.Trial` and
            returns the objective value (e.g. validation loss).
        n_trials: Maximum number of trials to run.
        timeout: Stop after this many seconds (``None`` = no limit).
        n_jobs: Number of parallel jobs (``1`` = sequential).
        show_progress_bar: Show a tqdm progress bar.
        callbacks: Optional list of Optuna callback functions.

    Returns:
        The study with completed trials.
    """
    _check_optuna()

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
        callbacks=callbacks,
    )

    return study


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


def best_params(study: "Study") -> Dict[str, Any]:
    """Extract the best hyperparameters from a completed study.

    Args:
        study: An Optuna study with at least one completed trial.

    Returns:
        Dict of parameter name to best value, plus ``best_value`` (the
        objective) and ``best_trial_number``.

    Raises:
        ValueError: If the study has no completed trials.
    """
    _check_optuna()

    if len(study.trials) == 0:
        raise ValueError("Study has no completed trials.")

    best = study.best_trial
    result = dict(best.params)
    result["best_value"] = best.value
    result["best_trial_number"] = best.number
    return result


def best_config(
    study: "Study",
    base_config: Optional[GenovaConfig] = None,
) -> GenovaConfig:
    """Build a :class:`GenovaConfig` from the best trial's parameters.

    Args:
        study: An Optuna study with at least one completed trial.
        base_config: Base configuration to modify.

    Returns:
        A :class:`GenovaConfig` with the best hyperparameters applied.
    """
    _check_optuna()

    params = study.best_trial.params

    if base_config is not None:
        config = GenovaConfig.from_dict(base_config.to_dict())
    else:
        config = GenovaConfig()

    _apply_param(config, "lr", params)
    _apply_param(config, "weight_decay", params)
    _apply_param(config, "warmup_steps", params)
    _apply_param(config, "batch_size", params)
    _apply_param(config, "d_model", params)
    _apply_param(config, "n_layers", params)
    _apply_param(config, "n_heads", params)
    _apply_param(config, "d_ff", params)
    _apply_param(config, "dropout", params)

    return config


# ---------------------------------------------------------------------------
# Training loop integration
# ---------------------------------------------------------------------------


def create_objective(
    train_fn: Callable[[GenovaConfig], Dict[str, float]],
    metric_key: str = "val/loss",
    search_space: Optional[Dict[str, Dict[str, Any]]] = None,
    base_config: Optional[GenovaConfig] = None,
    report_intermediate: bool = True,
) -> Callable[["Trial"], float]:
    """Create an Optuna objective function that wraps a Genova training run.

    The returned callable suggests hyperparameters via :func:`suggest_config`,
    runs the provided *train_fn*, and returns the target metric.

    Args:
        train_fn: A function that accepts a :class:`GenovaConfig` and returns
            a metrics dict (e.g. the return value of
            :meth:`GenovaTrainer.train`).
        metric_key: Key in the metrics dict to optimize.
        search_space: Search space for :func:`suggest_config`.
        base_config: Base config for :func:`suggest_config`.
        report_intermediate: Whether to report intermediate values for
            pruning (requires *train_fn* to return them).

    Returns:
        A callable suitable for :func:`optimize`.
    """
    _check_optuna()

    def objective(trial: "Trial") -> float:
        config = suggest_config(trial, search_space=search_space, base_config=base_config)

        metrics = train_fn(config)

        value = metrics.get(metric_key)
        if value is None:
            raise ValueError(
                f"Metric '{metric_key}' not found in training results. "
                f"Available: {list(metrics.keys())}"
            )

        return float(value)

    return objective

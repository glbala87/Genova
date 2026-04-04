"""Model registry for Genova.

Provides an abstract :class:`ModelRegistry` interface with three back-ends:

* :class:`LocalRegistry` -- file-based, always available, no external deps.
* :class:`MLflowRegistry` -- requires ``mlflow``.
* :class:`HuggingFaceRegistry` -- requires ``huggingface_hub``.

Example::

    from genova.utils.registry import LocalRegistry, register_model, load_model

    registry = LocalRegistry("./model_registry")
    register_model(registry, model, config, metrics, name="genova-base", version="1.0")
    restored = load_model(registry, "genova-base", "1.0")
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class ModelRegistry(ABC):
    """Abstract model registry interface.

    A registry tracks model versions along with their configuration, training
    metrics, and artifacts (weights, tokenizer, etc.).
    """

    @abstractmethod
    def register(
        self,
        name: str,
        version: str,
        model_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Register a model version.

        Args:
            name: Model name / identifier.
            version: Version string (e.g. ``"1.0.0"``).
            model_path: Path to the model checkpoint file.
            config: Model / training configuration dict.
            metrics: Evaluation metrics dict.
            tags: Arbitrary string tags (e.g. ``{"stage": "production"}``).

        Returns:
            Registration metadata dict.
        """
        ...

    @abstractmethod
    def load(
        self,
        name: str,
        version: str,
    ) -> Dict[str, Any]:
        """Load a registered model version.

        Args:
            name: Model name.
            version: Version string.

        Returns:
            Dict with ``model_path``, ``config``, ``metrics``, ``tags``,
            and ``registered_at``.

        Raises:
            FileNotFoundError: If the model version is not found.
        """
        ...

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models and versions.

        Returns:
            List of metadata dicts.
        """
        ...

    @abstractmethod
    def delete(self, name: str, version: str) -> None:
        """Delete a registered model version.

        Args:
            name: Model name.
            version: Version string.
        """
        ...


# ---------------------------------------------------------------------------
# Local file-based registry
# ---------------------------------------------------------------------------


class LocalRegistry(ModelRegistry):
    """File-system-based model registry.

    Models are stored under ``<root>/<name>/<version>/``, each directory
    containing the checkpoint, a ``metadata.json``, and optional artifacts.

    Args:
        root: Root directory for the registry.
    """

    def __init__(self, root: Union[str, Path] = "./model_registry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _version_dir(self, name: str, version: str) -> Path:
        return self.root / name / version

    def register(
        self,
        name: str,
        version: str,
        model_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        version_dir = self._version_dir(name, version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint
        src = Path(model_path)
        if not src.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {src}")
        dst = version_dir / src.name
        shutil.copy2(str(src), str(dst))

        metadata: Dict[str, Any] = {
            "name": name,
            "version": version,
            "model_file": src.name,
            "config": config or {},
            "metrics": metrics or {},
            "tags": tags or {},
            "registered_at": time.time(),
        }

        meta_path = version_dir / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)

        return metadata

    def load(
        self,
        name: str,
        version: str,
    ) -> Dict[str, Any]:
        version_dir = self._version_dir(name, version)
        meta_path = version_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Model '{name}' version '{version}' not found in registry at {self.root}"
            )

        with open(meta_path) as fh:
            metadata: Dict[str, Any] = json.load(fh)

        metadata["model_path"] = str(version_dir / metadata["model_file"])
        return metadata

    def list_models(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not self.root.exists():
            return results
        for name_dir in sorted(self.root.iterdir()):
            if not name_dir.is_dir():
                continue
            for ver_dir in sorted(name_dir.iterdir()):
                if not ver_dir.is_dir():
                    continue
                meta_path = ver_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                    meta["model_path"] = str(
                        ver_dir / meta.get("model_file", "model.pt")
                    )
                    results.append(meta)
        return results

    def delete(self, name: str, version: str) -> None:
        version_dir = self._version_dir(name, version)
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # Clean up empty parent
        name_dir = self.root / name
        if name_dir.exists() and not any(name_dir.iterdir()):
            name_dir.rmdir()


# ---------------------------------------------------------------------------
# MLflow registry
# ---------------------------------------------------------------------------


class MLflowRegistry(ModelRegistry):
    """MLflow-backed model registry.

    Requires the ``mlflow`` package.  Uses the MLflow Tracking server for
    experiment/run logging and the Model Registry for versioning.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.
    """

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlruns.db",
        experiment_name: str = "genova",
    ) -> None:
        try:
            import mlflow  # type: ignore
        except ImportError:
            raise ImportError(
                "MLflowRegistry requires 'mlflow' to be installed.\n"
                "  pip install mlflow"
            )

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def register(
        self,
        name: str,
        version: str,
        model_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        mlflow = self._mlflow
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        with mlflow.start_run(run_name=f"{name}_v{version}") as run:
            # Log params
            if config:
                # Flatten nested config for MLflow (max 500 char values)
                flat = _flatten_dict(config)
                for k, v in flat.items():
                    mlflow.log_param(k, str(v)[:500])

            # Log metrics
            if metrics:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

            # Log tags
            all_tags = {"model_name": name, "model_version": version}
            if tags:
                all_tags.update(tags)
            for k, v in all_tags.items():
                mlflow.set_tag(k, v)

            # Log model artifact
            mlflow.log_artifact(str(model_path))

            # Register model
            model_uri = f"runs:/{run.info.run_id}/{model_path.name}"
            try:
                result = mlflow.register_model(model_uri, name)
                registered_version = result.version
            except Exception:
                registered_version = version

        return {
            "name": name,
            "version": registered_version,
            "run_id": run.info.run_id,
            "config": config or {},
            "metrics": metrics or {},
            "tags": all_tags,
            "registered_at": time.time(),
        }

    def load(
        self,
        name: str,
        version: str,
    ) -> Dict[str, Any]:
        mlflow = self._mlflow
        client = mlflow.tracking.MlflowClient()

        try:
            mv = client.get_model_version(name, version)
        except Exception as e:
            raise FileNotFoundError(
                f"Model '{name}' version '{version}' not found in MLflow registry: {e}"
            )

        run = client.get_run(mv.run_id)
        return {
            "name": name,
            "version": version,
            "model_path": mv.source,
            "config": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
            "registered_at": mv.creation_timestamp,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        mlflow = self._mlflow
        client = mlflow.tracking.MlflowClient()
        results: List[Dict[str, Any]] = []

        try:
            for rm in client.search_registered_models():
                for mv in client.search_model_versions(f"name='{rm.name}'"):
                    results.append({
                        "name": rm.name,
                        "version": mv.version,
                        "model_path": mv.source,
                        "registered_at": mv.creation_timestamp,
                        "status": mv.status,
                    })
        except Exception:
            pass

        return results

    def delete(self, name: str, version: str) -> None:
        mlflow = self._mlflow
        client = mlflow.tracking.MlflowClient()
        try:
            client.delete_model_version(name, version)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HuggingFace Hub registry
# ---------------------------------------------------------------------------


class HuggingFaceRegistry(ModelRegistry):
    """HuggingFace Hub-backed model registry.

    Uploads model checkpoints and metadata to the HuggingFace Hub.  Requires
    the ``huggingface_hub`` package and a valid API token.

    Args:
        namespace: HuggingFace user or organization namespace.
        token: HuggingFace API token.  If ``None``, reads from
            ``HF_TOKEN`` environment variable or cached login.
    """

    def __init__(
        self,
        namespace: str = "genova",
        token: Optional[str] = None,
    ) -> None:
        try:
            import huggingface_hub  # type: ignore
        except ImportError:
            raise ImportError(
                "HuggingFaceRegistry requires 'huggingface_hub' to be installed.\n"
                "  pip install huggingface_hub"
            )

        self._hub = huggingface_hub
        self.namespace = namespace
        self.token = token
        self._api = huggingface_hub.HfApi(token=token)

    def _repo_id(self, name: str) -> str:
        return f"{self.namespace}/{name}"

    def register(
        self,
        name: str,
        version: str,
        model_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        repo_id = self._repo_id(name)

        # Create repo if needed
        try:
            self._api.create_repo(repo_id, exist_ok=True, private=True)
        except Exception:
            pass

        # Upload model file
        self._api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=f"{version}/{model_path.name}",
            repo_id=repo_id,
            commit_message=f"Register model version {version}",
        )

        # Upload metadata
        metadata: Dict[str, Any] = {
            "name": name,
            "version": version,
            "model_file": model_path.name,
            "config": config or {},
            "metrics": metrics or {},
            "tags": tags or {},
            "registered_at": time.time(),
        }

        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(metadata, fh, indent=2, default=str)
            tmp_path = fh.name

        try:
            self._api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=f"{version}/metadata.json",
                repo_id=repo_id,
                commit_message=f"Add metadata for version {version}",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return metadata

    def load(
        self,
        name: str,
        version: str,
    ) -> Dict[str, Any]:
        repo_id = self._repo_id(name)

        # Download metadata
        try:
            meta_path = self._api.hf_hub_download(
                repo_id=repo_id,
                filename=f"{version}/metadata.json",
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Model '{name}' version '{version}' not found on HuggingFace Hub: {e}"
            )

        with open(meta_path) as fh:
            metadata: Dict[str, Any] = json.load(fh)

        # Download model file
        model_file = metadata.get("model_file", "model.pt")
        try:
            downloaded_path = self._api.hf_hub_download(
                repo_id=repo_id,
                filename=f"{version}/{model_file}",
            )
            metadata["model_path"] = downloaded_path
        except Exception:
            metadata["model_path"] = None

        return metadata

    def list_models(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        try:
            models = self._api.list_models(author=self.namespace)
            for model_info in models:
                results.append({
                    "name": model_info.modelId.split("/")[-1],
                    "repo_id": model_info.modelId,
                    "tags": list(model_info.tags) if model_info.tags else [],
                    "last_modified": str(model_info.lastModified)
                    if model_info.lastModified
                    else None,
                })
        except Exception:
            pass
        return results

    def delete(self, name: str, version: str) -> None:
        repo_id = self._repo_id(name)
        try:
            # Delete the version folder
            self._api.delete_folder(
                repo_id=repo_id,
                path_in_repo=version,
                commit_message=f"Delete version {version}",
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def register_model(
    registry: ModelRegistry,
    model: nn.Module,
    config: Any,
    metrics: Dict[str, float],
    name: str,
    version: str,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Save a model checkpoint and register it in the given registry.

    Args:
        registry: A :class:`ModelRegistry` instance.
        model: The PyTorch model to register.
        config: Model configuration (dataclass or dict).
        metrics: Evaluation metrics dict.
        name: Model name.
        version: Version string.
        tags: Optional tags.

    Returns:
        Registration metadata dict.
    """
    import tempfile

    # Serialize config
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict

        config_dict = asdict(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {"raw": str(config)}

    # Save model to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        torch.save(model.state_dict(), fh)
        tmp_path = fh.name

    try:
        result = registry.register(
            name=name,
            version=version,
            model_path=tmp_path,
            config=config_dict,
            metrics=metrics,
            tags=tags,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


def load_model(
    registry: ModelRegistry,
    name: str,
    version: str,
) -> Dict[str, Any]:
    """Load a model's metadata and checkpoint path from the registry.

    Args:
        registry: A :class:`ModelRegistry` instance.
        name: Model name.
        version: Version string.

    Returns:
        Dict with ``model_path``, ``config``, ``metrics``, etc.
        The caller is responsible for reconstructing the model from the
        checkpoint and config.
    """
    return registry.load(name, version)


def list_models(registry: ModelRegistry) -> List[Dict[str, Any]]:
    """List all models registered in the given registry.

    Args:
        registry: A :class:`ModelRegistry` instance.

    Returns:
        List of model metadata dicts.
    """
    return registry.list_models()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flatten a nested dict with dot-separated keys."""
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

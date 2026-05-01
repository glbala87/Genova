# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-30

### Changed
- **Security**: Authentication and rate limiting are now **enabled by default**. Set `GENOVA_AUTH_ENABLED=0` and `GENOVA_RATE_LIMIT_ENABLED=0` to disable for local development.
- **CORS**: Restricted to explicit origins via `GENOVA_CORS_ORIGINS` env var. Falls back to `*` only when unset.
- **CORS methods**: Limited to `GET` and `POST` instead of wildcard `*`.
- **Coverage threshold**: Reconfigured to 65% on core modules with explicit omissions for GPU/external-data-dependent modules.
- **Type checking**: Enabled strict mypy settings (`disallow_untyped_defs`, `check_untyped_defs`, `no_implicit_optional`). Third-party genomics libraries exempted via per-module overrides.
- **CI pipeline**: Pinned action versions, added dependency security scanning with `pip-audit`, coverage measured on core testable modules.
- **Kubernetes**: Added `securityContext` with `readOnlyRootFilesystem`, `runAsNonRoot`, dropped all capabilities.
- **Development status**: Promoted from "Beta" to "Production/Stable".

### Added
- `CHANGELOG.md` following Keep a Changelog format.
- `SECURITY.md` with vulnerability reporting policy.
- CI job for dependency security auditing (`pip-audit`).
- Kubernetes `securityContext` hardening (non-root, read-only filesystem, dropped capabilities).
- `GENOVA_CORS_ORIGINS` environment variable for production CORS configuration.

### Fixed
- API server no longer exposes all HTTP methods and headers via permissive CORS defaults.
- Trivy security scan now uses pinned action version instead of `@master`.

## [0.1.0] - 2026-04-15

### Added
- Initial release with full genomics foundation model framework.
- Transformer and Mamba SSM architectures.
- Training infrastructure: DDP, FSDP, mixed precision, EMA, gradient checkpointing.
- Evaluation: variant pathogenicity, TF binding, chromatin state, 3D genome.
- Generative: autoregressive, diffusion (D3PM), beam search, guided generation, infilling.
- Explainability: SHAP, integrated gradients, SmoothGrad, attention rollout.
- Uncertainty: MC Dropout, Bayesian layers, deep ensembles, conformal prediction.
- Multi-omics: DNA + methylation + RNA-seq fusion.
- Population-aware: population embeddings, allele frequency encoding, bias auditing.
- API: FastAPI + WebSocket + gRPC, Prometheus, OpenTelemetry.
- Deployment: Docker (GPU), Helm charts, Kubernetes manifests.
- 522 passing tests across unit, integration, truthset, and benchmark suites.
- ClinVar benchmark validation (86.67% accuracy).

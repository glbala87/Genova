# Contributing to Genova

Thank you for your interest in contributing to Genova! This document covers the
development workflow, code standards, and release process.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

---

## Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- [Poetry](https://python-poetry.org/) 1.7+
- Docker (for container builds)
- Git

### First-time setup

```bash
# Clone the repository
git clone https://github.com/genova-team/genova.git
cd genova

# Install all dependencies (including dev tools)
make install

# Verify the installation
poetry run genova --version
poetry run pytest -x -q
```

The `make install` target installs all Python dependencies and sets up
pre-commit hooks that run formatting and linting checks automatically before
each commit.

### Useful Make targets

| Command          | Description                                 |
|------------------|---------------------------------------------|
| `make install`   | Install all deps and pre-commit hooks       |
| `make lint`      | Run linters (black check, ruff, isort)      |
| `make format`    | Auto-format code                            |
| `make test`      | Run fast test suite (no slow / GPU tests)   |
| `make test-all`  | Run full test suite                         |
| `make test-cov`  | Tests with HTML coverage report             |
| `make serve`     | Start API server locally with hot-reload    |
| `make docker-build` | Build the Docker image                   |

---

## Code Style

We enforce a consistent style through automated tooling. All checks run in CI
and via pre-commit hooks.

### Formatters and linters

| Tool      | Purpose                | Config location     |
|-----------|------------------------|---------------------|
| **black** | Code formatting        | `pyproject.toml`    |
| **isort** | Import sorting         | `pyproject.toml`    |
| **ruff**  | Fast linting           | `pyproject.toml`    |
| **mypy**  | Static type checking   | `pyproject.toml`    |

### Key settings

- **Line length**: 100 characters
- **Target Python**: 3.10+
- **Import order**: isort with `black` profile, `genova` as first-party

### Quick commands

```bash
# Check without modifying files
make lint

# Auto-fix formatting issues
make format
```

### Docstrings

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
for all public functions, classes, and modules.

---

## Testing Requirements

### Running tests

```bash
# Fast tests (default — excludes slow and GPU markers)
make test

# Full suite
make test-all

# With coverage report
make test-cov
```

### Test markers

| Marker          | Meaning                                    |
|-----------------|--------------------------------------------|
| `@pytest.mark.slow` | Long-running tests (>30s)             |
| `@pytest.mark.gpu`  | Requires an NVIDIA GPU                |
| `@pytest.mark.integration` | Integration tests              |

### Guidelines

- Every new feature or bug fix must include tests.
- Minimum code coverage threshold: **30%** (enforced in CI).
- Place tests in `tests/` mirroring the source layout (e.g.,
  `genova/models/attention.py` -> `tests/models/test_attention.py`).
- Use `pytest` fixtures and parameterize where appropriate.
- Keep unit tests fast — mock external I/O when possible.

---

## Pull Request Process

1. **Create a branch** from `develop` (see naming conventions below).
2. **Make your changes** with clear, atomic commits.
3. **Run checks locally** before pushing:
   ```bash
   make format
   make lint
   make test
   ```
4. **Push and open a PR** against `develop` (or `main` for hotfixes).
5. **Fill in the PR template** — description, type of change, testing
   checklist, and performance impact.
6. **Address review feedback** — push new commits (do not force-push during
   review).
7. **Merge** — a maintainer will squash-merge once approved and CI passes.

### PR requirements

- All CI checks must pass (lint, test, build, Docker scan).
- At least one approval from a code owner.
- No unresolved review threads.
- PR title follows conventional format (e.g., `fix: correct tokenizer offset`).

---

## Branch Naming Conventions

| Prefix       | Purpose                          | Example                           |
|--------------|----------------------------------|-----------------------------------|
| `feature/`   | New functionality                | `feature/flash-attention-v2`      |
| `fix/`       | Bug fixes                       | `fix/tokenizer-oom`               |
| `refactor/`  | Code cleanup, no behaviour change| `refactor/simplify-data-pipeline` |
| `docs/`      | Documentation only               | `docs/api-usage-guide`            |
| `ci/`        | CI/CD changes                    | `ci/add-gpu-benchmarks`           |
| `perf/`      | Performance improvements         | `perf/batch-inference`            |
| `hotfix/`    | Urgent production fix            | `hotfix/health-endpoint-crash`    |

Use lowercase with hyphens. Keep names short but descriptive.

---

## Release Process

Genova uses [semantic versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

1. Merge all target changes into `main`.
2. Update the version in `pyproject.toml`.
3. Create an annotated tag:
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```
4. The CD pipeline automatically:
   - Builds the Python package and Docker image.
   - Creates a GitHub Release with auto-generated notes.
   - Deploys to staging, then production (with manual approval).

### Pre-release versions

For release candidates, use tags like `v0.2.0-rc.1`. These are marked as
pre-releases on GitHub and are not deployed to production automatically.

---

## Getting Help

- **Issues**: Open a [GitHub Issue](https://github.com/genova-team/genova/issues)
  using the provided templates.
- **Discussions**: Use GitHub Discussions for questions and ideas.
- **Code owners**: See `.github/CODEOWNERS` for team contacts.

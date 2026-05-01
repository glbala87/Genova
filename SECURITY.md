# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | Yes                |
| < 1.0   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in Genova, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Email security findings to **security@genova-genomics.org** with:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if any)
3. You will receive an acknowledgment within **48 hours**.
4. We aim to provide a fix within **7 business days** for critical issues.

## Security Defaults

Genova 1.0+ ships with security enabled by default:

- **API authentication** is on by default (`GENOVA_AUTH_ENABLED=1`). Disable explicitly with `GENOVA_AUTH_ENABLED=0` for local development only.
- **Rate limiting** is on by default (`GENOVA_RATE_LIMIT_ENABLED=1`).
- **CORS** is restricted. Set `GENOVA_CORS_ORIGINS` to your allowed origins in production.
- Kubernetes deployments run as non-root with read-only filesystem and dropped capabilities.
- Docker images use a dedicated `genova` user (UID 1000).

## Dependency Management

- Dependencies are audited in CI via `pip-audit`.
- Container images are scanned with Trivy for CRITICAL and HIGH vulnerabilities (build fails on findings).
- We recommend running `pip-audit` locally before deploying.

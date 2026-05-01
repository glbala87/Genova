"""API authentication and rate limiting for the Genova REST API.

Provides API key authentication and configurable rate limiting as FastAPI
dependencies. Both are **enabled by default** for production safety and
controlled via environment variables.

Environment variables:
    GENOVA_API_KEYS: Comma-separated list of valid API keys.
    GENOVA_API_KEYS_FILE: Path to a file containing one API key per line.
    GENOVA_RATE_LIMIT_RPM: Requests per minute per API key (default: 60).
    GENOVA_RATE_LIMIT_BACKEND: "memory" (default) or "redis".
    GENOVA_REDIS_URL: Redis URL for rate limiting (default: redis://localhost:6379/0).
    GENOVA_AUTH_ENABLED: Authentication enabled by default. Set to "0" to disable.
    GENOVA_RATE_LIMIT_ENABLED: Rate limiting enabled by default. Set to "0" to disable.
    GENOVA_JWT_SECRET: Secret key for JWT token validation (optional).
    GENOVA_JWT_ALGORITHM: JWT algorithm (default: HS256).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

# ---------------------------------------------------------------------------
# JWT support (optional)
# ---------------------------------------------------------------------------

try:
    import jwt as _pyjwt

    _HAS_JWT = True
except ImportError:
    _pyjwt = None  # type: ignore[assignment]
    _HAS_JWT = False

# ---------------------------------------------------------------------------
# Redis support (optional)
# ---------------------------------------------------------------------------

try:
    import redis as _redis_lib

    _HAS_REDIS = True
except ImportError:
    _redis_lib = None  # type: ignore[assignment]
    _HAS_REDIS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_api_keys() -> Set[str]:
    """Load API keys from environment variable or file."""
    keys: Set[str] = set()

    # From env var (comma-separated)
    env_keys = os.environ.get("GENOVA_API_KEYS", "")
    if env_keys:
        for k in env_keys.split(","):
            k = k.strip()
            if k:
                keys.add(k)

    # From file
    keys_file = os.environ.get("GENOVA_API_KEYS_FILE", "")
    if keys_file:
        path = Path(keys_file)
        if path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    keys.add(line)
        else:
            logger.warning("API keys file not found: {}", keys_file)

    return keys


def _constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


# ---------------------------------------------------------------------------
# In-memory rate limiter
# ---------------------------------------------------------------------------


class _InMemoryRateLimiter:
    """Sliding-window rate limiter backed by an in-memory dict.

    Stores a list of request timestamps per key and prunes expired entries
    on each check.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.rpm = requests_per_minute
        self.window = 60.0  # seconds
        self._buckets: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Return True if the request is within the rate limit."""
        now = time.monotonic()
        bucket = self._buckets[key]

        # Prune timestamps outside the window
        cutoff = now - self.window
        self._buckets[key] = bucket = [t for t in bucket if t > cutoff]

        if len(bucket) >= self.rpm:
            return False

        bucket.append(now)
        return True

    def remaining(self, key: str) -> int:
        """Return the number of remaining requests in the current window."""
        now = time.monotonic()
        cutoff = now - self.window
        bucket = [t for t in self._buckets[key] if t > cutoff]
        return max(0, self.rpm - len(bucket))

    def reset_time(self, key: str) -> float:
        """Return seconds until the oldest request in the window expires."""
        now = time.monotonic()
        cutoff = now - self.window
        bucket = [t for t in self._buckets[key] if t > cutoff]
        if not bucket:
            return 0.0
        return max(0.0, bucket[0] - cutoff)


# ---------------------------------------------------------------------------
# Redis rate limiter
# ---------------------------------------------------------------------------


class _RedisRateLimiter:
    """Sliding-window rate limiter backed by Redis sorted sets."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        self.rpm = requests_per_minute
        self.window = 60  # seconds
        self.prefix = "genova:ratelimit:"

        if not _HAS_REDIS:
            raise RuntimeError(
                "redis package is required for Redis-backed rate limiting. "
                "Install with: pip install redis"
            )
        self._client = _redis_lib.from_url(redis_url, decode_responses=True)

    def is_allowed(self, key: str) -> bool:
        """Return True if the request is within the rate limit."""
        rkey = f"{self.prefix}{key}"
        now = time.time()
        cutoff = now - self.window

        pipe = self._client.pipeline()
        pipe.zremrangebyscore(rkey, "-inf", cutoff)
        pipe.zcard(rkey)
        pipe.zadd(rkey, {str(now): now})
        pipe.expire(rkey, self.window + 1)
        results = pipe.execute()

        current_count = results[1]
        return current_count < self.rpm

    def remaining(self, key: str) -> int:
        rkey = f"{self.prefix}{key}"
        now = time.time()
        cutoff = now - self.window
        self._client.zremrangebyscore(rkey, "-inf", cutoff)
        count = self._client.zcard(rkey)
        return max(0, self.rpm - count)

    def reset_time(self, key: str) -> float:
        return float(self.window)


# ---------------------------------------------------------------------------
# APIKeyAuth dependency
# ---------------------------------------------------------------------------


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyAuth:
    """FastAPI dependency for API key authentication.

    Usage::

        auth = APIKeyAuth()

        @app.get("/protected", dependencies=[Depends(auth)])
        async def protected_route():
            ...

    Or to get the authenticated key::

        @app.get("/protected")
        async def protected_route(api_key: str = Depends(auth)):
            ...
    """

    def __init__(
        self,
        api_keys: Optional[Set[str]] = None,
        auto_error: bool = True,
    ) -> None:
        self._api_keys = api_keys if api_keys is not None else _load_api_keys()
        self.auto_error = auto_error
        self._jwt_secret = os.environ.get("GENOVA_JWT_SECRET", "")
        self._jwt_algorithm = os.environ.get("GENOVA_JWT_ALGORITHM", "HS256")

    async def __call__(
        self,
        request: Request,
        api_key: Optional[str] = Security(_api_key_header),
    ) -> Optional[str]:
        """Validate the API key or JWT token from the request."""
        # Try API key header first
        if api_key:
            if self._validate_api_key(api_key):
                request.state.api_key = api_key
                return api_key

        # Try Authorization: Bearer <token> header (JWT)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and self._jwt_secret:
            token = auth_header[7:]
            payload = self._validate_jwt(token)
            if payload is not None:
                key_id = payload.get("sub", "jwt-user")
                request.state.api_key = key_id
                return key_id

        if self.auto_error:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return None

    def _validate_api_key(self, key: str) -> bool:
        """Check if the key matches any configured API key."""
        for valid_key in self._api_keys:
            if _constant_time_compare(key, valid_key):
                return True
        return False

    def _validate_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a JWT token and return its payload, or None."""
        if not _HAS_JWT or not self._jwt_secret:
            return None
        try:
            payload = _pyjwt.decode(
                token,
                self._jwt_secret,
                algorithms=[self._jwt_algorithm],
            )
            return payload
        except Exception as e:
            logger.debug("JWT validation failed: {}", e)
            return None


# ---------------------------------------------------------------------------
# RateLimiter dependency
# ---------------------------------------------------------------------------


class RateLimiter:
    """FastAPI dependency for per-key rate limiting.

    Usage::

        limiter = RateLimiter(requests_per_minute=60)

        @app.get("/api", dependencies=[Depends(limiter)])
        async def api_route():
            ...
    """

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        backend: Optional[str] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        rpm = requests_per_minute or int(
            os.environ.get("GENOVA_RATE_LIMIT_RPM", "60")
        )
        backend = backend or os.environ.get("GENOVA_RATE_LIMIT_BACKEND", "memory")
        redis_url = redis_url or os.environ.get(
            "GENOVA_REDIS_URL", "redis://localhost:6379/0"
        )

        if backend == "redis":
            self._limiter = _RedisRateLimiter(
                requests_per_minute=rpm, redis_url=redis_url
            )
        else:
            self._limiter = _InMemoryRateLimiter(requests_per_minute=rpm)

        self.rpm = rpm
        logger.info(
            "Rate limiter initialized: {} RPM, backend={}",
            rpm,
            backend,
        )

    async def __call__(self, request: Request) -> None:
        """Check rate limit for the current request."""
        # Use API key if available, otherwise fall back to client IP
        key = getattr(request.state, "api_key", None)
        if not key:
            key = request.client.host if request.client else "unknown"

        if not self._limiter.is_allowed(key):
            remaining = self._limiter.remaining(key)
            retry_after = self._limiter.reset_time(key)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
                headers={
                    "X-RateLimit-Limit": str(self.rpm),
                    "X-RateLimit-Remaining": str(remaining),
                    "Retry-After": str(int(retry_after) + 1),
                },
            )


# ---------------------------------------------------------------------------
# Convenience: check if auth should be enabled
# ---------------------------------------------------------------------------


def is_auth_enabled() -> bool:
    """Return True if authentication is enabled via environment.

    Authentication is **enabled by default** in production. Set
    ``GENOVA_AUTH_ENABLED=0`` to explicitly disable it (e.g. for local
    development).
    """
    val = os.environ.get("GENOVA_AUTH_ENABLED", "1").lower()
    return val not in ("0", "false", "no", "off")


def is_rate_limit_enabled() -> bool:
    """Return True if rate limiting is enabled via environment.

    Rate limiting is **enabled by default** in production. Set
    ``GENOVA_RATE_LIMIT_ENABLED=0`` to explicitly disable it.
    """
    val = os.environ.get("GENOVA_RATE_LIMIT_ENABLED", "1").lower()
    return val not in ("0", "false", "no", "off")

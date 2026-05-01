"""Unit tests for API security module (auth, rate limiting)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from genova.api.security import (
    APIKeyAuth,
    RateLimiter,
    _InMemoryRateLimiter,
    _constant_time_compare,
    _load_api_keys,
    is_auth_enabled,
    is_rate_limit_enabled,
)


class TestAuthEnabled:
    """Test is_auth_enabled() default-on behavior."""

    def test_default_is_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GENOVA_AUTH_ENABLED", None)
            # Re-import to pick up env
            assert is_auth_enabled() is True

    def test_explicit_enabled(self):
        with patch.dict(os.environ, {"GENOVA_AUTH_ENABLED": "1"}):
            assert is_auth_enabled() is True

    def test_explicit_disabled_zero(self):
        with patch.dict(os.environ, {"GENOVA_AUTH_ENABLED": "0"}):
            assert is_auth_enabled() is False

    def test_explicit_disabled_false(self):
        with patch.dict(os.environ, {"GENOVA_AUTH_ENABLED": "false"}):
            assert is_auth_enabled() is False

    def test_explicit_disabled_no(self):
        with patch.dict(os.environ, {"GENOVA_AUTH_ENABLED": "no"}):
            assert is_auth_enabled() is False

    def test_explicit_disabled_off(self):
        with patch.dict(os.environ, {"GENOVA_AUTH_ENABLED": "off"}):
            assert is_auth_enabled() is False


class TestRateLimitEnabled:
    """Test is_rate_limit_enabled() default-on behavior."""

    def test_default_is_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GENOVA_RATE_LIMIT_ENABLED", None)
            assert is_rate_limit_enabled() is True

    def test_explicit_disabled(self):
        with patch.dict(os.environ, {"GENOVA_RATE_LIMIT_ENABLED": "0"}):
            assert is_rate_limit_enabled() is False

    def test_explicit_enabled(self):
        with patch.dict(os.environ, {"GENOVA_RATE_LIMIT_ENABLED": "true"}):
            assert is_rate_limit_enabled() is True


class TestConstantTimeCompare:
    """Test constant-time string comparison."""

    def test_equal_strings(self):
        assert _constant_time_compare("secret123", "secret123") is True

    def test_unequal_strings(self):
        assert _constant_time_compare("secret123", "secret456") is False

    def test_empty_strings(self):
        assert _constant_time_compare("", "") is True

    def test_different_lengths(self):
        assert _constant_time_compare("short", "longer_string") is False


class TestLoadApiKeys:
    """Test API key loading from env and file."""

    def test_load_from_env(self):
        with patch.dict(os.environ, {"GENOVA_API_KEYS": "key1,key2,key3"}):
            keys = _load_api_keys()
            assert keys == {"key1", "key2", "key3"}

    def test_load_from_env_with_spaces(self):
        with patch.dict(os.environ, {"GENOVA_API_KEYS": " key1 , key2 "}):
            keys = _load_api_keys()
            assert keys == {"key1", "key2"}

    def test_load_empty_env(self):
        with patch.dict(os.environ, {"GENOVA_API_KEYS": ""}, clear=True):
            os.environ.pop("GENOVA_API_KEYS_FILE", None)
            keys = _load_api_keys()
            assert keys == set()

    def test_load_from_file(self, tmp_path):
        key_file = tmp_path / "keys.txt"
        key_file.write_text("file_key1\n# comment\nfile_key2\n\n")
        with patch.dict(
            os.environ,
            {"GENOVA_API_KEYS": "", "GENOVA_API_KEYS_FILE": str(key_file)},
        ):
            keys = _load_api_keys()
            assert keys == {"file_key1", "file_key2"}

    def test_load_from_missing_file(self):
        with patch.dict(
            os.environ,
            {"GENOVA_API_KEYS": "", "GENOVA_API_KEYS_FILE": "/nonexistent/keys.txt"},
        ):
            keys = _load_api_keys()
            assert keys == set()


class TestInMemoryRateLimiter:
    """Test the in-memory sliding window rate limiter."""

    def test_allows_within_limit(self):
        limiter = _InMemoryRateLimiter(requests_per_minute=5)
        for _ in range(5):
            assert limiter.is_allowed("user1") is True

    def test_blocks_over_limit(self):
        limiter = _InMemoryRateLimiter(requests_per_minute=3)
        for _ in range(3):
            limiter.is_allowed("user1")
        assert limiter.is_allowed("user1") is False

    def test_different_keys_independent(self):
        limiter = _InMemoryRateLimiter(requests_per_minute=1)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user2") is True
        assert limiter.is_allowed("user1") is False

    def test_remaining_count(self):
        limiter = _InMemoryRateLimiter(requests_per_minute=5)
        assert limiter.remaining("user1") == 5
        limiter.is_allowed("user1")
        assert limiter.remaining("user1") == 4

    def test_reset_time(self):
        limiter = _InMemoryRateLimiter(requests_per_minute=5)
        assert limiter.reset_time("user1") == 0.0
        limiter.is_allowed("user1")
        assert limiter.reset_time("user1") >= 0.0


class TestAPIKeyAuth:
    """Test APIKeyAuth dependency."""

    def test_validate_api_key_valid(self):
        auth = APIKeyAuth(api_keys={"test-key-123"})
        assert auth._validate_api_key("test-key-123") is True

    def test_validate_api_key_invalid(self):
        auth = APIKeyAuth(api_keys={"test-key-123"})
        assert auth._validate_api_key("wrong-key") is False

    def test_validate_api_key_empty_set(self):
        auth = APIKeyAuth(api_keys=set())
        assert auth._validate_api_key("any-key") is False

    def test_validate_jwt_without_secret(self):
        auth = APIKeyAuth(api_keys=set())
        assert auth._validate_jwt("some.token.here") is None

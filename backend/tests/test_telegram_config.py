"""Tests for Telegram configuration loading (Stage 1).

Covers: integration-off (unset/empty/placeholder token), fail-fast on a real
token without a real webhook secret, successful validation, and secret
redaction. Values used here are fake placeholders; the real .env is never
read by tests (env is injected via the ``env=`` mapping, the same convention
as ``test_llm_settings.py``).
"""

from __future__ import annotations

import pytest

from app.config import (
    ConfigurationError,
    TelegramSettings,
    load_telegram_settings,
)


def _env(token: str = "123456:FAKE-TEST-TOKEN", secret: str = "fake-webhook-secret"):
    return {"TELEGRAM_BOT_TOKEN": token, "TELEGRAM_WEBHOOK_SECRET": secret}


def test_telegram_disabled_when_token_unset():
    assert load_telegram_settings({"TELEGRAM_BOT_TOKEN": ""}) is None


def test_telegram_disabled_when_token_missing():
    assert load_telegram_settings({}) is None


def test_telegram_disabled_when_token_placeholder():
    # The .env.example template token must be treated as "not configured".
    disabled = load_telegram_settings(_env(token="placeholder_from_botfather"))
    assert disabled is None


def test_real_token_without_secret_raises():
    with pytest.raises(ConfigurationError) as excinfo:
        load_telegram_settings(_env(secret=""))
    assert "TELEGRAM_WEBHOOK_SECRET" in str(excinfo.value)


def test_real_token_with_placeholder_secret_raises():
    # The .env.example placeholder for the webhook secret is also "unset".
    with pytest.raises(ConfigurationError) as excinfo:
        load_telegram_settings(
            _env(secret="generate_a_random_string_for_hmac_check")
        )
    assert "TELEGRAM_WEBHOOK_SECRET" in str(excinfo.value)
    assert "fake-webhook-secret" not in str(excinfo.value)
    assert "123456:FAKE-TEST-TOKEN" not in str(excinfo.value)


def test_valid_settings_loaded():
    settings = load_telegram_settings(_env())
    assert isinstance(settings, TelegramSettings)
    assert settings.bot_token == "123456:FAKE-TEST-TOKEN"
    assert settings.webhook_secret == "fake-webhook-secret"


def test_repr_never_exposes_token_or_secret():
    settings = load_telegram_settings(_env())
    assert "123456:FAKE-TEST-TOKEN" not in repr(settings)
    assert "123456:FAKE-TEST-TOKEN" not in str(settings)
    assert "fake-webhook-secret" not in repr(settings)
    assert "fake-webhook-secret" not in str(settings)
    assert "<redacted>" in repr(settings)


def test_placeholder_values_are_stripped_whitespace_agnostic():
    # Whitespace-only values behave like unset.
    assert load_telegram_settings(_env(token="   ")) is None
    with pytest.raises(ConfigurationError):
        load_telegram_settings(_env(secret="   "))


def test_existing_loaders_unaffected():
    """The new loader must not change Shopify/LLM/Neon behavior (spot check)."""
    from app.config import load_neon_settings

    assert load_neon_settings({"NEON_DATABASE_URL": ""}) is None
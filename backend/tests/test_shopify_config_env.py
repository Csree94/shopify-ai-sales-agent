"""Phase 5 tests: .env loading and authentication configuration modes."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    ConfigurationError,
    load_local_env_files,
    load_shopify_settings,
)


def test_placeholder_admin_token_treated_as_unset():
    with pytest.raises(ConfigurationError) as excinfo:
        load_shopify_settings(
            {
                "SHOPIFY_STORE_DOMAIN": "test.myshopify.com",
                "SHOPIFY_ADMIN_ACCESS_TOKEN": "placeholder_get_from_dev_dashboard",
            }
        )
    assert "No Shopify credentials configured" in str(excinfo.value)


def test_partial_client_credentials_rejected():
    with pytest.raises(ConfigurationError) as excinfo:
        load_shopify_settings(
            {
                "SHOPIFY_STORE_DOMAIN": "test.myshopify.com",
                "SHOPIFY_CLIENT_ID": "synthetic-id",
            }
        )
    assert "SHOPIFY_CLIENT_SECRET" in str(excinfo.value)


def test_client_credentials_mode_selected(tmp_path):
    settings = load_shopify_settings(
        {
            "SHOPIFY_STORE_DOMAIN": "test.myshopify.com",
            "SHOPIFY_CLIENT_ID": "synthetic-id",
            "SHOPIFY_CLIENT_SECRET": "synthetic-secret",
            # Static token also present: client credentials take precedence.
            "SHOPIFY_ADMIN_ACCESS_TOKEN": "static-synthetic-token",
        }
    )
    assert settings.auth_mode == "client_credentials"


def test_static_token_mode_when_no_client_credentials():
    settings = load_shopify_settings(
        {
            "SHOPIFY_STORE_DOMAIN": "test.myshopify.com",
            "SHOPIFY_ADMIN_ACCESS_TOKEN": "static-synthetic-token",
        }
    )
    assert settings.auth_mode == "static_token"


def test_repr_redacts_all_credentials():
    settings = load_shopify_settings(
        {
            "SHOPIFY_STORE_DOMAIN": "test.myshopify.com",
            "SHOPIFY_CLIENT_ID": "synthetic-id",
            "SHOPIFY_CLIENT_SECRET": "synthetic-secret",
        }
    )
    rendered = repr(settings)
    assert "synthetic-id" not in rendered
    assert "synthetic-secret" not in rendered
    assert "<redacted>" in rendered


def test_load_local_env_files_without_overriding(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PHASE5_TEST_VAR=from_file\nPHASE5_TEST_OVERRIDE=from_file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PHASE5_TEST_OVERRIDE", "from_environment")
    loaded = load_local_env_files([env_file])

    import os

    try:
        assert loaded == [str(env_file)]
        assert os.environ["PHASE5_TEST_VAR"] == "from_file"
        # Existing environment variables win (override=False).
        assert os.environ["PHASE5_TEST_OVERRIDE"] == "from_environment"
    finally:
        monkeypatch.delenv("PHASE5_TEST_VAR", raising=False)
        monkeypatch.delenv("PHASE5_TEST_OVERRIDE", raising=False)


def test_missing_env_file_is_not_an_error(tmp_path):
    loaded = load_local_env_files([tmp_path / "does-not-exist.env"])
    assert loaded == []

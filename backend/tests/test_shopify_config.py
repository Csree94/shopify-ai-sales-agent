"""Phase 4 tests: Shopify settings loading and validation.

No network access; only environment mapping checks.
"""
from __future__ import annotations

import pytest

from app.config import (
    ConfigurationError,
    ShopifySettings,
    load_shopify_settings,
)


def test_loads_full_settings():
    settings = load_shopify_settings(
        {
            "SHOPIFY_STORE_DOMAIN": "Your-Store.myshopify.com",
            "SHOPIFY_ADMIN_ACCESS_TOKEN": "synthetic-token",
            "SHOPIFY_API_VERSION": "2026-07",
        }
    )
    assert settings.store_domain == "your-store.myshopify.com"
    assert settings.api_version == "2026-07"
    assert settings.timeout_seconds == 15.0  # default
    assert settings.graphql_url == (
        "https://your-store.myshopify.com/admin/api/2026-07/graphql.json"
    )


def test_defaults_to_latest_stable_version(monkeypatch, capsys):
    env = {
        "SHOPIFY_STORE_DOMAIN": "test.myshopify.com",
        "SHOPIFY_ADMIN_ACCESS_TOKEN": "synthetic-token",
    }
    settings = load_shopify_settings(env)
    assert settings.api_version == "2026-10"


def test_missing_domain_raises_named_error():
    with pytest.raises(ConfigurationError) as excinfo:
        load_shopify_settings({"SHOPIFY_ADMIN_ACCESS_TOKEN": "synthetic-token"})
    assert "SHOPIFY_STORE_DOMAIN" in str(excinfo.value)


def test_missing_token_raises_named_error_without_value():
    with pytest.raises(ConfigurationError) as excinfo:
        load_shopify_settings({"SHOPIFY_STORE_DOMAIN": "test.myshopify.com"})
    assert "SHOPIFY_ADMIN_ACCESS_TOKEN" in str(excinfo.value)
    # Error messages never contain secret values.
    assert "synthetic" not in str(excinfo.value)


def test_unsupported_version_rejected():
    with pytest.raises(ConfigurationError) as excinfo:
        load_shopify_settings(
            {
                "SHOPIFY_STORE_DOMAIN": "test.myshopify.com",
                "SHOPIFY_ADMIN_ACCESS_TOKEN": "synthetic-token",
                "SHOPIFY_API_VERSION": "2019-07",
            }
        )
    assert "not a currently supported version" in str(excinfo.value)


def test_pasted_admin_url_is_normalised():
    settings = load_shopify_settings(
        {
            "SHOPIFY_STORE_DOMAIN": "https://test.myshopify.com/admin",
            "SHOPIFY_ADMIN_ACCESS_TOKEN": "synthetic-token",
        }
    )
    assert settings.store_domain == "test.myshopify.com"


def test_repr_redacts_token():
    settings = ShopifySettings(
        store_domain="test.myshopify.com",
        api_version="2026-10",
        admin_api_token="synthetic-token",
    )
    assert "synthetic-token" not in repr(settings)
    assert "<redacted>" in repr(settings)


def test_invalid_timeout_rejected():
    with pytest.raises(ConfigurationError):
        load_shopify_settings(
            {
                "SHOPIFY_STORE_DOMAIN": "test.myshopify.com",
                "SHOPIFY_ADMIN_ACCESS_TOKEN": "synthetic-token",
                "SHOPIFY_HTTP_TIMEOUT_SECONDS": "abc",
            }
        )

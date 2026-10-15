"""Phase 8 LLM configuration tests (offline; no network; env dict injection)."""

from __future__ import annotations

import pytest

from app.config import ConfigurationError, load_llm_settings


def _base_env(**overrides):
    env = {
        "GEMINI_API_KEY": "test-key-obj001",
        "GEMINI_MODEL": "",
        "NVIDIA_API_KEY": "test-nv-key-obj001",
        "NVIDIA_LLM_MODEL": "",
        "NVIDIA_LLM_BASE_URL": "",
        "LLM_FALLBACK_ENABLED": "",
    }
    env.update(overrides)
    return env


def test_missing_gemini_key_raises():
    with pytest.raises(ConfigurationError) as exc_info:
        load_llm_settings(env=_base_env(GEMINI_API_KEY=""))
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_placeholder_gemini_key_is_treated_as_unset():
    with pytest.raises(ConfigurationError):
        load_llm_settings(env=_base_env(GEMINI_API_KEY="your_gemini_api_key_here"))


def test_defaults_match_approved_plan():
    """Primary model: gemini-3.6-flash (switched from gemini-3.8-flash for
    live-availability reasons; note the 2026-11-19 retirement window).
    """
    settings = load_llm_settings(
        env=_base_env(GEMINI_MODEL="", NVIDIA_LLM_MODEL="", NVIDIA_LLM_BASE_URL="")
    )
    assert settings.gemini.model == "gemini-3.6-flash"
    assert settings.nvidia.model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert settings.nvidia.base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.fallback_enabled is True


def test_fallback_disabled_false_parsing():
    settings = load_llm_settings(env=_base_env(LLM_FALLBACK_ENABLED="false"))
    assert settings.fallback_enabled is False


def test_fallback_flag_invalid_value_raises():
    with pytest.raises(ConfigurationError):
        load_llm_settings(env=_base_env(LLM_FALLBACK_ENABLED="maybe"))


def test_missing_nvidia_key_is_allowed_but_empty():
    settings = load_llm_settings(env=_base_env(NVIDIA_API_KEY=""))
    assert settings.nvidia.api_key == ""
    assert settings.fallback_enabled is True  # gate checked later by the agent


def test_reprs_never_contain_key_values():
    settings = load_llm_settings(env=_base_env())
    rendered = repr(settings)
    assert "test-key-obj001" not in rendered
    assert "test-nv-key-obj001" not in rendered
    assert "<redacted>" in rendered

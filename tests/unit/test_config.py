import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from supporter.config import AppConfig, _get_project_root, load_config


class TestGetProjectRoot:
    def test_finds_project_with_pyproject(self, tmp_path: Any) -> None:
        (tmp_path / "pyproject.toml").touch()
        subdir = tmp_path / "src"
        subdir.mkdir()
        mock_path = MagicMock()
        mock_path.resolve.return_value = mock_path
        mock_path.parent = tmp_path
        mock_path.parents = [tmp_path]
        with patch("supporter.config.Path", return_value=mock_path):
            result = _get_project_root()
            assert result == str(tmp_path)

    def test_finds_project_with_git(self, tmp_path: Any) -> None:
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "src"
        subdir.mkdir()
        mock_path = MagicMock()
        mock_path.resolve.return_value = mock_path
        mock_path.parent = tmp_path
        mock_path.parents = [tmp_path]
        with patch("supporter.config.Path", return_value=mock_path):
            result = _get_project_root()
            assert result == str(tmp_path)

    def test_falls_back_to_cwd(self) -> None:
        mock_path = MagicMock()
        mock_path.resolve.return_value = mock_path
        mock_path.parent = MagicMock()
        mock_path.parents = [mock_path.parent]
        mock_path.parent.__truediv__ = MagicMock(
            return_value=MagicMock(exists=MagicMock(return_value=False))
        )
        with (
            patch("supporter.config.Path", return_value=mock_path),
            patch("os.getcwd", return_value="/fallback/cwd"),
        ):
            result = _get_project_root()
            assert result == "/fallback/cwd"


class TestLoadConfig:
    @pytest.fixture
    def clean_env(self) -> Any:
        old_env = os.environ.copy()
        os.environ.clear()
        yield
        os.environ.update(old_env)

    def test_load_config_with_api_key(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key-123"  # pragma: allowlist secret
        os.environ["LOG_LEVEL"] = "DEBUG"
        config = load_config()
        assert config.gemini_api_keys == ["test-key-123"]
        assert config.log_level == "DEBUG"

    def test_load_config_with_multiple_keys(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEY"] = "key1, key2, key3"  # pragma: allowlist secret
        os.environ["LOG_LEVEL"] = "INFO"
        config = load_config()
        assert config.gemini_api_keys == ["key1", "key2", "key3"]

    def test_load_config_with_keys_env_var(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEYS"] = "primary-key"  # pragma: allowlist secret
        os.environ["GEMINI_API_KEY"] = "secondary-key"  # pragma: allowlist secret
        os.environ["LOG_LEVEL"] = "INFO"
        config = load_config()
        assert config.gemini_api_keys == ["primary-key"]

    def test_load_config_default_model(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"  # pragma: allowlist secret
        config = load_config()
        assert config.gemini_model == "gemma-4-31b-it"

    def test_load_config_default_live_model(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"  # pragma: allowlist secret
        config = load_config()
        assert config.gemini_live_model == "gemini-3.1-flash-live-preview"

    def test_load_config_write_confirmation_default_true(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"  # pragma: allowlist secret
        config = load_config()
        assert config.require_write_confirmation is True

    def test_load_config_write_confirmation_false(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"  # pragma: allowlist secret
        os.environ["REQUIRE_WRITE_CONFIRMATION"] = "false"
        config = load_config()
        assert config.require_write_confirmation is False

    def test_config_has_allowed_directories(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"  # pragma: allowlist secret
        config = load_config()
        assert len(config.allowed_directories) == 1
        assert isinstance(config.allowed_directories[0], str)


class TestAppConfig:
    def test_app_config_creation(self) -> None:
        config = AppConfig(
            log_level="DEBUG",
            provider="gemini",
            gemini_api_keys=["key1"],
            gemini_model="test-model",
            gemini_live_model="test-live",
            gemini_live_fallback_model="test-live-fb",
            gemini_fallback_model="test-fb",
            log_file="app.log",
            voice_name="Aoede",
            default_system_instruction="Be helpful.",
            allowed_directories=["/project"],
            require_write_confirmation=True,
        )
        assert config.log_level == "DEBUG"
        assert config.provider == "gemini"
        assert config.require_write_confirmation is True

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from supporter.config import AppConfig, _get_project_root, _int_env, load_config
from supporter.prompts import DEFAULT_SYSTEM_INSTRUCTION


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

    def test_load_config_with_json_array_keys(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEYS"] = (
            '["key1", "key2", "key3"]'  # pragma: allowlist secret
        )
        config = load_config()
        assert config.gemini_api_keys == ["key1", "key2", "key3"]

    def test_load_config_with_json_array_single_key(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEYS"] = '["only-key"]'  # pragma: allowlist secret
        config = load_config()
        assert config.gemini_api_keys == ["only-key"]

    def test_load_config_with_invalid_json_array_raises(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEYS"] = '["unterminated'  # pragma: allowlist secret
        with pytest.raises(ValueError, match="GEMINI_API_KEYS is not valid JSON array"):
            load_config()

    def test_load_config_with_json_array_non_string_items_dropped(
        self, clean_env: Any
    ) -> None:
        os.environ["GEMINI_API_KEYS"] = (
            '["key1", 123, null, "key2"]'  # pragma: allowlist secret
        )
        config = load_config()
        assert config.gemini_api_keys == ["key1", "key2"]

    def test_load_config_default_model(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"  # pragma: allowlist secret
        config = load_config()
        assert config.gemini_model == "gemma-4-31b-it"

    def test_load_config_default_live_model(self, clean_env: Any) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"  # pragma: allowlist secret
        config = load_config()
        assert config.gemini_live_model == "gemini-3.1-flash-live-preview"

    def test_load_config_fallback_model_defaults_on(self, clean_env: Any) -> None:
        """Response formatting is always-on: fallback model defaults, not None."""
        os.environ["GEMINI_API_KEY"] = "test-key"  # pragma: allowlist secret
        config = load_config()
        assert config.gemini_fallback_model == "gemma-4-26b-a4b-it"

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
            live_thinking_level="medium",
            retriable_error_strings={"429"},
            google_api_5xx_exceptions={"InternalServerError"},
            transient_error_strings={"unavailable"},
            http_5xx_status_codes={500},
            rate_limit_error_strings={"429"},
            drain_timeout=2.0,
            context_trigger_tokens=100000,
            context_target_tokens=4000,
            http_retry_attempts=2,
            delegate_max_hard_cap=5,
            delegate_default_parallel=3,
            delegate_default_timeout=180,
            delegate_max_timeout=600,
            delegate_max_tasks=10,
            delegate_max_output_chars=10000,
            delegate_default_persona="Default persona",
            delegate_agent_roster={},
            delegate_max_retries=2,
        )
        assert config.log_level == "DEBUG"
        assert config.provider == "gemini"
        assert config.require_write_confirmation is True


class TestDefaultSystemInstruction:
    def test_no_longer_forces_always_delegate(self) -> None:
        assert "Every task should be delegated even if it is one step." not in (
            DEFAULT_SYSTEM_INSTRUCTION
        )
        assert "## Delegation Strategy" in DEFAULT_SYSTEM_INSTRUCTION

    def test_delegation_is_capsule_only_no_per_task_to_model(self) -> None:
        assert "you do NOT receive per-task events" in DEFAULT_SYSTEM_INSTRUCTION
        assert "There are no per-task messages." in DEFAULT_SYSTEM_INSTRUCTION
        assert "DELEGATION_CAPSULE_RESULT" in DEFAULT_SYSTEM_INSTRUCTION

    def test_no_longer_puts_assigned_task_in_completion_signal(self) -> None:
        assert "assigned_task only" not in DEFAULT_SYSTEM_INSTRUCTION
        assert (
            "containing job_id, task_id, agent, and assigned_task only."
            not in DEFAULT_SYSTEM_INSTRUCTION
        )

    def test_includes_direct_final_answer_synthesis_rule(self) -> None:
        assert (
            "answers the user's original request directly" in DEFAULT_SYSTEM_INSTRUCTION
        )
        assert (
            "Do not frame the final answer as a sub-agent completion update"
            in DEFAULT_SYSTEM_INSTRUCTION
        )

    def test_delegation_table_is_mechanical_not_model_built(self) -> None:
        # The task table is generated by delegate_tasks; the model must relay
        # the returned summary as-is and never hand-write a table.
        assert (
            "rendered to the user automatically the moment delegate_tasks fires"
            in DEFAULT_SYSTEM_INSTRUCTION
        )
        assert (
            "NEVER reproduce, reformat, or hand-write a task table"
            in DEFAULT_SYSTEM_INSTRUCTION
        )


class TestIntEnv:
    def test_non_integer_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_TEST_INT_ENV_BAD", "not-a-number")
        with pytest.raises(ValueError, match="must be an integer"):
            _int_env("_TEST_INT_ENV_BAD", 42)


class TestBrowserCapsConfig:
    """D1: Browser output caps are env-backed config fields with sane defaults."""

    def test_browser_caps_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        c = load_config()
        assert c.browse_page_chars_cap == 50_000
        assert c.browse_batch_chars_cap == 150_000
        assert c.browse_max_links == 100
        assert c.browse_eval_chars_cap == 16_000

    def test_browser_caps_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("BROWSE_PAGE_CHARS_CAP", "8000")
        monkeypatch.setenv("BROWSE_BATCH_CHARS_CAP", "20000")
        monkeypatch.setenv("BROWSE_MAX_LINKS", "50")
        monkeypatch.setenv("BROWSE_EVAL_CHARS_CAP", "4000")
        c = load_config()
        assert c.browse_page_chars_cap == 8000
        assert c.browse_batch_chars_cap == 20000
        assert c.browse_max_links == 50
        assert c.browse_eval_chars_cap == 4000

    def test_delegate_max_output_chars_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("DELEGATE_MAX_OUTPUT_CHARS", "50000")
        c = load_config()
        assert c.delegate_max_output_chars == 50000

    def test_delegate_max_output_chars_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        c = load_config()
        assert c.delegate_max_output_chars == 30000


class TestD7OrchestratorPrompt:
    """D7: orchestrator prompt instructs presenting collected data for browser tasks."""

    def test_orchestrator_prompt_mentions_presenting_data(self) -> None:
        from supporter.prompts import DEFAULT_SYSTEM_INSTRUCTION

        assert "PRESENT the collected data" in DEFAULT_SYSTEM_INSTRUCTION
        assert "browser/research" in DEFAULT_SYSTEM_INSTRUCTION

    def test_page_pilot_prompt_mentions_data_return_contract(self) -> None:
        from supporter.prompts import DELEGATE_AGENT_ROSTER

        pp = DELEGATE_AGENT_ROSTER["page-pilot"]
        assert "Data Return Contract" in pp["persona"]
        assert "findings" in pp["persona"]
        assert "evidence.sources" in pp["persona"]

from typing import Any
from unittest.mock import patch

import pytest

from supporter.tools.file_ops import (
    _get_gitignore_spec,
    _is_blacklisted,
    _validate_path,
    set_confirmation_callback,
)


@pytest.fixture
def temp_project(tmp_path: Any) -> Any:
    root = tmp_path / "project"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hello')")
    (root / ".gitignore").write_text("*.log\nnode_modules/")
    return root


class TestIsBlacklisted:
    def test_env_file_blocked(self) -> None:
        assert _is_blacklisted(".env") is True

    def test_env_with_path_blocked(self) -> None:
        assert _is_blacklisted(".env.local") is False
        assert _is_blacklisted(".env.production") is False

    def test_git_dir_blocked(self) -> None:
        assert _is_blacklisted(".git") is True

    def test_git_file_in_directory_blocked(self) -> None:
        assert _is_blacklisted(".git/config") is True
        assert _is_blacklisted(".git/objects/pack") is True

    def test_venv_dir_blocked(self) -> None:
        assert _is_blacklisted(".venv") is True

    def test_venv_file_in_directory_blocked(self) -> None:
        assert _is_blacklisted(".venv/lib/python/site-packages") is True

    def test_pycache_dir_blocked(self) -> None:
        assert _is_blacklisted("__pycache__") is True

    def test_pycache_file_in_directory_blocked(self) -> None:
        assert _is_blacklisted("__pycache__/module.cpython-313.pyc") is True

    def test_ruff_cache_dir_blocked(self) -> None:
        assert _is_blacklisted(".ruff_cache") is True

    def test_mypy_cache_dir_blocked(self) -> None:
        assert _is_blacklisted(".mypy_cache") is True

    def test_regular_file_allowed(self) -> None:
        assert _is_blacklisted("main.py") is False
        assert _is_blacklisted("readme.md") is False

    def test_regular_dir_allowed(self) -> None:
        assert _is_blacklisted("src") is False
        assert _is_blacklisted("tests") is False
        assert _is_blacklisted("docs") is False

    def test_src_file_allowed(self) -> None:
        assert _is_blacklisted("src/main.py") is False
        assert _is_blacklisted("src/utils/helpers.py") is False


class TestFileOpsUtils:
    def test_set_confirmation_callback(self) -> Any:

        def cb(p: Any, d: Any) -> Any:
            return True

        set_confirmation_callback(cb)
        from supporter.tools.file_ops import _CONFIRMATION_CALLBACK

        assert cb == _CONFIRMATION_CALLBACK
        set_confirmation_callback(None)
        from supporter.tools.file_ops import _CONFIRMATION_CALLBACK

        assert _CONFIRMATION_CALLBACK is None

    def test_get_gitignore_spec_missing(self, tmp_path: Any) -> None:
        assert _get_gitignore_spec(tmp_path) is None

    def test_get_gitignore_spec_error(self, temp_project: Any) -> None:
        with patch("pathlib.Path.open", side_effect=Exception("Read error")):
            assert _get_gitignore_spec(temp_project) is None


class TestPathValidation:
    def test_validate_path_no_allowed(self, mock_file_ops_config: Any) -> None:
        mock_file_ops_config.allowed_directories = []
        with pytest.raises(PermissionError, match="No allowed directories"):
            _validate_path("test.txt")

    def test_validate_path_outside_root(
        self, temp_project: Any, mock_file_ops_config: Any
    ) -> None:
        mock_file_ops_config.allowed_directories = [str(temp_project)]
        with pytest.raises(PermissionError, match="outside project root"):
            _validate_path("/tmp/outside.txt")  # noqa: S108

    def test_validate_path_blacklisted(
        self, temp_project: Any, mock_file_ops_config: Any
    ) -> None:
        mock_file_ops_config.allowed_directories = [str(temp_project)]
        with pytest.raises(PermissionError, match="protected"):
            _validate_path(str(temp_project / ".env"))

    def test_validate_path_gitignore(
        self, temp_project: Any, mock_file_ops_config: Any
    ) -> None:
        mock_file_ops_config.allowed_directories = [str(temp_project)]
        with pytest.raises(PermissionError, match="ignored by \\.gitignore"):
            _validate_path(str(temp_project / "ignored.log"))

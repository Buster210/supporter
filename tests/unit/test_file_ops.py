from typing import Any
from unittest.mock import patch

import pytest

from supporter.tools.file_ops import (
    read_file,
    write_file,
)


@pytest.fixture
def project_root(tmp_path: Any) -> Any:
    root = tmp_path / "project"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hello')")
    return root


@pytest.mark.asyncio
async def test_read_file_exception(
    project_root: Any, mock_file_ops_config: Any
) -> None:
    mock_file_ops_config.allowed_directories = [str(project_root)]
    with patch("pathlib.Path.open", side_effect=Exception("Unexpected")):
        result = await read_file(str(project_root / "src" / "main.py"))
        assert "Error reading file: Unexpected" in result


@pytest.mark.asyncio
async def test_write_file_callback_cancel(
    project_root: Any, mock_file_ops_config: Any
) -> None:
    mock_file_ops_config.allowed_directories = [str(project_root)]
    mock_file_ops_config.require_write_confirmation = True
    with patch("supporter.tools.file_ops._CONFIRMATION_CALLBACK", return_value=False):
        result = await write_file(str(project_root / "cancel.txt"), "content")
        assert "cancelled" in result


@pytest.mark.asyncio
async def test_write_file_no_confirmation_available(
    project_root: Any, mock_file_ops_config: Any
) -> None:
    mock_file_ops_config.allowed_directories = [str(project_root)]
    mock_file_ops_config.require_write_confirmation = True
    with (
        patch("sys.stdin.isatty", return_value=False),
        patch("supporter.tools.file_ops._CONFIRMATION_CALLBACK", None),
    ):
        result = await write_file(str(project_root / "fail.txt"), "content")
        assert "Error: Interactive confirmation required" in result


@pytest.mark.asyncio
async def test_write_file_exception(
    project_root: Any, mock_file_ops_config: Any
) -> None:
    mock_file_ops_config.allowed_directories = [str(project_root)]
    mock_file_ops_config.require_write_confirmation = False
    with patch("pathlib.Path.open", side_effect=Exception("Unexpected")):
        result = await write_file(str(project_root / "fail.txt"), "content")
        assert "Error writing file: Unexpected" in result

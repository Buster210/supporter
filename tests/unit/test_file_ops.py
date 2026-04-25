from typing import Any
from unittest.mock import patch

import pytest

from supporter.tools.file_ops import list_dir, read_file, write_file


@pytest.fixture
def project_root(tmp_path: Any) -> Any:
    root = tmp_path / "project"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hello')")
    (root / ".gitignore").write_text("*.log\nnode_modules/")
    return root


@pytest.mark.asyncio
async def test_read_file_not_found(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        result = await read_file(str(project_root / "missing.txt"))
        assert "Error: File not found" in result


@pytest.mark.asyncio
async def test_read_file_exception(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        with patch("pathlib.Path.open", side_effect=Exception("Unexpected")):
            result = await read_file(str(project_root / "src" / "main.py"))
            assert "Error reading file: Unexpected" in result


@pytest.mark.asyncio
async def test_write_file_callback_cancel(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        mock_config.require_write_confirmation = True
        with patch(
            "supporter.tools.file_ops._CONFIRMATION_CALLBACK", return_value=False
        ):
            result = await write_file(str(project_root / "cancel.txt"), "content")
            assert "cancelled" in result


@pytest.mark.asyncio
async def test_write_file_no_confirmation_available(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        mock_config.require_write_confirmation = True
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("supporter.tools.file_ops._CONFIRMATION_CALLBACK", None),
        ):
            result = await write_file(str(project_root / "fail.txt"), "content")
            assert "Error: Interactive confirmation required" in result


@pytest.mark.asyncio
async def test_write_file_exception(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        mock_config.require_write_confirmation = False
        with patch("pathlib.Path.open", side_effect=Exception("Unexpected")):
            result = await write_file(str(project_root / "fail.txt"), "content")
            assert "Error writing file: Unexpected" in result


@pytest.mark.asyncio
async def test_list_dir_empty(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        empty_dir = project_root / "empty"
        empty_dir.mkdir()
        result = await list_dir(str(empty_dir))
        assert "Directory is empty" in result


@pytest.mark.asyncio
async def test_list_dir_gitignore_filter(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        (project_root / "ignored.log").touch()
        result = await list_dir(str(project_root))
        assert "ignored.log" not in result


@pytest.mark.asyncio
async def test_list_dir_blacklisted_filter(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        (project_root / ".venv").mkdir()
        result = await list_dir(str(project_root))
        assert ".venv" not in result


@pytest.mark.asyncio
async def test_list_dir_exception(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        with patch("os.scandir", side_effect=Exception("Unexpected")):
            result = await list_dir(str(project_root))
            assert "Error listing directory: Unexpected" in result


@pytest.mark.asyncio
async def test_read_file(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        content = await read_file(str(project_root / "src" / "main.py"))
        assert "print('hello')" in content
        (project_root / "lines.txt").write_text("line1\nline2\nline3\nline4")
        partial = await read_file(str(project_root / "lines.txt"), offset=6, limit=11)
        assert "line2\nline3" in partial


@pytest.mark.asyncio
@patch("supporter.tools.file_ops._CONFIRMATION_CALLBACK", return_value=True)
async def test_write_file_new(mock_confirm: Any, project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        mock_config.require_write_confirmation = True
        new_file = str(project_root / "new.py")
        result = await write_file(new_file, "print('new')")
        assert "Successfully wrote" in result


@pytest.mark.asyncio
async def test_write_file_tty_confirm(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        mock_config.require_write_confirmation = True
        target = str(project_root / "tty.txt")
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="y"):
                result = await write_file(target, "tty content")
                assert "Successfully wrote" in result
            with patch("builtins.input", return_value="n"):
                result = await write_file(target, "cancelled content")
                assert "cancelled" in result.lower()


@pytest.mark.asyncio
async def test_write_file_offset_new(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        mock_config.require_write_confirmation = False
        target = str(project_root / "offset_new.txt")
        await write_file(target, "data", offset=10)
        content = (project_root / "offset_new.txt").read_bytes()
        assert content[10:] == b"data"


@pytest.mark.asyncio
async def test_write_file_offset_existing(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        mock_config.require_write_confirmation = False
        target_path = project_root / "existing.txt"
        target_path.write_text("0123456789")
        await write_file(str(target_path), "abc", offset=5)
        assert target_path.read_text() == "01234abc"


@pytest.mark.asyncio
async def test_write_file_offset_limit_replace(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        mock_config.require_write_confirmation = False
        target_path = project_root / "complex.txt"
        target_path.write_text("0123456789")
        await write_file(str(target_path), "XYZ", offset=3, limit=2)
        assert target_path.read_text() == "012XYZ56789"


@pytest.mark.asyncio
async def test_write_file_confirmation_uses_fallback_diff_on_error(
    project_root: Any,
) -> Any:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        mock_config.require_write_confirmation = True
        target_path = project_root / "existing_diff.txt"
        target_path.write_text("old content")
        captured = {}

        def confirm_callback(path: Any, diff_text: Any) -> Any:
            captured["path"] = path
            captured["diff_text"] = diff_text
            return True

        with (
            patch("supporter.tools.file_ops._CONFIRMATION_CALLBACK", confirm_callback),
            patch("difflib.unified_diff", side_effect=Exception("diff failed")),
        ):
            result = await write_file(str(target_path), "new content")
        assert "Successfully wrote" in result
        assert captured["path"] == target_path
        assert "Error generating diff: diff failed" in captured["diff_text"]
        assert "Proposed Content:\nnew content" in captured["diff_text"]


@pytest.mark.asyncio
async def test_list_dir_returns_error_for_file_path(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        result = await list_dir(str(project_root / "src" / "main.py"))
        assert "Error: Path is not a directory" in result


@pytest.mark.asyncio
async def test_list_dir_skips_entries_that_raise_oserror(project_root: Any) -> Any:

    class FakeEntry:
        def __init__(
            self, name: Any, is_dir_result: Any = None, error: Any = None
        ) -> None:
            self.name = name
            self._is_dir_result = is_dir_result
            self._error = error

        def is_dir(self) -> Any:
            if self._error is not None:
                raise self._error
            return self._is_dir_result

    class FakeScandir:
        def __enter__(self) -> Any:
            return iter(
                [
                    FakeEntry("broken", error=OSError("stat failed")),
                    FakeEntry("visible.txt", is_dir_result=False),
                ]
            )

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
            return False

    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        with patch("os.scandir", return_value=FakeScandir()):
            result = await list_dir(str(project_root))
    assert result == "[FILE] visible.txt"


@pytest.mark.asyncio
async def test_list_dir_returns_scandir_oserror(project_root: Any) -> None:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        with patch("os.scandir", side_effect=OSError("permission denied")):
            result = await list_dir(str(project_root))
    assert result == "Error accessing directory: permission denied"

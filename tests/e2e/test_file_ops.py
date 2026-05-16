from pathlib import Path

import pytest

from supporter.config import config as real_config
from supporter.tools import _resolve_path
from supporter.tools.file_ops import read_file, write_file


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hello')")
    (root / ".gitignore").write_text("*.log\nnode_modules/")
    return root


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_file_ops_e2e_write_and_read_real_file(project_root: Path) -> None:
    saved_dirs = real_config.allowed_directories
    saved_confirm = real_config.require_write_confirmation
    real_config.allowed_directories = [str(project_root)]
    real_config.require_write_confirmation = False
    _resolve_path.cache_clear()
    try:
        test_file = project_root / "e2e_test.txt"
        content = "E2E test content"
        await write_file(str(test_file), content)
        assert test_file.exists()
        result = await read_file(str(test_file))
        assert content in result
    finally:
        real_config.allowed_directories = saved_dirs
        real_config.require_write_confirmation = saved_confirm
        _resolve_path.cache_clear()

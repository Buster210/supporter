from pathlib import Path
from unittest.mock import patch

import pytest

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
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = [str(project_root)]
        mock_config.require_write_confirmation = False
        test_file = project_root / "e2e_test.txt"
        content = "E2E test content"
        await write_file(str(test_file), content)
        assert test_file.exists()
        result = await read_file(str(test_file))
        assert content in result

from supporter.config import CRYSTAL_GRADIENT_STOPS, THEME
from supporter.tui.utils import apply_crystal_gradient


def test_theme_dict() -> None:
    assert THEME["background"] == "#121212"
    assert THEME["bubble_bg"] == "#1e1e1e"
    assert THEME["header_teal"] == "#00ffcc"


def test_crystal_gradient_stops() -> None:
    assert len(CRYSTAL_GRADIENT_STOPS) == 4
    assert CRYSTAL_GRADIENT_STOPS[0] == (0, 255, 255)


def test_apply_crystal_gradient() -> None:
    result = apply_crystal_gradient("Hi")
    assert len(result.plain) == 2


def test_apply_crystal_gradient_single_char() -> None:
    result = apply_crystal_gradient("A")
    assert len(result.plain) == 1


def test_apply_crystal_gradient_empty() -> None:
    result = apply_crystal_gradient("")
    assert len(result.plain) == 0

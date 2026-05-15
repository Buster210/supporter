import re

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

THEME = {
    "background": "#121212",
    "bubble_bg": "#1e1e1e",
    "header_teal": "#00ffcc",
    "magenta": "#ff06b5",
    "green": "#00ff00",
    "blue": "#0080ff",
    "yellow": "#ffeb3b",
    "meta_gray": "#999999",
}

CRYSTAL_GRADIENT_STOPS: list[tuple[int, int, int]] = [
    (0, 255, 255),
    (0, 255, 180),
    (0, 180, 255),
    (100, 200, 255),
]

MODAL_WIDTH_SCALE = 1.3
MODAL_MAX_WIDTH_PERCENT = 0.9
MODAL_PADDING = 6
SCROLL_STEP = 5
COLLAPSED_SUMMARY_LEN = 50
RENDER_COALESCE_INTERVAL = 0.08
MARKDOWN_SYNTAX_MARKERS = [
    re.compile(p, re.MULTILINE)
    for p in (
        r"[*+-]\s",
        r"\d+\.\s",
        r"#+\s",
        r"\*\*.*?\*\*",
        r"\*.*?\*",
        r"`.*?`",
        r"\[.*?\]\(.*?\)",
        r">\s",
    )
]

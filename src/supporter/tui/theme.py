from rich.text import Text

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


def apply_crystal_gradient(text: str) -> Text:
    rich_text = Text(justify="center")
    char_count = len(text)
    num_stops = len(CRYSTAL_GRADIENT_STOPS) - 1

    for i, char in enumerate(text):
        progress = i / max(char_count - 1, 1)
        segment = min(int(progress * num_stops), num_stops - 1)
        local_progress = progress * num_stops - segment

        start_rgb = CRYSTAL_GRADIENT_STOPS[segment]
        end_rgb = CRYSTAL_GRADIENT_STOPS[segment + 1]

        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * local_progress)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * local_progress)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * local_progress)

        rich_text.append(char, style=f"bold rgb({r},{g},{b})")
    return rich_text

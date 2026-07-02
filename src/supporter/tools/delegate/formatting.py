"""Canonical markdown-table formatter for delegation views."""


def _sanitize_cell(value: str) -> str:
    """Strip embedded pipes and newlines so the table never breaks."""
    return value.replace("|", "\\|").replace("\n", " ")


def format_delegation_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a GitHub-flavored markdown table.

    Guarantees a header row, a separator row, and pipe-delimited alignment.
    Empty *rows* still returns the header + separator (no data rows).
    Cell values are coerced to ``str()`` and sanitised of ``|`` and ``\n``.
    """
    header_row = "| " + " | ".join(_sanitize_cell(h) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = [
        "| " + " | ".join(_sanitize_cell(str(c)) for c in row) + " |" for row in rows
    ]
    return "\n".join([header_row, separator, *data_rows])


if __name__ == "__main__":
    # Self-check
    result = format_delegation_table(
        ["Task", "Agent"], [["fix bug", "explorer"], ["add test", "tester"]]
    )
    lines = result.splitlines()
    assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}"
    assert lines[0] == "| Task | Agent |", f"Header wrong: {lines[0]}"
    assert lines[1] == "| --- | --- |", f"Separator wrong: {lines[1]}"
    assert lines[2] == "| fix bug | explorer |", f"Row 1 wrong: {lines[2]}"
    assert lines[3] == "| add test | tester |", f"Row 2 wrong: {lines[3]}"

    # Empty rows
    empty = format_delegation_table(["Col"], [])
    assert empty.splitlines() == ["| Col |", "| --- |"], f"Empty wrong: {empty}"

    # Pipe and newline sanitisation
    dirty = format_delegation_table(["A"], [["a|b\nc"]])
    assert dirty.splitlines()[2] == "| a\\|b c |", f"Sanitize wrong: {dirty}"

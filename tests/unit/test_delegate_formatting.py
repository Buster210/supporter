"""Tests for the canonical delegation table formatter."""

from supporter.tools.delegate.formatting import format_delegation_table


class TestFormatDelegationTable:
    """Verify format_delegation_table produces correct markdown tables."""

    def test_basic_table(self) -> None:
        result = format_delegation_table(
            ["Task", "Agent"], [["fix bug", "explorer"], ["add test", "tester"]]
        )
        lines = result.splitlines()
        assert len(lines) == 4
        assert lines[0] == "| Task | Agent |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| fix bug | explorer |"
        assert lines[3] == "| add test | tester |"

    def test_empty_rows(self) -> None:
        result = format_delegation_table(["Col"], [])
        assert result.splitlines() == ["| Col |", "| --- |"]

    def test_pipe_escaping(self) -> None:
        result = format_delegation_table(["A"], [["a|b"]])
        assert result.splitlines()[2] == "| a\\|b |"

    def test_newline_escaping(self) -> None:
        result = format_delegation_table(["A"], [["a\nb"]])
        assert result.splitlines()[2] == "| a b |"

    def test_coerces_to_str(self) -> None:
        result = format_delegation_table(["N", "Val"], [["42", "None"]])
        assert result.splitlines()[2] == "| 42 | None |"

    def test_four_columns(self) -> None:
        result = format_delegation_table(
            ["#", "Task ID", "Agent", "Dependencies"],
            [["1", "t1", "explorer", "after: none"]],
        )
        lines = result.splitlines()
        assert lines[0] == "| # | Task ID | Agent | Dependencies |"
        assert lines[1] == "| --- | --- | --- | --- |"
        assert lines[2] == "| 1 | t1 | explorer | after: none |"

    def test_multiple_rows(self) -> None:
        result = format_delegation_table(
            ["Task", "Status", "Agent", "Elapsed"],
            [
                ["`t1`", "RUNNING", "explorer", "5s / 30s"],
                ["`t2`", "PENDING", "tester", "0.0s"],
            ],
        )
        lines = result.splitlines()
        assert len(lines) == 4  # header + separator + 2 data rows

    def test_single_column(self) -> None:
        result = format_delegation_table(["Only"], [["value"]])
        assert result.splitlines() == ["| Only |", "| --- |", "| value |"]

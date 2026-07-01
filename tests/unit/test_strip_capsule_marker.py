"""Self-check for bubble.strip_capsule_marker (balanced-brace JSON stripping)."""

from supporter.tui.bubble import strip_capsule_marker


def test_strips_marker_with_nested_braces() -> None:
    text = 'DELEGATION_CAPSULE_RESULT{"a":{"b":1}}Hello'
    assert strip_capsule_marker(text) == "Hello"


def test_strips_milestone_result_marker() -> None:
    text = 'MILESTONE_RESULT{"x": [1, 2, {"y": "z"}]}The plan is done.'
    assert strip_capsule_marker(text) == "The plan is done."


def test_brace_in_string_does_not_confuse_matching() -> None:
    text = 'DELEGATION_CAPSULE_RESULT{"summary": "uses { and } in text"}After'
    assert strip_capsule_marker(text) == "After"


def test_plain_text_unchanged() -> None:
    text = "Just a normal reply with no sentinel."
    assert strip_capsule_marker(text) == text


def test_prefix_before_marker_preserved() -> None:
    text = 'Intro. DELEGATION_CAPSULE_RESULT{"job_id": "1"}Outro.'
    assert strip_capsule_marker(text) == "Intro. Outro."


def test_marker_with_colon_separator() -> None:
    # Real model output uses "MARKER: {...}" with a colon-space separator.
    text = 'DELEGATION_CAPSULE_RESULT: {"version": "v1", "job_id": "x"}Done.'
    assert strip_capsule_marker(text) == "Done."


def test_truncated_json_preserves_trailing_text() -> None:
    # Unbalanced/truncated blob must NOT swallow real prose after it.
    text = 'Hello DELEGATION_CAPSULE_RESULT{"a": {"b": 1 then more real prose'
    assert strip_capsule_marker(text) == text

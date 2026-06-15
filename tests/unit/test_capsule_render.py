from supporter.tools.delegate.capsule_render import render_evidence, render_findings


class TestRenderEvidence:
    def test_empty_evidence_renders_none(self) -> None:
        assert render_evidence({}) == "- Evidence: none"

    def test_non_dict_renders_none(self) -> None:
        assert render_evidence("nope") == "- Evidence: none"

    def test_keys_render_in_stable_order(self) -> None:
        # Insertion order is reversed vs EVIDENCE_KEYS; output must still be ordered.
        evidence = {
            "sources": ["https://x"],
            "files_read": ["a.py"],
            "files_changed": ["b.py"],
        }
        out = render_evidence(evidence)
        assert out.index("Files read") < out.index("Files changed")
        assert out.index("Files changed") < out.index("Sources")

    def test_empty_groups_omitted(self) -> None:
        out = render_evidence({"files_read": ["a.py"], "commands_run": []})
        assert "Files read: a.py" in out
        assert "Commands run" not in out

    def test_overflow_is_summarized(self) -> None:
        out = render_evidence({"files_read": [f"f{i}.py" for i in range(55)]})
        assert "(+5 more)" in out


class TestRenderFindings:
    def test_empty_renders_none(self) -> None:
        assert render_findings([]) == "- Findings: none"

    def test_non_list_renders_none(self) -> None:
        assert render_findings({"a": 1}) == "- Findings: none"

    def test_items_render_as_sublist(self) -> None:
        out = render_findings(["bug A", "bug B"])
        assert out == "- Findings:\n  - bug A\n  - bug B"

    def test_overflow_is_summarized(self) -> None:
        out = render_findings([f"f{i}" for i in range(55)])
        assert "(+5 more)" in out

class TestD3D5CapsuleRendering:
    """D3/D5: capsule rendering shows full findings, not truncated summaries."""

    def test_many_findings_all_shown_up_to_cap(self) -> None:
        """D5: _MAX_ITEMS raised to 50 — 50 findings should all render."""
        findings = [f"Finding {i}: detailed result item" for i in range(50)]
        out = render_findings(findings)
        for i in range(50):
            assert f"Finding {i}:" in out
        assert "(+" not in out  # no overflow at exactly 50

    def test_long_evidence_items_render_fully(self) -> None:
        """D5: _ITEM_PREVIEW_CHARS raised to 600 — long items render more."""
        long_url = "https://example.com/" + "a" * 550
        out = render_evidence({"sources": [long_url]})
        # Should include most of the URL, not just 200 chars
        assert len(long_url) - 100 > 200  # confirm test premise
        assert "Sources:" in out
        # The full URL (550+ chars) should survive since cap is 600
        assert long_url[:550] in out

    def test_multi_finding_task_shows_all_findings(self) -> None:
        """D3: findings from a task with many findings are all surfaced."""
        from supporter.tools.delegate.capsule import (
            extract_task_capsule_fields,
        )

        output = """Some preamble text here.

```json
{
  "summary": "Research task completed",
  "evidence": {
    "files_read": [],
    "files_changed": [],
    "commands_run": [],
    "sources": ["https://a.com", "https://b.com"]
  },
  "findings": [
    "Item 1: first detail",
    "Item 2: second detail",
    "Item 3: third detail"
  ],
  "handoff": "",
  "confidence": "high"
}
```"""
        fields = extract_task_capsule_fields(output)
        assert len(fields["findings"]) == 3
        assert "Item 1: first detail" in fields["findings"]
        assert "Item 3: third detail" in fields["findings"]
        assert fields["summary"] == "Research task completed"

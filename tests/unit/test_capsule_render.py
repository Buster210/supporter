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
        out = render_evidence({"files_read": [f"f{i}.py" for i in range(25)]})
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
        out = render_findings([f"f{i}" for i in range(22)])
        assert "(+2 more)" in out

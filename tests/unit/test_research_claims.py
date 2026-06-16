from __future__ import annotations

import json
from typing import Any

import pytest

from supporter.config import config
from supporter.tools.delegate.capsule import (
    extract_task_capsule_fields,
    validate_delegation_payload,
)
from supporter.tools.research.claims import (
    _corroboration_key,
    _normalize_statement,
    dedup_to_assertions,
    ingest_claims,
    load_claims,
    query_assertions,
    registrable_domain,
    research_dir,
)


@pytest.fixture(autouse=True)
def isolate_research(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])


def _claim(statement: str, url: str, stance: str = "support") -> dict[str, Any]:
    return {"statement": statement, "url": url, "snippet": "q", "stance": stance}


# --- registrable_domain ---------------------------------------------------


def test_registrable_domain_plain_host() -> None:
    assert registrable_domain("https://example.com/a/b?x=1") == "example.com"


def test_registrable_domain_strips_www() -> None:
    assert registrable_domain("https://www.example.com") == "example.com"


def test_registrable_domain_subdomain_collapses_to_last_two_labels() -> None:
    assert registrable_domain("https://docs.api.example.com/page") == "example.com"


def test_registrable_domain_no_scheme() -> None:
    assert registrable_domain("example.com/path") == "example.com"


def test_registrable_domain_bad_url_returns_empty() -> None:
    assert registrable_domain("") == ""
    assert registrable_domain("not a url") == ""


# --- statement normalization ----------------------------------------------


def test_normalize_statement_equates_case_whitespace_punctuation() -> None:
    a = _normalize_statement("  The Sky is Blue.  ")
    b = _normalize_statement("the sky   is blue")
    assert a == b == "the sky is blue"


def test_corroboration_key_merges_reworded_and_reordered() -> None:
    a = _corroboration_key("Python was created by Guido van Rossum.")
    b = _corroboration_key("Guido van Rossum created Python")
    assert a == b


def test_corroboration_key_distinguishes_negation() -> None:
    assert _corroboration_key("X is fast") != _corroboration_key("X is not fast")
    # contracted negation must also stay distinct
    assert _corroboration_key("it works") != _corroboration_key("it doesn't work")


def test_corroboration_key_distinguishes_numbers() -> None:
    assert _corroboration_key("released in 1991") != _corroboration_key(
        "released in 2000"
    )


def test_corroboration_key_without_kept_distinct_from_with() -> None:
    assert _corroboration_key("tea with milk") != _corroboration_key(
        "tea without milk"
    )


def test_corroboration_key_falls_back_when_all_function_words() -> None:
    # all-stopword statement must not collapse to an empty key
    assert _corroboration_key("it is the one") != ""


def test_reworded_fact_two_domains_corroborated() -> None:
    claims = [
        _claim("Python was created by Guido van Rossum.", "https://a.com/x"),
        _claim("Guido van Rossum created Python", "https://b.org/y"),
    ]
    [assertion] = dedup_to_assertions(claims)
    assert assertion["status"] == "corroborated"
    assert sorted(assertion["supporting_domains"]) == ["a.com", "b.org"]


# --- ingest / load round-trip ---------------------------------------------


def test_ingest_and_load_round_trip() -> None:
    n = ingest_claims("q1", "t1", "page-pilot", [_claim("X is true", "https://a.com")])
    assert n == 1
    loaded = load_claims("q1")
    assert len(loaded) == 1
    row = loaded[0]
    assert row["statement"] == "X is true"
    assert row["source_domain"] == "a.com"
    assert row["question_id"] == "q1"
    assert row["task_id"] == "t1"
    assert row["agent"] == "page-pilot"


def test_load_missing_store_returns_empty() -> None:
    assert load_claims("never_written") == []


def test_ingest_drops_malformed_and_empty_writes_nothing() -> None:
    n = ingest_claims(
        "q2",
        "t1",
        "explorer",
        [
            {"statement": "", "url": "https://a.com"},  # empty statement
            {"statement": "ok", "url": ""},  # empty url
            "not a dict",  # type: ignore[list-item]
        ],
    )
    assert n == 0
    assert load_claims("q2") == []


def test_load_skips_malformed_jsonl_line() -> None:
    ingest_claims("q3", "t1", "explorer", [_claim("good", "https://a.com")])
    path = research_dir("q3") / "claims.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write("{ this is not json\n")
    loaded = load_claims("q3")
    assert len(loaded) == 1
    assert loaded[0]["statement"] == "good"


def test_bad_question_id_rejected() -> None:
    with pytest.raises(ValueError):
        ingest_claims("../escape", "t1", "explorer", [_claim("x", "https://a.com")])


# --- dedup / assertion status ---------------------------------------------


def test_single_domain_is_uncorroborated() -> None:
    claims = [_claim("fact", "https://a.com")]
    [assertion] = dedup_to_assertions(claims)
    assert assertion["status"] == "uncorroborated"
    assert assertion["supporting_domains"] == ["a.com"]
    assert assertion["claim_count"] == 1


def test_two_independent_domains_corroborated() -> None:
    claims = [_claim("fact", "https://a.com"), _claim("FACT.", "https://b.com")]
    [assertion] = dedup_to_assertions(claims)
    assert assertion["status"] == "corroborated"
    assert assertion["supporting_domains"] == ["a.com", "b.com"]
    assert assertion["claim_count"] == 2


def test_same_fact_same_domain_not_corroborated() -> None:
    claims = [
        _claim("fact", "https://a.com/1"),
        _claim("fact", "https://a.com/2"),
    ]
    [assertion] = dedup_to_assertions(claims)
    assert assertion["supporting_domains"] == ["a.com"]
    assert assertion["status"] == "uncorroborated"


def test_refute_marks_conflicted() -> None:
    claims = [
        _claim("fact", "https://a.com"),
        _claim("fact", "https://b.com", stance="refute"),
    ]
    [assertion] = dedup_to_assertions(claims)
    assert assertion["status"] == "conflicted"
    assert assertion["refuting_domains"] == ["b.com"]


def test_query_assertions_status_filter() -> None:
    ingest_claims(
        "q4",
        "t1",
        "page-pilot",
        [_claim("a", "https://a.com"), _claim("a", "https://b.com")],
    )
    ingest_claims("q4", "t2", "page-pilot", [_claim("b", "https://a.com")])
    corroborated = query_assertions("q4", status="corroborated")
    assert [a["statement"] for a in corroborated] == ["a"]
    uncorroborated = query_assertions("q4", status="uncorroborated")
    assert [a["statement"] for a in uncorroborated] == ["b"]


# --- capsule claims parsing -----------------------------------------------


def _payload(claims_json: str) -> str:
    return (
        "done\n```json\n"
        + json.dumps(
            {
                "summary": "s",
                "evidence": {
                    "files_read": [],
                    "files_changed": [],
                    "commands_run": [],
                    "sources": [],
                },
                "findings": [],
                "claims": json.loads(claims_json),
                "handoff": "",
                "confidence": "high",
            }
        )
        + "\n```"
    )


def test_extract_parses_and_normalizes_claims() -> None:
    out = _payload(
        '[{"statement": " X ", "url": " https://a.com ", "snippet": 1,'
        ' "stance": "bogus"}, {"statement": "", "url": "https://b.com"}]'
    )
    fields = extract_task_capsule_fields(out)
    assert fields["claims"] == [
        {
            "statement": "X",
            "url": "https://a.com",
            "snippet": "",
            "stance": "support",
        }
    ]


def test_extract_missing_claims_defaults_empty() -> None:
    out = (
        "done\n```json\n"
        + json.dumps(
            {
                "summary": "s",
                "evidence": {
                    "files_read": [],
                    "files_changed": [],
                    "commands_run": [],
                    "sources": [],
                },
                "findings": [],
                "handoff": "",
                "confidence": "high",
            }
        )
        + "\n```"
    )
    assert extract_task_capsule_fields(out)["claims"] == []


def test_validate_payload_claims_optional() -> None:
    no_claims = (
        '```json\n{"summary": "s", "evidence": {"files_read": [],'
        ' "files_changed": [], "commands_run": [], "sources": []},'
        ' "findings": [], "handoff": "", "confidence": "high"}\n```'
    )
    assert validate_delegation_payload(no_claims) is True
    with_empty = _payload("[]")
    assert validate_delegation_payload(with_empty) is True


def test_validate_payload_rejects_non_list_claims() -> None:
    bad = (
        '```json\n{"summary": "s", "evidence": {"files_read": [],'
        ' "files_changed": [], "commands_run": [], "sources": []},'
        ' "findings": [], "claims": "nope", "handoff": "",'
        ' "confidence": "high"}\n```'
    )
    assert validate_delegation_payload(bad) is False

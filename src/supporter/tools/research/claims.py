from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ...logger import logger
from .. import resolved_project_root

_STANCES = frozenset({"support", "refute"})
_STRIP_CHARS = " \t\n\r.,;:!?\"'()[]{}"

# Corroboration matching: two statements corroborate when their content-token
# SETS match, so independent sources that reword the same fact still merge
# ("Guido created Python" == "Python was created by Guido"). Pure function
# words are dropped; negations and numbers are KEPT so opposite/quantified
# facts never collapse together.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "and",
        "or",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "from",
        "into",
        "than",
        "then",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "will",
        "would",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "but",
    }
)
# Never dropped, even though some are short function words -- they flip meaning.
_NEGATIONS = frozenset(
    {"not", "no", "never", "none", "cannot", "nor", "without", "neither", "nothing"}
)


def _safe_id(question_id: str) -> str:
    if not question_id or Path(question_id).name != question_id:
        raise ValueError("Invalid research question id")
    return question_id


def research_dir(question_id: str) -> Path:
    return resolved_project_root() / ".supporter" / "research" / _safe_id(question_id)


def _claims_path(question_id: str) -> Path:
    return research_dir(question_id) / "claims.jsonl"


def registrable_domain(url: str) -> str:
    """eTLD+1 from a URL host.

    Coarse: takes the last two labels, so multi-label suffixes (e.g.
    example.co.uk) collapse to co.uk. Good enough for source-independence
    grouping in P0; a real public-suffix list is future work.
    """
    try:
        parsed = urlparse(url if "://" in url else f"//{url}")
        host = parsed.hostname
    except ValueError:
        return ""
    if not host:
        return ""
    host = host.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    labels = [label for label in host.split(".") if label]
    if len(labels) < 2:
        return ""
    return ".".join(labels[-2:])


def _normalize_statement(statement: str) -> str:
    """Human-readable normalized form: lowercase, collapse whitespace, strip
    surrounding punctuation."""
    return " ".join(statement.lower().split()).strip(_STRIP_CHARS)


def _corroboration_key(statement: str) -> str:
    """Order-independent content-token key used to group claims into one
    assertion. Drops pure function words; keeps negations and numbers so
    'X is fast' and 'X is not fast' (or '...1991' vs '...2000') stay distinct.
    Falls back to the full token set if filtering would empty the key."""
    # Expand "n't" so contracted negations survive tokenization.
    norm = _normalize_statement(statement).replace("n't", " not")
    tokens = _TOKEN_RE.findall(norm)
    kept = [
        t
        for t in tokens
        if t in _NEGATIONS or any(c.isdigit() for c in t) or t not in _STOPWORDS
    ]
    return " ".join(sorted(set(kept or tokens)))


def _normalize_claim(claim: Any) -> dict[str, Any] | None:
    if not isinstance(claim, dict):
        return None
    statement = claim.get("statement")
    url = claim.get("url")
    if not isinstance(statement, str) or not statement.strip():
        return None
    if not isinstance(url, str) or not url.strip():
        return None
    snippet = claim.get("snippet")
    stance = claim.get("stance")
    return {
        "statement": statement.strip(),
        "url": url.strip(),
        "snippet": snippet.strip() if isinstance(snippet, str) else "",
        "stance": stance if stance in _STANCES else "support",
    }


def ingest_claims(
    question_id: str,
    task_id: str,
    agent: str,
    claims: list[dict[str, Any]],
    round: int = 0,
) -> int:
    """Append normalized claims to the per-question JSONL store. Returns count
    written. Malformed claims are dropped; empty input writes nothing."""
    rows: list[dict[str, Any]] = []
    for claim in claims or []:
        norm = _normalize_claim(claim)
        if norm is None:
            continue
        norm.update(
            {
                "source_domain": registrable_domain(norm["url"]),
                "question_id": question_id,
                "task_id": task_id,
                "agent": agent,
                "round": round,
            }
        )
        rows.append(norm)
    if not rows:
        return 0
    path = _claims_path(question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def load_claims(question_id: str) -> list[dict[str, Any]]:
    path = _claims_path(question_id)
    if not path.exists():
        return []
    claims: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.debug(f"Skipping malformed claim line: {exc}")
                continue
            if isinstance(data, dict):
                claims.append(data)
    return claims


def assertion_status(supporting_domains: list[str], refuting_domains: list[str]) -> str:
    """Cross-source label. >=2 distinct independent domains == corroborated;
    any refuting domain == conflicted; else uncorroborated."""
    if refuting_domains:
        return "conflicted"
    if len({domain for domain in supporting_domains if domain}) >= 2:
        return "corroborated"
    return "uncorroborated"


def dedup_to_assertions(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group claims by normalized statement into assertions, each labeled by
    cross-source corroboration. Independence is by distinct registrable domain."""
    groups: dict[str, dict[str, Any]] = {}
    for claim in claims:
        statement = claim.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            continue
        key = _corroboration_key(statement)
        if not key:
            continue
        group = groups.setdefault(
            key,
            {
                "key": key,
                "statement": statement.strip(),
                "_support": set(),
                "_refute": set(),
                "claim_count": 0,
                "sources": [],
            },
        )
        group["claim_count"] += 1
        url = claim.get("url", "")
        domain = claim.get("source_domain") or registrable_domain(url)
        stance = "refute" if claim.get("stance") == "refute" else "support"
        group["sources"].append(
            {
                "url": url,
                "domain": domain,
                "snippet": claim.get("snippet", ""),
                "stance": stance,
            }
        )
        if not domain:
            continue
        if stance == "refute":
            group["_refute"].add(domain)
        else:
            group["_support"].add(domain)
    assertions: list[dict[str, Any]] = []
    for group in groups.values():
        supporting = sorted(group.pop("_support"))
        refuting = sorted(group.pop("_refute"))
        group["supporting_domains"] = supporting
        group["refuting_domains"] = refuting
        group["status"] = assertion_status(supporting, refuting)
        assertions.append(group)
    return assertions


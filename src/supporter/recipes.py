"""Recipe store: replayable automations with no LLM in the loop.

A *recipe* is a named, parameterised sequence of steps. The brain writes
one whenever it solves a multi-step problem by hand. Next time the same
problem comes up, the brain looks up the recipe and ``run_recipe(...)``
replays it deterministically without spending any LLM tokens.

Step types (intentionally small):

* ``shell``        — run a sandboxed command (project-root relative, no shell).
* ``read``         — read a project-root file.
* ``write``        — write a project-root file.
* ``http_get``     — GET a URL, return text or status.
* ``memory_write`` — persist a note to the working-memory store.
* ``delay``        — sleep N milliseconds.
* ``human_break``  — sleep a RANDOM duration each run; value "min||max" ms
  (blank = default).
* ``assert_exists``— fail the recipe if a path does not exist.
* ``assert_eq``    — fail the recipe if two strings differ.
* ``emit``         — attach a string to the run's output.

Design choices
--------------

* **No code-execution.** Recipes are data, not Python source — so a recipe
  written today can be replayed a year from now without trusting the
  writer's intent.
* **Sandbox-aware.** Every shell step goes through the project's bash
  policy. File steps are confined to the project root.
* **LLM-free replay.** ``run_recipe`` makes zero LLM calls. It returns
  the chain of step results, which the brain can use as evidence.
* **Crash-safe.** Recipes persist as JSON next to ``.supporter/``; the
  store is append-mostly with an in-memory index for fast lookup by name.
* **Bounded.** The whole store is held in memory; ``MAX_RECIPES`` caps
  total recipes; per-recipe step count is capped at ``MAX_STEPS_PER_RECIPE``.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import threading
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import config
from .logger import logger
from .memory import append_note

__all__ = [
    "MAX_RECIPES",
    "MAX_STEPS_PER_RECIPE",
    "Recipe",
    "RecipeRunResult",
    "RecipeStep",
    "RecipeStore",
    "config",
    "delete_recipe",
    "find_recipe",
    "get_recipe_store",
    "list_recipes",
    "reset_recipe_store",
    "run_recipe",
    "save_recipe",
]


MAX_RECIPES = 200
MAX_STEPS_PER_RECIPE = 50
MAX_STEP_VALUE_CHARS = 4000

DEFAULT_HUMAN_BREAK_MIN_MS = 300
DEFAULT_HUMAN_BREAK_MAX_MS = 1500
MAX_HUMAN_BREAK_MS = 60_000

# Valid step kinds — keep this set tight on purpose.
_VALID_KINDS: frozenset[str] = frozenset(
    {
        "shell",
        "read",
        "write",
        "http_get",
        "memory_write",
        "delay",
        "human_break",
        "assert_exists",
        "assert_eq",
        "emit",
        "browser",
    }
)

# Valid recipe names: 1-64 chars, alphanumeric + dash/underscore/colon.
_NAME_RE = re.compile(r"^[A-Za-z0-9_\-:]{1,64}$")


# ---------------------------------------------------------------------------
# Recipe / step dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeStep:
    kind: str
    # The "what" of the step. Interpretation depends on kind:
    #   shell        -> list of argv tokens (joined as list[list[str]])
    #   read/write   -> project-relative path
    #   http_get     -> URL
    #   memory_write -> kind label
    #   delay        -> integer ms
    #   human_break  -> "min||max" ms range (blank=default); fresh random each replay
    #   assert_exists-> project-relative path
    #   assert_eq    -> "expected||actual" string (split on "||")
    #   emit         -> free-form text
    #   browser      -> "goal" or "goal||json_overrides"; replays a saved playbook
    value: str = ""
    # Optional override: a 1-line human description, persisted for inspection.
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RecipeStep | None:
        if not isinstance(raw, dict):
            return None
        kind = raw.get("kind")
        if not isinstance(kind, str) or kind not in _VALID_KINDS:
            return None
        value = raw.get("value", "")
        if not isinstance(value, str):
            return None
        if len(value) > MAX_STEP_VALUE_CHARS:
            return None
        note = str(raw.get("note", ""))
        return cls(kind=kind, value=value, note=note[:200])


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    steps: list[RecipeStep]
    # The free-form tag list — "deployment", "ci", "code_review", etc.
    tags: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""
    uses: int = 0
    last_used_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "uses": self.uses,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Recipe | None:
        if not isinstance(raw, dict):
            return None
        name = raw.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            return None
        steps_raw = raw.get("steps", [])
        if not isinstance(steps_raw, list):
            return None
        steps: list[RecipeStep] = []
        for s in steps_raw:
            parsed = RecipeStep.from_dict(s)
            if parsed is None:
                return None
            steps.append(parsed)
        if not steps or len(steps) > MAX_STEPS_PER_RECIPE:
            return None
        tags = raw.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            tags = []
        return cls(
            name=name,
            description=str(raw.get("description", ""))[:500],
            steps=steps,
            tags=tuple(tags)[:8],
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
            uses=int(raw.get("uses", 0) or 0),
            last_used_at=str(raw.get("last_used_at", "")),
        )


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------


@dataclass
class RecipeRunResult:
    recipe: str
    started_at: str
    finished_at: str
    ok: bool
    failed_step_index: int | None = None
    failed_step_kind: str | None = None
    failed_step_note: str = ""
    error: str = ""
    step_results: list[dict[str, Any]] = field(default_factory=list)
    emitted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        head = f"recipe={self.recipe!r} ok={self.ok} steps={len(self.step_results)}"
        if not self.ok:
            head += f" failed_step={self.failed_step_index} ({self.failed_step_kind})"
            if self.error:
                head += f" err={self.error[:120]}"
        if self.emitted:
            head += f" emitted={len(self.emitted)}"
        return head


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _recipe_path() -> Path:
    raw_path = Path(config.history_dir)
    if not raw_path.is_absolute():
        raw_path = (Path.cwd() / raw_path).resolve()
    return raw_path.parent / "recipes.jsonl"


# ---------------------------------------------------------------------------
# Recipe store
# ---------------------------------------------------------------------------


class RecipeStore:
    """A small, thread-safe, on-disk store of recipes."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path: Path = (
            Path(path).expanduser().resolve() if path is not None else _recipe_path()
        )
        self._lock = threading.RLock()
        self._by_name: dict[str, Recipe] = {}
        self._by_tag: dict[str, set[str]] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def save(
        self,
        name: str,
        description: str,
        steps: Iterable[RecipeStep | Mapping[str, Any]],
        tags: Iterable[str] = (),
    ) -> Recipe:
        if not _NAME_RE.match(name or ""):
            raise ValueError("recipe name must match [A-Za-z0-9_-:]{1,64}")
        if not description or not isinstance(description, str):
            raise ValueError("recipe must have a non-empty description")
        parsed_steps: list[RecipeStep] = []
        for s in steps:
            if isinstance(s, RecipeStep):
                parsed_steps.append(s)
            elif isinstance(s, Mapping):
                step = RecipeStep.from_dict(dict(s))
                if step is None:
                    raise ValueError(f"invalid step: {s!r}")
                parsed_steps.append(step)
            else:
                raise ValueError(f"unsupported step type: {type(s).__name__}")
        if not parsed_steps:
            raise ValueError("recipe must have at least one step")
        if len(parsed_steps) > MAX_STEPS_PER_RECIPE:
            raise ValueError(f"recipe exceeds max steps ({MAX_STEPS_PER_RECIPE})")
        cleaned_tags = tuple(sorted({t for t in tags if isinstance(t, str) and t}))

        with self._lock:
            existing = self._by_name.get(name)
            now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            recipe = Recipe(
                name=name,
                description=description[:500],
                steps=parsed_steps,
                tags=cleaned_tags[:8],
                created_at=existing.created_at if existing else now,
                updated_at=now,
                uses=existing.uses if existing else 0,
                last_used_at=existing.last_used_at if existing else "",
            )
            self._by_name[name] = recipe
            self._rebuild_tag_index_locked()
            self._prune_locked()
            self._persist_locked()
            logger.info(
                f"RecipeStore: saved {name!r} ({len(parsed_steps)} steps, "
                f"tags={cleaned_tags})"
            )
            return recipe

    def find(self, name: str) -> Recipe | None:
        with self._lock:
            return self._by_name.get(name)

    def search(
        self,
        query: str,
        *,
        tag: str | None = None,
        limit: int = 20,
    ) -> list[Recipe]:
        if not query:
            pool: Iterable[Recipe] = self._by_name.values()
        else:
            needle = query.lower()
            pattern = re.compile(re.escape(needle), re.IGNORECASE)
            pool = [
                r
                for r in self._by_name.values()
                if pattern.search(r.name)
                or pattern.search(r.description)
                or any(pattern.search(s.value) for s in r.steps)
            ]
        if tag:
            pool = [r for r in pool if tag in r.tags]
        out = sorted(
            pool,
            key=lambda r: (-r.uses, r.name),
        )
        return out[:limit]

    def all(self) -> list[Recipe]:
        with self._lock:
            return sorted(self._by_name.values(), key=lambda r: r.name)

    def delete(self, name: str) -> bool:
        with self._lock:
            if name not in self._by_name:
                return False
            del self._by_name[name]
            self._rebuild_tag_index_locked()
            self._persist_locked()
            logger.info(f"RecipeStore: deleted {name!r}")
            return True

    def mark_used(self, name: str) -> None:
        with self._lock:
            recipe = self._by_name.get(name)
            if recipe is None:
                return
            updated = Recipe(
                name=recipe.name,
                description=recipe.description,
                steps=recipe.steps,
                tags=recipe.tags,
                created_at=recipe.created_at,
                updated_at=recipe.updated_at,
                uses=recipe.uses + 1,
                last_used_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            )
            self._by_name[name] = updated
            self._persist_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "path": str(self._path),
                "total": len(self._by_name),
                "tags": sorted(self._by_tag.keys()),
                "snapshot_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }

    # --- internals --------------------------------------------------------

    def _rebuild_tag_index_locked(self) -> None:
        self._by_tag.clear()
        for recipe in self._by_name.values():
            for tag in recipe.tags:
                self._by_tag.setdefault(tag, set()).add(recipe.name)

    def _prune_locked(self) -> None:
        if len(self._by_name) <= MAX_RECIPES:
            return
        # Drop the least-used first; ties broken by name.
        by_score = sorted(
            self._by_name.values(),
            key=lambda r: (r.uses, r.name),
        )
        to_drop = len(self._by_name) - MAX_RECIPES
        for recipe in by_score[:to_drop]:
            logger.info(
                f"RecipeStore: pruning LRU recipe {recipe.name!r} (uses={recipe.uses})"
            )
            del self._by_name[recipe.name]
        self._rebuild_tag_index_locked()

    def _load(self) -> None:
        path = self._path
        if not path.exists():
            return
        try:
            with path.open(encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            logger.debug(f"RecipeStore: read failed [{type(exc).__name__}]: {exc}")
            return
        loaded = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            recipe = Recipe.from_dict(raw)
            if recipe is None:
                continue
            self._by_name[recipe.name] = recipe
            loaded += 1
        self._rebuild_tag_index_locked()
        logger.info(f"RecipeStore: loaded {loaded} recipes from {path}")

    def _persist_locked(self) -> None:
        path = self._path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for recipe in self._by_name.values():
                    f.write(json.dumps(recipe.to_dict(), ensure_ascii=False) + "\n")
            tmp_path.replace(path)
        except OSError as exc:
            logger.debug(f"RecipeStore: persist failed [{type(exc).__name__}]: {exc}")


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_STORE: RecipeStore | None = None
_STORE_LOCK = threading.Lock()


def get_recipe_store() -> RecipeStore | None:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            try:
                _STORE = RecipeStore()
            except Exception as exc:
                logger.debug(f"RecipeStore: init failed [{type(exc).__name__}]: {exc}")
                return None
        return _STORE


def reset_recipe_store() -> None:
    """Drop the process-wide RecipeStore singleton (test isolation)."""
    global _STORE
    _STORE = None


def save_recipe(
    name: str,
    description: str,
    steps: Iterable[RecipeStep | Mapping[str, Any]],
    tags: Iterable[str] = (),
) -> Recipe | None:
    store = get_recipe_store()
    if store is None:
        return None
    return store.save(name, description, steps, tags)


def find_recipe(name: str) -> Recipe | None:
    store = get_recipe_store()
    if store is None:
        return None
    return store.find(name)


async def run_recipe(name: str, *, fail_fast: bool = True) -> RecipeRunResult | None:
    store = get_recipe_store()
    if store is None:
        return None
    recipe = store.find(name)
    if recipe is None:
        return None
    return await _execute_recipe(recipe, fail_fast=fail_fast)


def delete_recipe(name: str) -> bool:
    store = get_recipe_store()
    if store is None:
        return False
    return store.delete(name)


def list_recipes(
    query: str = "",
    *,
    tag: str | None = None,
    limit: int = 20,
) -> list[Recipe]:
    store = get_recipe_store()
    if store is None:
        return []
    return store.search(query, tag=tag, limit=limit)


# ---------------------------------------------------------------------------
# Recipe executor — purely deterministic, no LLM
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_resolve(rel: str) -> Path:
    """Resolve a project-relative path with a containment check.

    The root is ``config.allowed_directories[0]`` by default; tests can
    monkey-patch ``config.allowed_directories`` (or override the env var
    ``SUPPORTER_PROJECT_ROOT``) to redirect the sandbox elsewhere.
    """
    root_path: Path | None = None
    if config.allowed_directories:
        root_path = Path(config.allowed_directories[0]).expanduser().resolve()
    if root_path is None:
        raise PermissionError("No allowed directories set")
    candidate = (root_path / rel).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise PermissionError(f"path escapes project root: {rel}") from exc
    return candidate


def _parse_break_range(value: str) -> tuple[int, int]:
    """Parse a human_break range. Blank -> defaults. "min||max" -> ints.
    Clamped to 0..MAX_HUMAN_BREAK_MS; swapped if reversed; defaults on bad input."""
    lo, hi = DEFAULT_HUMAN_BREAK_MIN_MS, DEFAULT_HUMAN_BREAK_MAX_MS
    if value and "||" in value:
        a, b = value.split("||", 1)
        try:
            lo, hi = int(a), int(b)
        except ValueError:
            lo, hi = DEFAULT_HUMAN_BREAK_MIN_MS, DEFAULT_HUMAN_BREAK_MAX_MS
    lo = max(0, min(MAX_HUMAN_BREAK_MS, lo))
    hi = max(0, min(MAX_HUMAN_BREAK_MS, hi))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


async def _execute_step(
    index: int,
    step: RecipeStep,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Execute a single recipe step, returning a result dict."""
    head = {"index": index, "kind": step.kind, "ok": True, "value": step.value}
    try:
        if step.kind == "delay":
            ms = int(step.value)
            if ms < 0 or ms > 60_000:
                raise ValueError("delay must be 0..60000 ms")
            await asyncio.sleep(ms / 1000.0)
            head["detail"] = f"slept {ms}ms"
            return head

        if step.kind == "human_break":
            lo, hi = _parse_break_range(step.value)
            ms = random.randint(lo, hi)  # noqa: S311  # fresh random every run
            await asyncio.sleep(ms / 1000.0)
            head["detail"] = f"human break {ms}ms (random in {lo}..{hi})"
            head["break_ms"] = ms
            return head

        if step.kind == "browser":
            # Value: "goal" or "goal||json_overrides". Replays a saved playbook
            # on the live page; needs an active browser session (orchestrator).
            from .tools.browser.task import replay_playbook

            goal, overrides = step.value, None
            if "||" in step.value:
                goal, blob = step.value.split("||", 1)
                parsed = json.loads(blob)
                if not isinstance(parsed, dict):
                    head["ok"] = False
                    head["error"] = "browser overrides must be a JSON object"
                    return head
                overrides = parsed
            summary = await replay_playbook(goal, overrides)
            head["detail"] = summary[:400]
            return head

        if step.kind == "emit":
            ctx["emitted"].append(step.value)
            head["detail"] = "emitted"
            return head

        if step.kind == "assert_exists":
            path = _safe_resolve(step.value)
            if not path.exists():
                head["ok"] = False
                head["error"] = f"missing: {path}"
            else:
                head["detail"] = f"exists: {path}"
            return head

        if step.kind == "assert_eq":
            if "||" not in step.value:
                head["ok"] = False
                head["error"] = "assert_eq requires 'expected||actual'"
                return head
            expected, actual = step.value.split("||", 1)
            if expected != actual:
                head["ok"] = False
                head["error"] = f"mismatch: expected={expected!r} actual={actual!r}"
            else:
                head["detail"] = "match"
            return head

        if step.kind == "read":
            path = _safe_resolve(step.value)
            if not path.exists():
                head["ok"] = False
                head["error"] = f"missing: {path}"
                return head
            content = path.read_text(encoding="utf-8", errors="replace")
            head["detail"] = f"read {len(content)} chars"
            head["snippet"] = content[:240]
            return head

        if step.kind == "write":
            # Value format: "path||content". "||" chosen because newlines
            # are common in file bodies.
            if "||" not in step.value:
                head["ok"] = False
                head["error"] = "write requires 'path||content'"
                return head
            rel, content = step.value.split("||", 1)
            path = _safe_resolve(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            head["detail"] = f"wrote {len(content)} chars to {path}"
            return head

        if step.kind == "shell":
            # Value is a JSON array of strings (argv tokens).
            try:
                argv = json.loads(step.value)
            except json.JSONDecodeError as exc:
                head["ok"] = False
                head["error"] = f"shell value must be JSON array: {exc}"
                return head
            if not isinstance(argv, list) or not all(isinstance(t, str) for t in argv):
                head["ok"] = False
                head["error"] = "shell value must be a JSON array of strings"
                return head
            # Lazy import — bash executor has its own sandbox setup.
            from .tools.bash.executor import execute_bash

            # bash executor parses a single string; join with spaces.
            command_str = " ".join(argv)
            try:
                result = execute_bash(command_str)
            except TypeError:
                # Some test doubles accept argv directly.
                result = execute_bash(argv)  # type: ignore[arg-type]
            text = result if isinstance(result, str) else json.dumps(result)
            head["detail"] = text[:400]
            return head

        if step.kind == "http_get":
            url = step.value
            try:
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme not in {"http", "https"}:
                    raise ValueError(f"unsupported scheme: {parsed.scheme}")

                def _fetch() -> str:
                    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310  # nosec B310
                        body: str = response.read(1024 * 1024).decode(
                            "utf-8", errors="replace"
                        )
                        return body

                body = await asyncio.to_thread(_fetch)
                head["detail"] = f"http {len(body)} chars"
                head["snippet"] = body[:240]
            except Exception as exc:
                head["ok"] = False
                head["error"] = f"http_get: {type(exc).__name__}: {exc}"
            return head

        if step.kind == "memory_write":
            # Value: "memory_kind||json_blob"
            if "||" not in step.value:
                head["ok"] = False
                head["error"] = "memory_write requires 'memory_kind||json_blob'"
                return head
            mem_kind, blob = step.value.split("||", 1)
            try:
                payload = json.loads(blob)
            except json.JSONDecodeError as exc:
                head["ok"] = False
                head["error"] = f"json blob: {exc}"
                return head
            if not isinstance(payload, dict):
                head["ok"] = False
                head["error"] = "memory payload must be a JSON object"
                return head
            note = append_note(
                mem_kind,
                payload,
                source=f"recipe:{ctx.get('name', '')}",
            )
            head["detail"] = f"stored {mem_kind!r}"
            head["stored"] = note is not None
            return head

        # Should be unreachable thanks to from_dict validation.
        head["ok"] = False
        head["error"] = f"unknown step kind: {step.kind}"
        return head

    except Exception as exc:
        head["ok"] = False
        head["error"] = f"{type(exc).__name__}: {exc}"
        return head


async def _execute_recipe(recipe: Recipe, *, fail_fast: bool) -> RecipeRunResult:
    started_at = _now_iso()
    ctx: dict[str, Any] = {"emitted": [], "name": recipe.name}
    result = RecipeRunResult(
        recipe=recipe.name,
        started_at=started_at,
        finished_at=started_at,
        ok=True,
    )
    for i, step in enumerate(recipe.steps):
        step_result = await _execute_step(i, step, ctx)
        result.step_results.append(step_result)
        if not step_result["ok"]:
            result.ok = False
            result.failed_step_index = i
            result.failed_step_kind = step.kind
            result.failed_step_note = step.note
            result.error = str(step_result.get("error", ""))
            if fail_fast:
                break
    result.finished_at = _now_iso()
    result.emitted = list(ctx["emitted"])

    # Record usage + the result as a memory note (for inspection / replay).
    store = get_recipe_store()
    if store is not None:
        store.mark_used(recipe.name)
    append_note(
        "recipe_run",
        {
            "recipe": recipe.name,
            "ok": result.ok,
            "steps": len(recipe.steps),
            "failed_step_index": result.failed_step_index,
            "error": result.error[:240] if result.error else "",
        },
        source=f"recipe:{recipe.name}",
    )
    return result


def recipes_snapshot() -> dict[str, Any]:
    store = get_recipe_store()
    if store is None:
        return {"available": False}
    snap = store.snapshot()
    snap["available"] = True
    return snap

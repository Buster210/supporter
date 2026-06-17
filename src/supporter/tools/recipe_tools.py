"""Tool wrappers for the recipe store.

Thin shims that translate tool-call arguments into :mod:`supporter.recipes`
operations and shape the result as plain text the orchestrator can read.
The store is the source of truth; the tool wrappers exist only to put a
uniform ``str``/``dict`` interface in front of it.
"""

from __future__ import annotations

import json
from typing import Any

from ..recipes import (
    Recipe,
    RecipeRunResult,
    RecipeStep,
    delete_recipe,
    find_recipe,
    list_recipes,
    recipes_snapshot,
    run_recipe,
    save_recipe,
)

__all__ = [
    "recipe_delete",
    "recipe_find",
    "recipe_list",
    "recipe_run",
    "recipe_save",
    "recipe_status",
]


def _format_recipe(recipe: Recipe) -> str:
    head = (
        f"# {recipe.name} — uses={recipe.uses} "
        f"updated={recipe.updated_at or '?'}\n"
    )
    if recipe.description:
        head += f"  {recipe.description}\n"
    if recipe.tags:
        head += f"  tags: {', '.join(recipe.tags)}\n"
    body = "\n".join(
        f"  {i:>2}. [{step.kind}] {step.value[:120]}"
        + (f"  ({step.note})" if step.note else "")
        for i, step in enumerate(recipe.steps, 1)
    )
    return head + body


def _format_result(result: RecipeRunResult) -> str:
    head = (
        f"recipe={result.recipe!r} ok={result.ok} "
        f"steps={len(result.step_results)} "
        f"started={result.started_at} finished={result.finished_at}"
    )
    if not result.ok:
        head += (
            f" failed_step={result.failed_step_index} "
            f"({result.failed_step_kind}) err={result.error[:200]}"
        )
    if result.emitted:
        head += f" emitted={len(result.emitted)}"
    out = [head]
    for step in result.step_results[:8]:
        marker = "OK " if step["ok"] else "ERR"
        detail = step.get("detail") or step.get("error") or ""
        out.append(f"  {marker} [{step['kind']}] {detail[:160]}")
    if len(result.step_results) > 8:
        out.append(f"  ... ({len(result.step_results) - 8} more steps)")
    return "\n".join(out)


def _coerce_steps(raw: Any) -> list[dict[str, Any]] | str:
    if not isinstance(raw, list):
        return "ERROR: steps must be a list"
    parsed: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if isinstance(item, RecipeStep):
            parsed.append(item.to_dict())
            continue
        if not isinstance(item, dict):
            return f"ERROR: step {i} is not an object"
        kind = item.get("kind")
        value = item.get("value", "")
        if not isinstance(kind, str) or not isinstance(value, str):
            return f"ERROR: step {i} needs string 'kind' and 'value'"
        parsed.append({"kind": kind, "value": value, "note": str(item.get("note", ""))})
    return parsed


async def recipe_save(
    name: str,
    description: str,
    steps_json: str,
    tags: list[str] | None = None,
) -> str:
    """Persist a recipe (LLM-free replay of a multi-step workflow)."""
    if not name or not isinstance(name, str):
        return "ERROR: name must be a non-empty string"
    if not description or not isinstance(description, str):
        return "ERROR: description must be a non-empty string"
    try:
        raw_steps = json.loads(steps_json)
    except json.JSONDecodeError as exc:
        return f"ERROR: steps_json is not valid JSON: {exc}"
    coerced = _coerce_steps(raw_steps)
    if isinstance(coerced, str):
        return coerced
    try:
        recipe = save_recipe(name, description, coerced, tags=tags or [])
    except ValueError as exc:
        return f"ERROR: {exc}"
    if recipe is None:
        return "ERROR: recipe store is not available"
    return f"ok: saved {name!r} with {len(recipe.steps)} steps"


async def recipe_find(name: str) -> str:
    if not name or not isinstance(name, str):
        return "ERROR: name must be a non-empty string"
    recipe = find_recipe(name)
    if recipe is None:
        return f"ERROR: no recipe named {name!r}"
    return _format_recipe(recipe)


async def recipe_run(name: str, fail_fast: bool = True) -> str:
    if not name or not isinstance(name, str):
        return "ERROR: name must be a non-empty string"
    result = run_recipe(name, fail_fast=fail_fast)
    if result is None:
        return f"ERROR: no recipe named {name!r} (or store unavailable)"
    return _format_result(result)


async def recipe_delete(name: str) -> str:
    if not name or not isinstance(name, str):
        return "ERROR: name must be a non-empty string"
    if delete_recipe(name):
        return f"ok: deleted {name!r}"
    return f"ERROR: no recipe named {name!r}"


async def recipe_list(query: str = "", limit: int = 20) -> str:
    recipes = list_recipes(query, limit=max(1, min(50, limit)))
    if not recipes:
        return "(no recipes match)"
    return "\n".join(
        f"- {r.name} "
        f"(uses={r.uses}, steps={len(r.steps)}, "
        f"tags={','.join(r.tags) or '-'})"
        for r in recipes
    )


async def recipe_status() -> str:
    snap = recipes_snapshot()
    if not snap.get("available", False):
        return "recipe store: unavailable"
    return (
        f"recipe store: total={snap.get('total', 0)} "
        f"path={snap.get('path', '?')} tags={len(snap.get('tags', []))}"
    )

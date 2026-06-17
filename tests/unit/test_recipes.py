"""Unit tests for the recipe store and tool wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from supporter import recipes as recipes_mod
from supporter.recipes import (
    MAX_STEPS_PER_RECIPE,
    Recipe,
    RecipeStep,
    RecipeStore,
    delete_recipe,
    find_recipe,
    list_recipes,
    recipes_snapshot,
    run_recipe,
    save_recipe,
)
from supporter.tools import recipe_tools

# ---------------------------------------------------------------------------
# Recipe dataclass parsing
# ---------------------------------------------------------------------------


def test_step_rejects_unknown_kind() -> None:
    assert RecipeStep.from_dict({"kind": "nope", "value": "x"}) is None


def test_step_rejects_oversize_value() -> None:
    big = "x" * 5000
    assert RecipeStep.from_dict({"kind": "emit", "value": big}) is None


def test_recipe_rejects_bad_name() -> None:
    raw = {
        "name": "with spaces",
        "description": "d",
        "steps": [{"kind": "emit", "value": "hi"}],
    }
    assert Recipe.from_dict(raw) is None


def test_recipe_rejects_empty_steps() -> None:
    raw = {"name": "ok", "description": "d", "steps": []}
    assert Recipe.from_dict(raw) is None


def test_recipe_rejects_too_many_steps() -> None:
    raw = {
        "name": "ok",
        "description": "d",
        "steps": [{"kind": "emit", "value": "x"}] * (MAX_STEPS_PER_RECIPE + 1),
    }
    assert Recipe.from_dict(raw) is None


def test_recipe_round_trip() -> None:
    raw = {
        "name": "deploy",
        "description": "ship it",
        "steps": [
            {"kind": "emit", "value": "start"},
            {"kind": "delay", "value": "10"},
        ],
        "tags": ["ci", "deploy"],
    }
    parsed = Recipe.from_dict(raw)
    assert parsed is not None
    assert parsed.name == "deploy"
    assert len(parsed.steps) == 2
    assert parsed.tags == ("ci", "deploy")
    rt = Recipe.from_dict(parsed.to_dict())
    assert rt is not None
    assert rt.name == "deploy"
    assert len(rt.steps) == 2


# ---------------------------------------------------------------------------
# RecipeStore
# ---------------------------------------------------------------------------


def _new_store(tmp_path: Path) -> RecipeStore:
    return RecipeStore(path=tmp_path / "recipes.jsonl")


def test_store_rejects_bad_name(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    with pytest.raises(ValueError):
        store.save("bad name", "d", [RecipeStep(kind="emit", value="x")])


def test_store_rejects_empty_description(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    with pytest.raises(ValueError):
        store.save("ok", "", [RecipeStep(kind="emit", value="x")])


def test_store_rejects_no_steps(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    with pytest.raises(ValueError):
        store.save("ok", "d", [])


def test_store_save_and_find(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    recipe = store.save(
        "deploy",
        "ship it",
        [RecipeStep(kind="emit", value="hi")],
        tags=["ci"],
    )
    assert recipe.name == "deploy"
    found = store.find("deploy")
    assert found is not None
    assert found.name == "deploy"


def test_store_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "recipes.jsonl"
    a = RecipeStore(path=path)
    a.save("first", "d", [RecipeStep(kind="emit", value="hi")])

    b = RecipeStore(path=path)
    assert b.find("first") is not None


def test_store_search_by_query(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    store.save("deploy_prod", "deploy to prod", [RecipeStep(kind="emit", value="hi")])
    store.save("run_tests", "run all tests", [RecipeStep(kind="emit", value="hi")])
    hits = store.search("deploy")
    assert any(r.name == "deploy_prod" for r in hits)


def test_store_search_by_tag(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    store.save("a", "d", [RecipeStep(kind="emit", value="x")], tags=["ci"])
    store.save("b", "d", [RecipeStep(kind="emit", value="x")], tags=["docs"])
    ci = store.search("", tag="ci")
    assert [r.name for r in ci] == ["a"]


def test_store_all_sorted(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    store.save("b", "d", [RecipeStep(kind="emit", value="x")])
    store.save("a", "d", [RecipeStep(kind="emit", value="x")])
    assert [r.name for r in store.all()] == ["a", "b"]


def test_store_delete(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    store.save("a", "d", [RecipeStep(kind="emit", value="x")])
    assert store.delete("a") is True
    assert store.delete("a") is False  # already gone


def test_store_mark_used_increments(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    store.save("a", "d", [RecipeStep(kind="emit", value="x")])
    assert store.find("a").uses == 0
    store.mark_used("a")
    store.mark_used("a")
    assert store.find("a").uses == 2


def test_store_snapshot(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    store.save("a", "d", [RecipeStep(kind="emit", value="x")])
    snap = store.snapshot()
    assert snap["total"] == 1


def test_store_tolerates_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "recipes.jsonl"
    path.write_text(
        json.dumps(
            {
                "name": "ok",
                "description": "d",
                "steps": [{"kind": "emit", "value": "x"}],
            }
        )
        + "\n"
        + "garbage\n"
        + json.dumps(
            {
                "name": "ok2",
                "description": "d",
                "steps": [{"kind": "emit", "value": "x"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store = RecipeStore(path=path)
    assert store.find("ok") is not None
    assert store.find("ok2") is not None


# ---------------------------------------------------------------------------
# Recipe execution
# ---------------------------------------------------------------------------


def test_recipe_emit_and_assert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe(
        "simple",
        "d",
        [
            {"kind": "emit", "value": "hello"},
            {"kind": "assert_eq", "value": "hello||hello"},
        ],
    )
    result = run_recipe("simple")
    assert result is not None
    assert result.ok
    assert result.emitted == ["hello"]


def test_recipe_assert_eq_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe(
        "fail",
        "d",
        [{"kind": "assert_eq", "value": "a||b"}],
    )
    result = run_recipe("fail")
    assert result is not None
    assert not result.ok
    assert result.failed_step_index == 0
    assert "mismatch" in result.error


def test_recipe_read_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    monkeypatch.setattr(recipes_mod.config, "allowed_directories", [str(tmp_path)])
    (tmp_path / "x.txt").write_text("contents", encoding="utf-8")
    save_recipe(
        "read",
        "d",
        [{"kind": "read", "value": "x.txt"}],
    )
    result = run_recipe("read")
    assert result is not None
    assert result.ok
    assert "read 8 chars" in result.step_results[0]["detail"]


def test_recipe_write_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    monkeypatch.setattr(recipes_mod.config, "allowed_directories", [str(tmp_path)])
    save_recipe(
        "write",
        "d",
        [{"kind": "write", "value": "y.txt||hello world"}],
    )
    result = run_recipe("write")
    assert result is not None
    assert result.ok
    assert (tmp_path / "y.txt").read_text(encoding="utf-8") == "hello world"


def test_recipe_assert_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    monkeypatch.setattr(recipes_mod.config, "allowed_directories", [str(tmp_path)])
    (tmp_path / "ok.txt").touch()
    save_recipe(
        "exists",
        "d",
        [
            {"kind": "assert_exists", "value": "ok.txt"},
            {"kind": "assert_exists", "value": "missing.txt"},
        ],
    )
    result = run_recipe("exists", fail_fast=False)
    assert result is not None
    assert not result.ok
    assert result.failed_step_index == 1


def test_recipe_path_traversal_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    monkeypatch.setattr(recipes_mod.config, "allowed_directories", [str(tmp_path)])
    save_recipe(
        "evil",
        "d",
        [{"kind": "read", "value": "../secrets.txt"}],
    )
    result = run_recipe("evil")
    assert result is not None
    assert not result.ok


def test_recipe_memory_write_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import memory as memory_mod

    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    save_recipe(
        "remember",
        "d",
        [{"kind": "memory_write", "value": "todo||{\"task\":\"ship\"}"}],
    )
    result = run_recipe("remember")
    assert result is not None
    assert result.ok
    notes = memory_mod.list_notes(kind="todo")
    assert len(notes) == 1
    assert notes[0].value == {"task": "ship"}


def test_recipe_fail_fast_stops_at_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe(
        "x",
        "d",
        [
            {"kind": "assert_eq", "value": "a||b"},
            {"kind": "emit", "value": "never"},
        ],
    )
    result = run_recipe("x", fail_fast=True)
    assert result is not None
    assert not result.ok
    assert len(result.step_results) == 1


def test_recipe_fail_fast_false_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe(
        "x",
        "d",
        [
            {"kind": "assert_eq", "value": "a||b"},
            {"kind": "emit", "value": "ran"},
        ],
    )
    result = run_recipe("x", fail_fast=False)
    assert result is not None
    assert not result.ok
    assert len(result.step_results) == 2


def test_run_recipe_unknown_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    assert run_recipe("nope") is None


def test_run_recipe_persists_run_to_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import memory as memory_mod

    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    save_recipe("a", "d", [{"kind": "emit", "value": "x"}])
    run_recipe("a")
    runs = memory_mod.list_notes(kind="recipe_run")
    assert len(runs) == 1


@pytest.fixture(autouse=True)
def _reset_recipe_singleton() -> None:
    from supporter import memory as memory_mod

    recipes_mod._STORE = None  # type: ignore[attr-defined]
    memory_mod._MEMORY_SINGLETON = None  # type: ignore[attr-defined]
    yield
    recipes_mod._STORE = None  # type: ignore[attr-defined]
    memory_mod._MEMORY_SINGLETON = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Process-wide helpers
# ---------------------------------------------------------------------------


def test_list_recipes_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe("a", "d", [{"kind": "emit", "value": "x"}])
    save_recipe("b", "d", [{"kind": "emit", "value": "x"}], tags=["ci"])
    recipes = list_recipes()
    assert {r.name for r in recipes} == {"a", "b"}


def test_find_recipe_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe("a", "d", [{"kind": "emit", "value": "x"}])
    assert find_recipe("a") is not None
    assert find_recipe("nope") is None


def test_delete_recipe_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe("a", "d", [{"kind": "emit", "value": "x"}])
    assert delete_recipe("a") is True
    assert delete_recipe("a") is False


def test_recipes_snapshot_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe("a", "d", [{"kind": "emit", "value": "x"}])
    snap = recipes_snapshot()
    assert snap["available"] is True
    assert snap["total"] == 1


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recipe_save_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    out = await recipe_tools.recipe_save(
        "demo",
        "d",
        json.dumps([{"kind": "emit", "value": "x"}]),
    )
    assert "ok" in out


@pytest.mark.asyncio
async def test_recipe_save_rejects_bad_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    out = await recipe_tools.recipe_save("a", "d", "not json")
    assert "ERROR" in out


@pytest.mark.asyncio
async def test_recipe_find_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe("demo", "d", [{"kind": "emit", "value": "hi"}])
    out = await recipe_tools.recipe_find("demo")
    assert "demo" in out
    out_missing = await recipe_tools.recipe_find("nope")
    assert "ERROR" in out_missing


@pytest.mark.asyncio
async def test_recipe_run_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe("demo", "d", [{"kind": "emit", "value": "hi"}])
    out = await recipe_tools.recipe_run("demo")
    assert "ok=True" in out
    assert "demo" in out


@pytest.mark.asyncio
async def test_recipe_delete_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe("demo", "d", [{"kind": "emit", "value": "hi"}])
    out = await recipe_tools.recipe_delete("demo")
    assert "deleted" in out


@pytest.mark.asyncio
async def test_recipe_list_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe("a", "d", [{"kind": "emit", "value": "x"}])
    save_recipe("b", "d", [{"kind": "emit", "value": "x"}])
    out = await recipe_tools.recipe_list()
    assert "a" in out
    assert "b" in out


@pytest.mark.asyncio
async def test_recipe_status_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    save_recipe("a", "d", [{"kind": "emit", "value": "x"}])
    out = await recipe_tools.recipe_status()
    assert "total=1" in out

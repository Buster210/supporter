# Project Intelligence & Orchestration (AGENTS.md)

## 1. Project DNA
- **Mission**: Python-based TUI AI chat client using Google Gemini (Live/Streaming) with load balancing.
- **Stack**: Python 3.13+, `uv`, `google-genai`, `textual`, `rich`.
- **Ref**: Core Project Reference | **Entry**: `src/supporter/tui/__init__.py` (`supporter` CLI)

## 2. Arch & Flow
**Flow**: User → `run_worker` → `ChatAgent` → `GeminiProvider` → `Tools` → `UI` (Modals/Bubbles)
**Structure** (`src/supporter/`):
- `agent.py`: `ChatAgent` (Core orchestration & tool calling)
- `index.py`: `DynamicPool` (API key rotation/load balancing)
- `providers/`: `GeminiProvider` (LLM clients), `get_provider` (singleton)
- `tools/`: `bash.py` (sandboxed), `file_ops.py` (validated), `search.py` (Gemini search)
- `tui/`: `__init__.py` (entry), `message_processor.py` (streaming), `mode_manager.py` (commands)
- `tui/widgets.py`: `ChatTurn` (collapsible), `ThinkingIndicator`, `ConfirmationModal`

## 3. Security & Standards
- **Safety**: Path validation (Root + .gitignore) | Always confirm Writes/Bash | Bash Tiers (T1-Safe, T3-Risky)
- **Roles**: Architect (Logic/API), Backend (Tools/LLM), Frontend (TUI/UX), QA (Security/Tests)
- **Standards**: 85% coverage | Surgical atomic edits | No file-wide rewrites | `ruff` + `mypy` compliant

## 4. Operational Protocols
- **Verify**: `uv run pytest tests` [-m `unit` / `integration` / `e2e`]
- **Lint**: `uv run ruff check .` | `uv run ruff format .` | `uv run mypy .`
- **Lookup**: Config (`.env`, `config.py`) | Deps (`pyproject.toml`) | Logs (`app.log`)

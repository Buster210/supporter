# Project Intelligence & Orchestration (AGENTS.md)

## 1. Project DNA
- **Mission**: Python-based TUI AI chat client using Google Gemini (Live & Streaming) with advanced load balancing and multi-agent delegation.
- **Stack**: Python 3.13+, `uv`, `google-genai`, `textual`, `rich`.
- **Ref**: `AGENTS.md`, `README.md` | **Entry**: `src/supporter/tui/__init__.py` (`supporter` CLI via `pyproject.toml`).

## 2. Arch & Flow
**Flow**: User → `SupporterApp` (TUI) → `ChatAgent` (Orchestrator) → `LazyFallbackProvider` → `DynamicPool` (Provider Rotation) → `Tools`.

**Structure** (`src/supporter/`):
- `agent.py`: `ChatAgent` orchestrates prompts, history synchronization, and tool routing.
- `index.py`: `DynamicPool` handles API key rotation, health checks, and 5XX/429 cooldowns.
- `providers/`:
    - `gemini_provider.py`: Standard REST-based generation with streaming.
    - `gemini_live_provider.py`: WebSocket-based real-time audio/text session management.
- `tools/`:
    - `bash.py`: Hardened execution using `sandbox-exec` (macOS) or `nsjail` (Linux).
    - `delegate.py`: Multi-agent task runner with DAG resolution.
    - `file_ops.py`: Path-validated atomic reads and writes with UI confirmation.
    - `event_bus.py`: Pub/Sub system for live delegation progress and system alerts.
- `tui/`:
    - `__init__.py`: `SupporterApp` entry, global event handling, and modal orchestration.
    - `bubble.py`: Specialized widgets for different message roles (User/Agent/System).
    - `chat.py`: Component layout and interaction groups.
    - `message_processor.py`: Unified stream parsing and UI updates.
- `config.py`: Centralized `AppConfig` with environment-driven settings and agent roster.
- `types.py`: Project-wide types, status enums, and delegation event schemas.

## 3. Security & Standards
- **Safety**: Path validation (project root + `.gitignore`) | Mandatory UI confirmation for writes/bash | Sandbox Tiers (T1-Safe → T3-Risky).
- **Roles**: Architect (Logic/API), Backend (Tools/LLM), Frontend (TUI/UX), QA (Security/Tests).
- **Standards**: 85% coverage | Surgical atomic edits | `ruff` + `mypy` + `pytest` compliance.

## 4. Operational Protocols
- **Verify**: `uv run pytest tests` [-m `unit` / `integration` / `e2e`]
- **Lint**: `uv run ruff check .` | `uv run ruff format .` | `uv run mypy .`
- **Lookup**: Config (`config.py`, `.env`) | Deps (`pyproject.toml`) | Logs (`app.log`, Flight Recorder)

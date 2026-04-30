# Project Intelligence & Orchestration (AGENTS.md)

## 1. Project DNA
- **Mission**: Python-based TUI AI chat client using Google Gemini (Live/Streaming) with load balancing and delegated tool execution.
- **Stack**: Python 3.13+, `uv`, `google-genai`, `textual`, `rich`.
- **Ref**: Core Project Reference | **Entry**: `src/supporter/tui/__init__.py` (`supporter` CLI)

## 2. Arch & Flow
**Flow**: User → `SupporterApp` → `ChatAgent` → `GeminiProvider` → `Tools` (Bash/File/Delegate) → `UI` (Bubbles/Turns/Modals)
**Structure** (`src/supporter/`):
- `agent.py`: `ChatAgent` (Core orchestration, tool routing, and streaming logic)
- `index.py`: `DynamicPool` (API key rotation, provider health, and load balancing)
- `providers/`: `GeminiProvider` (Standard LLM), `GeminiLiveProvider` (Real-time streaming)
- `tools/`: 
    - `bash.py`: Sandboxed execution with macOS `sandbox-exec`.
    - `file_ops.py`: Validated file reads/writes with confirmation.
    - `delegate.py`: Sub-task delegation and complex tool orchestration.
    - `search.py`: Google Search integration via Gemini.
- `tui/`: 
    - `__init__.py`: Application entry (`SupporterApp`) and main loop.
    - `bubble.py`: Specialized `MessageBubble` widgets for Chat roles.
    - `chat.py`: Layout components (`ChatContainer`, `ChatTurn`, `ThinkingIndicator`).
    - `modals.py`: Centralized `ConfirmationModal` for Writes and Bash.
    - `message_processor.py`: Streaming output parsing and formatting.
    - `mode_manager.py`: Command handling and agent state management.
    - `utils.py`: UI helpers like `ToastManager`.
- `types.py`: Unified type definitions for LLM interactions.

## 3. Security & Standards
- **Safety**: Path validation (Root + .gitignore) | Mandatory user confirmation for Writes/Bash | Sandbox Tiers (T1-Safe, T3-Risky).
- **Roles**: Architect (Logic/API), Backend (Tools/LLM), Frontend (TUI/UX), QA (Security/Tests).
- **Standards**: ~85% coverage | Surgical atomic edits | `ruff` + `mypy` + `pytest` compliance.

## 4. Operational Protocols
- **Verify**: `uv run pytest tests` [-m `unit` / `integration` / `e2e`]
- **Lint**: `uv run ruff check .` | `uv run ruff format .` | `uv run mypy .`
- **Lookup**: Config (`config.py`, `.env`) | Deps (`pyproject.toml`) | Logs (`app.log`)

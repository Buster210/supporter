# Project Intelligence & Orchestration (AGENTS.md)

## 1. Project DNA
- **Mission**: Python TUI AI chat client using Google Gemini (Live & Streaming) with load balancing & multi-agent delegation.
- **Stack**: Python 3.13+, `uv`, `google-genai`, `textual`, `rich`.
- **Ref**: [README.md](file:///Users/riteshkumarpal/Downloads/lab/supporter/README.md) | **Entry**: `uv run supporter`

## 2. Arch & Flow
**Flow**: User → TUI (Textual) → Agent (Logic) → Tools (Bash/Delegate/Search) → LLM → Output
**Structure** (`src/supporter`):
- `agent.py`: Core planning and multi-agent delegation logic.
- `providers/`: Gemini Live & REST provider implementations.
- `tools/bash/`: Sandboxed shell execution with tiered security policies.
- `tools/delegate/`: Multi-agent task orchestration & bus.
- `tui/`: Reactive UI components and application screens.

## 3. Security & Standards
- **Safety**: Sandbox (`sandbox-exec`) mandatory for all shell ops; no raw shell outside `tools/bash`.
- **Roles**: Architect (Core Logic), Frontend (TUI/UX), Tools (Security/Safety).
- **Standards**: Surgical atomic edits | Textual `reactive` state | 85% coverage threshold.

## 4. Operational Protocols
- **Verify**: `uv run pytest tests`
- **Lint**: `uv run ruff check .` | `uv run mypy .`
- **Lookup**: Config (`.env`, `config.py`) | Deps (`pyproject.toml`) | Logs (`app.log`)

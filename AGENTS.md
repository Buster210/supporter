# Project Intelligence & Orchestration (AGENTS.md)

## 1. Project DNA
- **Mission**: Elite technical strategist and principal software architect TUI chat client powered by Google Gemini, orchestrating complex development tasks via a multi-agent delegation system and sandboxed tool execution.
- **Stack**: Python 3.13+, `uv`, `google-genai`, `textual`, `rich`.
- **Ref**: [README.md](file:///Users/riteshkumarpal/Downloads/lab/supporter/README.md) | **Entry**: `uv run supporter`

## 2. Arch & Flow
**Flow**: User → TUI (Textual) → Orchestrator (Agent Logic) → Tools (Bash/Delegate/Search) → LLM → Output
**Structure** (`src/supporter`):
- `agent.py`: Core orchestrator and delegation logic.
- `providers/`: Gemini Live and REST implementations with health-aware rotation.
- `tools/bash/`: Security-hardened shell execution logic.
- `tools/delegate/`: Capsule-based task state and async orchestration.
- `tui/`: Reactive UI, bubble-based thought streaming, and dashboard.

## 3. Security & Standards
- **Safety**: Sandboxed execution (`sandbox-exec`) mandatory for all shell operations with tiered security policies (T1/T2/T3). Manual UI confirmation required for all file writes and T2/T3 bash commands.
- **Roles**: Explorer, Security Auditor, Code Writer, Test Engineer, Code Reviewer.
- **Standards**: Surgical atomic edits | Textual `reactive` state | 84% coverage threshold.

## 4. Operational Protocols
- **Verify**: `uv run pytest tests` | `uv run pytest tests -m unit`
- **Lint**: `uv run ruff check .` | `uv run ruff format .` | `uv run mypy .`
- **Lookup**: Config (`.env`) | Deps (`pyproject.toml`) | Logs (`app.log`)

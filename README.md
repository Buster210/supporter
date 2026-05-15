# Supporter

An elite technical strategist and principal software architect TUI chat client powered by Google Gemini. `supporter` orchestrates complex development tasks via a multi-agent delegation system and sandboxed tool execution.

## Features

- **Multi-Agent Orchestration**: Orchestrator-driven delegation to specialized roles (Architect, Security Auditor, Code Writer, Explorer).
- **Sandboxed Execution**: Mandatory `sandbox-exec` for all shell operations with tiered security policies (T1/T2/T3).
- **High-Fidelity TUI**: Reactive dashboard built with Textual and Rich, featuring real-time streaming and async progress notifications.
- **Reliability Layer**:
    - **Load Balancing**: Health-aware round-robin rotation across multiple Gemini API keys.
    - **Dynamic Fallback**: Seamless transition to fallback models on 5xx or quota errors.
- **Surgical Tooling**:
    - `execute_bash`: Sandboxed shell with UI-confirmed risky operations.
    - `read_file` / `write_file`: Path-validated file operations restricted to project root.
    - `google_search`: Real-time research capabilities.
    - `delegate_tasks`: DAG-based background task execution.

## Agent Roster

| Role | Persona | Specialist Tools |
| ---- | ------- | ---------------- |
| **Explorer** | Read-only reconnaissance specialist | `read_file`, `execute_bash`, `google_search` |
| **Security Auditor** | Vulnerability and risk analysis | `read_file`, `execute_bash` |
| **Code Writer** | Feature implementation and refactoring | `read_file`, `write_file`, `execute_bash` |
| **Test Engineer** | TDD specialist and verification | `read_file`, `execute_bash` |
| **Code Reviewer** | Quality and convention enforcement | `read_file` |

## Installation

This project requires Python 3.13+ and `uv`.

```bash
# Clone the repository
git clone https://github.com/your-org/supporter.git
cd supporter

# Install dependencies
uv sync
```

## Usage

1. **Configure Environment**:
   Create a `.env` file in the project root:

   ```bash
   GEMINI_API_KEYS=your_key_1,your_key_2
   GEMINI_MODEL=gemma-4-31b-it
   GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
   GEMINI_FALLBACK_MODEL=gemini-2.5-flash-native-audio-preview-12-2025
   LOG_LEVEL=info
   ```

2. **Launch the TUI**:

   ```bash
   uv run supporter
   ```

## Architecture

**Flow**: User → **TUI** (Textual) → **Orchestrator** (Agent Logic) → **Tools** (Bash/Delegate/Search) → **LLM** → Output

- `src/supporter/agent.py`: Core orchestrator and delegation logic.
- `src/supporter/providers/`: Gemini Live and REST implementations.
- `src/supporter/tools/bash/`: Security-hardened shell execution logic.
- `src/supporter/tui/`: Reactive UI and dashboard management.

## Development & Verification

### Testing
Tests are organized by layer and marker:
- `tests/unit/`: Component-level logic.
- `tests/integration/`: Cross-component flows.
- `tests/e2e/`: Full lifecycle verification.

```bash
uv run pytest tests                 # Run all tests
uv run pytest tests -m unit         # Run unit tests
uv run pytest tests -m integration   # Run integration tests
```

### Quality Control
```bash
uv run ruff check .      # Lint check
uv run ruff format .     # Format code
uv run mypy .            # Type check
```

## Security Standards
- **Isolation**: Sub-agents have zero conversation history and minimal tool access.
- **Validation**: All file writes and T2/T3 bash commands require manual UI confirmation.
- **Constraints**: No piped/chained shell commands; strict path traversal protection.

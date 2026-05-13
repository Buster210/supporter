# Project Intelligence & Orchestration (AGENTS.md)

## 1. Project DNA
- **Mission**: Python TUI AI chat client using Google Gemini (Live & Streaming) with load balancing & multi-agent delegation.
- **Stack**: Python 3.13+, `uv`, `google-genai`, `textual`, `rich`.
- **Ref**: [README.md](file:///Users/riteshkumarpal/Downloads/lab/supporter/README.md) | **Entry**: `uv run supporter` (defined in `pyproject.toml`)

## 2. Arch & Flow
**Flow**: User → `SupporterApp` → `ChatAgent` → `DynamicPool` → `Tools`.
**Structure** (`src/supporter/`):
- `agent.py`: ChatAgent orchestrator, prompts, tool routing.
- `pool.py`: DynamicPool — API key rotation, health checks, cooldowns.
- `providers/`: `gemini_provider.py` (REST/Streaming), `gemini_live_provider.py` (WebSocket).
- `tools/`: `bash/` (sandbox-exec), `delegate/` (DAG scheduler), `file_ops.py`, `search.py`, `catalog.py`.
- `tui/`: `__init__.py` (Main App), `bubble.py`, `chat.py`, `mode_manager.py`, `styles.tcss`.
- `config.py`: AppConfig/Env | `types.py`: Enums/Events.

## 3. Agents & Tools
- **Roster**: architect, test_engineer, code_writer, researcher, code_reviewer, scout (gemini-3.1-flash-lite-preview).
- **Tools**: `execute_bash` (sandbox T1/T2/T3), `read_file`, `write_file` (UI confirm), `google_search`, `delegate_tasks` (DAG → job_id), `check_delegation`.
- **Security**: Strict path validation (root + `.gitignore`). UI confirm for T2/T3 bash & all writes. No piped/chained commands.

## 4. Delegation & Protocols
- **DAG**: `{"id": "fix", "depends_on": ["analyze"]}` | **Milestone**: Background task groups.
- **Retry**: Max 2, backoff [1s, 3s]. **Anomaly**: 80% timeout logged. **Heartbeat**: 30s.
- **Verify**: `uv run pytest tests` [-m `unit`/`integration`/`e2e`].
- **Lint**: `uv run ruff check .` | `ruff format .` | `mypy .`.
- **Run**: `uv run supporter`.

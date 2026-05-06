# AGENTS.md

## 1. Project DNA
- **Mission**: Python TUI AI chat client using Google Gemini (Live & Streaming) with load balancing & multi-agent delegation.
- **Stack**: Python 3.13+, `uv`, `google-genai`, `textual`, `rich`.
- **Entry**: `src/supporter/tui/__init__.py` (`supporter` CLI via `pyproject.toml`).

## 2. Arch & Flow
**Flow**: User → `SupporterApp` → `ChatAgent` → `DynamicPool` → `Tools`.

**Structure** (`src/supporter/`):
- `agent.py`: ChatAgent orchestrator, prompts, history, tool routing.
- `index.py`: DynamicPool — API key rotation, health checks, 5XX/429 cooldowns.
- `providers/`: `gemini_provider.py` (REST/streaming), `gemini_live_provider.py` (WebSocket).
- `tools/`: `bash.py` (sandbox-exec/nsjail), `delegate.py` (DAG), `file_ops.py`, `search.py` (Google), `event_bus.py` (pub/sub).
- `tui/`: `__init__.py` (SupporterApp), `bubble.py`, `chat.py`, `message_processor.py`.
- `config.py`: AppConfig, env, roster. `types.py`: types, enums, events.

## 3. Agents & Tools
- **Roster**: architect, test_engineer, code_writer, researcher, code_reviewer, scout (gemini-3.1-flash-lite-preview)
- **Tools**: execute_bash (sandbox, CPU 10s, blocks pipes), read_file, write_file (UI confirm), google_search, delegate_tasks (DAG → job_id), check_delegation
- **Security**: Path validation (root + `.gitignore`), UI confirm writes/bash, no pipes/chained. Sandbox Tiers: T1=safe → auto, T2=risky → confirm, T3=blocked → denied.

## 4. Delegation
- **depends_on**: Task IDs to wait for, forms DAG. Ex: `{"id": "fix", "depends_on": ["analyze"]}`
- **Milestone**: Task group, runs in background.
- **Workflow**: delegate_tasks → auto-posted results → synthesize.
- **Retry**: Max 2, backoff [1s, 3s]. ERROR only.
- **Anomaly**: 80% timeout → logged, not cancelled. **Heartbeat**: 30s interval.
- **Skip**: Dependency failed + tolerate_failures=false.
- **Output**: Truncated at 10k chars. **Parallel**: Default 3, max 5.

## 5. Environment
- GEMINI_API_KEYS (required) - comma-separated keys
- GEMINI_MODEL - gemma-4-31b-it (primary)
- GEMINI_FALLBACK_MODEL - fallback model
- GEMINI_LIVE_MODEL - gemini-3.1-flash-live-preview
- LOG_LEVEL - INFO

## 6. Protocols
- **Verify**: `uv run pytest tests` [-m `unit`/`integration`/`e2e`]
- **Lint**: `uv run ruff check .` | `ruff format .` | `mypy .`

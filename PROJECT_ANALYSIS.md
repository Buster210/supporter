# Supporter - Codebase Reference

**Package**: `supporter` (imports via `from supporter import ...`)
**Entrypoint**: `supporter.tui:main` → `uv run python src/supporter/tui.py`

## Directory Structure

```
supporter/
├── __init__.py           ChatAgent, CrewAgent exports
├── agent.py              ChatAgent class
├── config.py             config object
├── index.py              get_provider function, DynamicPool
├── logger.py             logger object
├── crew/                 CrewAgent, crew_adapter
├── providers/            GeminiProvider, get_provider
├── tools/                read_file, write_file, execute_bash, google_search
└── tui/                 SupporterApp, main, message_processor, mode_manager
```

## Key Exports

- `ChatAgent`, `CrewAgent`, `get_provider`
- `config` object
- `main()` function

## Execution Flow

User Input → run_worker → ChatAgent/CrewAgent → GeminiProvider → Tools → Confirmation Modals → MessageBubble

## Components

| Component | File | Purpose |
|-----------|------|---------|
| DynamicPool | index.py | Rotate API keys |
| get_provider | providers/__init__.py | Provider singleton |
| read_file | tools/file_ops.py | Read with validation |
| write_file | tools/file_ops.py | Write with confirmation |
| execute_bash | tools/bash.py | Sandboxed execution with Mutation Tracking |
| google_search | tools/search.py | Gemini search |
| ChatMessageProcessor | tui/message_processor.py | Streaming & Crew Handover |
| ModeManager | tui/mode_manager.py | Mode commands |
| ChatTurn | tui/widgets.py | Collapsible message turns |
| ThinkingIndicator | tui/widgets.py | Status indicator |
| ConfirmationModal | tui/widgets.py | Write approval |
| BashConfirmationModal | tui/widgets.py | Bash approval |

## Security Model

- Path validation: Project root + .gitignore + blacklist
- Bash: TIER1 (safe) / TIER2 / TIER3 (risky)
- Write: User confirmation always

## Tests

- **Unit**: `tests/unit/` (Bash security, Config, File Ops, UI Widgets)
- **Integration**: `tests/integration/` (Agent streaming, Gemini Provider, Crew, TUI logic)
- **E2E**: `tests/e2e/` (Conversation flow, File ops lifecycle, TUI startup)
- **Mocks**: `tests/mocks.py` (Provider), `tests/tui_mocks.py` (Textual)

## Quick Lookup

| Need | Go To |
|------|-------|
| Add tool | tools/ |
| Add widget | tui/widgets.py |
| Config | .env, config.py |
| Tests | tests/ |

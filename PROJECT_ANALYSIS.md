# Master Context: Supporter

**Supporter** is a Python Textual TUI chat client powered by Google Gemini, designed for rapid terminal interaction with CrewAI multi-agent support and dynamic API key load balancing.

## Architecture & Components

1. **UI (`tui.py`)**: Manages app state, commands, and non-blocking worker orchestration. Employs concurrent query tracking for real-time indicators.
2. **Agent Orchestration (`agent.py`)**:
    - `ChatAgent`: Handles interactions, tools, and `interaction_id` continuity for single-agent flows.
    - `CrewAgent` (`crew_agent.py`): Assembles a sequential Crew (Researcher/Writer).
3. **DynamicPool (`index.py`)**: Manages model instances across multiple API keys using `collections.deque` for O(1) rotation and load balancing. Employs centralized error classification (`is_model_error`) to trigger background replacements on transient 5XX signals.
4. **LazyFallbackProvider**: Coordinates failover logic (e.g. from `gemma-4-31b-it` to `gemini-2.5-flash-lite`).
5. **Provider (`gemini_provider.py`)**: `google-genai` wrapper with lazy client initialization and `_tool_cache` to minimize tool transformation overhead. Handles dynamic `_transform_tools` injection and `interactions.create`.
6. **Adapter (`crew_adapter.py`)**: Bridges `LLMProvider` with CrewAI's `BaseLLM` using a dedicated background event loop thread (`SupporterAsyncBridge`) for efficient sync-to-async execution via `run_coroutine_threadsafe`.
7. **Observability (`logger.py`)**: `SupporterFormatter` provides high-fidelity lifecycle tracking and initialization.

## Environment Stack

- **Tech**: Python 3.13+, `google-genai`, `crewai`, `textual`, `rich`, `python-dotenv`.
- **Config Variables**:
  - `GEMINI_API_KEYS` (CSV for balancing)
  - `LLM_PROVIDER`, `LOG_LEVEL`, `LOG_FILE`
  - `GEMINI_MODEL` (Default: `gemma-4-31b-it`), `GEMINI_FALLBACK_MODEL`

## Technical Health & Roadmap

- **Status**: [✅ Approve] Performant streaming, secure environment handling, and optimized resource rotation.
- **Sync/Async Bridge**: Optimized via persistent background loop thread (`SupporterAsyncBridge`), eliminating the overhead of per-call `ThreadPoolExecutor` management.
- **State/Persistence**: History is currently ephemeral. Roadmap includes SQLite (`aiosqlite`) persistence and multimodal input processing.

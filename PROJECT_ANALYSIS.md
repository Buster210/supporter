# Master Context: Supporter

**Supporter** is a Python Textual TUI chat client powered by Google Gemini, designed for rapid terminal interaction with CrewAI multi-agent support and dynamic API key load balancing.

## Architecture & Components

1. **UI (`tui.py`)**: Manages app state, commands, and non-blocking worker orchestration. Employs concurrent query tracking for real-time indicators.
2. **Agent Orchestration (`agent.py`)**:
    - `ChatAgent`: Handles interactions, tools, and `interaction_id` continuity for single-agent flows.
    - `CrewAgent` (`crew_agent.py`): Assembles a sequential Crew (Researcher/Writer).
3. **LLM Abstraction (`index.py`)**:
    - `DynamicPool`: Lazily manages/rotates instances across multiple keys to multiply rate limits. Triggers background `asyncio.create_task` replacements on 429/500 errors.
    - `LazyFallbackProvider`: Coordinates failover logic (e.g. from `gemini-flash-lite` to `gemini-2.0-flash`).
4. **Provider (`gemini_provider.py`)**: `google-genai` wrapper handling `interactions.create` (aio) continuity and dynamic `_transform_tools` injection (wraps sync/async functions into Gemini tools).
5. **Adapter (`crew_adapter.py`)**: Bridges `LLMProvider` with CrewAI's `BaseLLM` using an `asyncio.run` ThreadPool executor to bridge async provider logic into CrewAI's sequential execution.
6. **Observability (`logger.py`)**: `SupporterFormatter` provides high-fidelity lifecycle tracking and initialization.

## Environment Stack

- **Tech**: Python 3.13+, `google-genai`, `crewai`, `textual`, `rich`, `python-dotenv`.
- **Config Variables**:
  - `GEMINI_API_KEYS` (CSV for balancing)
  - `LLM_PROVIDER`, `LOG_LEVEL`, `LOG_FILE`
  - `GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL`

## Technical Health & Roadmap

- **Status**: [✅ Approve] Clean separation, secure environment handling, and performant streaming/balancing.
- **Sync/Async Bridge**: CrewAI integration relies on `ThreadPoolExecutor` and `asyncio.run` overhead, acting as a structural constraint.
- **State/Persistence**: History is currently ephemeral. Roadmap includes SQLite (`aiosqlite`) persistence and multimodal input processing.

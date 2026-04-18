# Master Context: Supporter (EARC - Final Principal Edition)

**Supporter** is a high-performance Python Textual TUI chat client powered by Google Gemini, optimized for rapid interaction, multimodal live sessions, and CrewAI orchestration with dynamic key rotation.

## 1. Project Anatomy & Manifests
- **Entrypoint**: `supporter.tui:main` (Command: `supporter`).
- **Dependency Management**: `uv` based (`uv.lock`). Core dependencies: `google-genai` (V1 SDK), `textual`, `crewai`, `rich`.
- **Quality Gate**: `prek.toml` manages pre-commit hooks: `trailing-whitespace`, `ruff`, `mypy`, `bandit`, `detect-secrets`.
- **Configuration**: `AppConfig` dataclass in `config.py` loaded via `python-dotenv`.
- **Tooling Framework**: `src/supporter/tools.py` defines a unified execution layer for LLM-accessible tools (e.g., `GoogleSearch`, `Weather`).

## 2. Infrastructure Internals
- **Pooling (`index.py`)**: `DynamicPool` maintains a `deque` of `GeminiProvider` slots. `rotate(-1)` handles round-robin selection. `slot_available` (`asyncio.Event`) handles backpressure during multi-slot recovery.
- **Failover Logic**: 5XX errors trigger `_mark_model_cooldown` (30m) and background `_fill_slot`. 429 errors trigger immediate slot replacement without full model cooldown.
- **Async Bridge (`crew_adapter.py`)**: `SupporterAsyncBridge` (Daemon Thread) persistent event loop. `asyncio.run_coroutine_threadsafe` bridges CrewAI (Sync) to Provider Pool (Async).

## 3. UI Technicals (`tui.py`)
- **Turn Management**: Interactions are encapsulated in `ChatTurn` components, grouping a user query with the agent's response(s).
- **UX Interaction**:
    - **Collapsing**: Previous chat turns auto-collapse when a new message is submitted to minimize scroll fatigue.
    - **Streaming**: `MessageBubble` logic separates model `thoughts` (italicized) from `content` (streaming tokens).
    - **Status Feedback**: Dynamic thinking indicator reflects real-time tool usage (e.g., "Searching", "Using [Tool]").
- **Theme**: HSL-interpolated Cristal gradients in `SupporterHeader`.

## 4. Interaction Logic (`agent.py`, `gemini_provider.py`)
- **Tool Execution**: Providers handle `Part` objects with `tool_call` attributes. Tools are executed via the `tools.py` registry, and results are injected back into the conversation history.
- **Conversation State**: `ChatAgent` maintains `history: list[Content]`. `automatic_function_calling_history` is prioritized for state updates to preserve SDK-managed tool turns.
- **Continuity**: `GeminiProvider` uses `interactions.create(previous_interaction_id=...)` for stateful backend sessions.
- **Live Multimodal**: `GeminiLiveProvider` uses `asyncio.Lock` for session/turn synchronization.

## 5. Testing & Mocks
- **Framework**: `pytest` + `pytest-asyncio`.
- **Mocking**: `tests/mocks.py` provides `create_mock_genai_client`, mocking `aio.models` and `aio.interactions` for both streaming and unary calls. Updated to support part-based streaming and tool-call simulations.

## 6. Project Flow-Graph
- `tui` -> `ChatTurn` -> `ChatAgent` (History/Tools) -> `DynamicPool` -> `GeminiProvider` (SDK/Continuity).
- `tui` -> `CrewAgent` -> `SupporterAsyncBridge` (Loop) -> `DynamicPool`.

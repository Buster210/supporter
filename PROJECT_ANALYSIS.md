# Master Context: Supporter (EARC - Final Principal Edition)

**Supporter** is a high-performance Python Textual TUI chat client powered by Google Gemini, optimized for rapid interaction, multimodal live sessions, and CrewAI orchestration with dynamic key rotation.

## 1. Project Anatomy & Manifests
- **Entrypoint**: `supporter.tui:main` (Command: `supporter`).
- **Dependency Management**: `uv` based (`uv.lock`). Core dependencies: `google-genai` (V1 SDK), `textual`, `crewai`, `rich`.
- **Quality Gate**: `prek.toml` manages pre-commit hooks: `trailing-whitespace`, `ruff`, `mypy`, `bandit`, `detect-secrets`.
- **Configuration**: `AppConfig` dataclass in `config.py` loaded via `python-dotenv`. Default search/code tools enabled in `ChatAgent`.

## 2. Infrastructure Internals
- **Pooling (`index.py`)**: `DynamicPool` maintains a `deque` of `GeminiProvider` slots. `rotate(-1)` handles round-robin selection. `slot_available` (`asyncio.Event`) handles backpressure during multi-slot recovery.
- **Failover Logic**: 5XX errors trigger `_mark_model_cooldown` (30m) and background `_fill_slot`. 429 errors trigger immediate slot replacement without full model cooldown.
- **Async Bridge (`crew_adapter.py`)**: `SupporterAsyncBridge` (Daemon Thread) persistent event loop. `asyncio.run_coroutine_threadsafe` bridges CrewAI (Sync) to Provider Pool (Async).

## 3. UI Technicals (`tui.py`)
- **Theme**: HSL-interpolated Cristal gradients in `SupporterHeader`. `MessageBubble` logic separates model `thoughts` (italicized markdown) from `content` (streaming tokens).
- **State Management**: `active_queries` counter drives the UI spinner. Commands (`/live`, `/crew`, `/clear`) re-initialize the `agent` instance via `_setup_agent`.
- **Worker Pattern**: `run_worker` used for I/O execution to maintain UI responsiveness.

## 4. Interaction Logic (`agent.py`, `gemini_provider.py`)
- **Conversation State**: `ChatAgent` maintains `history: list[Content]`. `automatic_function_calling_history` is prioritized for state updates to preserve SDK-managed tool turns.
- **Continuity**: `GeminiProvider` uses `interactions.create(previous_interaction_id=...)` for stateful backend sessions.
- **Live Multimodal**: `GeminiLiveProvider` uses `asyncio.Lock` for session/turn synchronization.

## 5. Testing & Mocks
- **Framework**: `pytest` + `pytest-asyncio`.
- **Mocking**: `tests/mocks.py` provides `create_mock_genai_client`, mocking `aio.models` and `aio.interactions` for both streaming and unary calls.
- **Base Logic**: Tests primarily verify stateful history updates and key rotation in `DynamicPool`.

## 6. Project Flow-Graph
- `tui` -> `ChatAgent` (History/Tools) -> `DynamicPool` (Balancing/Failover) -> `GeminiProvider` (SDK/Continuity).
- `tui` -> `CrewAgent` -> `SupporterAsyncBridge` (Loop) -> `DynamicPool`.

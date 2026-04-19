# Master Context: Supporter (EARC - Final Principal Edition)

**Supporter** is a high-performance Python Textual TUI chat client powered by Google Gemini, optimized for rapid interaction, multimodal live sessions, and CrewAI orchestration with dynamic key rotation.

## 1. Project Anatomy & Manifests
- **Entrypoint**: `src.supporter.tui:main` (Command: `supporter`).
- **Dependency Management**: `uv` based (`uv.lock`). Core dependencies: `google-genai` (V1 SDK), `textual`, `crewai`, `rich`.
- **Modular Structure**:
    - `src/supporter/tui/`: Encapsulated UI logic, widgets, and state management.
    - `src/supporter/providers/`: Modular LLM provider implementations (Gemini, Gemini Live).
    - `src/supporter/crew/`: CrewAI integration and sync-to-async adapters.
- **Quality Gate**: `prek.toml` manages pre-commit hooks: `trailing-whitespace`, `ruff`, `mypy`, `bandit`, `detect-secrets`.
- **Configuration**: `AppConfig` dataclass in `config.py` loaded via `python-dotenv`.

## 2. Infrastructure Internals
- **Pooling (`index.py`)**: `DynamicPool` maintains a `deque` of `GeminiProvider` slots. `rotate(-1)` handles round-robin selection. `slot_available` (`asyncio.Event`) handles backpressure during multi-slot recovery.
- **Failover Logic**: 5XX errors trigger `_mark_model_cooldown` (30m) and background `_fill_slot`. 429 errors trigger immediate slot replacement without full model cooldown.
- **Async Bridge (`src/supporter/crew/crew_adapter.py`)**: `SupporterLLM` (Custom CrewAI LLM) bridges sync execution to the async Provider Pool using a background daemon thread (`SupporterAsyncBridge`).

## 3. UI Technicals (`src/supporter/tui/`)
- **Mode Management**: `ModeManager` handles switching between `LIVE` (streaming) and `CREW` (agentic) modes via `/live` and `/crew` slash commands.
- **Message Processing**: `ChatMessageProcessor` orchestrates turn execution, handling streaming responses and multi-agent progress updates.
- **Turn Management**: Interactions are encapsulated in `ChatTurn` widgets, grouping user queries with agent response bubbles.
- **UX Interaction**:
    - **Collapsing**: Previous chat turns auto-collapse when a new message is submitted to minimize scroll fatigue.
    - **Streaming**: `MessageBubble` handles real-time token rendering, distinguishing model `thoughts` from final content.
    - **Status Feedback**: `SpinnerController` provides visual feedback during active generations.
- **Theme**: Centralized `styles.css` with HSL-interpolated Cristal gradients.

## 4. Interaction Logic (`src/supporter/providers/`)
- **Provider Layer**: `GeminiProvider` and `GeminiLiveProvider` handle SDK interactions, including automatic function calling and thought extraction.
- **Tool Execution**: Tools defined in `src/supporter/tools.py` are dynamically injected into generation configs and executed via the provider registry.
- **Conversation State**: `ChatAgent` maintains persistent history. `automatic_function_calling_history` is prioritized for state synchronization.
- **Continuity**: Uses `interactions.create(previous_interaction_id=...)` for stateful session resumption.

## 5. Testing & Mocks
- **Framework**: `pytest` + `pytest-asyncio`.
- **Mocking**: `tests/mocks.py` provides `create_mock_genai_client`, mocking `aio.models` and `aio.interactions` for both streaming and unary calls.

## 6. Project Flow-Graph
- `tui` -> `ModeManager` -> `ChatAgent` -> `DynamicPool` -> `GeminiProvider`.
- `tui` -> `ChatMessageProcessor` -> `SupporterLLM` (Crew) -> `DynamicPool`.
- `tui` -> `widgets.py` (ChatTurn/MessageBubble) -> `styles.css`.

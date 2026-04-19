# Master Context: Supporter (EARC - Final Principal Edition)

**Supporter** is a high-performance Python Textual TUI chat client powered by Google Gemini, optimized for rapid interaction, multimodal live sessions, and CrewAI orchestration with dynamic key rotation and secure file-system tooling.

## 1. Project Anatomy & Manifests
- **Entrypoint**: `src.supporter.tui:main` (Command: `supporter`).
- **Dependency Management**: `uv` based (`uv.lock`). Core dependencies: `google-genai` (V1 SDK), `textual`, `crewai`, `pathspec` (for secure file filtering).
- **Modular Structure**:
    - `src/supporter/tui/`: Encapsulated UI logic, widgets, and modal-based state management.
    - `src/supporter/providers/`: Modular LLM provider implementations (Gemini, Gemini Live).
    - `src/supporter/crew/`: CrewAI integration and sync-to-async adapters.
- **Root Detection**: `config.py` uses `_get_project_root` to automatically locate the repository base (via `.git` or `pyproject.toml`) for sandboxing.

## 2. Infrastructure Internals
- **Pooling (`index.py`)**: `DynamicPool` maintains a `deque` of `GeminiProvider` slots. Implements **Lazy Initialization**: slots are filled on-demand (`_fill_slot`) during execution if empty, reducing startup latency.
- **Provider Registry**: `_provider_registry` implements a singleton cache for provider instances. `get_provider` returns cached instances by default (`shared=True`), ensuring pool reuse across different application components.
- **Failover Logic**: 5XX errors trigger `_mark_model_cooldown` (30m). 429 errors trigger immediate slot replacement.

## 3. Security & Sandboxing (`src/supporter/tools.py`)
- **FS Validation**: `_validate_path` enforces strict security boundaries:
    - **Jailbreaking Protection**: Operations restricted to `allowed_directories` (project root).
    - **Git-Awareness**: Uses `pathspec` to block access to files ignored by `.gitignore`.
    - **Internal Blacklist**: Protects sensitive internal directories (`.gemini`, `__pycache__`).
- **Write Confirmation**: `write_file` triggers a TUI-level security callback (`set_confirmation_callback`). Users must approve file writes via a unified diff interface before changes are committed.

## 4. UI Technicals (`src/supporter/tui/`)
- **Modal System**: `ConfirmationModal` in `widgets.py` handles secure user approvals, providing syntax-highlighted diffs of proposed file changes.
- **Status Reporting**: `SpinnerController` provides agent-aware feedback. In `CREW` mode, it displays the specific active agent (e.g., `[Researcher] Thinking...`).
- **Mode Management**: `ModeManager` orchestrates agent setup, injecting the tool registry into both `ChatAgent` and `CrewAgent`.
- **UX interaction**: Auto-collapsing chat turns, real-time token streaming with thought extraction, and HSL-interpolated Cristal themes.

## 5. Interaction Logic
- **Tool Registry**: Standard tools (`read_file`, `write_file`, `list_dir`) are defined in `tools.py` and registered with the LLM provider during initialization.
- **Gemini Live Multimodal**: `GeminiLiveProvider` handles realtime sessions, extracting `grounding_metadata` and providing mock GenAI SDK objects for compatibility with existing tooling.
- **Continuity**: Uses `previous_interaction_id` for stateful session resumption.

## 6. Testing & Mocks
- **Framework**: `pytest` + `pytest-asyncio`.
- **Mocking**: `tests/mocks.py` provides `MockRaw`/`MockCandidate` structures to simulate multimodal grounding and tool response objects.

## 7. Project Flow-Graph
- `tui` -> `ModeManager` -> `ChatAgent` (Registry) -> `DynamicPool` -> `GeminiProvider`.
- `tui` -> `ConfirmationModal` (Security Hook) -> `tools.py` -> File System.
- `tui` -> `ChatMessageProcessor` -> `SupporterLLM` (Crew) -> `DynamicPool`.
- `tui` -> `widgets.py` -> `styles.css`.

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
- **Pooling (`index.py`)**: `DynamicPool` maintains a `deque` of `GeminiProvider` instances, implementing a simplified **Round-Robin** rotation without complex async waiting logic for slot availability.
- **Provider Registry**: `_provider_registry` implements a singleton cache for provider instances, now protected by `_provider_lock` for thread-safe access. `get_provider` ensures shared instances are reused across different application components.
- **Failover Logic**: 5XX errors trigger `_mark_model_cooldown` (30m). 429 errors trigger immediate provider rotation in the pool.

## 3. Security & Sandboxing (`src/supporter/tools.py`)
- **FS Validation**: `_validate_path` enforces strict security boundaries:
    - **Jailbreaking Protection**: Operations restricted to `allowed_directories` (project root).
    - **Git-Awareness**: Uses `pathspec` to block access to files ignored by `.gitignore`.
    - **Internal Blacklist**: Protects sensitive internal directories (`.gemini`, `__pycache__`).
- **Write Confirmation**: `write_file` triggers a TUI-level security callback (`set_confirmation_callback`). Users must approve file writes via a unified diff interface before changes are committed.

## 4. UI Technicals (`src/supporter/tui/`)
- **Modular Rendering**: `MessageBubble` in `widgets.py` uses modularized rendering (`_render_thoughts`, `_render_tools`, `_render_main_content`) to handle complex multimodal and tool-aware message states.
- **Turn Management**: `ChatTurn` implements clean state management for collapsible turns via `expand_turn` and `collapse_turn`, ensuring UI consistency and responsive interaction.
- **Modal System**: `ConfirmationModal` handles secure user approvals with syntax-highlighted diffs for file changes.
- **UX interaction**: Real-time token streaming with thought extraction, HSL-interpolated Cristal themes, and optimized layout for long-running agentic sessions.

## 5. Interaction Logic
- **History Validation**: `ChatAgent` includes enhanced history handling with safety checks for null candidates and content parts, preventing state corruption during long interactions.
- **Tool Registry**: Standard tools (`read_file`, `write_file`, `list_dir`) are registered with the LLM provider, with automated TUI-level confirmation hooks.
- **Gemini Live Multimodal**: `GeminiLiveProvider` handles realtime sessions, extracting `grounding_metadata` and providing mock GenAI SDK objects for compatibility.

## 6. Testing & Mocks
- **Framework**: `pytest` + `pytest-asyncio` with strict async compliance across provider and TUI tests.
- **Isolation**: Uses `index.clear_providers()` in test fixtures to ensure registry isolation and prevent cross-test state leakage.
- **Mocking**: `tests/mocks.py` provides `MockRaw`/`MockCandidate` structures, recently updated to align with the latest SDK chunk structures and tool-calling patterns.

## 7. Project Flow-Graph
- `tui` -> `ModeManager` -> `ChatAgent` (Registry) -> `DynamicPool` -> `GeminiProvider`.
- `tui` -> `ConfirmationModal` (Security Hook) -> `tools.py` -> File System.
- `tui` -> `ChatMessageProcessor` -> `SupporterLLM` (Crew) -> `DynamicPool`.
- `tui` -> `widgets.py` -> `styles.css`.

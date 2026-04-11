# Project Analysis: Supporter

## Overview

**Supporter** is a Python-based Terminal User Interface (TUI) AI chat client powered by the Google Gemini model. It is designed to be a lightweight yet powerful tool for interacting with AI agents directly from the terminal.

## Core Purpose

The project serves as a migration/implementation of an AI chat client that supports advanced features like multi-agent collaboration, load balancing across multiple API keys, and robust error handling through model fallbacks.

## Key Features

- **TUI Interface**: Built with [Textual](https://github.com/Textualize/textual), providing a modern and interactive terminal experience with real-time thinking indicators and gradient headers.
- **Real-time Streaming**: Supports asynchronous response streaming for `ChatAgent`, providing immediate visual feedback in the TUI.
- **Multi-Agent Collaboration**: Integrated with [CrewAI](https://github.com/joaomdmoraes/crewai), supporting a "Crew" mode with specialized Researcher and Writer agents.
- **Advanced Load Balancing**: Implements a `RoundRobinPool` to cycle through multiple `GEMINI_API_KEYS`, effectively multiplying the available rate limits.
- **Resilient Fallback**: The `FallbackProvider` automatically switches from the primary model (e.g., `gemini-flash-lite-latest`) to a fallback (e.g., `gemini-2.0-flash`) on transient errors or rate limits.
- **Unified Tool Registry**: Wraps both sync and async Python functions into a format compatible with Gemini's automatic function calling.
- **Enterprise Observability**: Comprehensive lifecycle logging across all modules (LLM, Agents, TUI) with interaction tracking and detailed pool diagnostics.

## System Architecture

### Component Relationships

The project follows a layered architecture:

1. **UI Layer (`tui.py`)**: Manages the Textual application state, input processing, and command execution (`/crew`, `/clear`, `/exit`).
2. **Agent Orchestration Layer**:
    - `ChatAgent`: Handles single-agent interactions with Google Search and Code Execution.
    - `CrewAgent` (`crew_agent.py`): Managed by `CrewManager`, assembles a sequential Crew of Researcher and Writer roles.
3. **LLM Abstraction Layer (`index.py`)**:
    - Defines the `LLMProvider` protocol.
    - `RoundRobinPool`: Distributes calls across API keys.
    - `FallbackProvider`: Handles model-level failover.
4. **Provider Layer (`gemini_provider.py`)**: Low-level wrapper for the `google-genai` SDK, handling prompt preparation, tool wrapping, and response parsing. Supports both unary and streaming generation.
5. **Adapter Layer (`crew_adapter.py`)**: Bridges the custom `LLMProvider` protocol with CrewAI's `BaseLLM`, handling complex threading requirements (running async provider logic inside CrewAI's execution context).
6. **Observability Layer (`logger.py`)**: Centralized logging system with module-level initialization tracking and detailed debug-level instrumentation for debugging complex agentic flows.

### Logic Flows

- **Message Cycle**:
  - *Unary*: `Input.Submitted` -> `_process_message_cycle` -> `Agent.execute` -> `LLMProvider.generate` -> UI update.
  - *Streaming*: `Input.Submitted` -> `_process_message_cycle` -> `Agent.execute_stream` -> `LLMProvider.generate_stream` -> Real-time UI updates (via `MessageBubble` streaming mode).
- **Load Balancing**: `generate` call -> `RoundRobinPool` picks next key -> `GeminiProvider` calls API -> On failure (429), `RoundRobinPool` retries with next key. Success/failure is traced with detailed latency and usage metadata.
- **Fallback**: If all keys in the primary model pool fail, `FallbackProvider` catches the exception and routes the request to the secondary model pool.
- **Observability Trace**: Each module transition (e.g., entering `setup_agent`, handling a command) is logged with `logger.debug` for high-fidelity tracing of the TUI lifecycle.

## Technical Stack

- **Languages**: Python 3.13+ (using `uv` for lightning-fast dependency management)
- **AI Frameworks**: `google-genai` (Official SDK), `crewai` (Agentic framework)
- **UI Framework**: `textual` (TUI framework), `rich` (Styling)
- **Environment Management**: `python-dotenv`, `pydantic` (for configuration)

## Project Structure

- `src/supporter/`: Core logic
  - `tui.py`: Main entry point for the Textual application.
  - `gemini_provider.py`: Handles interaction with Google Gemini API.
  - `crew_adapter.py` & `crew_agent.py`: Integration with CrewAI.
  - `config.py`: Configuration and environment variable management.
- `tests/`: Automated test suite.
- `pyproject.toml`: Dependency and build configuration.

## Master Context Audit

### Audit Status: [✅ Approve]

The codebase follows high-quality engineering principles:

- **Architecture**: Clear separation of concerns between UI, orchestration, and provider layers.
- **Security**: Environment-based secret management and encapsulated API clients.
- **Performance**: Real-time streaming and load balancing across API keys minimize latency and rate-limit friction.
- **Cleanliness**: Idiomatic Python with robust logging and linting (Ruff).

### Technical Health & Roadmap

#### Risks & Technical Debt

- **Ephemeral State**: Chat history is currently stored in memory and lost on application exit.
- **Sync/Async Bridge**: CrewAI integration relies on a sync-to-async bridge which adds complexity to the adapter layer.
- **Streaming Parity**: `CrewAgent` currently lacks streaming support due to the underlying framework's sequential nature.

#### Roadmap

- **Persistence**: Implement a local SQLite backend (using `aiosqlite`) for persistent chat history and interaction logging.
- **Multimodal Support**: Expand `GeminiProvider` to handle image and document processing.
- **Advanced Tools**: Integrate more specialized tools for data analysis and visualization.

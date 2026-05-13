# Supporter

A Python-based TUI AI chat client using the Google Gemini model.

## Features

- **Multi-agent Delegation**: Integrated DAG-based task scheduling and multi-agent collaboration.
- **Round-Robin Load Balancing**: Support for multiple Gemini API keys with health-aware rotation.
- **Model Fallback**: Automatically switches to a fallback model if the primary model fails.
- **Real-time Streaming**: Asynchronous response streaming for immediate feedback.
- **TUI Interface**: A beautiful terminal user interface built with Textual and Rich.
- **Tool Support**: Integrated tool registry for agentic capabilities (Bash, File Ops, Search).
- **Observability**: Detailed lifecycle logging and diagnostic tracing.

## Installation

This project requires Python 3.13+.

```bash
# Install dependencies
uv sync
```

For development:

```bash
uv sync --extra dev
```

## Usage

1. Configure your environment:
   Create a `.env` file with:

   ```bash
   GEMINI_API_KEYS=your_key_1,your_key_2
   GEMINI_MODEL=gemma-4-31b-it
   GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
   GEMINI_FALLBACK_MODEL=gemini-2.5-flash-lite
   LOG_LEVEL=info
   ```

2. Run the TUI:

   ```bash
   uv run supporter
   ```

## Testing

Tests are organized into three layers:
- `tests/unit/`: Individual component tests (load balancer, mode manager, bash, logger)
- `tests/integration/`: Multi-component tests (search, indexing)
- `tests/e2e/`: End-to-end tests (file operations)

Run all tests or by marker:

```bash
uv run pytest tests                 # all tests
uv run pytest tests -m unit         # unit tests only
uv run pytest tests -m integration   # integration tests only
uv run pytest tests -m e2e           # e2e tests only
```

## Linting

```bash
uv run ruff check .      # Lint check
uv run ruff format .     # Format code
uv run mypy .            # Type check
```

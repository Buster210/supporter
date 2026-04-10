# Suppporter

A Python-based TUI AI chat client using the Google Gemini model.

## Features
- **Multi-Agent Execution**: CrewAI-powered multi-agent collaboration with researcher and writer roles.
- **Round-Robin Load Balancing**: Support for multiple Gemini API keys.
- **Model Fallback**: Automatically switches to a fallback model if the primary model fails.
- **TUI Interface**: A beautiful terminal user interface built with Textual.
- **Tool Support**: Integrated tool registry for agentic capabilities.

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
   GEMINI_MODEL=gemini-flash-lite-latest
   GEMINI_FALLBACK_MODEL=gemini-2.0-flash
   LOG_LEVEL=info
   ```

2. Run the TUI:
   ```bash
   supporter
   ```
   Or directly:
   ```bash
   uv run python src/supporter/tui.py
   ```

## Testing

```bash
uv run pytest tests
```

## Linting

```bash
uv run ruff check .
```

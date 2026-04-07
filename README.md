# Suppporter

A Python-based TUI AI chat client using the Google Gemini model.

## Features
- **Round-Robin Load Balancing**: Support for multiple Gemini API keys.
- **Model Fallback**: Automatically switches to a fallback model if the primary model fails.
- **TUI Interface**: A beautiful terminal user interface built with Textual.
- **Tool Support**: Integrated tool registry for agentic capabilities.

## Installation

This project requires Python 3.13+.

```bash
# Install dependencies
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

1. Configure your environment:
   Create a `.env` file with:
   ```bash
   GEMINI_API_KEYS=your_key_1,your_key_2
   GEMINI_MODEL=gemini-3.1-flash-lite-preview
   LOG_LEVEL=info
   ```

2. Run the TUI:
   ```bash
   python3 tui.py
   ```

## Testing

```bash
python3 -m pytest python_tests
```

## Linting

```bash
ruff check .
```
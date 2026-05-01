import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .types import AppConfig

__all__ = ["AppConfig", "config"]

load_dotenv()

HTTP_RATE_LIMIT = 429
HTTP_INTERNAL_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_RETRY_ATTEMPTS = 2

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

THEME = {
    "background": "#121212",
    "bubble_bg": "#1e1e1e",
    "header_teal": "#00ffcc",
    "magenta": "#ff06b5",
    "green": "#00ff00",
    "blue": "#0080ff",
    "yellow": "#ffeb3b",
    "meta_gray": "#999999",
}

CRYSTAL_GRADIENT_STOPS: list[tuple[int, int, int]] = [
    (0, 255, 255),
    (0, 255, 180),
    (0, 180, 255),
    (100, 200, 255),
]

TOOL_ARG_MAX_LEN = 40
TOOL_ARG_TRUNC_LEN = 37
MODAL_WIDTH_SCALE = 1.3
MODAL_MAX_WIDTH_PERCENT = 0.9
MODAL_PADDING = 6
BASH_MODAL_MAX_WIDTH = 80
SCROLL_STEP = 5
COLLAPSED_SUMMARY_LEN = 50
MARKDOWN_SYNTAX_MARKERS = [
    r"[*+-]\s",
    r"\d+\.\s",
    r"#+\s",
    r"\*\*.*?\*\*",
    r"\*.*?\*",
    r"`.*?`",
    r"\[.*?\]\(.*?\)",
    r">\s",
]

INTERNAL_BLACKLIST = [
    ".env",
    ".git",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
]

DEFAULT_MODEL = "gemma-4-31b-it"

RETRIABLE_CODES = {
    "1000",
    "1006",
    "1011",
    "1007",
    "1008",
    "429",
    "quota",
    "exhausted",
}

GOOGLE_5XX_ERRORS = {
    "InternalServerError",
    "ServiceUnavailable",
    "BadGateway",
    "GatewayTimeout",
    "APIError",
}

TRANSIENT_SIGNALS = {"unavailable", "overloaded", "internal error", "service level"}
HTTP_ERRORS_5XX = {HTTP_SERVICE_UNAVAILABLE, HTTP_INTERNAL_ERROR}
RATE_LIMIT_SIGNALS = {"quota", "too many requests", "429"}

DRAIN_TIMEOUT = 2.0
CONTEXT_TRIGGER_TOKENS = 100_000
CONTEXT_TARGET_TOKENS = 4_000

DELEGATE_MAX_HARD_CAP = 5
DELEGATE_DEFAULT_PARALLEL = 3
DELEGATE_DEFAULT_TIMEOUT = 180
DELEGATE_MAX_TIMEOUT = 600
DELEGATE_MAX_TASKS = 10
DELEGATE_MAX_OUTPUT_CHARS = 10000
DELEGATE_ALLOWED_TOOLS = {"read_file", "write_file", "execute_bash", "google_search"}
DELEGATE_MAX_RETRIES = 2

DELEGATE_HEARTBEAT_INTERVAL = 30
DELEGATE_ANOMALY_THRESHOLD = 0.8
DELEGATE_JOB_ID_LEN = 8
DELEGATE_RETRY_BACKOFF = [1.0, 3.0]

DELEGATE_DEFAULT_PERSONA = (
    "You are a focused task executor. You have been delegated a specific sub-task. "
    "Execute it precisely and completely. Report your findings and actions clearly. "
    "Do not ask clarifying questions -- work with what you have been given. If you "
    "encounter an error, report it and any partial progress. Be concise but thorough."
)

DELEGATE_AGENT_ROSTER: dict[str, dict[str, Any]] = {
    "security_auditor": {
        "persona": (
            "You are a Senior Security Auditor. Focus exclusively on: "
            "injection vulnerabilities, path traversal, privilege escalation, "
            "and resource leaks. Flag severity as CRITICAL/HIGH/MEDIUM/LOW. "
            "Cite exact line numbers. No false positives."
        ),
        "tools": {"read_file", "execute_bash"},
        "model": None,
    },
    "test_engineer": {
        "persona": (
            "You are a Test Engineer. Write or run tests. Report pass/fail "
            "with exact error output. Suggest fixes for failures. "
            "Never modify production code."
        ),
        "tools": {"read_file", "execute_bash"},
        "model": None,
    },
    "code_writer": {
        "persona": (
            "You are an Implementation Engineer. Write clean, production-ready "
            "code following existing project conventions. Include docstrings. "
            "Validate your changes compile before reporting."
        ),
        "tools": {"read_file", "write_file", "execute_bash"},
        "model": None,
    },
    "researcher": {
        "persona": (
            "You are a Research Analyst. Search for information, read docs, "
            "and synthesize findings into concise, actionable summaries. "
            "Always cite sources."
        ),
        "tools": {"read_file", "google_search"},
        "model": None,
    },
    "code_reviewer": {
        "persona": (
            "You are a Senior Code Reviewer. Analyze code for correctness, "
            "readability, maintainability, and adherence to project conventions. "
            "Provide specific, actionable feedback with line references."
        ),
        "tools": {"read_file"},
        "model": None,
    },
    "scout": {
        "persona": (
            "You are a Reconnaissance Scout. Your sole purpose is to read files and "
            "provide a highly token-efficient 'map' to other agents. When given a file "
            "and an intended action (e.g., 'fix bug', 'add feature'), you must: "
            "1) Identify the total line count. 2) Map the key structures (classes, "
            "functions, imports). 3) Extract ONLY the specific lines or code blocks "
            "relevant to the action. Never return the whole file. Your output must "
            "be a dense summary designed to minimize token usage for the next agent."
        ),
        "tools": {"read_file", "execute_bash"},
        "model": "gemini-3.1-flash-lite-preview",
    },
}

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an elite technical strategist and principal software architect "
    "with the ability to orchestrate parallel sub-agents via the "
    "delegate_tasks tool. This root directory is your configuration; you are "
    "authorized to self-improve via surgical edits. Consult AGENTS.md and "
    "README.md for protocols before modifying.\n\n"
    "## Core Identity\n"
    "You are the ORCHESTRATOR. Your primary job is to understand intent, plan, "
    "and route work. You THINK first, then decide: do it yourself OR delegate.\n\n"
    "## THE GOLDEN RULE: Always Delegate\n"
    "Count the independent steps needed to fulfill the request:\n\n"
    "**DELEGATE immediately:**\n"
    "- Every tasks should be delegated even if its one step.\n\n"
    "- Read multiple files -> delegate parallel reads\n"
    "- Analyze then fix -> delegate with depends_on\n"
    "- Edit across multiple files -> delegate to code_writer(s)\n"
    "- Research + implement -> delegate researcher then code_writer\n"
    "- Run tests across modules -> delegate parallel test runs\n"
    "- Any combination of read/write/bash across different targets\n\n"
    "CRITICAL: If you catch yourself about to make a SECOND tool call, STOP. "
    "You should have delegated instead. The orchestrator's hands touch tools "
    "only for true single-step work. Everything else goes to sub-agents.\n\n"
    "## MANDATORY DELEGATION WORKFLOW\n"
    "Delegation ALWAYS follows these steps in sequence:\n\n"
    "STEP 1: Call delegate_tasks(milestone, tasks, max_parallel)\n"
    "  -> Returns INSTANTLY with a plan summary and a job_id.\n"
    "  -> Sub-agents are now running in the background.\n"
    "  -> DO NOT call check_delegation immediately; wait for progress updates.\n\n"
    "STEP 2: Narrate to the user BEFORE collecting. Use this format:\n"
    "  ---\n"
    "  Delegating **[milestone]** to [N] sub-agent(s):\n"
    "  | # | Agent | Task |\n"
    "  | - | ----- | ---- |\n"
    "  | 1 | [role] | [summary] |\n"
    "  [parallel/sequential explanation]\n"
    "  Waiting for them to complete...\n"
    "  ---\n\n"
    "STEP 3: Call collect_delegation(job_id=<id from step 1>)\n"
    "  -> Blocks until all sub-agents finish.\n"
    "  -> Returns the full milestone report.\n\n"
    "STEP 4: Synthesize the report and respond to the user.\n\n"
    "The narration in Step 2 is MANDATORY. The user must see what agents were "
    "dispatched BEFORE the collect call blocks.\n\n"
    "## How to Delegate Effectively\n"
    "1. DECOMPOSE: Break the request into independent sub-tasks.\n"
    "2. IDENTIFY DEPENDENCIES: Use depends_on for sequential chains "
    "(e.g., analyze -> fix -> test).\n"
    "3. SELECT AGENTS from the roster:\n"
    "   - security_auditor: vulnerability analysis, injection risks\n"
    "   - test_engineer: writing/running tests, reporting failures\n"
    "   - code_writer: implementing features, production code\n"
    "   - researcher: searching for information, reading docs\n"
    "   - code_reviewer: code quality, conventions, readability\n"
    "   - custom: novel tasks -- provide a specific persona\n"
    "4. CRAFT SELF-CONTAINED TASKS: Sub-agents have NO conversation history. "
    "Include all file paths, context, and requirements in the task description.\n"
    "5. SCOPE TOOLS: Grant only what each agent needs. "
    "A researcher never needs write_file. A reviewer never needs execute_bash.\n"
    "6. SET TIMEOUTS: Complex multi-step work up to 600s, simple reads ~60s.\n\n"
    "## After Delegation\n"
    "When you receive the milestone report:\n"
    "- REVIEW each sub-agent's output critically\n"
    "- SYNTHESIZE findings into a coherent response for the user\n"
    "- IDENTIFY gaps or errors and fix yourself (if 1-step) or delegate follow-up\n"
    "- Never dump raw sub-agent output without synthesis\n\n"
    "## CRITICAL: NEVER DO TASKS YOURSELF\n"
    "You are the ORCHESTRATOR, not a worker. Your hands NEVER touch tools. "
    "No matter how trivial the task — a single read, a one-liner edit, a simple "
    "command — you MUST delegate it to a sub-agent. If you find yourself about to "
    "call ANY tool, STOP and delegate instead. The only thing you do is: plan, "
    "decompose, delegate, synthesize. Period.\n\n"
    "## Technical Excellence\n"
    "Analyze complex problems through the lens of scalability, maintainability, "
    "and efficiency. Anticipate edge cases and performance bottlenecks. "
    "Provide rigorous, architecturally sound guidance."
)


def _get_project_root() -> str:
    current = Path(__file__).resolve().parent
    for parent in [current, *list(current.parents)]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return str(parent)
    return os.getcwd()


def load_config() -> AppConfig:
    raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    project_root = _get_project_root()

    return AppConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        provider=os.getenv("LLM_PROVIDER", "gemini"),
        gemini_api_keys=keys,
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        gemini_live_model=os.getenv(
            "GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview"
        ),
        gemini_live_fallback_model=os.getenv(
            "GEMINI_LIVE_FALLBACK_MODEL",
            "gemini-2.5-flash-native-audio-preview-12-2025",
        ),
        gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL"),
        log_file=os.getenv("LOG_FILE", "app.log"),
        voice_name=os.getenv("GEMINI_VOICE_NAME", "Puck"),
        default_system_instruction=os.getenv(
            "DEFAULT_SYSTEM_INSTRUCTION", DEFAULT_SYSTEM_INSTRUCTION
        ),
        allowed_directories=[project_root],
        require_write_confirmation=os.getenv(
            "REQUIRE_WRITE_CONFIRMATION", "true"
        ).lower()
        == "true",
        live_thinking_level=os.getenv("GEMINI_LIVE_THINKING_LEVEL", "high").lower(),
        retriable_codes=RETRIABLE_CODES,
        google_5xx_errors=GOOGLE_5XX_ERRORS,
        transient_signals=TRANSIENT_SIGNALS,
        http_errors_5xx=HTTP_ERRORS_5XX,
        rate_limit_signals=RATE_LIMIT_SIGNALS,
        drain_timeout=DRAIN_TIMEOUT,
        context_trigger_tokens=CONTEXT_TRIGGER_TOKENS,
        context_target_tokens=CONTEXT_TARGET_TOKENS,
        http_retry_attempts=int(
            os.getenv("HTTP_RETRY_ATTEMPTS", str(HTTP_RETRY_ATTEMPTS))
        ),
        delegate_max_hard_cap=DELEGATE_MAX_HARD_CAP,
        delegate_default_parallel=DELEGATE_DEFAULT_PARALLEL,
        delegate_default_timeout=DELEGATE_DEFAULT_TIMEOUT,
        delegate_max_timeout=DELEGATE_MAX_TIMEOUT,
        delegate_max_tasks=DELEGATE_MAX_TASKS,
        delegate_max_output_chars=DELEGATE_MAX_OUTPUT_CHARS,
        delegate_allowed_tools=DELEGATE_ALLOWED_TOOLS,
        delegate_default_persona=DELEGATE_DEFAULT_PERSONA,
        delegate_agent_roster=DELEGATE_AGENT_ROSTER,
        delegate_max_retries=int(
            os.getenv("DELEGATE_MAX_RETRIES", str(DELEGATE_MAX_RETRIES))
        ),
    )


config = load_config()

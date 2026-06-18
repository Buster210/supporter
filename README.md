# Supporter

**Supporter** is a terminal-UI AI chat client powered by Google Gemini. It runs as an
*orchestrator* — an "elite technical strategist and principal software architect" persona
that converses with you, decides whether to answer directly or delegate, and drives a fleet
of specialized sub-agents to execute multi-step engineering tasks. Every shell command runs
inside a mandatory OS sandbox, every file write is path-constrained to the project root, and
the agent can drive a stealth, human-like browser when a task needs the web.

- **Package:** `supporter` `0.2.0` · **License:** Apache-2.0
- **Python:** 3.14+ (pinned via `.python-version`)
- **Entry point:** `supporter = "supporter.tui:main"` → `uv run supporter`

---

## Table of contents

- [Supporter](#supporter)
  - [Table of contents](#table-of-contents)
  - [What problem it solves](#what-problem-it-solves)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation / setup](#installation--setup)
  - [Usage / quickstart](#usage--quickstart)
  - [Configuration](#configuration)
  - [How it works / architecture](#how-it-works--architecture)
    - [Request flow](#request-flow)
    - [Package layout](#package-layout)
    - [Core runtime](#core-runtime)
    - [Providers \& connection pooling](#providers--connection-pooling)
    - [Tool subsystem](#tool-subsystem)
      - [Bash — sandboxed shell](#bash--sandboxed-shell)
      - [File ops \& search](#file-ops--search)
      - [Browser — stealth automation](#browser--stealth-automation)
      - [Delegate — sub-agent orchestration](#delegate--sub-agent-orchestration)
    - [TUI](#tui)
    - [Agent roster](#agent-roster)
    - [Reliability \& self-healing](#reliability--self-healing)
    - [Security model](#security-model)
  - [Development](#development)
  - [License](#license)

---

## What problem it solves

Driving an LLM through real engineering work from a terminal usually means stitching together
several rough edges yourself: a single conversation that drowns in its own history, one API
key that rate-limits mid-task, a model that can either *talk* or *act* but not both, shell and
file access that is one typo away from wrecking your machine, and web research that hallucinates
because the model never actually loaded the page. Doing several independent sub-tasks at once —
and *trusting* the results — is harder still.

Supporter packages all of that into one tool. It gives you:

- **One orchestrator, many hands.** The agent decides per request whether to answer directly or
  fan work out to specialized sub-agents running concurrently as a dependency DAG — and only
  accepts their coding output after an automated two-tier QA gate (objective tests + a unanimous
  three-reviewer panel).
- **Safety by construction.** Every shell command runs inside a mandatory OS sandbox with a
  layered allow/confirm/block policy; every file touch is pinned to the project root; secrets
  are redacted from output. You can hand it real power without holding your breath.
- **Web research that actually browses.** A stealth, human-like Playwright driver loads and
  reads pages (and records/replays known workflows) instead of guessing from training data.
- **It keeps running.** Multi-key health-aware rotation, model fallback on quota/5xx, persistent
  live sessions with auto-reconnect, and crash-safe durable history mean a long session survives
  the failures that normally kill it.

In short: it turns "chat with a model" into "delegate a milestone to a supervised, sandboxed,
self-healing engineering team" — from your terminal.

---

## Features

- **Multi-agent orchestration** — `delegate_tasks` launches a background DAG of role-specialized
  sub-agents (explorer, security auditor, test engineer, code reviewer, page-pilot) with
  dependency resolution, per-task timeouts, retries, and live progress in the UI.
- **Two-tier QA gate** — completed coding tasks must pass objective tier-1 checks (auto-detected
  ruff/mypy/pytest or `package.json` scripts, real exit codes) *and* a unanimous tier-2 panel of
  three Gemini reviewers, with automatic correction rounds.
- **Mandatory sandboxed shell** — `execute_bash` is gated by a stateless layered policy (block /
  confirm / allow tiers, AST inspection of inline interpreter payloads) and wrapped in
  `sandbox-exec` (macOS) or `nsjail` (Linux); no sandbox means no bash.
- **Path-guarded file ops** — `read_file` / `write_file` constrained to the project root, honoring
  `.gitignore` + an internal blacklist, with optional diff-confirmation on writes.
- **Stealth browser automation** — Patchright-based `browse` with human-like input (Bézier mouse,
  typo-and-correct typing, fatigue/idle behaviors), ad/tracker blocking, Cloudflare Turnstile
  solving, accessibility-tree snapshots, and a record/replay **playbook** system.
- **Grounded web fallback** — `google_search` returns an explicitly-unverified grounded answer when
  full browsing isn't warranted.
- **Reliability layer** — health-aware multi-key rotation with cooldowns, dynamic model fallback,
  persistent live sessions with idle-reconnect and pre-warm, and self-healing metrics.
- **Durable, compacting history** — crash-safe JSONL persistence with session rotation and
  LLM-based context compaction; interrupted delegation jobs resume on restart.
- **Working memory + recipes** — crash-safe JSONL stores for cross-session working state
  and parameterised, LLM-free replayable automations; the assistant reads the recent
  working-memory block on every turn and writes recipes when it solves a multi-step
  problem so it can replay them next time.
- **AutoRecover + Verification** — a unifying catch-and-heal wrapper for any async call
  site (rotates the keypool on transient 5xx) and a verification loop that retries a
  generation against pure-Python checks (length, garble, JSON shape, files-exist,
  recipe-passes) before persisting the result.
- **Reactive TUI** — Textual dashboard with streamed bubbles, thought streaming, tool-call
  rendering, confirmation modals, and live delegation progress.
- **Audit trail** — every tool-routing decision, recovery attempt, and delegation capsule
  is persisted for inspection.

---

## Prerequisites

- **Python 3.14+** (pinned in `.python-version`).
- **[`uv`](https://docs.astral.sh/uv/)** for dependency management and running.
- **One or more Google Gemini API keys** (`GEMINI_API_KEYS`).
- **An OS sandbox** for the bash tool — `sandbox-exec` (built into macOS) or `nsjail` (Linux).
  Without one, bash is automatically disabled; the rest of the app still runs.
- **Optional:** a Chromium install for the `browse` tool (provided via `patchright`), and the
  `opencode` CLI if you want the opencode delegation backend.

Runtime dependencies (installed by `uv sync`): `google-genai` (REST + Live API), `textual` (TUI),
`rich` (rendering), `python-dotenv` (`.env` loading), `patchright` (stealth-patched Playwright fork).

---

## Installation / setup

```bash
git clone <repo-url> supporter
cd supporter
uv sync                 # install runtime + dev dependencies
```

Then create a `.env` in the project root (see [Configuration](#configuration)) with at least your
`GEMINI_API_KEYS`.

---

## Usage / quickstart

Launch the TUI:

```bash
uv run supporter
```

Type to chat. The orchestrator answers directly or delegates; tool calls that mutate state (risky
bash, file writes, browser `eval`/upload/download) surface a confirmation modal. Slash commands:
`/live`, `/agent`, `/clear`, `/exit`.

Helper scripts:

```bash
uv run python scripts/live_smoke.py    # end-to-end smoke test of GeminiLiveProvider
uv run python scripts/live_probe.py    # probe the Live API connection
uv run python scripts/browser_drive.py # drive the browser via the page-pilot agent
```

---

## Configuration

All configuration is environment-driven (loaded from `.env`). A minimal `.env`:

```bash
GEMINI_API_KEYS=your_key_1,your_key_2          # comma-separated or JSON array
GEMINI_MODEL=gemma-4-31b-it                     # REST model (orchestrator / sub-agents)
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview # primary live WebSocket model
GEMINI_LIVE_FALLBACK_MODEL=gemini-2.5-flash-native-audio-preview-12-2025
LOG_LEVEL=info
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEYS` / `GEMINI_API_KEY` | *required* | One or more API keys (comma-separated or JSON array) |
| `GEMINI_MODEL` | `gemma-4-31b-it` | REST model for orchestrator and sub-agents |
| `GEMINI_LIVE_MODEL` | `gemini-3.1-flash-live-preview` | Primary Live (WebSocket) model |
| `GEMINI_LIVE_FALLBACK_MODEL` | `gemini-2.5-flash-native-audio-preview-12-2025` | Live fallback model |
| `GEMINI_FALLBACK_MODEL` | *(none)* | REST fallback model |
| `LLM_PROVIDER` | `gemini` | Provider type (only `gemini` supported) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LOG_FILE` | `app.log` | Log file path |
| `GEMINI_VOICE_NAME` | `Puck` | Voice for Live API audio |
| `GEMINI_LIVE_THINKING_LEVEL` | `high` | Live thinking budget (`low`/`medium`/`high`) |
| `RECONNECT_BACKOFF_BASE` / `RECONNECT_BACKOFF_CAP` | `0.5` / `8.0` | Live reconnect backoff bounds (seconds) |
| `PREWARM_SAFETY_MARGIN` | `5.0` | Seconds before idle timeout to pre-warm next session |
| `BROWSER_PROFILE_PATH` / `BROWSER_PROFILE_NAME` | *(none)* | Persistent browser profile dir / name |
| `BROWSER_ACTIONS_PER_MIN` | `36` | Browser action rate cap |
| `OPENCODE_BIN` / `OPENCODE_MODEL` | `~/.opencode/bin/opencode` | `opencode` CLI backend path / model |
| `EMPTY_RESUME_POLICY` | `trust` | How to treat empty history on session resume |
| `SUPPORTER_SESSION_ID` | *(auto UUID)* | Override the durable-history session ID |

Additional `AppConfig` flags (env-driven) tune durable history, history compaction, write
confirmation, and delegation behavior (`DELEGATE_*` keys for repair rounds, confidence floor,
heartbeat interval, etc.). Defaults live in `src/supporter/config.py`.

---

## How it works / architecture

### Request flow

```
User input
    │
    ▼
TUI (Textual)  ── src/supporter/tui/
    │   streams chunks, renders bubbles, surfaces confirmations & delegation progress
    ▼
ChatAgent.execute_stream()  ── src/supporter/agent.py
    │   compacts history, assembles LLMOptions, logs each routing decision
    ▼
LLMProvider  ── src/supporter/providers/  +  pool.py
    ├── REST path:  LazyFallbackProvider → DynamicPool → GeminiProvider  (key rotation, cooldown)
    └── Live path:  LazyFallbackProvider → GeminiLiveProvider            (persistent WebSocket)
    │
    ▼
Tool dispatch (resolved from the catalog)
    ├── execute_bash      → sandboxed shell (sandbox-exec / nsjail)
    ├── read_file / write_file → path-validated, project-root-only
    ├── google_search     → grounded fallback web answer
    ├── browse            → stealth Playwright automation (page-pilot)
    └── delegate_tasks    → DAG of background sub-agents → capsule result
    │
    ▼
Result streamed back to the TUI
```

The orchestrator "thinks first, then decides: do it itself or delegate." Web research is
preferentially routed to the `browse` tool (and the `page-pilot` sub-agent); `google_search`
is a quick, explicitly-labeled *unverified* fallback.

### Package layout

```
src/supporter/
├── __init__.py
├── agent.py               # ChatAgent — the orchestrator loop
├── config.py              # AppConfig singleton + policy constants
├── prompts.py             # model names, system prompts, agent roster
├── types.py               # shared dataclasses, Protocols, TypedDicts, events
├── session.py             # HistoryStore — durable JSONL turn persistence
├── pool.py                # DynamicPool, LazyFallbackProvider, get_provider()
├── history_summarizer.py  # LLM-based context compaction
├── decision_log.py        # per-turn tool-routing audit trail
├── recovery_metrics.py    # self-healing counters
├── logger.py              # async "flight recorder" logging
├── providers/
│   ├── gemini_provider.py      # REST (stateless) provider
│   └── gemini_live_provider.py # Live (persistent WebSocket) provider
├── tools/
│   ├── base.py            # ToolError
│   ├── catalog.py         # ToolSpec registry + role-gated selection
│   ├── resolver.py        # assemble Gemini tool declarations
│   ├── file_ops.py        # read_file / write_file (path-guarded)
│   ├── search.py          # google_search (grounded fallback)
│   ├── bash/              # sandboxed shell (defs, executor, policy, sandbox)
│   ├── browser/           # stealth Playwright automation + playbooks
│   └── delegate/          # sub-agent orchestration (DAG, capsules, QA gate)
└── tui/                   # Textual UI (app, bubbles, modals, modes, listener)
```

### Core runtime

| Module | Responsibility |
| --- | --- |
| `agent.py` — `ChatAgent` | The orchestrator. Holds the provider, history, and tool registry. `_build_compacted_history()` sends `[cached summary] + recent turns` instead of full history; `_prepare_execution_context()` assembles per-turn `LLMOptions`; `_record_brain_decision()` logs which tool (or `text_response`) the model chose. Loads/reloads durable history when enabled. |
| `types.py` | Every shared type: `LLMProvider` (Protocol), `LLMOptions` (TypedDict), `LLMChunk` / `LLMResult` (streaming + complete results), `AppConfig`, `TaskStatus`, the `DelegationEvent` hierarchy (`MilestoneStarted/Completed`, `TaskStarted/Completed/Failed/TimedOut/Retrying/Anomaly`, `HeartbeatTick`), and the `ModeChanged` TUI message. |
| `prompts.py` | Single source for model constants (`MODEL_GEMMA_31B`, `MODEL_GEMINI_LIVE`, `MODEL_GEMINI_LIVE_FALLBACK`), the orchestrator `DEFAULT_SYSTEM_INSTRUCTION`, the generic `DELEGATE_DEFAULT_PERSONA`, and the `DELEGATE_AGENT_ROSTER` (per-role persona + tools + model + `live` flag). |
| `config.py` | Builds the process-wide `AppConfig` singleton from env vars. Also defines error-classification constants, `INTERNAL_BLACKLIST` (paths banned from tools), and `DELEGATE_*` tuning. |
| `session.py` — `HistoryStore` | Durable, append-only JSONL turn log with `fsync` crash safety; images written to a per-session `images/` dir and stored as base64 references. `append()`, `load(limit)` (tolerates a corrupt trailing line), `rotate()` (fresh session, old data preserved). |
| `history_summarizer.py` | `render_turns()` flattens `Content` to a transcript; `summarize_turns()` asks Gemini for a compact narrative, injected by `ChatAgent` as a synthetic `[PREVIOUS_CONTEXT_SUMMARY]` turn. |
| `decision_log.py` | `log_decision(site, chosen, reason, interaction_id)` appends a `DecisionEntry` to a JSONL audit log after every generation; `recent_decisions()` reads them back. |
| `recovery_metrics.py` | Process-scoped `RecoveryCounters` — `key_rotations` and `re_snapshots_survived` — incremented by the providers and browser tool when they self-heal. |
| `logger.py` | Async queue-based logging with a `_FlightRecorderLogger` ring buffer that can dump recent records on error; `init_logger()` / `shutdown_logger()`. |

### Providers & connection pooling

- **Uniform interface** — all callers use one `LLMProvider` API and don't care how a turn is served.
- **REST** — stateless, per-request turns; used by all sub-agents.
- **Live** — a WebSocket session kept warm across turns for the orchestrator and browser pilot, with multimodal/voice support, idle-reconnect, and background pre-warming.
- **Resilience layer** — health-aware rotation across multiple API keys, model cooldowns, and transparent fallback to a backup model on quota/5xx/exhaustion.

### Tool subsystem

- **Central catalog** — tools are async functions; each declares whether sub-agents and which roles may call it.
- **Role-scoped** — the orchestrator gets the full set, each sub-agent only its role-appropriate slice.
- **Model-shaped** — selected tools are translated into the shape the active model expects at call time.
- **Confined** — every tool is restricted to the project root.

#### Bash — sandboxed shell

- **Default-deny** — each command is parsed (never passed to a shell) and scored into **allow / confirm / block**.
- **Policy checks** — rejects dangerous binaries, temp-dir execution, nuclear deletes, shell/network pipe chains, sensitive paths, and risky inline interpreter payloads.
- **Hardened execution** — argument lists, minimal env, CPU/memory limits, kill timeout, capped output, secret redaction.
- **OS sandbox** — `sandbox-exec` (macOS) / `nsjail` (Linux); **no sandbox, no bash.**

#### File ops & search

- **`read_file` / `write_file`** — confined to the project root, respect `.gitignore` + an internal blacklist; writes can require diff-confirmation.
- **`google_search`** — a lightweight, explicitly-unverified grounded fallback; `browse` is preferred for real web work.

#### Browser — stealth automation

- **`browse`** — drives real Chromium (via Patchright) behind a "look human, not bot" layer.
- **Humanized** — natural mouse/keyboard/scroll input, paced guardrails with confirmation on risky actions, ad/tracker blocking, Cloudflare Turnstile solving.
- **Perception** — reads pages via a cleaned accessibility-tree snapshot (not raw HTML); state is isolated per agent for concurrent pilots.
- **Playbooks** — records and replays known workflows by re-resolving elements against the live page.
- **Actions** — navigation, observation, interaction, JS eval, frames, tabs, storage, uploads/downloads.

#### Delegate — sub-agent orchestration

- **Supervised DAG** — a milestone becomes background sub-agents; the task list is validated and cycle-checked up front, then runs concurrently under a global cap with dependency ordering.
- **Backends** — each task runs as a role-scoped Gemini sub-agent or a headless `opencode` subprocess.
- **Observable + durable** — an event bus streams live progress to the UI; each job's full state persists as a **capsule** (status, evidence, findings, synthesized answer) that survives restarts.
- **Two-tier QA gate** — objective checks (ruff/mypy/pytest, real exit codes) plus unanimous approval from three reviewer roles, with automatic correction rounds before acceptance.

### TUI

- **Dashboard** — a full-screen Textual app streaming the conversation as collapsible bubbles (text, thoughts, tool calls).
- **Startup** — resumes crash-interrupted delegation jobs and pre-warms the live session and browser.
- **Confirmation** — mutating tool calls pause for a modal.
- **Controls** — switch **LIVE** / **SINGLE** mode and issue slash commands (`/live`, `/agent`, `/clear`, `/exit`).
- **Delegation** — progress and results surface inline and feed back into the agent's next turn.

### Agent roster

Sub-agents are role-defined — a persona with a restricted tool set and model — and run with **zero
shared conversation history**, seeing only the context handed to them.

| Role | Focus | Typical tools |
| --- | --- | --- |
| **Explorer** | Read-only reconnaissance | `read_file`, `execute_bash`, `google_search` |
| **Security Auditor** | Vulnerability / risk analysis | `read_file`, `execute_bash` |
| **Test Engineer** | TDD + verification (tier-2 QA reviewer) | `read_file`, `execute_bash` |
| **Code Reviewer** | Quality / convention enforcement (tier-2 QA reviewer) | `read_file` |
| **page-pilot** | Stealth web research / browser automation (live) | `browse` (full browser toolset) |

### Autonomous surface

The assistant has three persistence / automation surfaces beyond the conversation history,
so multi-step work survives restarts and known tasks replay without burning LLM tokens:

- **Working memory** (`memory_write` / `memory_read` / `memory_search` / `memory_list_kinds` /
  `memory_compact` / `memory_clear` / `memory_status`) — a small crash-safe JSONL store at
  `.supporter/working_memory.jsonl` for the *current session's* working state: half-finished
  tasks, recent URLs, the user's preferred working directory, fingerprints of the last
  working build. Append-only with atomic writes; the assistant auto-injects a compact
  recent-notes block into its own context.
- **Recipes** (`recipe_save` / `recipe_find` / `recipe_run` / `recipe_delete` / `recipe_list`
  / `recipe_status`) — named, parameterised lists of steps (`shell`, `read`, `write`,
  `http_get`, `memory_write`, `delay`, `assert_exists`, `assert_eq`, `emit`). When the
  orchestrator solves a multi-step problem itself, it `recipe_save`s the flow; next time
  `recipe_run(name)` replays it deterministically with **zero LLM tokens**. Recipes are
  JSONL at `.supporter/recipes.jsonl`; each step is data, not Python, so they replay safely
  without trusting the writer's intent.
- **AutoRecover watchdog** (`supporter.recover.AutoRecover`) — a unifying catch-and-heal
  wrapper around any async call site (`provider.generate`, `browser.browse`,
  `agent.execute`, `recipe.run`). Classifies exceptions, runs pluggable recovery actions
  in order (the built-in `rotate_api_key` action marks the failing key sick and lets the
  keypool pick a fresh one on next `acquire()`), and retries with the same arguments.
  `ChatAgent.execute_with_verification(recover=...)` wires it in by default, so a
  transient 5xx / network blip rotates the keypool and retries without burning the
  verification budget.

Plus a small verifier layer (`supporter.verify.VerificationLoop`) that powers
`ChatAgent.execute_with_verification`: run → pure-Python checks (length, garble, JSON
shape, files-exist, recipe-passes) → re-attempt with the failure context if any check
fails, bounded by `max_attempts`. The final result is what gets persisted to history;
intermediate attempts stay off the user-visible transcript.

### Reliability & self-healing

Built to survive the failures that normally end a long session:

- **Provider errors** — absorbed by key rotation, cooldowns, and model fallback.
- **Dropped live connections** — reconnect and replay history.
- **Crashes** — don't lose work, via durable history + resumable delegation jobs.
- **Browser stale-reference flakiness** — self-heals via re-snapshotting.
- **Observability** — key rotations and recoveries are counted; every tool-routing decision is logged.

### Security model

Safety is enforced structurally, not by trusting the model:

- **Mandatory OS sandbox** for all shell — no sandbox means no bash, full stop.
- **No shell interpreter** — commands run as argument lists, so there are no pipes or chains into
  a shell or the network.
- **Minimal environment** — only `PATH`/`TERM`/`LANG` reach a subprocess, so ambient secrets
  can't leak.
- **Tiered confirmation** — risky commands and all file writes need UI approval; nuclear/exfil patterns are hard-blocked.
- **Path containment** — all file and shell access stays inside the project root.
- **Output redaction** — credentials and API keys are stripped from tool output.
- **Sub-agent isolation** — no shared history, role-scoped tools only.

---

## Development

Layered test suite (`pytest`, `asyncio_mode=auto`, coverage gate **84%**):

```bash
uv run pytest tests            # all tests
uv run pytest tests/unit       # component-level logic
uv run pytest tests/integration # cross-component flows
uv run pytest tests/e2e        # full-lifecycle verification
```

Quality control:

```bash
uv run ruff check .   # lint (E,W,F,I,N,UP,S,B,C4,ASYNC,PIE,RSE,RET,SIM,T20,RUF; line-length 88)
uv run ruff format .  # format
uv run mypy .         # strict type checking (py3.14)
uv run bandit -r src  # security lint
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
</content>

# Architecture

## Project Structure

AndrewCLI separates **shipped code** (the Python package) from **runtime configuration** (domains, events, the global `config.yaml`). Runtime configuration lives under `~/.config/andrewcli/` and is seeded from bundled defaults the first time `andrewcli` runs, so the user can edit, add, or remove domains and events without touching the installed package.

```
~/.config/andrewcli/                # Runtime configuration (auto-seeded)
├── config.yaml                     # Global configuration
├── events/                         # Event definitions — auto-discovered, activated via /name
│   ├── timer.py
│   ├── file.py
│   ├── monitor.py
│   ├── project.py
│   ├── loop.py
│   └── schedule.py
└── domains/                        # Each domain is a self-contained config folder
    └── general/                    # — auto-discovered, no manual registration
        ├── config.yaml             # Per-domain overrides (api_base_url, model, …)
        ├── system_prompt.md        # Domain prompt
        ├── tools/                  # Tool subclasses auto-loaded from *.py
        │   └── common.py
        └── skills/                 # Skill markdown files auto-loaded as Skill instances

AndrewCLI/                          # Installed package (this repository)
├── andrewcli.py                    # Unified entry point (CLI, tray, server)
├── pyproject.toml
└── src/
    ├── defaults/                   # Bundled defaults — copied to ~/.config/andrewcli/ on first run
    │   ├── config.yaml
    │   ├── domains/...
    │   └── events/...
    ├── shared/
    │   ├── config.py               # Config class — loads ~/.config/andrewcli/config.yaml
    │   └── paths.py                # Centralized paths (PROJECT_ROOT, CONFIG_DIR, LAUNCH_DIR, DATA_DIR, …)
    ├── core/
    │   ├── andrew.py               # AndrewCore — shared event log, dispatch, slash commands (CLI + tray)
    │   ├── server.py               # FastAPI middleware + shared bridge (inbox queue, session store)
    │   ├── domain.py               # Domain class (async generator, event bus, busy_lock)
    │   ├── event.py                # Event ABC + EventBus (add, remove, running, stop)
    │   ├── llm.py                  # Async LLM client with streaming + tool-calling loop
    │   ├── memory.py               # Rolling memory with background summarization
    │   ├── registry.py             # Unified auto-discovery: domains, events (+ /slash parsing), tools, skills
    │   ├── router.py               # ToolRouter — LLM-based tool/skill router
    │   ├── skill.py                # Base Skill class (markdown-defined tools)
    │   └── tool.py                 # Base Tool class — auto-generates OpenAI schemas from type hints
    ├── cli/                        # CLI surface
    │   ├── app.py                  # AndrewCLI — subclasses AndrewCore, adds terminal I/O
    │   ├── animations.py           # Spinner (async, dynamic status)
    │   ├── filter.py               # ThinkFilter — parses <think> tags for reasoning display
    │   └── renderer.py             # StreamRenderer — spinner + filtering + typewriter streaming + ESC-to-stop
    └── tray/                       # System tray GUI (PyQt6)
        ├── __main__.py
        ├── bootstrap.py            # Pre-Qt init (sets QT_QPA_PLATFORM from config)
        ├── app.py                  # Thin shell — constructs TrayController and QApplication
        ├── controller.py           # TrayController — holds AndrewCore, domain loading, worker lifecycle
        ├── worker.py               # StreamWorker QThread — runs domain.generate() on shared async loop
        ├── panel.py                # ChatPanel widget — input (with history), streamed output, spinner, controls
        ├── icon.py                 # Tray icon and context menu
        ├── style.css               # Qt stylesheet (Catppuccin Mocha)
        └── md.css                  # CSS for markdown rendering in QTextBrowser
```

---

## Config

A centralized `Config` class (`src/shared/config.py`) loads `~/.config/andrewcli/config.yaml` and exposes settings as attributes. Used by `app.py` to select the active domain, and by the tray app for window dimensions, position, opacity, and platform backend.

On first import, `src/shared/paths.py` seeds `~/.config/andrewcli/` from the package's bundled defaults (`src/defaults/`) and inserts that directory at the front of `sys.path`, so `domains.<name>` and `events.<name>` resolve as regular Python packages no matter where AndrewCLI is installed.

---

## Domains

A **Domain** is a folder under `~/.config/andrewcli/domains/<name>/` that groups a system prompt, tools, and skills into a single persona. There is no per-domain Python class — every domain is pure configuration:

- **Settings** — loaded from `domains/<name>/config.yaml`. Any key set there overrides the matching key from the global `config.yaml` while this domain is active; missing keys fall through to the global value, then to the class-level default. Recognized keys: `api_base_url`, `model`, `routing_enabled`.
- **System prompt** — loaded from `domains/<name>/system_prompt.md`. Plain markdown, no frontmatter.
- **Tools** — auto-discovered from `domains/<name>/tools/*.py` (every concrete `Tool` subclass is instantiated and registered).
- **Skills** — auto-discovered from `domains/<name>/skills/*.md`.

All four are **reloaded before every user turn** via `Domain.reload()`. Tool modules are re-imported with `importlib.reload`, skill files are re-scanned from disk, the system prompt and `config.yaml` are re-read. Flipping `routing_enabled` or pointing the domain at a new endpoint takes effect immediately — no restart required. If `api_base_url` or `model` changes, a new LLM client is created and the existing conversation memory is transplanted into it. Memory and the event bus are never touched by a reload.

The active domain is chosen from the global `config.yaml` (`domain: "general"`) and can be **switched at runtime** with TAB. Domains can optionally override the global LLM endpoint and model per-domain.

Each domain owns an `EventBus` instance that starts empty — events are independent of domains and are added at runtime through slash commands.

---

## Tools

A **Tool** is a Python class the LLM can call. Tools auto-generate their OpenAI function schema from `execute()`'s type hints — no manual schema boilerplate. The base `Tool.run()` wrapper catches exceptions and returns a `[Tool Error]` string so the agent can recover without crashing.

**Built-in tools** (`~/.config/andrewcli/domains/general/tools/`): `WriteFile`, `ReadFile`, `ExecuteCommand`, `GetCurrentDate`. Every shell command is spawned with `cwd=LAUNCH_DIR` so `execute_command` always targets the directory you launched `andrewcli` from.

---

## Skills

A **Skill** is a markdown-defined tool. Instead of executing code, it returns natural-language instructions the LLM follows using the available tools. Skill files live inside each domain's `skills/` folder and are auto-loaded as `Skill` instances — no Python subclass needed. Each file starts with a YAML frontmatter block:

```markdown
---
name: my_skill
description: What this skill does
tools: [tool_name_1, tool_name_2]
---

# Instructions
1. Step one using the available tools
2. Step two using the available tools
```

The optional `tools:` field lists tools the skill requires — they are injected into the generation call even if the router didn't select them.

When the LLM invokes a skill, its body is **promoted into the system prompt** as a turn-scoped `<skill:NAME>...</skill:NAME>` block. The tool-call response is just a short acknowledgement pointing the model at the new system instructions. This puts skill steps at the same authority tier as the domain's base system prompt — far stickier than embedding them in a `role: tool` message, where local models tend to summarize or skip steps. The block is cleared in a `try/finally` at turn end so it never leaks into the next turn's routing or context.

---

## CLI Layer

- **`AndrewCLI`** (`cli/app.py`) — subclasses `AndrewCore`. Adds terminal I/O: `_read_input` (cbreak mode, TAB domain switch, UP/DOWN history, ESC-to-stop), `_bg_print` (atomically clears prompt, prints text, restores prompt), and the main async `run()` loop.
- **`Spinner`** (`cli/animations.py`) — async spinner with a dynamic `.status` property. Shows what the agent is doing: `⠴ Thinking...`, `⠧ Running execute_command: ls -la`, `⠋ Loading: write_file, read_file`.
- **`ThinkFilter`** (`cli/filter.py`) — streaming parser for `<think>...</think>` tags. Handles tags split across token boundaries. Renders reasoning in dim italic while keeping the final answer in normal text.
- **`StreamRenderer`** (`cli/renderer.py`) — orchestrates the full output pipeline: spinner lifecycle, `RouteEvent` and `ToolEvent` processing, think filtering, typewriter-effect streaming, ESC-to-stop.

## AndrewCore

`AndrewCore` (`core/andrew.py`) is the shared logic layer used by both the CLI and the tray. It holds the event output log (`_event_log`, `_event_tool_log`) and live-generation buffers (`_event_live`, `_event_live_tools`), implements `_event_dispatch` (token collection, tool tracking, log update), and exposes `handle_slash` for `/events`, `/stop`, and `/status`.

Surfaces override three hooks:

| Hook | CLI | Tray |
|---|---|---|
| `_on_event_token(iid, token)` | no-op (prints atomically at end) | puts token into Qt queue (live streaming) |
| `_on_event_output(iid, desc, response)` | `_bg_print` banner + response | no-op (already streamed) |
| `_on_event_done(iid)` | no-op | puts `None` sentinel into Qt queue |

`/status [id]` shows: iteration count, per-iteration tool calls with arguments and result previews, and a live `[generating]` block with partial text and in-progress tool calls while generation is running.

---

## Tray App

A **PyQt6 system tray application** that uses the same domain classes and async logic as the CLI. `app.py` is a thin shell that constructs the `QApplication` and delegates all orchestration to `TrayController` (`controller.py`).

- **`controller.py`** — holds an `AndrewCore` instance that handles all event logging and slash commands (`/events`, `/stop`, `/status`). Manages domain loading and `StreamWorker` lifecycle. Wires `AndrewCore`'s hooks so event tokens are queued for the Qt main thread.
- **`worker.py`** — `StreamWorker` QThread runs `domain.generate()` on a shared asyncio event loop via `asyncio.run_coroutine_threadsafe`. Emits `token_received`, `tool_status`, `finished`, and `error` signals. Cancellation cancels the asyncio future and sets a flag checked during streaming.
- **`panel.py`** — `ChatPanel` with a `QLineEdit` input, `QTextBrowser` for streamed markdown output, and header controls. UP/DOWN arrow keys navigate input history. TAB cycles domains. Braille spinner driven by `QTimer` shows tool names during execution and routing. Conversation history is persisted to `$LAUNCH_DIR/.andrewcli/data/conversation.md` and restored on next launch.
- **Event bridge** — the `EventBus` runs on the shared asyncio loop. A `queue.SimpleQueue` bridges it to the Qt main thread. A `QTimer` polling at 100 ms drains the queue, shows balloon notifications via `QSystemTrayIcon.showMessage`, and routes tokens to the panel — all without blocking the Qt or asyncio loops.
- Submitting a new message while generating **cancels the previous generation** and waits before starting a new one.

---

## Async Pipeline

The entire I/O pipeline is non-blocking:

1. **`AndrewCLI`** (`cli/app.py`) — runs under `asyncio.run()`. Input read via a custom async `_read_input()` using cbreak mode, supporting TAB domain switch and UP/DOWN history.
2. **Router** — async LLM-as-classifier call that returns the minimal tool/skill set for the prompt.
3. **Spinner** — `asyncio` task that animates and updates status text from `RouteEvent` and `ToolEvent`s.
4. **Streaming** — `LLM.generate()` is an async generator. Tokens are yielded as they arrive. `ToolEvent` objects are also yielded to update the spinner.
5. **Tool calls** — accumulated from streamed chunks, executed via `tool.run()`, looped back automatically. Malformed JSON arguments are caught and reported instead of crashing.
6. **Memory summarization** — fires in the background after the response completes; no user-facing delay. Skipped entirely for short turns.
7. **Event bus** — runs as a set of concurrent asyncio tasks alongside the main loop. Each event waits on its own `condition()` coroutine independently, then acquires `Domain.busy_lock` to dispatch (FIFO across events and user turns).

---

## Server

The FastAPI server (`src/core/server.py`) is a **thin middleware**: it does not call the LLM directly. `POST /chat` enqueues the message into a shared bridge inbox and returns a `session_id`; the CLI/tray picks it up within 100 ms and processes it through the normal submit path — the same code path as a typed message, so slash commands, events, and the full domain pipeline all work identically over HTTP. Response tokens are accumulated in a per-session store and returned via `GET /chat/{session_id}` (tokens are consumed on each call; poll until `done: true`).

The CLI and tray automatically start the FastAPI server in a background thread on launch (configurable via `server.enabled` in `config.yaml`). `andrewcli --server` starts the server standalone without a UI.

All three surfaces (CLI, system tray, HTTP API) share the same core: same domains, same memory, same router, and the same `Domain.busy_lock` that serializes user turns and event dispatches identically on every surface.

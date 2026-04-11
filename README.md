# AndrewCLI

A lightweight, fully async Python agent — designed to **keep your context clean**.

Local models degrade fast as context grows: reasoning gets muddier, tool calls go off the rails, and token budgets hit the ceiling. AndrewCLI fights this with two mechanisms that run on every turn:

- **Rolling memory** — after each response, messages are trimmed to just the last exchange and replaced with a compact `~300-word` summary injected into the system prompt. The model always has context, but never a bloated history.
- **Context-aware router** — before each generation, a fast LLM call selects only the tools and skills actually needed for the current request. Irrelevant schemas never reach the generation prompt.

Both modes (CLI and system tray) share the same core: same domains, same memory, same router.

---

## Project Structure

```
AndrewCLI/
├── andrewcli.py                    # Unified entry point (CLI, tray, server)
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
└── src/
    ├── shared/
    │   └── config.py               # Config class — loads config.yaml
    ├── core/
    │   ├── domain.py               # Base Domain class (async generator, event bus)
    │   ├── event.py                # Event ABC + EventBus
    │   ├── llm.py                  # Async LLM client with streaming + tool-calling loop
    │   ├── memory.py               # Rolling memory with background summarization
    │   ├── router.py               # ToolRouter — selects only needed tools/skills per request
    │   ├── skill.py                # Base Skill class (markdown-defined tools)
    │   └── tool.py                 # Base Tool class — auto-generates OpenAI schemas from type hints
    ├── events/                     # Event definitions (auto-loaded by domain)
    │   ├── timer.py                # TimerEvent — fires on a fixed interval
    │   └── file.py                 # FileEvent — fires when a watched file is modified
    ├── ui/                         # CLI rendering layer
    │   ├── animations.py           # Spinner (async, dynamic status)
    │   ├── filter.py               # ThinkFilter — parses <think> tags for reasoning display
    │   └── renderer.py             # StreamRenderer — spinner + filtering + typewriter streaming
    ├── tray/                       # System tray GUI (PyQt6)
    │   ├── __main__.py             # Entry point for `python -m src.tray`
    │   ├── bootstrap.py            # Pre-Qt init (sets QT_QPA_PLATFORM from config)
    │   ├── app.py                  # Orchestrator — domain loading, worker lifecycle, signals
    │   ├── worker.py               # StreamWorker QThread — runs domain.generate() on shared async loop
    │   ├── panel.py                # ChatPanel widget — input, streamed output, spinner, controls
    │   ├── icon.py                 # Tray icon and context menu
    │   ├── style.css               # Qt stylesheet (Catppuccin Mocha)
    │   └── md.css                  # CSS for markdown rendering in QTextBrowser
    ├── tools/
    │   └── common.py               # WriteFile, ReadFile, ExecuteCommand, GetCurrentDate
    ├── skills/
    │   ├── myskills.py             # Skill subclass definitions
    │   └── skills_files/           # Skill instruction markdown files
    └── domains/
        ├── general.py              # General-purpose domain
        ├── experimental.py         # Shell-only domain (execute_command)
        └── coding.py               # Coding-focused domain (WIP)
```

---

## How Context Stays Clean

### Rolling Memory

After each completed turn:

1. The last 1500 characters of conversation are extracted as an excerpt.
2. If no summary exists yet, the excerpt is saved as-is (capped at 2000 chars).
3. If a summary exists, it is **merged with the excerpt** via a background LLM call that produces a single `~300-word` summary — facts, decisions, code written, tools used, user preferences.
4. Messages are **trimmed to just the last exchange** (last user message + last assistant response). The summary is injected into the system prompt inside `<memory>` tags.
5. The merge runs as a **fire-and-forget `asyncio` task** — the next prompt is ready immediately. Sequential merges are serialized with a lock to prevent race conditions.

The model always has prior context, but the message history never grows beyond one exchange. This keeps the effective prompt size bounded regardless of conversation length.

**Storage:** `~/.andrewcli/data/memory.json`

### Context-Aware Router

Before each generation, `ToolRouter` runs a mini LLM call:

- Input: user prompt + memory summary + last exchange + full tool/skill catalog (names and descriptions only)
- Output: JSON array of needed tool/skill names
- The generation call receives **only the selected schemas**, not the full catalog

This keeps the tool section of the prompt proportional to the request. A "what time is it?" question won't include file or shell tool schemas. A skill that requires specific tools can declare them in its YAML frontmatter — those are injected even if the router didn't select them.

If the routing call fails or returns nothing useful, all tools pass through unchanged.

---

## Architecture

### Config

A centralized `Config` class (`src/shared/config.py`) loads `config.yaml` and exposes settings as attributes. Used by `app.py` to select the active domain, by `ExecuteCommand` to read `execute_bash_automatically`, and by the tray app for window dimensions, position, opacity, and platform backend.

### Domains

A **Domain** groups a system prompt, a set of tools, a set of skills, and a list of events into a single persona. Defined as Python classes in `src/domains/`, loaded dynamically from `config.yaml`. The `generate()` method is an async generator that yields tokens as they stream in. Domains can be **switched at runtime** with TAB.

Each domain owns an `EventBus` instance built from its `events` list. The bus is started by the app layer alongside the main loop.

### Tools

A **Tool** is a Python class the LLM can call. Tools auto-generate their OpenAI function schema from `execute()`'s type hints — no manual schema boilerplate. The base `Tool.run()` wrapper catches exceptions and returns a `[Tool Error]` string so the agent can recover without crashing.

**Built-in tools** (`src/tools/common.py`): `WriteFile`, `ReadFile`, `ExecuteCommand`, `GetCurrentDate`

### Skills

A **Skill** is a markdown-defined tool. Instead of executing code, it returns natural-language instructions that the LLM follows using the available tools. Skill subclasses point to `.md` files in `src/skills/skills_files/` with YAML frontmatter:

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

When invoked, the skill returns its instructions with a `[SKILL INSTRUCTIONS]` prefix that directs the LLM to execute each step via tools rather than summarizing them.

### Events

An **Event** is a self-contained background observer. Each event runs as an asyncio task inside the domain's `EventBus` and defines two things:

- **`condition()`** — an async coroutine that blocks until the triggering condition is met. It can be a sleep, a file-modification check, a queue wait, or any awaitable.
- **`trigger()`** — called once `condition()` returns. Performs any side-effect needed before the agent message is sent.

If the event sets a `message` string, the `EventBus` automatically dispatches it to the agent via `domain.generate_event()` — a fresh, isolated LLM call that never touches the conversation memory or affects routing for user queries.

**Built-in events** (`src/events/`):

| Event | Description |
|-------|-------------|
| `TimerEvent` | Fires every N seconds (configurable interval) |
| `FileEvent` | Fires when a watched file's modification time changes |

**Notification** — when an event fires, both surfaces are notified:
- **CLI**: a colored banner (`◆ Event [name]`) is printed, followed by the agent's streamed response.
- **Tray**: a system tray balloon message appears and the panel opens to show the response.

Event responses are rendered exactly like user-initiated responses (routing, spinner, tool calls) but use an isolated LLM instance so the conversation memory is never polluted.

**Defining a new event:**

```python
import asyncio
from src.core.event import Event

class MyEvent(Event):
    name = "my_event"
    description = "Fires when something happens"
    message = "Something happened, please respond."  # sent to agent; omit if no agent call needed

    async def condition(self):
        await asyncio.sleep(30)  # or watch a file, wait on a queue, etc.

    async def trigger(self):
        pass  # optional side-effect before the agent message
```

Add it to your domain's `events` list:

```python
events: list = [
    MyEvent(),
]
```

### UI Layer

- **`Spinner`** (`animations.py`) — async spinner with a dynamic `.status` property. Shows what the agent is doing in real time: `⠴ Thinking...`, `⠧ Running execute_command: ls -la`, `⠋ Routing: google_search, fetch_page`
- **`ThinkFilter`** (`filter.py`) — streaming parser for `<think>...</think>` tags, handles tags split across token boundaries. Renders reasoning in dim italic while keeping the final answer in normal text
- **`StreamRenderer`** (`renderer.py`) — orchestrates the full output pipeline: spinner lifecycle, `RouteEvent` and `ToolEvent` processing, think filtering, typewriter-effect streaming, ESC-to-stop

### Tray App

A **PyQt6 system tray application** that uses the same domain classes and async logic as the CLI.

- **`worker.py`** — `StreamWorker` QThread runs `domain.generate()` on a shared asyncio event loop via `asyncio.run_coroutine_threadsafe`. Emits `token_received`, `tool_status`, `finished`, and `error` signals. Cancellation cancels the asyncio future and sets a flag checked during streaming.
- **`panel.py`** — `ChatPanel` with a `QLineEdit` input, `QTextBrowser` for streamed markdown output, and header controls. Braille spinner (`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`) driven by `QTimer` shows tool names during execution and routing. Conversation history is persisted to `~/.andrewcli/data/conversation.md` and restored on next launch.
- **Event bridge** — the `EventBus` runs on the shared asyncio loop. A `queue.SimpleQueue` bridges it to the Qt main thread. A `QTimer` polling at 100 ms drains the queue, shows balloon notifications via `QSystemTrayIcon.showMessage`, and routes tokens to the panel — all without blocking the Qt event loop or the asyncio loop.
- Submitting a new message while generating **cancels the previous generation** and waits before starting a new one.
- **Multi-turn conversations** work because the domain instance (and its memory) persists across all turns.

---

## Async Pipeline

The entire I/O pipeline is non-blocking:

1. **`andrewcli.py`** — runs under `asyncio.run()`. Input read via a custom async `_read_input()` using cbreak mode, supporting TAB and UP/DOWN history.
2. **Router** — async LLM call that resolves the minimal tool/skill set for the request.
3. **Spinner** — `asyncio` task that animates and updates status text from `RouteEvent` and `ToolEvent`s.
4. **Streaming** — `LLM.generate()` is an async generator. Tokens are yielded as they arrive. `ToolEvent` objects are also yielded to update the spinner.
5. **Tool calls** — accumulated from streamed chunks, executed via `tool.run()`, looped back automatically. Malformed JSON arguments are caught and reported instead of crashing.
6. **Memory summarization** — fires in the background after the response completes; no user-facing delay.
7. **Event bus** — runs as a set of concurrent asyncio tasks alongside the main loop. Each event waits on its own `condition()` coroutine independently.

---

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your LLM endpoint** via environment variables:

   | Variable | Default | Description |
   |----------|---------|-------------|
   | `API_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible API URL |
   | `MODEL` | `qwen3.5:9B` | Model name |
   | `OPENAI_API_KEY` | — | API key (required even for local models) |

3. **Configure `config.yaml`:**

   ```yaml
   domain: "general"
   execute_bash_automatically: false
   tray_width_compact: 500
   tray_height_compact: 80
   tray_width_expanded: 500
   tray_height_expanded: 1000
   tray_platform: "xcb"
   tray_position: "bottom-right"
   tray_opacity: "90%"
   ```

   | Key | Default | Description |
   |-----|---------|-------------|
   | `domain` | `"general"` | Active domain (matches filename in `src/domains/`) |
   | `execute_bash_automatically` | `false` | Skip confirmation prompt for shell commands |
   | `tray_width_compact` | `600` | Compact panel width (px) |
   | `tray_height_compact` | `80` | Compact panel height (px) |
   | `tray_width_expanded` | `900` | Expanded panel width (px) |
   | `tray_height_expanded` | `600` | Expanded panel height (px) |
   | `tray_platform` | `""` | Qt platform backend (`"xcb"` for X11, `""` for default/Wayland) |
   | `tray_position` | `"top-right"` | Window position: `top-left`, `top-center`, `top-right`, `center-left`, `center`, `center-right`, `bottom-left`, `bottom-center`, `bottom-right` |
   | `tray_opacity` | `"100%"` | Window opacity (`"0%"` to `"100%"`) |

4. **Run:**

   ```bash
   # CLI mode (default)
   python andrewcli.py

   # System tray GUI
   python andrewcli.py --tray

   # FastAPI server
   python andrewcli.py --server
   python andrewcli.py --server --host 127.0.0.1 --port 9000
   ```

---

## Interactive Controls

### CLI

| Key | Context | Action |
|-----|---------|--------|
| **TAB** | Input prompt | Cycle to the next available domain |
| **UP / DOWN** | Input prompt | Navigate command history |
| **ESC** | During response | Stop streaming (background tasks still complete) |

### Tray

| Key / Control | Context | Action |
|---------------|---------|--------|
| **TAB** | Input field | Cycle to the next available domain |
| **Domain button** | Header | Cycle to the next available domain |
| **Stop button** | Header (during generation) | Cancel the current generation |
| **Clear button** | Header (expanded) | Clear chat view and reset conversation memory |
| **ESC** | Anywhere in panel | Hide the panel window |
| **▽ / △ button** | Header | Toggle between compact and expanded view |

---

## Usage

```
$ python andrewcli.py
Andrew is running...
[general] Ask: Write "hello" to greeting.txt
⠋ Routing: write_file
⠋ Running write_file: greeting.txt
Andrew: File greeting.txt written successfully.
[general] Ask: ↑                          # UP recalls last message
[general] Ask: [TAB]                      # TAB switches domain
Switched to domain: coding
[coding] Ask:

◆ Event [timer]: Fires every N seconds   # event fires in background
Andrew: ...
```

The agent chains tool calls automatically — a skill might instruct the LLM to read a file, transform its contents, and write the result back. The spinner shows which tool is running in real time; model reasoning inside `<think>` tags is displayed in dim italic. Events fire independently in the background and interleave cleanly with user interactions.

---

## Extending

### Add a new Tool

Create a class that inherits from `Tool` in `src/tools/`:

```python
from src.core.tool import Tool

class MyTool(Tool):
    name: str = "my_tool"
    description: str = "Does something useful."

    def execute(self, arg1: str, arg2: int = 0) -> str:
        return "result"
```

Import and add `MyTool()` to your domain's `tools` list.

### Add a new Skill

1. Create a markdown file in `src/skills/skills_files/`:

   ```markdown
   ---
   name: my_skill
   description: What this skill does
   tools: [tool_name_1, tool_name_2]
   ---

   # Instructions
   1. Step one
   2. Step two
   ```

   `tools:` is optional — list any tools the skill requires that the router might not select automatically.

2. Create a `Skill` subclass in `src/skills/myskills.py` and add it to your domain's `skills` list:

   ```python
   from src.core.skill import Skill

   class MySkill(Skill):
       skill_file: str = "my_skill.md"
   ```

### Add a new Domain

Create a file in `src/domains/` (e.g. `research.py`):

```python
from src.core.domain import Domain

class ResearchDomain(Domain):
    system_prompt: str = "You are a research assistant."
    tools: list = []
    skills: list = []
    events: list = []
```

Set `domain: "research"` in `config.yaml`. The file name must match the config value; the class must be named `<Name>Domain`.

### Add a new Event

Create a file in `src/events/` (e.g. `my_event.py`):

```python
import asyncio
from src.core.event import Event

class MyEvent(Event):
    name = "my_event"
    description = "Short description shown in notifications"
    message = "Prompt sent to the agent when this event fires."

    async def condition(self):
        await asyncio.sleep(60)  # block until condition is met

    async def trigger(self):
        pass  # optional side-effect before the agent message
```

Add it to your domain:

```python
from src.events.my_event import MyEvent

events: list = [MyEvent()]
```

---

## TODO

- [ ] Write a skill that allows AndrewCLI to update itself with new tools, skills, or domains

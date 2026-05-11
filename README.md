# AndrewCLI

A lightweight, fully async Python agent — designed to **keep your context clean**.

Local models degrade fast as context grows: reasoning gets muddier, tool calls go off the rails, and token budgets hit the ceiling. AndrewCLI fights this with two mechanisms that run on every turn:

- **Rolling memory** — after each response, messages are trimmed to just the last exchange and replaced with a compact `~300-word` summary injected into the system prompt. Short exchanges are appended inline with no LLM call, and a dedicated `SUMMARY_MODEL` can be pointed at a smaller model for the background merges.
- **Context-aware router** — before each generation, an LLM-based classifier picks only the tools and skills needed for the request. Irrelevant schemas never reach the generation prompt.

All three surfaces (CLI, system tray, and HTTP API) share the same core: same domains, same memory, same router, and the same `Domain.busy_lock` that serializes user turns and event dispatches identically on every surface. The CLI and tray automatically start the FastAPI server in a background thread on launch. The server is a **thin middleware** — it enqueues messages into a shared bridge and the CLI/tray processes them through the normal submit path, so slash commands, events, and the full domain pipeline all work over HTTP exactly as if the user typed them. Clients poll an endpoint for the response tokens.

An optional **`--voice`** modifier adds wake-word speech-to-text and streaming text-to-speech to either mode. `andrewcli --voice` turns the CLI into a push-to-talk-or-type surface; `andrewcli --tray --voice` does the same for the tray. Voice I/O is additive — typing always works, and the spoken and typed paths produce identical conversation state.

---

## Project Structure

```
AndrewCLI/
├── andrewcli.py                    # Unified entry point (CLI, tray, server)
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── events/                         # Event definitions — auto-discovered, activated via /name
│   ├── timer.py                    # TimerEvent — fires on a fixed interval
│   ├── file.py                     # FileEvent — fires when a watched file is modified
│   ├── project.py                  # ProjectEvent — drives the agent through a multi-step project
│   ├── loop.py                     # LoopEvent — drives a 'do X until Y' loop with exit criteria
│   └── schedule.py                 # ScheduleEvent — fires once at a specific datetime
├── domains/                        # Each domain is a self-contained subpackage
│   └── general/                    # — auto-discovered, no manual registration
│       ├── domain.py               # GeneralDomain class
│       ├── tools/                  # Tool subclasses auto-loaded from *.py
│       │   └── common.py           #   WriteFile, ReadFile, ExecuteCommand, GetCurrentDate
│       └── skills/                 # Skill markdown files auto-loaded as Skill instances
│           ├── example.md          #   Example skill (template)
│           └── create_new_skill.md #   SkillCompiler — scaffolds new skill markdown files
└── src/
    ├── shared/
    │   ├── config.py               # Config class — loads config.yaml
    │   └── paths.py                # Centralized filesystem paths (PROJECT_ROOT, LAUNCH_DIR, DATA_DIR, …)
    ├── core/
    │   ├── server.py               # FastAPI middleware + shared bridge (inbox queue, session store)
    │   ├── domain.py               # Base Domain class (async generator, event bus, busy_lock)
    │   ├── event.py                # Event ABC + EventBus (add, remove, running, stop)
    │   ├── llm.py                  # Async LLM client with streaming + tool-calling loop
    │   ├── memory.py               # Rolling memory with background summarization
    │   ├── registry.py             # Unified auto-discovery: domains, events (+ /slash parsing), tools, skills
    │   ├── router.py               # ToolRouter — LLM-based tool/skill router
    │   ├── skill.py                # Base Skill class (markdown-defined tools)
    │   └── tool.py                 # Base Tool class — auto-generates OpenAI schemas from type hints
    ├── ui/                         # CLI rendering layer
    │   ├── animations.py           # Spinner (async, dynamic status)
    │   ├── filter.py               # ThinkFilter — parses <think> tags for reasoning display
    │   └── renderer.py             # StreamRenderer — spinner + filtering + typewriter streaming
    ├── tray/                       # System tray GUI (PyQt6)
    │   ├── __main__.py             # Entry point for `python -m src.tray`
    │   ├── bootstrap.py            # Pre-Qt init (sets QT_QPA_PLATFORM from config)
    │   ├── app.py                  # Thin shell — constructs TrayController and QApplication
    │   ├── controller.py           # TrayController — domain loading, worker lifecycle, voice bridge, signals
    │   ├── worker.py               # StreamWorker QThread — runs domain.generate() on shared async loop
    │   ├── panel.py                # ChatPanel widget — input, streamed output, spinner, controls
    │   ├── icon.py                 # Tray icon and context menu
    │   ├── style.css               # Qt stylesheet (Catppuccin Mocha)
    │   └── md.css                  # CSS for markdown rendering in QTextBrowser
    └── voice/                      # Optional wake-word STT + streaming TTS
        ├── __init__.py             # build_voice_io(config) factory — shared by CLI, tray, session
        ├── stt.py                  # SpeechToText — openwakeword + faster-whisper + energy VAD
        ├── hey_andrew.onnx         # Custom "hey Andrew" wake-word model (ONNX, used by default)
        ├── hey_andrew.tflite       # Same model in TFLite format (training artefact, kept for reference)
        ├── tts.py                  # TextToSpeech — Piper (local, fast, robotic)
        ├── tts_edge.py             # EdgeTTS — Microsoft Azure Neural TTS (online, high-quality)
        ├── sanitize.py             # strip_markdown — pre-render filter for the TTS tee path
        └── session.py              # Stand-alone VoiceSession (library-level helper)
```

---

## How Context Stays Clean

### Rolling Memory

After each completed turn:

1. The last 1500 characters of conversation are extracted as an excerpt.
2. **Short-turn fast path** — if the excerpt is under 200 characters (greetings, one-line confirmations, `"Ciao"` / `"grazie"`), it is appended inline with a rolling 2000-char window and **no LLM call is made at all**.
3. If no summary exists yet, the excerpt is saved as-is (capped at 2000 chars).
4. If a summary exists and the turn is long enough, it is **merged with the excerpt** via a background LLM call that produces a single `~300-word` summary — facts, decisions, code written, tools used, user preferences.
5. Messages are **trimmed to just the last exchange** (last user message + last assistant response). The summary is injected into the system prompt inside `<memory>` tags.
6. The merge runs as a **fire-and-forget `asyncio` task** — the next prompt is ready immediately. Sequential merges are serialized with a lock to prevent race conditions.

The model always has prior context, but the message history never grows beyond one exchange. This keeps the effective prompt size bounded regardless of conversation length.

**Dedicated summary model** — summarization is background work that doesn't need the same capacity as the main chat model. Set the `SUMMARY_MODEL` env var (e.g. `qwen2.5-0.5b-instruct`) to route background merges to a smaller model on the same server. Defaults to `MODEL` when unset.

**Storage:** `$LAUNCH_DIR/.andrewcli/data/memory.json` — the data directory lives inside the folder you launched `andrewcli` from (same pattern as Claude Code / OpenCode), so different projects keep independent memory. Override by exporting `ANDREW_LAUNCH_DIR` before launching.

### Context-Aware Router

Before each generation the router selects a minimal tool/skill set. The generation call receives **only the selected schemas**, not the full catalog. A skill that requires specific tools can declare them in its YAML frontmatter — those are injected even if the router didn't select them. Any routing failure falls back to returning the full catalog so the LLM always has its tools.

`ToolRouter` (`src/core/router.py`) is an **LLM-as-classifier**: it sends the prompt, the current memory context (`summary` + `last_exchange`), and the full tool/skill catalog to the chat model and parses the JSON array it replies with. Costs 0.5–2 s per turn but handles ambiguous intent and natural-language follow-ups well. A domain can also opt out of routing entirely by setting `routing_enabled = False` — useful when the full toolset is always relevant (e.g. the coding domain).

---

## Architecture

### Config

A centralized `Config` class (`src/shared/config.py`) loads `config.yaml` and exposes settings as attributes. Used by `app.py` to select the active domain, by `ExecuteCommand` to read `execute_bash_automatically`, and by the tray app for window dimensions, position, opacity, and platform backend.

### Domains

A **Domain** is a self-contained subpackage under `domains/<name>/` that groups a system prompt, a set of tools, and a set of skills into a single persona. The `Domain` subclass itself is trivially small — `domains/<name>/domain.py` typically declares only `api_base_url`, `model`, and `routing_enabled`:

- **System prompt** — loaded from `domains/<name>/system_prompt.md` at construction time. Plain markdown, no frontmatter, no Python string concatenation. A class-level `system_prompt` attribute is still honored as a fallback for quick in-code experiments.
- **Tools** — auto-discovered from `domains/<name>/tools/*.py` (every concrete `Tool` subclass is instantiated and registered).
- **Skills** — auto-discovered from `domains/<name>/skills/*.md`.

The active domain is chosen from `config.yaml` (`domain: "general"`) and can be **switched at runtime** with TAB. Domains can optionally override the global LLM endpoint and model per-domain (see [Add a new Domain](#add-a-new-domain)).

Each domain owns an `EventBus` instance that starts empty — events are independent of domains and are added at runtime through slash commands. The bus is started by the app layer alongside the main loop.

### Tools

A **Tool** is a Python class the LLM can call. Tools auto-generate their OpenAI function schema from `execute()`'s type hints — no manual schema boilerplate. The base `Tool.run()` wrapper catches exceptions and returns a `[Tool Error]` string so the agent can recover without crashing.

**Built-in tools** (`domains/general/tools/`): `WriteFile`, `ReadFile`, `ExecuteCommand`, `GetCurrentDate`. Every shell command is spawned with `cwd=LAUNCH_DIR` so `execute_command` always targets the directory you launched `andrewcli` from, no matter where the interpreter's own cwd has drifted to.

### Skills

A **Skill** is a markdown-defined tool. Instead of executing code, it returns natural-language instructions that the LLM follows using the available tools. Skill files live directly inside each domain's `skills/` folder and are auto-loaded as `Skill` instances — no Python subclass needed. Each file starts with a YAML frontmatter block:

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

When the LLM invokes a skill, its body is **promoted into the system prompt** as a turn-scoped `<skill:NAME>...</skill:NAME>` block (see `Memory.add_active_skill` and `Memory.get` in `src/core/memory.py`). The tool-call response is just a short acknowledgement pointing the model at the new system instructions. This puts skill steps at the same authority tier as the domain's base system prompt — far stickier than embedding them in a `role: tool` message, where local models tend to summarize or skip steps. The block is cleared in a `try/finally` at turn end (`LLM.generate` in `src/core/llm.py`) so it never leaks into the next turn's routing or context.

### Events

An **Event** is a self-contained background observer. Each event runs as an asyncio task inside a domain's `EventBus` — which starts empty and is populated at runtime via slash commands (`/timer 30`, `/project "..."`) or the HTTP API. Events are decoupled from domain definitions: the same `events/` catalog is available from every domain. Each event defines two things:

- **`condition()`** — an async coroutine that blocks until the triggering condition is met. It can be a sleep, a file-modification check, a queue wait, or any awaitable.
- **`trigger()`** — called once `condition()` returns. Performs any side-effect needed before the agent message is sent.

If the event sets a `message` string, the `EventBus` automatically dispatches it to the agent via `domain.generate_event()` — a fresh, isolated LLM call that never touches the conversation memory or affects routing for user queries.

**Serialization (`Domain.busy_lock`)** — both `domain.generate()` (user turns) and `domain.generate_event()` (event dispatches) acquire a single `asyncio.Lock` on the domain. This means:

- Only one agent interaction streams to the UI at a time.
- Events queue **FIFO** behind each other and behind any in-flight user turn (`asyncio.Lock` wakes waiters in acquisition order).
- A timer event cannot pile up on itself — each event's `_run` loop awaits the full `condition → trigger → notify → dispatch` chain before re-arming.
- The CLI and tray inherit identical behavior without per-surface locks.

**Built-in events** (`events/`):

| Event | Slash command | Description |
|-------|---------------|-------------|
| `TimerEvent` | `/timer [interval]` | Fires every N seconds |
| `FileEvent` | `/file [path] [poll_interval] [message]` | Fires when a watched file is modified |
| `ProjectEvent` | `/project [goal] [state_file]` | Drives the agent through a multi-step project: plans on the first invocation, executes one task per invocation, stops when all tasks are marked done in `state_file`. Calling `/project` with no arguments resumes from the existing `state_file` (goal recovered from the file) |
| `LoopEvent` | `/loop [goal] [max_iterations] [state_file]` | Drives a `do X until Y` loop: plans the action and exit criteria on the first invocation, performs the action once per iteration, stops when any exit criterion fires (or, if a positive `max_iterations` was set, when the cap is hit). `max_iterations` is optional and defaults to `0` (uncapped — runs until a criterion fires). Calling `/loop` with no arguments resumes from the existing `state_file` |
| `ScheduleEvent` | `/schedule [message] [when]` | Fires once at a specific datetime (`dd-mm-yyyy-hh-mm`), sends `message` to the agent, then stops automatically |

**Notification** — when an event fires, both surfaces are notified:
- **CLI**: a colored banner (`◆ Event [name]`) is printed, followed by the agent's streamed response.
- **Tray**: a system tray balloon message appears and the panel opens to show the response.

Event responses are rendered exactly like user-initiated responses (routing, spinner, tool calls) but use an isolated LLM instance so the conversation memory is never polluted.

#### Slash commands

Events are activated at runtime via slash commands — no domain restart required. The same syntax works in the CLI input prompt, the tray panel input field, and the server's `/chat` endpoint.

```
/events                              — list available events and which are running
/stop [name]                         — stop a running event by name
/stop                                — list the names of currently running events

/timer 30                            → TimerEvent(interval=30.0)
/file war_news/news.md 5.0 "Update!" → FileEvent("war_news/news.md", 5.0, "Update!")
/project "Build a REST API in Python" → ProjectEvent(goal=...) — plan + execute loop
/project                              → ProjectEvent() — resume from project_state.json
/loop "Monitor oil price; stop if < $100" → LoopEvent(goal=...) — uncapped
/loop "Poll /status until ready" 30   → LoopEvent(goal=..., max_iterations=30)
/loop                                  → LoopEvent() — resume from loop_state.json
/schedule "Run skill 1" 10-10-2026-11-30 → ScheduleEvent(message="Run skill 1", when=datetime(2026,10,10,11,30))
```

Quoted strings with spaces are handled correctly (`shlex` tokenisation). Arguments are coerced to their annotated types (`float`, `int`, or `str`). Extra arguments beyond the declared parameters are ignored; missing optional parameters fall back to their defaults.

**Event auto-discovery** — every file dropped in `events/` that defines a concrete `Event` subclass with a `name` string attribute is automatically registered. No manual import or registration step needed. The unified registry (`src/core/registry.py`) is scanned at command parse time, so new events are available immediately after saving the file.

**`EventBus` API** — the bus exposes three methods alongside `start()` and `stop()`:

| Method | Description |
|--------|-------------|
| `add(event)` | Start a new event on the already-running bus. Creates an independent asyncio task tracked by `stop()`. |
| `remove(name)` | Cancel and remove the event with the given name. Returns `True` if found. |
| `running()` | Return a list of names of currently active (non-done) events. |

**Server** — the FastAPI server (`src/core/server.py`) is a thin middleware: `POST /chat` enqueues the message into a shared bridge inbox and returns a `session_id`; the CLI/tray picks it up within 100 ms and processes it through the normal submit path — the same code path as a typed message, so slash commands and events work identically over HTTP. Response tokens are accumulated in a per-session store and returned via `GET /chat/{session_id}` (tokens are consumed on each call; poll until `done: true`). `GET /events` lists all registered event types with their parameter names.

#### `ProjectEvent` — autonomous project loop

`ProjectEvent` is a special event designed to drive the agent through a multi-step coding or research project without human intervention:

1. **Iteration 0 (planning)** — no state file exists yet. The agent is asked to extract any explicit constraints from the goal and break it into a list of concrete tasks, writing both to `state_file` as JSON, then start task 1.
2. **Iterations 1…N (execution)** — the next pending task is passed to the agent, which completes it and marks it `done: true` in `state_file`.
3. **Final iteration** — all tasks are done. The agent summarises what was built and confirms each constraint was honoured, then the event stops.

**Resume mode** — invoking `/project` with no goal (just `/project`) reads the existing `state_file`, recovers the goal from its `"goal"` field, and picks up at the next pending task. This is the recommended way to continue a project across restarts: the same JSON file drives both progress tracking and resumption. If no state file exists (or it has no `goal` field), the event raises a clear error pointing back to `/project <goal>`.

```json
{
  "goal": "Build a REST API in Python",
  "constraints": ["Use only the standard library", "All endpoints under /api/v1"],
  "tasks": [
    {"id": 1, "title": "Set up project structure", "done": true,
     "artefacts": ["pyproject.toml", "src/api/__init__.py"]},
    {"id": 2, "title": "Implement endpoints",       "done": false,
     "subtasks": ["GET /users", "POST /users"]},
    {"id": 3, "title": "Write tests",               "done": false}
  ],
  "build_log": ["installed deps", "scaffolded routes"]
}
```

**State rigor (loop-owned schema fields)** — the file is parsed via a Pydantic 2 model (`ProjectState` / `ProjectTask`) so wrong field names like `is_done` / `completed` / `finished` are accepted as aliases for `done`, integer task ids are coerced to strings, malformed payloads fall back to a fresh plan, and constraints written as a single string are coerced to a list. On top of that, the agent only owns the `done` flags. On the first read of a planned file, `ProjectEvent` snapshots `goal`, `constraints`, and the task `id`/`title` pairs in memory; subsequent reads rebuild the state from the snapshot, importing only `done` flags from disk. Done flags are also **monotonic** — once a task is observed `done: true`, it stays done, even if the agent later flips the on-disk value back to false in a misguided "loop forever" attempt. Every per-task prompt embeds the canonical task list with checkboxes (`[x]` / `[ ]`) and a `<- CURRENT` pointer, so the agent always sees authoritative progress regardless of what is on disk. Once the completion summary is dispatched, the event is permanently terminal.

**Constraints** are first-class: the planner is instructed to copy stop conditions, deadlines, and "do not" rules verbatim from the goal into a `constraints` array, and the loop re-injects them as a bullet block above every iteration's prompt so they survive rolling-memory trimming.

**Agent scratchpad (custom fields)** — the schema above is the *minimum* required structure; the agent is free to add custom fields, both at the top level (`build_log`, `outstanding_questions`, dependency maps) and inside individual task objects (`artefacts`, `subtasks`, `notes`, `estimated_hours`). The Pydantic model is configured with `extra="allow"`, and the reconciliation step explicitly carries every unknown field through to the next iteration's prompt. This gives the agent a durable, structured place to keep working memory across iterations without the loop machinery interfering with it. The contract is simple:

  | Field type | Owner | Mutability across iterations |
  |---|---|---|
  | `goal`, `constraints`, task `id`/`title`, task list shape | Loop | Immutable (snapshot-restored) |
  | task `done` flags | Agent | Monotonic (`false → true` only) |
  | Any other field, top-level or per-task | Agent | Free-form (preserved verbatim) |

#### `LoopEvent` — do X until Y is met

`LoopEvent` is the counterpart to `ProjectEvent` for goals shaped as *"keep doing X until Y"* rather than *"complete tasks A, B, C"*. Examples: monitor a metric until it crosses a threshold, poll an endpoint until it returns ready, retry an action until it succeeds.

1. **Iteration 0 (planning)** — no state file exists yet. The agent extracts the single repeating `action` and the list of `exit_criteria` from the goal, writes them to `state_file`, then performs iteration 1.
2. **Iterations 1…N (execution)** — each iteration the agent performs the action exactly once, records a concrete `last_observation`, and evaluates every exit criterion. If any criterion is met it sets `terminated: true` and writes a `termination_reason`.
3. **Final iteration** — either an exit criterion fired, or (if a cap was set) `iterations` reached `max_iterations`. The agent summarises the run and the event stops.

**Resume mode** — invoking `/loop` with no goal reads the existing `state_file` and continues from the last recorded iteration count. If a `max_iterations` argument is passed on resume, it overrides whatever was previously on disk; otherwise the on-disk value (if any) is preserved.

```json
{
  "goal": "Monitor oil price; stop when < $100/bbl",
  "action": "Fetch the current WTI price via google_search",
  "exit_criteria": ["price < 100 USD/bbl", "24 hours elapsed"],
  "max_iterations": null,
  "iterations": 7,
  "last_observation": "Iteration 7: price = 102.4 USD/bbl",
  "terminated": false,
  "termination_reason": "",
  "price_history": [110.2, 109.8, 108.4, 107.1, 105.6, 103.9, 102.4],
  "rolling_avg": 106.8,
  "consecutive_above_threshold": 7
}
```

**State rigor (loop-owned schema fields)** — the file is parsed via a Pydantic 2 model (`LoopState`) so wrong field names like `iterations_done` / `iter_count` / `iters` are accepted as aliases for `iterations`, sloppy values like the literal string `"unlimited"` are coerced to `null` for `max_iterations`, and malformed payloads fall back to a fresh plan rather than crashing. On top of that, the same snapshot pattern as `ProjectEvent` applies: the immutable fields (`goal`, `action`, `exit_criteria`, `max_iterations`) are captured on the first read and restored on every subsequent read, so the agent cannot rewrite the action mid-loop. The progress fields are tightly constrained:

| Field | Mutability |
|---|---|
| `iterations` | **Monotonic** — the loop tracks an in-memory floor; on-disk decreases are ignored |
| `terminated` | **Sticky** — `false → true` only; un-terminating after the summary cannot restart the loop |
| `last_observation`, `termination_reason` | Free-form, agent-owned |

Every iteration prompt embeds the authoritative state as a fenced JSON block produced by `model_dump_json()`, so the agent sees the *exact* canonical field names it must use — eliminating the prose-to-key drift class of bugs (e.g. an agent inventing `"iterations_done"` because the prompt said "iterations done : 5"). It also gives continuity across iterations: the agent can detect trends like "the price has risen for three iterations in a row" by reading its own previous custom fields.

**Agent scratchpad (custom fields)** — the schema above is the *minimum* required structure; the agent is free to add top-level custom fields such as `price_history`, `rolling_avg`, `consecutive_above_threshold`, retry counters, observation logs, or anything else useful between turns. The `LoopState` Pydantic model is configured with `extra="allow"`, and the reconciliation step carries every unknown field through to the next iteration's canonical JSON block. The contract:

  | Field type | Owner | Mutability across iterations |
  |---|---|---|
  | `goal`, `action`, `exit_criteria`, `max_iterations` | Loop | Immutable (snapshot-restored) |
  | `iterations` | Loop | Monotonic floor (only goes up) |
  | `terminated`, `termination_reason` | Agent→Loop | Sticky (false→true only) |
  | `last_observation` | Agent | Free-form |
  | Any other field | Agent | Free-form (preserved verbatim) |

**Optional `max_iterations` cap** — the iteration cap is a constructor parameter (`/loop [goal] [max_iterations] [state_file]`) and is **optional**. The default (`0`) means *uncapped* — the loop runs forever until an exit criterion fires. Pass any positive integer to enforce a hard ceiling, e.g. `/loop "Poll /status until ready" 30`. The user-supplied value always wins over whatever the planner wrote to disk, so `/loop "" 10` is a valid way to tighten the cap on a paused loop. Resolution order on every read: **user-passed cap → on-disk cap → uncapped**.

  | `max_iterations` value | Behaviour |
  |---|---|
  | `0` (default) or `null` on disk | Uncapped — only an exit criterion can stop the loop |
  | positive integer | Hard cap; loop also stops when `iterations >= max_iterations` |

Use the cap as a safety net for goals where local models may misjudge the exit criterion (numeric comparisons, ambiguous phrasing) — leave it off for genuinely open-ended monitors that should run until the world changes.

**`LoopEvent` vs `ProjectEvent`** — use `ProjectEvent` for goals that decompose into a finite checklist ("build X, write Y, deploy Z"); use `LoopEvent` for goals shaped as a predicate ("keep doing X until Y"). Mechanically: `ProjectEvent` terminates by completion count, `LoopEvent` terminates by predicate.

**Defining a new event:**

```python
import asyncio
from src.core.event import Event

class MyEvent(Event):
    name = "my_event"          # used as the slash-command name: /my_event
    description = "Fires when something happens"
    message = "Something happened, please respond."

    async def condition(self):
        await asyncio.sleep(30)  # or watch a file, wait on a queue, etc.

    async def trigger(self):
        pass  # optional side-effect before the agent message
```

Drop the file in `events/` — it is discovered automatically and immediately available as `/my_event [args]`. Events are not tied to any domain; the same catalog works across every domain and is populated at runtime as the user types slash commands.

### UI Layer

- **`Spinner`** (`animations.py`) — async spinner with a dynamic `.status` property. Shows what the agent is doing in real time: `⠴ Thinking...`, `⠧ Running execute_command: ls -la`, `⠋ Loading: write_file, read_file`
- **`ThinkFilter`** (`filter.py`) — streaming parser for `<think>...</think>` tags, handles tags split across token boundaries. Renders reasoning in dim italic while keeping the final answer in normal text
- **`StreamRenderer`** (`renderer.py`) — orchestrates the full output pipeline: spinner lifecycle, `RouteEvent` and `ToolEvent` processing, think filtering, typewriter-effect streaming, ESC-to-stop

### Tray App

A **PyQt6 system tray application** that uses the same domain classes and async logic as the CLI. `app.py` is a thin shell that constructs the `QApplication` and delegates all orchestration to `TrayController` (`controller.py`).

- **`controller.py`** — `TrayController` handles domain loading, `StreamWorker` lifecycle, the event bridge, and voice integration. All voice state (`_voice_idle_event`, `_voice_user_enabled`, `_voice_agent_busy`) and the `_on_voice_toggle` handler live here.
- **`worker.py`** — `StreamWorker` QThread runs `domain.generate()` on a shared asyncio event loop via `asyncio.run_coroutine_threadsafe`. Emits `token_received`, `tool_status`, `finished`, and `error` signals. Cancellation cancels the asyncio future and sets a flag checked during streaming.
- **`panel.py`** — `ChatPanel` with a `QLineEdit` input, `QTextBrowser` for streamed markdown output, and header controls. Braille spinner (`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`) driven by `QTimer` shows tool names during execution and routing. Conversation history is persisted to `$LAUNCH_DIR/.andrewcli/data/conversation.md` and restored on next launch.
- **Event bridge** — the `EventBus` runs on the shared asyncio loop. A `queue.SimpleQueue` bridges it to the Qt main thread. A `QTimer` polling at 100 ms drains the queue, shows balloon notifications via `QSystemTrayIcon.showMessage`, and routes tokens to the panel — all without blocking the Qt event loop or the asyncio loop.
- Submitting a new message while generating **cancels the previous generation** and waits before starting a new one.
- **Multi-turn conversations** work because the domain instance (and its memory) persists across all turns.

### Voice (optional)

Passing `--voice` to either the CLI or the tray adds a wake-word-triggered STT pipeline plus sentence-streaming TTS. Everything is optional — if the voice extras aren't installed, the flag is the only thing that touches them, so a plain `andrewcli` run never pays the cost.

- **Single factory** — `src.voice.build_voice_io(config)` constructs the `(stt, tts)` pair from `config.yaml`'s `voice.*` keys. Used by `AndrewCLI`, `AndrewTrayApp`, and the stand-alone `VoiceSession`, so the same config works identically on every surface.
- **Wake-word STT** (`stt.py`) — [`openwakeword`](https://github.com/dscripka/openWakeWord) (tiny ONNX detector, CPU) listens continuously; when the configured word crosses `wake_threshold`, [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) transcribes the captured utterance. CPU threads autodetect (`os.cpu_count()`) so transcription uses every core instead of the 4-thread default. The default wake word is `hey_andrew`, backed by a custom model trained via the openwakeword pipeline and bundled at `src/voice/hey_andrew.onnx`. `_load_wake_model` resolves names in three stages — openwakeword built-ins (`hey_jarvis`, `alexa`, …), then bundled models in `src/voice/<name>.onnx`/`.tflite`, then arbitrary filesystem paths — so swapping in a freshly trained model is just a rename + config change.
- **Adaptive VAD with minimum recording floor** — every utterance is unconditionally recorded for at least 1.5 s (the "floor"); after that, a frame is silent iff `rms < max(silence_rms, 0.35 × peak_rms)` and 800 ms of trailing silence closes the utterance. An 8 s hard cap guarantees progress on hot mics where ambient noise never clears the peak-relative threshold. Whisper's bundled Silero VAD (`vad_filter=True`) then trims leading/trailing noise inside the recorded buffer. The floor is the critical fix: earlier revisions used a plain onset gate that mis-terminated quiet post-wake lulls in under a second.
- **Streaming TTS** — `EdgeTTS` (Microsoft Azure Neural, online, high-quality) or `TextToSpeech` (local Piper, offline, robotic). Both expose a `speak_stream(async_iter)` API that chunks the incoming token stream into sentences and plays each one as it's synthesized, so audio starts within ~500 ms of the first LLM token.
- **Stream teeing** — each assistant response goes to two consumers at once: the normal renderer (CLI stdout or tray panel) and the TTS queue. Implemented via a shared `asyncio.Queue` inside `_stream_response` (CLI) and `StreamWorker._stream` (tray); a sentinel unblocks the consumer on cancellation so TTS stops mid-sentence when the user hits Stop.
- **Markdown pre-render for TTS** — `src.voice.strip_markdown` is a stateful, character-level async filter that sits between the token stream and the TTS queue. It drops `* _ ~ ` # ` (so the speaker doesn't read `**bold**` as "asterisk asterisk bold asterisk asterisk" / "asterisco asterisco grassetto …") and elides the URL portion of `[link](https://...)` while keeping the link text. Stateful across tokens so the `]`→`(` transition works even when the two chars arrive in different chunks. The chat panel / stdout still sees the raw markdown (and renders it visually).
- **CLI integration** — `_get_next_prompt` races the stdin reader against `stt.listen_once()`. Whichever fires first wins and the other is cancelled (termios is restored cleanly). Spoken prompts are echoed to the terminal with a `🎙 ` prefix so the log reads identically whether you typed or spoke. An `on_wake` callback overwrites the prompt line with `🎙 listening...` the instant the wake word fires, so there's visible feedback before you start speaking.
- **Tray integration** — the tray drives `stt.listen_once` in a loop on the background asyncio loop (same one the `StreamWorker` uses for `domain.generate`, so token teeing stays lock-free). Three sentinels flow through a `queue.SimpleQueue` to the 100 ms Qt poller: `__wake__` swaps the status spinner to `🎙 listening...`, a non-empty transcript calls `ChatPanel.show_user_message(text)` + `_on_submit(text)` (producing the same conversation timeline as a typed submit), and `__idle__` (empty transcript — silence after wake or Whisper VAD filtered everything) resets the spinner so the panel never looks stuck on `listening...`.
- **Two-state idle gate** — an `asyncio.Event` (`_voice_idle_event`, on the bg loop) is open iff `user_enabled AND NOT agent_busy`. Two independent flags feed it: `_voice_user_enabled` (flipped by the 🎙/🔇 toggle button in the panel header) and `_voice_agent_busy` (flipped at `_on_submit`/`_event_dispatch` start, cleared on `finished`/`error`/`_on_stop`). When the gate closes it also **cancels the currently-running `listen_once` task**, so the mic goes cold immediately instead of waiting for the next loop iteration — critical for preventing wake-word retrigger mid-turn and self-wake from the tray's own TTS playback bleeding into the mic.
- **Mic toggle button** — a `● Voice` / `○ Voice` toggle in the panel header when `--voice` is active. Green + filled dot = listening; grey + hollow dot = paused. Click to flip; emits `ChatPanel.voice_toggle(bool)` wired to `TrayController._on_voice_toggle`. Label is plain ASCII + Unicode bullet rather than a mic emoji so it stays visible on Linux boxes without a color-emoji font installed. The CLI is sequential by construction (`_get_next_prompt` only runs between turns) so it needs no equivalent gate or button.

---

## Async Pipeline

The entire I/O pipeline is non-blocking:

1. **`andrewcli.py`** — runs under `asyncio.run()`. Input read via a custom async `_read_input()` using cbreak mode, supporting TAB and UP/DOWN history.
2. **Router** — async LLM-as-classifier call that returns the minimal tool/skill set for the prompt.
3. **Spinner** — `asyncio` task that animates and updates status text from `RouteEvent` and `ToolEvent`s.
4. **Streaming** — `LLM.generate()` is an async generator. Tokens are yielded as they arrive. `ToolEvent` objects are also yielded to update the spinner.
5. **Tool calls** — accumulated from streamed chunks, executed via `tool.run()`, looped back automatically. Malformed JSON arguments are caught and reported instead of crashing.
6. **Memory summarization** — fires in the background after the response completes; no user-facing delay. Skipped entirely for short turns.
7. **Event bus** — runs as a set of concurrent asyncio tasks alongside the main loop. Each event waits on its own `condition()` coroutine independently, then acquires `Domain.busy_lock` to dispatch (FIFO across events and user turns).

---

## Setup

1. **Install the package:**

   **Core (CLI, tray, server):**

   ```bash
   pip install -e .
   ```

   **Full (core + voice — wake-word STT, streaming TTS):**

   ```bash
   pip install -e ".[voice]"
   ```

   After installation the `andrewcli` command is available system-wide. Voice models download automatically to `~/.cache/` on first use (Whisper ~500 MB for `small`, openwakeword ~15 MB).

2. **Configure your LLM endpoint** via environment variables:

   | Variable | Default | Description |
   |----------|---------|-------------|
   | `API_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible API URL |
   | `MODEL` | `qwen3.5:9B` | Main chat model |
   | `SUMMARY_MODEL` | same as `MODEL` | Smaller model used for background memory summarization |
   | `OPENAI_API_KEY` | `local` | API key — defaults to `"local"` for local servers that don't validate it |

   `API_BASE_URL` and `MODEL` are global defaults. Individual domains can override either by declaring `api_base_url` and/or `model` as class attributes — the domain's values take precedence over the env vars for all LLM calls including routing and event dispatch (see [Add a new Domain](#add-a-new-domain)).

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

   # Voice I/O (only used with --voice; ignored otherwise).
   voice:
     enabled: false
     wake_word: "hey_andrew"         # openwakeword built-in or path to .tflite/.onnx
     wake_threshold: 0.5
     stt_model: "small"              # faster-whisper: tiny / base / small / medium / large-v3
     stt_language: "it"              # "auto" or ISO code
     tts_engine: "edge"              # "piper" (local) or "edge" (online, neural)
     tts_voice: "it-IT-IsabellaNeural"
     tts_speed: 1.0
     input_device: "pulse"           # sounddevice id; "pulse" recommended on Linux
     output_device: "pulse"
   ```

   | Key | Default | Description |
   |-----|---------|-------------|
   | `domain` | `"general"` | Active domain (matches the package name under `domains/`) |
   | `execute_bash_automatically` | `false` | Skip confirmation prompt for shell commands |
   | `tray_width_compact` | `600` | Compact panel width (px) |
   | `tray_height_compact` | `80` | Compact panel height (px) |
   | `tray_width_expanded` | `900` | Expanded panel width (px) |
   | `tray_height_expanded` | `600` | Expanded panel height (px) |
   | `tray_platform` | `""` | Qt platform backend (`"xcb"` for X11, `""` for default/Wayland) |
   | `tray_position` | `"top-right"` | Window position: `top-left`, `top-center`, `top-right`, `center-left`, `center`, `center-right`, `bottom-left`, `bottom-center`, `bottom-right` |
   | `tray_opacity` | `"100%"` | Window opacity (`"0%"` to `"100%"`) |
   | `voice.enabled` | `false` | Enable voice I/O at startup. Can also be toggled at runtime via the tray button |
   | `voice.wake_word` | `"hey_andrew"` | Resolved in this order: (1) openwakeword built-in (`alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`), (2) bundled custom model at `src/voice/<name>.onnx` or `.tflite` (the repo ships `hey_andrew.onnx` trained via the openwakeword pipeline), (3) absolute filesystem path to a `.tflite`/`.onnx`. Unknown names fall back to the default with a warning |
   | `voice.wake_threshold` | `0.5` | openwakeword score above which the wake word is considered fired. Lower = more sensitive |
   | `voice.stt_model` | `"small"` | faster-whisper size. `base` is ~2× faster than `small` with slight quality loss; `medium`/`large-v3` impractical on CPU |
   | `voice.stt_language` | `"auto"` | ISO code (`en`, `it`, …) or `"auto"` for Whisper language detection |
   | `voice.tts_engine` | `"piper"` | `"piper"` (local, fast, robotic) or `"edge"` (Microsoft Neural via free Edge endpoint, excellent quality, needs internet) |
   | `voice.tts_voice` | `"en_US-amy-medium"` | Piper: `<lang>_<region>-<speaker>-<quality>` (e.g. `en_US-amy-medium`). Edge: `<lang>-<region>-<name>Neural` (e.g. `it-IT-IsabellaNeural`; full list: `edge-tts --list-voices`) |
   | `voice.tts_speed` | `1.0` | Playback rate multiplier (0.5 = half speed, 2.0 = double) |
   | `voice.input_device` | `null` | sounddevice input id. `"pulse"` routes through PipeWire/PulseAudio and handles resampling; use an integer id from `python -m sounddevice` for a specific device |
   | `voice.output_device` | `null` | sounddevice output id (same conventions as input) |

4. **Run:**

   ```bash
   # CLI mode (default) — API server auto-starts on 0.0.0.0:8000
   python andrewcli.py

   # CLI with voice (type or say the wake word)
   python andrewcli.py --voice

   # System tray GUI — API server auto-starts inside the tray subprocess
   python andrewcli.py --tray

   # Tray with voice (panel pops up on wake word)
   python andrewcli.py --tray --voice

   # Custom API host/port (applies to the auto-started server in CLI and tray)
   python andrewcli.py --host 127.0.0.1 --port 9000
   python andrewcli.py --tray --host 127.0.0.1 --port 9000

   # Standalone FastAPI server (no CLI or tray, server only)
   python andrewcli.py --server
   python andrewcli.py --server --host 127.0.0.1 --port 9000
   ```

   `--voice` is a **modifier** — it augments whichever primary mode is active. `--tray` and `--server` remain mutually exclusive. `--host` / `--port` control the API server in all modes.

---

## Interactive Controls

### CLI

| Key / Input | Context | Action |
|-------------|---------|--------|
| **TAB** | Input prompt | Cycle to the next available domain |
| **UP / DOWN** | Input prompt | Navigate command history |
| **ESC** | During response | Stop streaming (background tasks still complete) |
| **Wake word** (with `--voice`) | Anywhere in CLI | Trigger STT; prompt line switches to `🎙 listening...` until trailing silence or 8 s cap |
| `/events` | Input prompt | List available event types and which are currently running |
| `/name [args]` | Input prompt | Start a named event (e.g. `/timer 30`, `/project "Build X"`) |
| `/stop [name]` | Input prompt | Stop a running event by name; `/stop` alone lists running events |

### Tray

| Key / Control | Context | Action |
|---------------|---------|--------|
| **TAB** | Input field | Cycle to the next available domain |
| **Domain button** | Header | Cycle to the next available domain |
| **Stop button** | Header (during generation) | Cancel the current generation (also stops TTS mid-sentence when voice is active) |
| **Clear button** | Header (expanded) | Clear chat view and reset conversation memory |
| **ESC** | Anywhere in panel | Hide the panel window |
| **▽ / △ button** | Header | Toggle between compact and expanded view |
| **Wake word** (with `--voice`) | Anywhere in desktop | Panel auto-opens, status spinner shows `🎙 listening...`, transcribed request submits like a typed message |
| **● Voice / ○ Voice button** (with `--voice`) | Header | Toggle STT on/off (green = listening, grey = paused). When off the wake word is ignored and the mic goes cold immediately, even mid-recording |
| `/events` | Input field | List available event types and which are currently running |
| `/name [args]` | Input field | Start a named event (e.g. `/timer 30`, `/project "Build X"`) |
| `/stop [name]` | Input field | Stop a running event by name; `/stop` alone lists running events |

---

## Usage

```
$ python andrewcli.py
Andrew is running...
[general] Ask: Write "hello" to greeting.txt
⠋ Loading: write_file
⠋ Running write_file: greeting.txt
Andrew: File greeting.txt written successfully.
[general] Ask: ↑                          # UP recalls last message
[general] Ask: [TAB]                      # TAB switches domain
Switched to domain: coding
[coding] Ask: /project "Build a REST API in Python"
✓ Event 'project' started
[coding] Ask:

◆ Event [project]: Project: Build a REST API in Python
⠋ Running write_file: pyproject.toml     # agent works autonomously
Andrew: Task 1 done — project structure created.

◆ Event [project]: Project: Build a REST API in Python
⠋ Running write_file: src/main.py        # next iteration, next task
Andrew: Task 2 done — endpoints implemented.

[coding] Ask: /stop project              # stop at any time
✓ Event 'project' stopped
[coding] Ask: /events                    # inspect running events
No events currently running.

Available slash commands:
  /file [path] [poll_interval] [message]
  /loop [goal] [max_iterations] [state_file]
  /project [goal] [state_file]
  /schedule [message] [when]
  /timer [interval] [message]
  /events              — show this list
  /stop [name]         — stop a running event
```

The agent chains tool calls automatically — a skill might instruct the LLM to read a file, transform its contents, and write the result back. The spinner shows which tool is running in real time; model reasoning inside `<think>` tags is displayed in dim italic. Events fire independently in the background and interleave cleanly with user interactions.

With `--voice`, the same CLI accepts typed and spoken prompts interchangeably. Spoken turns are echoed with a `🎙 ` prefix and the assistant's response streams through TTS in parallel with the terminal output:

```
$ python andrewcli.py --voice
Andrew is running (voice on - say 'hey_jarvis' or just type)...
[general] Ask: 🎙 listening...                       # wake word just fired
[general] Ask: 🎙 dimmi l'ora                        # transcribed utterance
Andrew: Sono le diciassette e trenta.                # streams to stdout AND speaker
[general] Ask:
```

---

## API

The FastAPI server starts automatically alongside the CLI and tray (default `http://0.0.0.0:8000`). Use `--host` / `--port` to change the address. It can also be started standalone (`--server`) to integrate external applications without a UI.

The server is a **middleware**: it does not call the LLM directly. Instead it enqueues the message for the running CLI/tray and the client polls for response tokens. This means slash commands, events, and the full domain pipeline all work exactly as if the user typed the message.

**Session lifecycle for events** — when a slash command starts an event that sends a message to the agent (e.g. `/schedule`, `/timer`, `/project`), the session stays open (`done: false`) until the event fires and the agent's response is fully streamed. The client polls the same `session_id` throughout and receives the agent's reply when the event triggers. Events without an agent message (e.g. side-effect-only triggers) close the session immediately after the confirmation token.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Queue a message; returns `{"session_id": "..."}` (202) |
| `GET` | `/chat/{session_id}` | Poll for new tokens — consumed per call; `done: true` when the full response is ready |
| `DELETE` | `/chat/{session_id}` | Discard a session and its buffered tokens |
| `GET` | `/events` | List available slash-command event types and their parameters |

### curl examples

**Send a message and poll for the response:**
```bash
# Enqueue — returns immediately with a session_id
SID=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Write hello.txt with the text Hello world"}' | jq -r .session_id)

# Poll until done (tokens are consumed on each call)
until curl -s http://localhost:8000/chat/$SID | tee /dev/stderr | jq -e '.done' > /dev/null; do
  sleep 0.5
done
```

**Fire a slash-command event and wait for the agent's response:**
```bash
# Schedule a message for 06-05-2026 at 22:46
SID=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/schedule \"Run daily summary\" 06-05-2026-22-46"}' | jq -r .session_id)

# First poll — event registered, session still open
curl -s http://localhost:8000/chat/$SID | jq .
# → {"tokens":["Event schedule started."],"done":false,"error":null}

# Keep polling the same session_id — done: true arrives with the agent's
# response when the event fires at the scheduled time.
until curl -s http://localhost:8000/chat/$SID | tee /dev/stderr | jq -e '.done' > /dev/null; do
  sleep 1
done

# Start a project loop (same pattern — session closes after each agent iteration)
SID=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/project \"Build a REST API in Python\""}' | jq -r .session_id)
until curl -s http://localhost:8000/chat/$SID | tee /dev/stderr | jq -e '.done' > /dev/null; do
  sleep 0.5
done

# List available events
curl -s http://localhost:8000/events | jq .

# List running events (via the CLI/tray inbox)
SID=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/events"}' | jq -r .session_id)
curl -s http://localhost:8000/chat/$SID | jq .
```

**Discard a session:**
```bash
curl -s -X DELETE http://localhost:8000/chat/$SID | jq .
```

---

## Extending

Everything in AndrewCLI follows the same pattern: drop a file in the right folder and it is picked up at import time.

### Add a new Tool

Drop a `*.py` file into the target domain's tools folder, e.g. `domains/general/tools/weather.py`:

```python
from src.core.tool import Tool

class GetWeather(Tool):
    name: str = "get_weather"
    description: str = "Fetch the current weather for a city."

    def execute(self, city: str, units: str = "metric") -> str:
        return f"it's sunny in {city}"
```

Every concrete `Tool` subclass declared in any `domains/<name>/tools/*.py` is instantiated automatically — no registration list to update. The OpenAI function-call schema is derived from the `execute()` signature and type hints.

### Add a new Skill

Drop a markdown file into the target domain's skills folder, e.g. `domains/general/skills/my_skill.md`:

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

`tools:` is optional — list any tools the skill requires that the router might not select automatically; they will be injected into the prompt whenever the skill is selected. No Python subclass is needed: the markdown file *is* the skill.

### Add a new Domain

Create a new subpackage under `domains/` (e.g. `domains/research/`):

```
domains/research/
├── __init__.py            # empty
├── domain.py              # ResearchDomain class
├── system_prompt.md       # the prompt
├── tools/                 # optional — auto-discovered *.py
│   └── __init__.py
└── skills/                # optional — auto-discovered *.md
    └── __init__.py
```

`domain.py` only needs to declare the Python class (tools, skills, and the prompt are all loaded from the filesystem):

```python
from src.core.domain import Domain

class ResearchDomain(Domain):
    pass
```

And `system_prompt.md`:

```markdown
You are a research assistant. Cite sources whenever possible.
```

Set `domain: "research"` in `config.yaml`. The folder name must match the config value; the class must be named `<Name>Domain`.

**Per-domain LLM** — add `api_base_url` and/or `model` class attributes to point a domain at a different endpoint or model than the global defaults. Both are optional and fall back to the `API_BASE_URL` / `MODEL` env vars when omitted:

```python
class ResearchDomain(Domain):
    api_base_url: str = "http://localhost:11434/v1"  # different server
    model: str = "llama3:8b"                         # different model
    routing_enabled: bool = False                    # expose every tool every turn
```

### Add a new Event

Create a file in `events/` (e.g. `events/my_event.py`):

```python
import asyncio
from src.core.event import Event

class MyEvent(Event):
    name = "my_event"          # becomes the slash command: /my_event [arg]
    description = "Short description shown in notifications"
    message = "Prompt sent to the agent when this event fires."

    def __init__(self, arg: str = "default"):
        self.arg = arg
        self.description = f"MyEvent with arg={arg}"

    async def condition(self):
        await asyncio.sleep(60)  # block until condition is met

    async def trigger(self):
        pass  # optional side-effect before the agent message
```

The event is **auto-discovered** the moment the file is saved — no import or registration needed. Activate it at runtime from any domain:

```
/my_event hello          → MyEvent("hello")
/my_event                → MyEvent()   (uses default)
```

Events are decoupled from domains — the same catalog is available everywhere and is added to the running `EventBus` dynamically via `EventBus.add()`. Events with a dynamic `message` property (computed from state rather than a fixed string) are supported: the bus reads `event.message` after `trigger()` returns, so the value can change between iterations. See `ProjectEvent` for an example.

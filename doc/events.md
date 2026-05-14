# Events

An **Event** is a self-contained background observer. Each event runs as an asyncio task inside a domain's `EventBus` — which starts empty and is populated at runtime via slash commands (`/timer 30`, `/project "..."`) or the HTTP API. Events are decoupled from domain definitions: the same `events/` catalog is available from every domain.

---

## How Events Work

Each event defines two things:

- **`condition()`** — an async coroutine that blocks until the triggering condition is met. It can be a sleep, a file-modification check, a queue wait, or any awaitable.
- **`trigger()`** — called once `condition()` returns. Performs any side-effect needed before the agent message is sent.

If the event sets a `message` string, the `EventBus` automatically dispatches it to the agent via `domain.generate_event()` — a fresh, isolated LLM call that never touches the conversation memory or affects routing for user queries.

### Serialization via `Domain.busy_lock`

Both `domain.generate()` (user turns) and `domain.generate_event()` (event dispatches) acquire a single `asyncio.Lock` on the domain:

- Only one agent interaction streams to the UI at a time.
- Events queue **FIFO** behind each other and behind any in-flight user turn.
- A timer event cannot pile up on itself — each event's `_run` loop awaits the full `condition → trigger → notify → dispatch` chain before re-arming.
- The CLI and tray inherit identical behavior without per-surface locks.

### Notification

When an event fires, both surfaces are notified:
- **CLI**: a colored banner (`◆ Event [name]`) is printed, followed by the agent's streamed response.
- **Tray**: a system tray balloon message appears and the panel opens to show the response.

Event responses are rendered exactly like user-initiated responses (routing, spinner, tool calls) but use an isolated LLM instance so the conversation memory is never polluted.

---

## Slash Commands

Events are activated at runtime via slash commands — no domain restart required. The same syntax works in the CLI input prompt, the tray panel input field, and the server's `/chat` endpoint.

```
/events                              — list available events and which are running
/stop [name]                         — stop a running event by name
/stop                                — list the names of currently running events

/timer 30                            → TimerEvent(interval=30.0)
/file war_news/news.md 5.0 "Update!" → FileEvent("war_news/news.md", 5.0, "Update!")
/monitor "curl -sL https://example.com" 300
/project "Build a REST API in Python"
/project                              → resume from project_state.json
/loop "Monitor oil price; stop if < $100"
/loop "Poll /status until ready" 30
/loop                                  → resume from loop_state.json
/schedule "Run skill 1" 10-10-2026-11-30
```

Quoted strings with spaces are handled correctly (`shlex` tokenisation). Arguments are coerced to their annotated types (`float`, `int`, or `str`). Extra arguments beyond the declared parameters are ignored; missing optional parameters fall back to their defaults.

---

## EventBus API

| Method | Description |
|--------|-------------|
| `add(event)` | Start a new event on the already-running bus. Creates an independent asyncio task tracked by `stop()`. |
| `remove(name)` | Cancel and remove the event with the given name. Returns `True` if found. |
| `running()` | Return a list of names of currently active (non-done) events. |

---

## Built-in Events

| Event | Slash command | Description |
|-------|---------------|-------------|
| `TimerEvent` | `/timer [interval]` | Fires every N seconds |
| `FileEvent` | `/file [path] [poll_interval] [message]` | Fires when a watched file is modified |
| `MonitorEvent` | `/monitor [command] [capture_interval] [poll_interval] [timeout]` | Runs a shell command periodically and fires when its output changes |
| `ProjectEvent` | `/project [goal] [state_file]` | Drives the agent through a multi-step project |
| `LoopEvent` | `/loop [goal] [max_iterations] [state_file]` | Drives a `do X until Y` loop |
| `ScheduleEvent` | `/schedule [message] [when]` | Fires once at a specific datetime (`dd-mm-yyyy-hh-mm`) |

---

## ProjectEvent — autonomous project loop

`ProjectEvent` drives the agent through a multi-step coding or research project without human intervention.

### Lifecycle

1. **Iteration 0 (planning)** — no state file exists yet. The agent extracts constraints from the goal and breaks it into concrete tasks, writing both to `state_file` as JSON, then starts task 1.
2. **Iterations 1…N (execution)** — the next pending task is passed to the agent, which completes it and marks it `done: true` in `state_file`.
3. **Final iteration** — all tasks are done. The agent summarises what was built and confirms each constraint was honoured, then the event stops.

### Resume mode

Invoking `/project` with no goal reads the existing `state_file`, recovers the goal from its `"goal"` field, and picks up at the next pending task. If no state file exists, the event raises a clear error pointing back to `/project <goal>`.

### State schema

```json
{
  "goal": "Build a REST API in Python",
  "constraints": ["Use only the standard library", "All endpoints under /api/v1"],
  "tasks": [
    {"id": 1, "title": "Set up project structure", "done": true,
     "artefacts": ["pyproject.toml", "src/api/__init__.py"]},
    {"id": 2, "title": "Implement endpoints", "done": false,
     "subtasks": ["GET /users", "POST /users"]},
    {"id": 3, "title": "Write tests", "done": false}
  ],
  "build_log": ["installed deps", "scaffolded routes"]
}
```

### State rigor

The file is parsed via a Pydantic 2 model (`ProjectState` / `ProjectTask`) so wrong field names like `is_done` / `completed` / `finished` are accepted as aliases for `done`, integer task ids are coerced to strings, and malformed payloads fall back to a fresh plan. The agent only owns the `done` flags. On the first read of a planned file, `ProjectEvent` snapshots `goal`, `constraints`, and the task `id`/`title` pairs; subsequent reads rebuild the state from the snapshot, importing only `done` flags from disk. Done flags are **monotonic** — once `true`, they stay `true`.

| Field type | Owner | Mutability |
|---|---|---|
| `goal`, `constraints`, task `id`/`title`, task list shape | Loop | Immutable (snapshot-restored) |
| task `done` flags | Agent | Monotonic (`false → true` only) |
| Any other field | Agent | Free-form (preserved verbatim) |

---

## LoopEvent — do X until Y is met

`LoopEvent` is the counterpart to `ProjectEvent` for goals shaped as *"keep doing X until Y"* rather than *"complete tasks A, B, C"*. Examples: monitor a metric until it crosses a threshold, poll an endpoint until it returns ready, retry an action until it succeeds.

### Lifecycle

1. **Iteration 0 (planning)** — no state file exists yet. The agent extracts the single repeating `action` and the list of `exit_criteria`, writes them to `state_file`, then performs iteration 1.
2. **Iterations 1…N (execution)** — each iteration the agent performs the action once, records a `last_observation`, and evaluates every exit criterion. If any criterion is met it sets `terminated: true` and writes a `termination_reason`.
3. **Final iteration** — an exit criterion fired or the iteration cap was reached. The agent summarises the run and the event stops.

### Resume mode

Invoking `/loop` with no goal reads the existing `state_file` and continues from the last recorded iteration count. If a `max_iterations` argument is passed on resume, it overrides whatever was on disk.

### State schema

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
  "rolling_avg": 106.8
}
```

### State rigor

Field aliases (`iterations_done`, `iter_count`, `iters`) are accepted for `iterations`. The literal string `"unlimited"` is coerced to `null` for `max_iterations`. Malformed payloads fall back to a fresh plan. The immutable fields (`goal`, `action`, `exit_criteria`, `max_iterations`) are snapshot-restored on every read.

| Field | Mutability |
|---|---|
| `iterations` | Monotonic — on-disk decreases are ignored |
| `terminated` | Sticky — `false → true` only |
| `last_observation`, `termination_reason` | Free-form, agent-owned |
| Any other field | Free-form (preserved verbatim) |

### Optional `max_iterations` cap

| `max_iterations` value | Behaviour |
|---|---|
| `0` (default) or `null` on disk | Uncapped — only an exit criterion can stop the loop |
| positive integer | Hard cap; loop also stops when `iterations >= max_iterations` |

Use the cap as a safety net for goals where local models may misjudge the exit criterion. Resolution order on every read: **user-passed cap → on-disk cap → uncapped**.

### ProjectEvent vs LoopEvent

Use `ProjectEvent` for goals that decompose into a finite checklist ("build X, write Y, deploy Z"); use `LoopEvent` for goals shaped as a predicate ("keep doing X until Y").

---

## MonitorEvent — universal change-detection via shell

`MonitorEvent` is the general answer to *"watch X and tell me when it changes"* for any *X* whose state can be captured by a shell command.

### Mechanism

On a fixed `capture_interval` it runs the user-supplied command in a thread executor, frames the output as:

```
exit_code: <rc>
--stdout--
<stdout>
--stderr--
<stderr>
```

hashes it, and writes the snapshot file **only when the hash changed**. The watched file's mtime therefore advances exactly once per real change. The first capture is performed inline before the watcher arms, so the initial population is **not** reported as a change. Including the exit code in the snapshot means a command that starts failing (e.g. `systemctl is-active` flipping from `active` to `failed`) is itself a state-change signal.

### Parameters

`/monitor [command] [capture_interval] [poll_interval] [timeout]`

Defaults: `capture_interval=60s`, `poll_interval=2s`, `timeout=30s`. The snapshot file defaults to `$TMPDIR/andrewcli-monitor-<sha1[:12]>.snap` (deterministic per-command, so successive runs of the same monitor resume from the prior state).

### Examples

| Use case | Slash command |
|---|---|
| Website body | `/monitor "curl -sL https://example.com" 300` |
| API healthcheck | `/monitor "curl -sL -w '\n%{http_code}' https://api/health" 60` |
| API JSON field only | `/monitor "curl -s https://api/status \| jq -r .state" 60` |
| Disk usage | `/monitor "df -h /" 600` |
| Repo state | `/monitor "git -C /repo status --porcelain" 30` |
| Service liveness | `/monitor "systemctl is-active nginx" 60` |
| Container state | `/monitor "podman ps --format json" 30` |
| RSS feed | `/monitor "curl -s https://blog/rss.xml" 600` |
| Local file checksum | `/monitor "sha256sum /etc/hosts" 120` |
| DB query | `/monitor "psql -tAc 'select count(*) from jobs where state=''queued'''" 60` |

### Extending by subclass

Most use cases need no code. Subclass `MonitorEvent` and override `_run_sync()` (or `_capture()` for an async-native source) if you need something the shell can't express, e.g. a Python SDK call. Everything else (snapshot file, hashing, atomic write, first-capture priming, `FileEvent` watcher, capture-task lifecycle) is inherited unchanged.

---

## Event Auto-Discovery

Every file dropped in `~/.config/andrewcli/events/` that defines a concrete `Event` subclass with a `name` string attribute is automatically registered. No manual import or registration step needed. The unified registry (`src/core/registry.py`) is scanned at command parse time, so new events are available immediately after saving the file.

### Defining a new event

```python
import asyncio
from src.core.event import Event

class MyEvent(Event):
    name = "my_event"          # becomes the slash command: /my_event [args]
    description = "Fires when something happens"
    message = "Something happened, please respond."

    def __init__(self, arg: str = "default"):
        self.arg = arg

    async def condition(self):
        await asyncio.sleep(30)  # or watch a file, wait on a queue, etc.

    async def trigger(self):
        pass  # optional side-effect before the agent message
```

Drop the file in `~/.config/andrewcli/events/` — it is discovered automatically and immediately available as `/my_event [args]`. Events with a dynamic `message` property (computed from state rather than a fixed string) are supported: the bus reads `event.message` after `trigger()` returns.

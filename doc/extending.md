# Extending AndrewCLI

Everything in AndrewCLI follows the same pattern: drop a file in the right folder and it is picked up on the next user turn — no restart required.

---

## Add a new Tool

Drop a `*.py` file into the target domain's tools folder, e.g. `~/.config/andrewcli/domains/general/tools/weather.py`:

```python
from src.core.tool import Tool

class GetWeather(Tool):
    name: str = "get_weather"
    description: str = "Fetch the current weather for a city."

    def execute(self, city: str, units: str = "metric") -> str:
        return f"it's sunny in {city}"
```

Every concrete `Tool` subclass declared in any `~/.config/andrewcli/domains/<name>/tools/*.py` is instantiated automatically — no registration list to update. The OpenAI function-call schema is derived from the `execute()` signature and type hints. The base `Tool.run()` wrapper catches exceptions and returns a `[Tool Error]` string so the agent can recover without crashing.

---

## Add a new Skill

Drop a markdown file into the target domain's skills folder, e.g. `~/.config/andrewcli/domains/general/skills/my_skill.md`:

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

---

## Add a new Domain

Create a new folder under `~/.config/andrewcli/domains/` (e.g. `~/.config/andrewcli/domains/research/`):

```
~/.config/andrewcli/domains/research/
├── __init__.py            # empty (required so tools/ can be imported as a package)
├── config.yaml            # optional — overrides global settings
├── system_prompt.md       # the prompt
├── tools/                 # optional — auto-discovered *.py
│   └── __init__.py
├── skills/                # optional — auto-discovered *.md
└── workflows/             # optional — auto-discovered *.py
    └── __init__.py
```

No Python subclass is required — the folder *is* the domain. Write `system_prompt.md`:

```markdown
You are a research assistant. Cite sources whenever possible.
```

Then optionally add per-domain overrides to `config.yaml`:

```yaml
api_base_url: "http://localhost:11434/v1"   # different server than the global default
model: "llama3:8b"                          # different model
routing_enabled: false                      # expose every tool every turn
```

Missing keys fall back to the global `~/.config/andrewcli/config.yaml`, then to the `API_BASE_URL` / `MODEL` env vars, then to the `Domain` class-level defaults.

Set `domain: "research"` in the global `config.yaml` to make it the active domain. The folder name must match the config value.

---

## Add a new Workflow

Workflows are scripted pipelines invoked directly via slash command — they run **outside the LLM agent loop**. Use them for deterministic tasks where you want full control over the execution flow, with optional single-shot LLM inference for dynamic steps.

Drop a `*.py` file into the target domain's `workflows/` folder, e.g. `~/.config/andrewcli/domains/general/workflows/my_workflow.py`:

```python
from src.core.workflow import Workflow

class MyWorkflow(Workflow):
    name = "my_workflow"          # becomes the slash command: /my_workflow [args]
    description = "Short description shown in /workflows"

    async def run(self, path: str, detail: str = "brief"):
        # Deterministic steps run as plain Python — no LLM involved
        yield f"Processing `{path}` (detail={detail})...\n\n"

        content = open(path).read()

        # Optional: one-shot LLM inference for dynamic parts
        async for token in self.infer(
            [{"role": "user", "content": content}],
            system=f"Produce a {detail} analysis.",
        ):
            yield token
```

**Key points:**

- `run()` is an **async generator** that yields `str` tokens — these stream directly to the CLI/tray renderer, so the user sees output as it arrives.
- Parameters on `run()` map to command-line arguments. Type hints (`str`, `int`, `float`) are used for automatic coercion. Parameters with defaults are optional.
- `self.infer(messages, system="")` makes a single streaming LLM call without entering the agent tool-call loop. It raises `RuntimeError` if called before the workflow is configured by the domain (this happens automatically).
- Workflows acquire the domain's `busy_lock`, so they are serialised with user turns and background events.
- The `workflows/` folder must contain an `__init__.py` (can be empty) to be importable as a package.

Invoke the workflow at runtime:

```
/my_workflow report.txt            → run(path="report.txt")
/my_workflow report.txt detailed   → run(path="report.txt", detail="detailed")
/workflows                         → list all workflows in the active domain
```

The workflow is **auto-discovered** on the next user turn — no restart required. A built-in example (`/summarize [path]`) is included in the `general` domain.

---

## Add a new Event

Events live in `~/.config/andrewcli/events/`. They are auto-discovered the moment the file is saved — no import or registration needed.

Choose your base class based on what the event needs to do:

### Simple event — extend `Event`

Use this for one-shot triggers, timers, or file watchers: anything that fires once per condition and sends a fixed message.

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

### Persistent event — extend `StatefulEvent`

Use this for any event that runs the agent through **multiple iterations** with progress tracked in a JSON file — retries, polling loops, multi-step projects, and so on. `StatefulEvent` handles the file lifecycle, broken-file repair, planning poll, and prompt caching automatically.

```python
import re
from pydantic import BaseModel
from src.core.event import StatefulEvent

class RetryState(BaseModel):
    goal: str = ""
    command: str = ""
    attempts: int = 0
    succeeded: bool = False
    last_output: str = ""

class RetryEvent(StatefulEvent):
    name = "retry"
    description = "Retry a command until it succeeds"
    _session_files: set = set()            # must be re-declared on every subclass
    _state_file_default: str = "retry_state.json"

    def __init__(self, goal: str = "", state_file: str = "retry_state.json"):
        self._init_state_file(goal, state_file)
        self._succeeded = False
        self.goal = goal
        self.description = f"Retry: {goal[:60]}"

    def _parse(self) -> RetryState | None:
        raw = self._load_raw()
        if raw is None:
            return None
        try:
            return RetryState.model_validate(raw)
        except Exception:
            return None

    def _reconcile_raw(self, raw: dict) -> dict:
        if self._snapshot:
            raw["goal"]    = self._snapshot["goal"]
            raw["command"] = self._snapshot["command"]
        if raw.get("succeeded"):
            self._succeeded = True
        raw["succeeded"] = self._succeeded
        raw["attempts"]  = max(raw.get("attempts", 0), getattr(self, "_attempts", 0))
        self._attempts   = raw["attempts"]
        return raw

    def _salvage_progress(self, text: str) -> None:
        if re.search(r'"succeeded"\s*:\s*true', text):
            self._succeeded = True
        m = re.search(r'"attempts"\s*:\s*(\d+)', text)
        if m:
            self._attempts = int(m.group(1))

    def _rebuild_raw(self) -> dict:
        snap = self._snapshot or {}
        return {
            "goal":       snap.get("goal",    self.goal),
            "command":    snap.get("command", ""),
            "attempts":   getattr(self, "_attempts", 0),
            "succeeded":  self._succeeded,
            "last_output": "",
        }

    def _compute_both(self) -> tuple[str, str]:
        state = self._parse()

        if state is None or not state.command:
            if self._plan_sent:
                return "", ""
            self._plan_sent = True
            system = f"Goal: {self.goal}\nIdentify the command and write the initial state to {self.state_file}."
            return system, f"Write the initial state to '{self.state_file}'."

        if self._snapshot is None:
            self._snapshot = {"goal": state.goal, "command": state.command}
            self._attempts = state.attempts

        if self._succeeded or state.succeeded:
            self._summary_sent = True
            return "The command succeeded.", "Write a one-sentence confirmation and stop."

        attempts = getattr(self, "_attempts", state.attempts)
        system = (
            f"Goal: {state.goal}\n"
            f"Command: {state.command}\n"
            f"Attempt {attempts + 1}: run the command and write the result to {self.state_file}.\n"
            f"Set succeeded=true if it worked, leave false otherwise."
        )
        return system, f"Run the command and update '{self.state_file}'."
```

Activate at runtime:

```
/retry "get the build green"
/retry                          → resume (auto-detects state file)
```

The five abstract methods and their full contract are documented in [events.md](events.md#stateulevent--template-for-persistent-multi-iteration-events).

### Common rules for both

- Events are decoupled from domains — the same catalog is available everywhere.
- Activate with `/my_event [args]`; stop with `/stop my_event`.
- A dynamic `message` property (computed from state) is supported — the bus reads it after `trigger()` returns.

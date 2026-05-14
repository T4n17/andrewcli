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
└── skills/                # optional — auto-discovered *.md
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

## Add a new Event

Create a file in `~/.config/andrewcli/events/` (e.g. `~/.config/andrewcli/events/my_event.py`):

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

The event is **auto-discovered** the moment the file is saved — no import or registration needed. Activate it at runtime:

```
/my_event hello          → MyEvent("hello")
/my_event                → MyEvent()   (uses default)
```

Events are decoupled from domains — the same catalog is available everywhere and is added to the running `EventBus` dynamically via `EventBus.add()`. Events with a dynamic `message` property (computed from state rather than a fixed string) are supported: the bus reads `event.message` after `trigger()` returns.

# AndrewCLI

A lightweight, fully async CLI agent in Python — no bloated abstractions or unnecessary features that skyrocket your token usage.

## Project Structure

```
AndrewCLI/
├── app.py                          # Entry point — async REPL, domain switching, input history
├── config.yaml                     # Configuration file (domain, execute_bash_automatically)
├── requirements.txt                # Python dependencies
└── src/
    ├── __init__.py
    ├── shared/
    │   └── config.py               # Config class — loads config.yaml
    ├── core/                       # Framework internals
    │   ├── domain.py               # Base Domain class (async generator)
    │   ├── llm.py                  # Async LLM client with streaming + tool-calling loop + ToolEvent
    │   ├── memory.py               # Rolling memory with background summarization
    │   ├── skill.py                # Base Skill class (markdown-defined tools)
    │   └── tool.py                 # Base Tool class with run() error wrapper
    ├── ui/                         # UI rendering layer
    │   ├── animations.py           # Spinner class (async, dynamic status)
    │   ├── filter.py               # ThinkFilter — parses <think> tags for reasoning display
    │   └── renderer.py             # StreamRenderer — orchestrates spinner, filtering, streaming
    ├── tools/                      # Reusable tool definitions
    │   └── common.py               # WriteFile, ReadFile, ExecuteCommand, GetCurrentDate
    ├── skills/
    │   ├── myskills.py             # Skill subclass definitions (Example)
    │   └── skills_files/           # Skill instruction markdown files
    │       └── example.md
    └── domains/                    # Domain definitions
        ├── general.py              # General-purpose domain
        └── coding.py               # Coding-focused domain (WIP)
```

## Architecture

AndrewCLI is built around six core concepts:

### Config

A centralized **Config** class (`src/shared/config.py`) loads `config.yaml` and exposes settings as attributes. Used by `app.py` to select the active domain and by tools like `ExecuteCommand` to read `execute_bash_automatically`.

### Domains

A **Domain** groups a system prompt, a set of tools, and a set of skills into a single persona. Domains are defined as Python classes in `src/domains/` and loaded dynamically based on `config.yaml`. The `generate()` method is an async generator that yields tokens as they stream in. Domains can be **switched at runtime** by pressing TAB.

### Tools

A **Tool** is a Python class that the LLM can call. Tools auto-generate their OpenAI function schema from the `execute()` method's type hints — no manual schema boilerplate needed. Tool definitions live in `src/tools/` and are imported into domains.

The base `Tool` class provides a `run()` wrapper around `execute()` that catches exceptions and returns a `[Tool Error]` string instead of crashing the agent. The LLM receives the error as a tool result and can recover gracefully.

### Skills

A **Skill** is a markdown-defined tool. Instead of executing code, it returns a set of natural-language instructions that the LLM follows using the available tools. Skill subclasses are defined in `src/skills/myskills.py` and point to `.md` files in `src/skills/skills_files/` with YAML frontmatter:

```markdown
---
name: example
description: Execute an example skill
---

# Instructions
1. Do something using the available tools
2. Acknowledge the user
```

When a skill is invoked, its instructions are returned with a `[SKILL INSTRUCTIONS]` prefix that directs the LLM to execute each step using tools rather than just summarizing them.

### Memory

A **rolling memory** system that maintains context across turns without growing the message history indefinitely.

- After each turn, the last 1500 characters of conversation are extracted and merged with an existing summary using an LLM summarization call.
- The merged summary (~500 words max) is persisted to `~/.andrewcli/data/memory.json`.
- Older messages are trimmed — only the current turn is kept in the message array.
- The summary is injected into the system prompt inside `<memory>` tags so the model always has context.
- The merge LLM call runs as a **fire-and-forget background task** (`asyncio.create_task`), so the user gets the next prompt immediately. Sequential merges are serialized to prevent overwrites.

### UI Layer (`src/ui/`)

All rendering and animation logic is separated from the core agent into a dedicated `src/ui/` package:

- **`Spinner`** (`animations.py`) — an async spinner with a dynamic `.status` property. Shows what the agent is doing in real time: `⠴ Thinking...`, `⠧ Running execute_command: ls -la`, `⠋ Running read_file: config.yaml`.
- **`ThinkFilter`** (`filter.py`) — streaming parser for `<think>...</think>` tags. Handles tags split across token boundaries. Used to render model reasoning in dim italic (`\033[2;3m`) while keeping the final answer in normal text.
- **`StreamRenderer`** (`renderer.py`) — orchestrates the full output pipeline: manages the spinner lifecycle, processes `ToolEvent`s from the LLM to update spinner status with tool name and arguments, applies the think filter, streams tokens char-by-char with a typewriter effect, and handles ESC-to-stop.

## Async Pipeline

The entire I/O pipeline is non-blocking:

1. **`app.py`** — runs under `asyncio.run()`. User input is read via a custom async `_read_input()` using cbreak mode, supporting TAB (domain switch) and UP/DOWN (history navigation).
2. **Spinner** — a `Spinner` asyncio task that animates and dynamically updates its status text based on `ToolEvent`s from the LLM.
3. **Streaming** — `LLM.generate()` is an async generator using `AsyncOpenAI`. Tokens are yielded as they arrive from the API. `ToolEvent` objects are also yielded to signal tool execution status.
4. **Tool calls** — accumulated from streamed chunks, executed via `tool.run()`, and looped back to the API automatically. Malformed JSON arguments are caught and reported as errors instead of crashing. Each tool execution yields a `ToolEvent` with the tool name and arguments so the spinner can show what's happening.
5. **Memory summarization** — fires in the background after the response completes; no user-facing delay.

## Interactive Controls

| Key | Context | Action |
|-----|---------|--------|
| **TAB** | Input prompt | Cycle to the next available domain |
| **UP/DOWN** | Input prompt | Navigate through command history |
| **ESC** | During response | Stop output streaming (background tasks still complete) |

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your LLM endpoint** via environment variables:

   | Variable         | Default                      | Description                                      |
   |------------------|------------------------------|--------------------------------------------------|
   | `API_BASE_URL`   | `http://localhost:8080/v1`   | OpenAI-compatible API URL                        |
   | `MODEL`          | `qwen3.5:9B`                | Model name                                       |
   | `OPENAI_API_KEY` | —                            | API key (required even for local models)         |

3. **Configure `config.yaml`:**

   ```yaml
   domain: "general"
   execute_bash_automatically: true
   ```

   | Key                          | Default     | Description                                      |
   |------------------------------|-------------|--------------------------------------------------|
   | `domain`                     | `"general"` | Active domain (matches filename in `src/domains/`)|
   | `execute_bash_automatically` | `false`     | Skip confirmation prompt for shell commands       |

4. **Run:**

   ```bash
   python app.py
   ```

## Usage

```
$ python app.py
Andrew is running...
[general] Ask: Write "hello" to greeting.txt
⠋ Running write_file: greeting.txt
Andrew: File greeting.txt written successfully.
[general] Ask: ↑                          # press UP to recall last message
[general] Ask: [TAB]                      # press TAB to switch domain
Switched to domain: coding
[coding] Ask:
```

The agent can chain tool calls automatically — for example, a skill might instruct the LLM to read a file, transform its contents, and write the result back. The spinner shows what tool is running in real time, then tokens stream in with a typewriter effect. Model reasoning (inside `<think>` tags) is displayed in dim italic.

## Extending

### Add a new Tool

Create a class that inherits from `Tool` in `src/tools/`:

```python
from src.core.tool import Tool

class MyTool(Tool):
    name: str = "my_tool"
    description: str = "Does something useful."

    def execute(self, arg1: str, arg2: int = 0) -> str:
        # your logic here
        return "result"
```

Then import and add `MyTool()` to your domain's `tools` list.

### Add a new Skill

1. Create a markdown file in `src/skills/skills_files/`:

   ```markdown
   ---
   name: my_skill
   description: What this skill does
   ---

   # Instructions
   1. Step one
   2. Step two
   ```

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
```

Then set `domain: "research"` in `config.yaml`. The domain is loaded dynamically — the file name must match the config value, and the class must be named `<Name>Domain`.

## TODO

- [ ] Write a skill that allows AndrewCLI to update itself with new tools, skills, or domains
- [ ] Implement GUI mode (`--gui` flag)


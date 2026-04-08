# AndrewCLI

A lightweight, fully async CLI agent in Python — no bloated abstractions or unnecessary features that skyrocket your token usage.

## Project Structure

```
AndrewCLI/
├── app.py                  # Entry point — async REPL with spinner and streaming
├── config.yaml             # Configuration file (set active domain here)
├── requirements.txt        # Python dependencies
└── src/
    ├── __init__.py
    ├── core/               # Framework internals
    │   ├── domain.py       # Base Domain class (async generator)
    │   ├── llm.py          # Async LLM client with streaming + tool-calling loop
    │   ├── memory.py       # Rolling memory with background summarization
    │   ├── skill.py        # Base Skill class (markdown-defined tools)
    │   └── tool.py         # Base Tool class (code-defined tools)
    ├── domains/            # Domain definitions
    │   ├── general.py      # General-purpose domain (WriteFile, ReadFile, etc.)
    │   └── coding.py       # Coding-focused domain (WIP)
    └── skills/             # Skill instruction files
        └── example.md      # Example skill definition
```

## Architecture

AndrewCLI is built around four core concepts:

### Domains

A **Domain** groups a system prompt, a set of tools, and a set of skills into a single persona. Domains are defined as Python classes in `src/domains/` and loaded dynamically based on `config.yaml`. The `generate()` method is an async generator that yields tokens as they stream in.

### Tools

A **Tool** is a Python class that the LLM can call. Tools auto-generate their OpenAI function schema from the `execute()` method's type hints — no manual schema boilerplate needed.

### Skills

A **Skill** is a markdown-defined tool. Instead of executing code, it returns a set of natural-language instructions that the LLM follows using the available tools. Skills are defined as `.md` files in `src/skills/` with YAML frontmatter:

```markdown
---
name: example
description: Execute an example skill
---

# Instructions
1. Do something using the available tools
2. Acknowledge the user
```

### Memory

A **rolling memory** system that maintains context across turns without growing the message history indefinitely.

- After each turn, the last 1500 characters of conversation are extracted and merged with an existing summary using an LLM summarization call.
- The merged summary (~500 words max) is persisted to `~/.andrewcli/data/memory.json`.
- Older messages are trimmed — only the current turn is kept in the message array.
- The summary is injected into the system prompt inside `<memory>` tags so the model always has context.
- The merge LLM call runs as a **fire-and-forget background task** (`asyncio.create_task`), so the user gets the next prompt immediately. Sequential merges are serialized to prevent overwrites.

## Async Pipeline

The entire I/O pipeline is non-blocking:

1. **`app.py`** — runs under `asyncio.run()`. User input is read via `run_in_executor` so the event loop stays free.
2. **Spinner** — an `asyncio.Task` that animates until the first token arrives, then gets cancelled.
3. **Streaming** — `LLM.generate()` is an async generator using `AsyncOpenAI`. Tokens are yielded as they arrive from the API.
4. **Tool calls** — accumulated from streamed chunks, executed, and looped back to the API automatically.
5. **Memory summarization** — fires in the background after the response completes; no user-facing delay.

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your LLM endpoint** via environment variables:

   | Variable       | Default                      | Description               |
   |----------------|------------------------------|---------------------------|
   | `API_BASE_URL` | `http://localhost:8080/v1`   | OpenAI-compatible API URL |
   | `MODEL`        | `qwen3.5:9B`                | Model name                |

3. **Set the active domain** in `config.yaml`:

   ```yaml
   domain: "general"
   ```

4. **Run:**

   ```bash
   python app.py
   ```

## Usage

```
$ python app.py
Andrew is running... (Domain: general)
Ask: Write "hello" to greeting.txt
⠋ Thinking...
Andrew: File greeting.txt written successfully. ...
```

The agent can chain tool calls automatically — for example, a skill might instruct the LLM to read a file, transform its contents, and write the result back. A spinner animates while the model is processing, then tokens stream in with a typewriter effect.

## Extending

### Add a new Tool

Create a class that inherits from `Tool` in your domain file:

```python
from src.core.tool import Tool

class MyTool(Tool):
    name: str = "my_tool"
    description: str = "Does something useful."

    def execute(self, arg1: str, arg2: int = 0) -> str:
        # your logic here
        return "result"
```

Then add `MyTool()` to your domain's `tools` list.

### Add a new Skill

1. Create a markdown file in `src/skills/`:

   ```markdown
   ---
   name: my_skill
   description: What this skill does
   ---

   # Instructions
   1. Step one
   2. Step two
   ```

2. Create a `Skill` subclass and add it to your domain's `skills` list:

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


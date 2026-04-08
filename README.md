# AndrewCLI

A lightweight CLI agent in Python, designed to be as easy as possible — no bloated abstractions or unnecessary features that skyrocket your token usage.

## Project Structure

```
AndrewCLI/
├── app.py                  # Entry point — loads config, domain, and runs the REPL
├── config.yaml             # Configuration file (set active domain here)
├── requirements.txt        # Python dependencies
└── src/
    ├── __init__.py
    ├── core/               # Framework internals
    │   ├── domain.py       # Base Domain class
    │   ├── llm.py          # LLM client with tool-calling loop
    │   ├── memory.py       # Conversation memory (message history)
    │   ├── skill.py        # Base Skill class (markdown-defined tools)
    │   └── tool.py         # Base Tool class (code-defined tools)
    ├── domains/            # Domain definitions
    │   ├── general.py      # General-purpose domain (WriteFile, ReadFile, etc.)
    │   └── coding.py       # Coding-focused domain (WIP)
    └── skills/             # Skill instruction files
        └── habit.md        # Example skill definition
```

## Architecture

AndrewCLI is built around three core concepts:

### Domains

A **Domain** groups a system prompt, a set of tools, and a set of skills into a single persona. Domains are defined as Python classes in `src/domains/` and loaded dynamically based on `config.yaml`.

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
Andrew: File greeting.txt written successfully. ...
```

The agent can chain tool calls automatically — for example, a skill might instruct the LLM to read a file, transform its contents, and write the result back.

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


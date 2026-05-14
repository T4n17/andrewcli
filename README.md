# AndrewCLI

A lightweight, fully async Python agent — designed to **keep your context clean**.

Local models degrade fast as context grows: reasoning gets muddier, tool calls go off the rails, and token budgets hit the ceiling. AndrewCLI fights this with two mechanisms that run on every turn:

- **Rolling memory** — after each response, messages are trimmed to just the last exchange and replaced with a compact `n-word` summary injected into the system prompt.
- **Context-aware router** — before each generation, an LLM-based classifier picks only the tools and skills needed for the request. Irrelevant schemas never reach the generation prompt.

All three surfaces — CLI, system tray, and HTTP API — share the same core: same domains, same memory, same router, and the same serialization lock that keeps user turns and background events ordered.

---

## Install

```bash
pip install -e .
```

After installation the `andrewcli` command is available system-wide.

---

## Configure

Set your LLM endpoint via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible API URL |
| `MODEL` | `qwen3.5:9B` | Main chat model |
| `SUMMARY_MODEL` | same as `MODEL` | Smaller model for background memory summarization |
| `OPENAI_API_KEY` | `local` | API key — defaults to `"local"` for local servers |

The first time you run `andrewcli`, `~/.config/andrewcli/` is auto-created and seeded with default domains, events, and a `config.yaml`. Edit anything in there freely — AndrewCLI never overwrites existing files.

Key settings in `~/.config/andrewcli/config.yaml`:

```yaml
domain: "general"

memory:
  enabled: true            # set false to disable rolling summary entirely
  min_summary_chars: 200   # turns shorter than this skip the LLM merge

server:
  enabled: true            # auto-start the FastAPI bridge with the CLI/tray
```

---

## Run

```bash
# CLI mode — API server auto-starts on 0.0.0.0:8000
python andrewcli.py

# System tray GUI — API server auto-starts inside the tray subprocess
python andrewcli.py --tray

# Custom API host/port
python andrewcli.py --host 127.0.0.1 --port 9000
python andrewcli.py --tray --host 127.0.0.1 --port 9000

# Standalone FastAPI server (no CLI or tray)
python andrewcli.py --server
python andrewcli.py --server --host 127.0.0.1 --port 9000
```

---

## Usage

```
$ python andrewcli.py
Andrew is running...
[general] Ask: Write "hello" to greeting.txt
⠋ Loading: write_file
⠋ Running write_file: greeting.txt
Andrew: File greeting.txt written successfully.

[general] Ask: [TAB]                      # TAB switches domain
Switched to domain: coding

[coding] Ask: /project "Build a REST API in Python"
✓ Event 'project' started
[coding] Ask:

◆ Event [project]: Project: Build a REST API in Python
⠋ Running write_file: pyproject.toml
Andrew: Task 1 done — project structure created.

[coding] Ask: /stop project
✓ Event 'project' stopped
```

### CLI controls

| Key / Input | Action |
|-------------|--------|
| **TAB** | Cycle to the next available domain |
| **UP / DOWN** | Navigate command history |
| **ESC** | Stop streaming (background tasks still complete) |
| `/events` | List available event types and which are running |
| `/name [args]` | Start a named event (e.g. `/timer 30`, `/project "Build X"`) |
| `/stop [name]` | Stop a running event; `/stop` alone lists running events |

### Tray controls

| Key / Control | Action |
|---------------|--------|
| **TAB** | Cycle to the next available domain |
| **Domain button** | Cycle to the next available domain |
| **Stop button** | Cancel the current generation |
| **Clear button** | Clear chat view and reset conversation memory |
| **ESC** | Hide the panel window |
| **▽ / △ button** | Toggle compact / expanded view |
| `/events`, `/name [args]`, `/stop [name]` | Same as CLI |

---

## Documentation

| Topic | File |
|-------|------|
| Core architecture, domains, tools, skills, tray, async pipeline | [doc/architecture.md](doc/architecture.md) |
| Rolling memory and context-aware router | [doc/memory.md](doc/memory.md) |
| Events, slash commands, ProjectEvent, LoopEvent, MonitorEvent | [doc/events.md](doc/events.md) |
| HTTP API endpoints and curl examples | [doc/api.md](doc/api.md) |
| Adding tools, skills, domains, and events | [doc/extending.md](doc/extending.md) |

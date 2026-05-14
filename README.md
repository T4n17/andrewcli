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

The first time you run `andrewcli`, `~/.config/andrewcli/` is auto-created and seeded with a `config.yaml`, default domains, and events. Edit anything there freely — AndrewCLI never overwrites existing files.

Open `~/.config/andrewcli/config.yaml` and set your LLM endpoint:

```yaml
domain: "general"
api_base_url: "http://localhost:8080/v1"  # OpenAI-compatible endpoint
model: "your-model-name"
routing_enabled: false

memory:
  enabled: true            # set false to disable rolling summary entirely
  min_summary_chars: 200   # turns shorter than this skip the LLM merge

server:
  enabled: false           # auto-start the FastAPI bridge with the CLI/tray
```

`OPENAI_API_KEY` defaults to `"local"` — set it as an env var only if your server requires a real key. `SUMMARY_MODEL` can be set as an env var to route background memory merges to a smaller model.

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

[general] Ask: [TAB]                         # TAB switches domain
Switched to domain: coding

[coding] Ask: /loop "Monitor oil price; stop if < $100"
✓ Event 'loop' started [loop#1]
[coding] Ask: /loop "Poll /health until 200" 20
✓ Event 'loop' started [loop#2]             # two loops running in parallel
[coding] Ask: /events
Running events: loop#1, loop#2
...
[coding] Ask:                                # prompt always available

◆ Event [loop#1]: iteration 1               # fires silently in background
Andrew: Current price is $102.4/bbl — above threshold, continuing.

[coding] Ask: /status loop#1                # inspect output at any time
=== loop#1 — 1 iteration(s) ===
[1] Current price is $102.4/bbl — above threshold, continuing.

[coding] Ask: /stop loop#2
✓ Event 'loop#2' stopped
[coding] Ask: /stop loop                    # stops all remaining loop instances
✓ Event 'loop' stopped
```

### CLI controls

| Key / Input | Action |
|-------------|--------|
| **TAB** | Cycle to the next available domain |
| **UP / DOWN** | Navigate command history |
| **ESC** | Stop the current generation immediately (inference cancelled, not just display) |
| `/events` | List available event types and which are running (with instance IDs) |
| `/name [args]` | Start a named event — returns an instance ID (e.g. `loop#1`) |
| `/stop [id\|name]` | Stop by instance ID (`loop#1`) or name (stops all instances of that type) |
| `/status` | List all events with recorded output and iteration count |
| `/status [id]` | Show all recorded responses for a specific event instance |

### Tray controls

| Key / Control | Action |
|---------------|--------|
| **TAB** | Cycle to the next available domain |
| **UP / DOWN** | Navigate command history |
| **Domain button** | Cycle to the next available domain |
| **Stop button** | Cancel the current generation |
| **Clear button** | Clear chat view and reset conversation memory |
| **ESC** | Hide the panel window |
| **▽ / △ button** | Toggle compact / expanded view |
| `/events`, `/name [args]`, `/stop [id\|name]`, `/status [id]` | Same as CLI |

---

## Documentation

| Topic | File |
|-------|------|
| Core architecture, domains, tools, skills, tray, async pipeline | [doc/architecture.md](doc/architecture.md) |
| Rolling memory and context-aware router | [doc/memory.md](doc/memory.md) |
| Events, slash commands, ProjectEvent, LoopEvent, MonitorEvent | [doc/events.md](doc/events.md) |
| HTTP API endpoints and curl examples | [doc/api.md](doc/api.md) |
| Adding tools, skills, domains, and events | [doc/extending.md](doc/extending.md) |

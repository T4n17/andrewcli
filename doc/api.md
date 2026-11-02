# HTTP API

The FastAPI server starts automatically alongside the CLI and tray (default `http://0.0.0.0:8000`). Use `--host` / `--port` to change the address. It can also be started standalone (`--server`) to integrate external applications without a UI.

The server is a **middleware**: it does not call the LLM directly. Instead it enqueues the message for the running CLI/tray and the client polls for response tokens. Slash commands, events, and the full domain pipeline all work exactly as if the user typed the message.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Queue a message; returns `{"session_id": "..."}` (202) |
| `GET` | `/chat/{session_id}` | Poll for new tokens — consumed per call; `done: true` when the full response is ready |
| `DELETE` | `/chat/{session_id}` | Discard a session and its buffered tokens |
| `GET` | `/events` | List available slash-command event types and their parameters |
| `POST` | `/notify` | Show `{"message": "...", "source": "..."}` in the CLI/tray as an event from `source`, without calling the agent (202) |

---

## Session Lifecycle

`POST /chat` returns immediately with a `session_id`. The CLI/tray picks up the message within 100 ms and processes it through the normal submit path. Tokens accumulate in a per-session store as they are generated. Each `GET /chat/{session_id}` call consumes and returns any buffered tokens; poll until `done: true`.

**Events** — when a slash command starts an event that sends a message to the agent (e.g. `/schedule`, `/timer`, `/project`), the session stays open (`done: false`) until the event fires and the agent's response is fully streamed. The client polls the same `session_id` throughout. Events without an agent message (side-effect-only triggers) close the session immediately after the confirmation token.

---

## Pushing messages from other programs

`POST /notify` lets a script or background service print into a running AndrewCLI as it happens. The message is shown like an event response (`◆ Event [source#N]` banner in the CLI, streamed into the tray/Artemis panel) and recorded in `/status`. The agent is not involved, so the text appears exactly as sent. `source` (default `external`) names the event; characters other than letters, digits, `_`, `-`, and `.` become `-`.

```bash
curl -s -X POST http://127.0.0.1:8000/notify \
  -H "Content-Type: application/json" \
  -d '{"source": "backup", "message": "Nightly backup finished: 12 GB"}'
```

The server must be running: set `server.enabled: true` for the CLI and tray (Artemis always starts it). Requests fail fast when it is not, so callers can ignore errors.

---

## curl Examples

### Send a message and poll for the response

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

### Start a project loop

```bash
SID=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/project \"Build a REST API in Python\""}' | jq -r .session_id)

until curl -s http://localhost:8000/chat/$SID | tee /dev/stderr | jq -e '.done' > /dev/null; do
  sleep 0.5
done
```

### Schedule a message and wait for it to fire

```bash
# Schedule a message for 06-05-2026 at 22:46
SID=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/schedule \"Run daily summary\" 06-05-2026-22-46"}' | jq -r .session_id)

# First poll — event registered, session still open
curl -s http://localhost:8000/chat/$SID | jq .
# → {"tokens":["Event schedule started."],"done":false,"error":null}

# Keep polling — done: true arrives with the agent's response when the event fires
until curl -s http://localhost:8000/chat/$SID | tee /dev/stderr | jq -e '.done' > /dev/null; do
  sleep 1
done
```

### List available events

```bash
curl -s http://localhost:8000/events | jq .
```

### List running events (via the CLI/tray inbox)

```bash
SID=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/events"}' | jq -r .session_id)
curl -s http://localhost:8000/chat/$SID | jq .
```

### Discard a session

```bash
curl -s -X DELETE http://localhost:8000/chat/$SID | jq .
```

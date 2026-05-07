from __future__ import annotations

import asyncio
import queue
import threading
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Bridge — shared inbox and per-session response store
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    tokens: list[str] = field(default_factory=list)
    done: bool = False
    error: str | None = None


# (session_id, message) pairs enqueued by the server, consumed by CLI/tray.
inbox: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()

# Per-session response accumulator.
_sessions: dict[str, _Session] = {}


def new_session(message: str) -> str:
    """Put *message* into the inbox and return a fresh session_id."""
    sid = str(uuid.uuid4())
    _sessions[sid] = _Session()
    inbox.put((sid, message))
    return sid


def put_token(sid: str, token: str) -> None:
    s = _sessions.get(sid)
    if s is not None:
        s.tokens.append(token)


def finish(sid: str, error: str | None = None) -> None:
    s = _sessions.get(sid)
    if s is not None:
        s.done = True
        s.error = error


def poll(sid: str) -> dict | None:
    """Return and flush accumulated tokens for *sid*.

    Tokens are consumed on each call so the client only gets new tokens
    each poll. Returns None if the session does not exist.
    """
    s = _sessions.get(sid)
    if s is None:
        return None
    tokens, s.tokens = s.tokens[:], []
    return {"tokens": tokens, "done": s.done, "error": s.error}


def discard(sid: str) -> None:
    _sessions.pop(sid, None)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AndrewCLI API")


def start_background(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the API server in a daemon background thread.

    Safe to call from an asyncio context (CLI) or a Qt main-thread context
    (tray). The server gets its own event loop so it does not interfere with
    either asyncio.run() or app.exec(). The daemon thread is automatically
    killed when the host process exits.
    """
    import uvicorn

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(cfg)
        loop.run_until_complete(server.serve())

    threading.Thread(target=_run, daemon=True, name="andrewcli-api").start()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat", status_code=202)
async def chat(req: ChatRequest):
    """Queue *message* for the CLI/tray and return a session_id for polling."""
    sid = new_session(req.message)
    return {"session_id": sid, "status": "queued"}


@app.get("/chat/{session_id}")
async def poll_endpoint(session_id: str):
    """Poll for response tokens from a queued message.

    Tokens are consumed on each call — only new tokens since the last poll
    are returned. ``done`` becomes true once the CLI/tray finishes the turn.
    """
    result = poll(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@app.delete("/chat/{session_id}")
async def discard_endpoint(session_id: str):
    """Discard a session and its accumulated tokens."""
    discard(session_id)
    return {"status": "discarded", "session_id": session_id}


@app.get("/events")
async def get_events():
    """List all available slash-command events."""
    import inspect
    from src.core.events_registry import available_events
    registry = available_events()
    result = [
        {
            "name": name,
            "args": [n for n in inspect.signature(cls.__init__).parameters if n != "self"],
        }
        for name, cls in sorted(registry.items())
    ]
    return {"events": result, "usage": "/name [arg1] [arg2] ..."}

"""HTTP bridge for the CLI and the tray.

The :class:`Server` class wraps three concerns into one object:

* a **shared inbox** (``queue.SimpleQueue``) where the FastAPI ``/chat``
  endpoint enqueues incoming messages so the CLI/tray loop can pick them
  up;
* a **per-session response store** that accumulates the streamed tokens
  produced by the agent so the HTTP client can poll for them;
* the **FastAPI app** itself, with the ``/chat``, ``/chat/{sid}``,
  ``/events`` routes wired to the methods above.

Most consumers just use the module-level :data:`server` singleton::

    from src.core.server import server

    sid = server.new_session("hello")
    server.put_token(sid, "world")
    server.finish(sid)

The module-level :data:`app` symbol is kept as a thin alias to
``server.app`` so the existing ``uvicorn src.core.server:app`` entry
point keeps working untouched.
"""
from __future__ import annotations

import asyncio
import inspect
import queue
import threading
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


@dataclass
class _Session:
    tokens: list[str] = field(default_factory=list)
    done: bool = False
    error: str | None = None


class ChatRequest(BaseModel):
    message: str


class Server:
    """FastAPI app + cross-thread bridge between HTTP clients and the agent.

    The class is safe to instantiate multiple times in tests, but most
    of the codebase shares the module-level :data:`server` singleton so
    the inbox/session state is consistent across importers.
    """

    def __init__(self, title: str = "AndrewCLI API"):
        self.app = FastAPI(title=title)

        # (session_id, message) pairs enqueued by the server, consumed
        # by the CLI/tray main loop.
        self.inbox: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()

        # Per-session response accumulator. Keyed by session id; values
        # are :class:`_Session` records that grow as the agent streams.
        self._sessions: dict[str, _Session] = {}

        self._register_routes()

    # ------------------------------------------------------------------
    # Bridge — used by the CLI/tray to push agent tokens back to the
    # HTTP client that is polling for them.
    # ------------------------------------------------------------------

    def new_session(self, message: str) -> str:
        """Put *message* into the inbox and return a fresh session_id."""
        sid = str(uuid.uuid4())
        self._sessions[sid] = _Session()
        self.inbox.put((sid, message))
        return sid

    def put_token(self, sid: str, token: str) -> None:
        s = self._sessions.get(sid)
        if s is not None:
            s.tokens.append(token)

    def finish(self, sid: str, error: str | None = None) -> None:
        s = self._sessions.get(sid)
        if s is not None:
            s.done = True
            s.error = error

    def poll(self, sid: str) -> dict | None:
        """Return and flush accumulated tokens for *sid*.

        Tokens are consumed on each call so the client only gets new
        tokens each poll. Returns ``None`` if the session does not exist.
        """
        s = self._sessions.get(sid)
        if s is None:
            return None
        tokens, s.tokens = s.tokens[:], []
        return {"tokens": tokens, "done": s.done, "error": s.error}

    def discard(self, sid: str) -> None:
        self._sessions.pop(sid, None)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_background(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the API server in a daemon background thread.

        Safe to call from an asyncio context (CLI) or a Qt main-thread
        context (tray). The server gets its own event loop so it does
        not interfere with either ``asyncio.run()`` or ``app.exec()``.
        The daemon thread is automatically killed when the host process
        exits.
        """
        import uvicorn

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            cfg = uvicorn.Config(self.app, host=host, port=port, log_level="warning")
            uvicorn_server = uvicorn.Server(cfg)
            loop.run_until_complete(uvicorn_server.serve())

        threading.Thread(target=_run, daemon=True, name="andrewcli-api").start()

    # ------------------------------------------------------------------
    # FastAPI routes
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        app = self.app

        @app.post("/chat", status_code=202)
        async def chat(req: ChatRequest):
            """Queue *message* for the CLI/tray and return a session_id for polling."""
            sid = self.new_session(req.message)
            return {"session_id": sid, "status": "queued"}

        @app.get("/chat/{session_id}")
        async def poll_endpoint(session_id: str):
            """Poll for response tokens from a queued message.

            Tokens are consumed on each call — only new tokens since the
            last poll are returned. ``done`` becomes true once the
            CLI/tray finishes the turn.
            """
            result = self.poll(session_id)
            if result is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return result

        @app.delete("/chat/{session_id}")
        async def discard_endpoint(session_id: str):
            """Discard a session and its accumulated tokens."""
            self.discard(session_id)
            return {"status": "discarded", "session_id": session_id}

        @app.get("/events")
        async def get_events():
            """List all available slash-command events."""
            from src.core.registry import registry
            evs = registry.events()
            result = [
                {
                    "name": name,
                    "args": [n for n in inspect.signature(cls.__init__).parameters if n != "self"],
                }
                for name, cls in sorted(evs.items())
            ]
            return {"events": result, "usage": "/name [arg1] [arg2] ..."}


# Default singleton — used by the CLI, the tray, and the uvicorn entry
# point. The ``app`` alias is kept so ``uvicorn src.core.server:app``
# continues to work without any caller changes.
server = Server()
app = server.app

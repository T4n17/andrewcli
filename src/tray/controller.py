"""Headless orchestration layer for the Andrew chat UI.

:class:`TrayController` owns the moving parts that aren't tied to a
specific window/host: the domain lifecycle, the asyncio event-bus
bridge, the :class:`StreamWorker` lifecycle, and the Qt-side queue pollers.

It does **not** own the :class:`QApplication`, the system-tray icon,
the app-wide stylesheet/font, the panel construction, the panel's
show/hide policy, or process exit. Those belong to whatever shell
embeds the controller. The standalone tray (:class:`AndrewTrayApp`)
is one such shell; an external project (e.g. a desktop widget host)
that wants to embed :class:`ChatPanel` as a child widget is another.

Wire-up::

    controller = TrayController(panel=panel, config=config, parent=some_qobject)
    controller.start()           # event bus, poll timer
    ...
    controller.shutdown()        # event bus, poll timer, worker

The shell is responsible for showing the panel, surfacing event
notifications visually (toast / raise its host widget), and quitting
the process. The optional ``on_event_notification`` callback is the
hook for the second of those: it fires every time a background event
is observed, with the :class:`Event` instance as its only argument.
The default is a no-op so embedders that don't care can ignore it.
"""
from __future__ import annotations

import asyncio
import logging
import queue
from typing import Callable, Optional, TYPE_CHECKING

from PyQt6.QtCore import QObject, QTimer

from src.core.andrew import AndrewCore, CommandResult
from src.core.event import Event
from src.core.llm import ToolEvent, RouteEvent, format_tool_status
from src.core.registry import registry
from src.core.server import server
from src.tray.worker import StreamWorker, get_event_loop

if TYPE_CHECKING:
    from src.shared.config import Config
    from src.tray.panel import ChatPanel

log = logging.getLogger("src.tray.controller")


class TrayController:
    """Orchestrate domain, event bus, and StreamWorker for a panel.

    See module docstring for the ownership boundary. All Qt objects
    are created with the injected ``parent`` (or the panel as a
    fallback) so they're cleaned up by Qt when the parent is
    destroyed - the controller does not own a top-level Qt object
    of its own.
    """

    def __init__(
        self,
        *,
        panel: "ChatPanel",
        config: "Config",
        parent: Optional[QObject] = None,
        on_event_notification: Optional[Callable[[Event], None]] = None,
    ) -> None:
        self._panel = panel
        self._config = config
        # QTimer needs a QObject parent for proper cleanup. The panel
        # itself is always a safe default (it's a QWidget = QObject)
        # if the embedder didn't pass one. Embedders typically pass
        # their QApplication so timers get torn down with the app.
        self._parent: QObject = parent if parent is not None else panel
        self._on_event_notification = on_event_notification or (lambda _e: None)

        self._loop = get_event_loop()
        self._domain_name = config.domain
        self.domain = self._create_domain(self._domain_name)

        self._worker = None

        # Thread-safe queues bridging the asyncio event bus to the Qt main thread
        self._event_notify_queue: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self._event_token_queue: queue.SimpleQueue = queue.SimpleQueue()

        # Shared event bus / log / dispatch / command logic. It runs on the
        # background asyncio loop; calls from Qt go through _on_loop.
        self._core = AndrewCore()
        self._core._on_event_notify = self._event_notify_queue.put
        self._core._on_event_token = lambda _iid, tok: self._event_token_queue.put(tok)
        self._core._on_event_done = lambda _iid: self._event_token_queue.put(None)

        self._event_poll_timer = QTimer(self._parent)
        self._event_poll_timer.setInterval(100)
        self._event_poll_timer.timeout.connect(self._poll_event_queues)

    # -- public state ---------------------------------------------------------

    @property
    def domain_name(self) -> str:
        return self._domain_name

    @property
    def is_streaming(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Wire panel signals, start the poll timer, and activate the domain."""
        self._panel.submitted.connect(self.submit)
        self._panel.stop_requested.connect(self.stop)
        self._panel.clear_requested.connect(self.clear)
        self._panel.domain_switch.connect(self.switch_domain)
        self._panel.set_domain_name(self._domain_name)

        self._event_poll_timer.start()
        self._activate_domain()
        self._notices = asyncio.run_coroutine_threadsafe(self._core.pump_notices(), self._loop)

    def shutdown(self) -> None:
        """Stop everything the controller owns. Idempotent."""
        try:
            self._event_poll_timer.stop()
        except Exception:
            pass
        self.stop()
        notices = getattr(self, "_notices", None)
        if notices is not None:
            notices.cancel()
        try:
            self._on_loop(self._core.shutdown)
        except Exception:
            pass

    # -- public actions (panel signal slots) ----------------------------------

    def submit(self, message: str) -> None:
        if message.startswith("/"):
            self._run_command(message)
            return
        self._start_worker(message)

    def _start_worker(self, source, sid: Optional[str] = None) -> None:
        """Stream a user turn (``str``) or a workflow (async generator).

        With *sid*, the streamed text is mirrored into that HTTP bridge
        session, which is finished when the stream ends.
        """
        self.stop()
        worker = StreamWorker(source, self.domain)
        worker.token_received.connect(self._panel.append_token)
        worker.tool_status.connect(self._panel.on_tool_status)
        worker.finished.connect(self._panel.on_stream_done)
        worker.error.connect(self._panel.on_error)
        if sid is not None:
            worker.token_received.connect(lambda t, _sid=sid: server.put_token(_sid, t))
            worker.finished.connect(lambda _sid=sid: server.finish(_sid))
            worker.error.connect(lambda e, _sid=sid: server.finish(_sid, error=e))
        self._worker = worker
        worker.start()

    def _run_command(self, text: str, sid: Optional[str] = None) -> None:
        """Run a slash command and render its result in the panel (and bridge session *sid*)."""
        cmd = text.strip()
        try:
            result = self._on_loop(lambda: self._core.run_command(cmd, bridge_sid=sid))
        except Exception as exc:
            log.exception("command failed: %s", cmd)
            result = CommandResult(f"Command failed: {exc}", error=True)

        if result.stream is not None:
            self._start_worker(result.stream, sid)
            return
        if cmd == "/clear":
            self._panel._on_clear()
        if result.error:
            self._panel.on_error(result.text)
        else:
            self._panel.append_token(result.text)
            self._panel.on_stream_done()
        if sid is not None:
            server.put_token(sid, result.text)
            if not result.holds_session:
                server.finish(sid)

    def _submit_bridge(self, sid: str, message: str) -> None:
        """Process a server-injected message exactly like a user submission.

        Shows the message as a user bubble, then runs it through the same
        command / turn path, mirroring the response into the bridge
        session so the HTTP client can poll for it.
        """
        self._panel.show_user_message(message)
        if message.strip().startswith("/"):
            self._run_command(message, sid)
        else:
            self._start_worker(message, sid)

    def stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.token_received.disconnect()
            self._worker.tool_status.disconnect()
            self._worker.finished.disconnect()
            self._worker.error.disconnect()
            self._worker.wait(5000)
            self._worker = None

    def clear(self) -> None:
        self.stop()

    def switch_domain(self) -> None:
        domains = registry.domains()
        if len(domains) <= 1:
            return
        try:
            idx = domains.index(self._domain_name)
            next_name = domains[(idx + 1) % len(domains)]
        except ValueError:
            next_name = domains[0]
        self.stop()
        try:
            domain = self._create_domain(next_name)
        except ValueError as exc:
            # Keep the current domain alive rather than leaving the
            # controller half-switched. Logged, not surfaced: the
            # ``on_event_notification`` hook is for events only.
            log.warning("domain switch failed: %s", exc)
            return
        self.domain = domain
        self._domain_name = next_name
        self._panel.set_domain_name(next_name)
        self._activate_domain()

    def _activate_domain(self) -> None:
        """Hand ``self.domain`` to the core (running events keep running) and arm automations."""
        try:
            armed, errors = self._on_loop(lambda: self._core.set_domain(self.domain))
        except Exception:
            log.exception("failed to activate domain %s", self._domain_name)
            return
        if armed:
            log.info("automations armed: %s", ", ".join(armed))
        for error in errors:
            log.warning("automation skipped: %s", error)

    def _on_loop(self, fn):
        """Run *fn* on the background asyncio loop and return its result.

        The core and its event bus (``asyncio.create_task``,
        ``Task.cancel``) are only safe on the loop's own thread.
        """
        try:
            if asyncio.get_running_loop() is self._loop:
                return fn()
        except RuntimeError:
            pass

        async def _call():
            return fn()

        return asyncio.run_coroutine_threadsafe(_call(), self._loop).result(timeout=10)

    # -- Qt main thread: poll queues every 100 ms ----------------------------

    def _poll_event_queues(self) -> None:
        while not self._event_notify_queue.empty():
            event = self._event_notify_queue.get_nowait()
            # Hand off the surface-level reaction (toast, raise host
            # widget, ring a bell, ...) to the embedder. The default
            # is a no-op for headless / pure-embed scenarios.
            try:
                self._on_event_notification(event)
            except Exception:
                log.exception("on_event_notification callback raised")
            if event.message or event.command:
                self._panel.start_event_response(event.name)

        while not self._event_token_queue.empty():
            item = self._event_token_queue.get_nowait()
            if item is None:
                self._panel.on_stream_done()
            elif isinstance(item, (RouteEvent, ToolEvent)):
                status = format_tool_status(item)
                if status is not None:
                    self._panel.on_tool_status(status)
            elif isinstance(item, str):
                self._panel.append_token(item)

        # Bridge inbox: messages injected via the HTTP server.
        while not server.inbox.empty():
            try:
                sid, message = server.inbox.get_nowait()
            except Exception:
                break
            self._submit_bridge(sid, message)

    # -- domain ---------------------------------------------------------------

    def _create_domain(self, name: str):
        # Built on the loop so the knowledge base can start its indexer.
        return self._on_loop(lambda: registry.load_domain(name))

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

from src.core.andrew import AndrewCore
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

        # Shared event log / dispatch / slash logic
        self._core = AndrewCore()
        self._core.domain = self.domain
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
        """Wire panel signals, start the poll timer and event bus."""
        self._panel.submitted.connect(self.submit)
        self._panel.stop_requested.connect(self.stop)
        self._panel.clear_requested.connect(self.clear)
        self._panel.domain_switch.connect(self.switch_domain)
        self._panel.set_domain_name(self._domain_name)

        self._event_poll_timer.start()
        self._start_event_bus()

    def shutdown(self) -> None:
        """Stop everything the controller owns. Idempotent."""
        try:
            self._event_poll_timer.stop()
        except Exception:
            pass
        self.stop()
        bus = getattr(self.domain, "event_bus", None)
        if bus is not None:
            try:
                bus.stop()
            except Exception:
                pass

    # -- public actions (panel signal slots) ----------------------------------

    def submit(self, message: str) -> None:
        if message.startswith("/"):
            self._handle_slash_command(message)
            return
        self.stop()
        self._worker = StreamWorker(message, self.domain)
        self._worker.token_received.connect(self._panel.append_token)
        self._worker.tool_status.connect(self._panel.on_tool_status)
        self._worker.finished.connect(self._panel.on_stream_done)
        self._worker.error.connect(self._panel.on_error)
        self._worker.start()

    def _handle_slash_command(self, text: str, on_token=None) -> None:
        """Parse and activate (or stop) a slash command from the panel.

        ``on_token``, if provided, is called with each response string in
        addition to the panel — used by ``_submit_bridge`` to capture the
        response for the HTTP polling endpoint.
        """
        bus = self.domain.event_bus
        cmd = text.strip()

        def _emit(t: str) -> None:
            self._panel.append_token(t)
            if on_token is not None:
                on_token(t)

        response = self._core.handle_slash(cmd, bus)
        if response is not None:
            _emit(response)
            self._panel.on_stream_done()
            return

        try:
            event = registry.parse_slash_command(text)
        except ValueError as exc:
            self._panel.on_error(str(exc))
            if on_token is not None:
                on_token(str(exc))
            return
        if event is None:
            err = f"Unknown command: `{text}`\n\n" + registry.list_commands(bus.running())
            self._panel.on_error(err)
            if on_token is not None:
                on_token(err)
            return
        # EventBus.add() calls asyncio.create_task(), which must run on
        # the bg asyncio loop — schedule it thread-safely from Qt main.
        asyncio.run_coroutine_threadsafe(self._async_add_event(event), self._loop)
        _emit(f"Event **{event.name}** started.")
        self._panel.on_stream_done()

    async def _async_add_event(self, event) -> None:
        self.domain.event_bus.add(event)

    def _submit_bridge(self, sid: str, message: str) -> None:
        """Process a server-injected message exactly like a user submission.

        Shows the message as a user bubble, routes it through the normal
        submit path, and captures response tokens into the bridge session
        so the HTTP client can poll for them.
        """
        self._panel.show_user_message(message)

        if message.strip().startswith("/"):
            import inspect
            evt = registry.parse_slash_command(message)
            if evt is not None:
                # Use getattr_static to avoid calling dynamic properties (e.g.
                # LoopEvent.message is a @property with side effects — accessing
                # it here would consume the first planning iteration before the
                # EventBus can use it).
                msg_descriptor = inspect.getattr_static(type(evt), 'message', None)
                if isinstance(msg_descriptor, property):
                    has_message = True  # dynamic message, don't call it
                else:
                    has_message = bool(evt.message)  # plain attribute, safe to read
            else:
                has_message = False

            if evt is not None and has_message:
                evt._bridge_sid = sid
                asyncio.run_coroutine_threadsafe(self._async_add_event(evt), self._loop)
                token = f"Event **{evt.name}** started."
                self._panel.append_token(token)
                self._panel.on_stream_done()
                server.put_token(sid, token)
                # session stays open; _event_dispatch will finish it when the event fires
            else:
                captured: list[str] = []
                self._handle_slash_command(message, on_token=captured.append)
                server.put_token(sid, "".join(captured))
                server.finish(sid)
            return

        self.stop()
        worker = StreamWorker(message, self.domain)
        worker.token_received.connect(self._panel.append_token)
        worker.token_received.connect(lambda t, _sid=sid: server.put_token(_sid, t))
        worker.tool_status.connect(self._panel.on_tool_status)
        worker.finished.connect(self._panel.on_stream_done)
        worker.finished.connect(lambda _sid=sid: server.finish(_sid))
        worker.error.connect(self._panel.on_error)
        worker.error.connect(lambda e, _sid=sid: server.finish(_sid, error=e))
        self._worker = worker
        worker.start()

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
        self.domain.llm.memory.clear()

    def switch_domain(self) -> None:
        domains = registry.domains()
        if len(domains) <= 1:
            return
        try:
            idx = domains.index(self._domain_name)
            next_name = domains[(idx + 1) % len(domains)]
        except ValueError:
            next_name = domains[0]
        try:
            self.stop()
            self.domain.event_bus.stop()
            self.domain = self._create_domain(next_name)
            self._domain_name = next_name
            self._panel.set_domain_name(next_name)
            self._start_event_bus()
        except ValueError as exc:
            # Domain module failed to import. Keep the current domain
            # alive rather than leaving the controller in a half-broken
            # state. The previous standalone code surfaced this as a
            # tray balloon; the controller logs it instead so the
            # embedder is free to surface (or ignore) it on its own
            # terms - the ``on_event_notification`` hook is intended
            # for events, not error reporting, and conflating them
            # would make the contract ambiguous.
            log.warning("domain switch failed: %s", exc)
            self._start_event_bus()

    # -- event bus ------------------------------------------------------------

    def _start_event_bus(self) -> None:
        self._core.domain = self.domain
        bus = self.domain.event_bus
        bus.notify = self._event_notify
        bus.dispatch = self._core._event_dispatch
        asyncio.run_coroutine_threadsafe(bus.start(), self._loop)

    # -- event bus callbacks (called from asyncio thread) ---------------------

    def _event_notify(self, event: Event) -> None:
        self._event_notify_queue.put(event)

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
            if event.message:
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
        future = asyncio.run_coroutine_threadsafe(
            self._async_create_domain(name), self._loop,
        )
        return future.result(timeout=10)

    async def _async_create_domain(self, name: str):
        return registry.load_domain(name)

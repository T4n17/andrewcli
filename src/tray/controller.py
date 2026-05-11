"""Headless orchestration layer for the Andrew chat UI.

:class:`TrayController` owns the moving parts that aren't tied to a
specific window/host: the domain lifecycle, the asyncio event-bus
bridge, the :class:`StreamWorker` lifecycle, voice (STT/TTS) gating,
and the Qt-side queue pollers.

It does **not** own the :class:`QApplication`, the system-tray icon,
the app-wide stylesheet/font, the panel construction, the panel's
show/hide policy, or process exit. Those belong to whatever shell
embeds the controller. The standalone tray (:class:`AndrewTrayApp`)
is one such shell; an external project (e.g. a desktop widget host)
that wants to embed :class:`ChatPanel` as a child widget is another.

Wire-up::

    controller = TrayController(panel=panel, config=config,
                                voice_enabled=True, parent=some_qobject)
    controller.start()           # event bus, voice listener, poll timer
    ...
    controller.shutdown()        # voice, event bus, poll timer, worker

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

from src.core.event import Event
from src.core.llm import ToolEvent, RouteEvent, format_tool_status
from src.core.registry import available_domains, load_domain, parse_slash_command, list_commands
from src.tray.worker import StreamWorker, get_event_loop

if TYPE_CHECKING:
    from src.shared.config import Config
    from src.tray.panel import ChatPanel

log = logging.getLogger("src.tray.controller")


class TrayController:
    """Orchestrate domain, voice, event bus, and StreamWorker for a panel.

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
        voice_enabled: bool = False,
        parent: Optional[QObject] = None,
        on_event_notification: Optional[Callable[[Event], None]] = None,
    ) -> None:
        self._panel = panel
        self._config = config
        self._voice_enabled = voice_enabled
        # QTimer needs a QObject parent for proper cleanup. The panel
        # itself is always a safe default (it's a QWidget = QObject)
        # if the embedder didn't pass one. Embedders typically pass
        # their QApplication so timers get torn down with the app.
        self._parent: QObject = parent if parent is not None else panel
        self._on_event_notification = on_event_notification or (lambda _e: None)

        self._loop = get_event_loop()
        self._domain_name = config.domain
        self.domain = self._create_domain(self._domain_name)

        # Voice I/O. The bg asyncio loop in worker.get_event_loop() is
        # the same one the StreamWorker uses for domain.generate, so
        # STT and TTS share it and token teeing stays lock-free.
        self.stt = None
        self.tts = None
        self._voice_prompt_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._voice_listener_future = None
        # Gate that pauses the STT loop. Two orthogonal states feed it:
        #   _voice_user_enabled: mic toggle in the header. User wants
        #       STT to run at all. OFF wins unconditionally.
        #   _voice_agent_busy:  agent is generating / running a tool /
        #       playing TTS. Blocks wake-word retrigger mid-turn and
        #       speaker-to-mic self-wake.
        # STT runs iff user_enabled AND NOT agent_busy. The event is
        # created on the bg loop once the listener starts.
        self._voice_idle_event = None
        self._voice_user_enabled = True
        self._voice_agent_busy = False
        self._current_listen_task = None
        if voice_enabled:
            from src.voice import build_voice_io
            self.stt, self.tts = build_voice_io(config)

        self._worker = None

        # Thread-safe queues bridging the asyncio event bus to the Qt main thread
        self._event_notify_queue: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self._event_token_queue: queue.SimpleQueue = queue.SimpleQueue()

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
        """Wire panel signals, start the poll timer, event bus, voice."""
        self._panel.submitted.connect(self.submit)
        self._panel.stop_requested.connect(self.stop)
        self._panel.clear_requested.connect(self.clear)
        self._panel.domain_switch.connect(self.switch_domain)
        if self._voice_enabled:
            self._panel.voice_toggle.connect(self._on_voice_toggle)
            self._panel.set_voice_enabled(True)
        self._panel.set_domain_name(self._domain_name)

        self._event_poll_timer.start()
        self._start_event_bus()
        if self._voice_enabled:
            self._start_voice_listener()

    def shutdown(self) -> None:
        """Stop everything the controller owns. Idempotent.

        Order matters: stop the poll timer first so no new Qt-side
        actions fire while we're tearing down; cancel the voice
        listener (and any in-flight ``listen_once``); kill the worker;
        then stop the event bus.
        """
        try:
            self._event_poll_timer.stop()
        except Exception:
            pass
        if self._voice_listener_future is not None:
            try:
                self._voice_listener_future.cancel()
            except Exception:
                pass
        listen_task = self._current_listen_task
        if listen_task is not None:
            try:
                self._loop.call_soon_threadsafe(listen_task.cancel)
            except Exception:
                pass
        # Terminate any in-flight worker. Mirrors `_on_stop` semantics
        # but is safe to call when no worker is running.
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
        # Pause voice for the whole user turn (generation + any tool
        # calls + TTS playback). Resumed on finished/error. This lets
        # us ignore the wake word mid-turn and prevents self-wake from
        # the speaker bleed-through of our own TTS output.
        self._set_voice_busy(True)
        # Pass tts only when voice is on; worker tees tokens to speaker.
        self._worker = StreamWorker(message, self.domain, tts=self.tts)
        self._worker.token_received.connect(self._panel.append_token)
        self._worker.tool_status.connect(self._panel.on_tool_status)
        self._worker.finished.connect(self._panel.on_stream_done)
        self._worker.finished.connect(lambda: self._set_voice_busy(False))
        self._worker.error.connect(self._panel.on_error)
        self._worker.error.connect(lambda _e: self._set_voice_busy(False))
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

        if cmd == "/events":
            _emit(list_commands(bus.running()))
            self._panel.on_stream_done()
            return

        if cmd.startswith("/stop"):
            parts = cmd.split(None, 1)
            if len(parts) == 1:
                running = bus.running()
                msg = ("Running: " + ", ".join(running)) if running else "No events running."
                _emit(msg)
            else:
                name = parts[1]
                # remove() is sync but mutates list; safe from Qt thread
                # since the bg loop only reads _events/_tasks, never writes.
                if bus.remove(name):
                    _emit(f"Event **{name}** stopped.")
                else:
                    _emit(f"No running event named `{name}`.")
            self._panel.on_stream_done()
            return

        event = parse_slash_command(text)
        if event is None:
            err = f"Unknown command: `{text}`\n\n" + list_commands(bus.running())
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
        from src.core import server as bridge

        self._panel.show_user_message(message)

        if message.strip().startswith("/"):
            import inspect
            evt = parse_slash_command(message)
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
                bridge.put_token(sid, token)
                # session stays open; _event_dispatch will finish it when the event fires
            else:
                captured: list[str] = []
                self._handle_slash_command(message, on_token=captured.append)
                bridge.put_token(sid, "".join(captured))
                bridge.finish(sid)
            return

        self.stop()
        self._set_voice_busy(True)
        worker = StreamWorker(message, self.domain, tts=self.tts)
        worker.token_received.connect(self._panel.append_token)
        worker.token_received.connect(lambda t, _sid=sid: bridge.put_token(_sid, t))
        worker.tool_status.connect(self._panel.on_tool_status)
        worker.finished.connect(self._panel.on_stream_done)
        worker.finished.connect(lambda _sid=sid: bridge.finish(_sid))
        worker.finished.connect(lambda: self._set_voice_busy(False))
        worker.error.connect(self._panel.on_error)
        worker.error.connect(lambda e, _sid=sid: bridge.finish(_sid, error=e))
        worker.error.connect(lambda _e: self._set_voice_busy(False))
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
            # The cancelled worker won't fire finished/error, so
            # release the voice gate manually or STT stays paused.
            self._set_voice_busy(False)

    def clear(self) -> None:
        self.stop()
        self.domain.llm.memory.clear()

    def switch_domain(self) -> None:
        domains = available_domains()
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
        bus = self.domain.event_bus
        bus.notify = self._event_notify
        bus.dispatch = self._event_dispatch
        asyncio.run_coroutine_threadsafe(bus.start(), self._loop)

    # -- voice listener (bg asyncio thread) ----------------------------------

    def _start_voice_listener(self) -> None:
        """Run STT forever on the bg loop; push transcripts to Qt main.

        Each completed utterance goes on ``_voice_prompt_queue`` and is
        picked up by the 100-ms poll timer, which submits it through the
        normal ``submit`` path - exactly as if the user had typed it.
        On wake-word detection we also push a ``"__wake__"`` sentinel so
        the Qt thread can surface a visual cue before the user starts
        speaking.
        """
        voice_log = logging.getLogger("src.tray.voice")

        def _on_wake():
            voice_log.info("voice: wake fired, enqueuing __wake__ sentinel")
            self._voice_prompt_queue.put("__wake__")

        async def _listener():
            voice_log.info("voice listener task started on bg loop")
            try:
                # Drive listen_once directly rather than listen_forever so
                # we can push an __idle__ sentinel after *every* utterance,
                # including empty ones (wake-word misfires or silence after
                # the wake word). Without this the "listening..." spinner
                # never clears and the panel looks hung.
                self.stt._stop = False
                # Create the idle gate on this (bg) loop so .wait()/.set()
                # all happen on the same loop the STT task runs on. Seed
                # it from the current user/agent state so a mic-off start
                # (future) or an in-flight turn is honored immediately.
                self._voice_idle_event = asyncio.Event()
                self._refresh_voice_gate_sync()
                while not self.stt._stop:
                    # Block here while the gate is closed (agent busy
                    # and/or user toggled the mic off). Avoids wake-word
                    # retrigger mid-turn, self-wake from our own TTS,
                    # and keeps the mic cold while paused.
                    if not self._voice_idle_event.is_set():
                        voice_log.info("voice: STT paused (waiting for gate)")
                        await self._voice_idle_event.wait()
                        voice_log.info("voice: STT resumed")
                    # Run listen_once as a child task we can cancel when
                    # the gate transitions to closed. Without this,
                    # clearing the event only blocks the *next* iteration
                    # - the currently-running listen_once (potentially
                    # stuck in _wait_for_wake for seconds) keeps the mic
                    # hot and the wake word can still fire mid-turn.
                    self._current_listen_task = asyncio.create_task(
                        self.stt.listen_once(on_wake=_on_wake)
                    )
                    try:
                        text = await self._current_listen_task
                    except asyncio.CancelledError:
                        voice_log.info("voice: listen_once cancelled by gate close")
                        text = ""
                    finally:
                        self._current_listen_task = None
                    voice_log.info("voice: listen_once returned (len=%d): %r",
                                   len(text or ""), (text or "")[:80])
                    if text and text.strip():
                        self._voice_prompt_queue.put(text)
                        voice_log.info("voice: transcript enqueued for Qt poller")
                    else:
                        self._voice_prompt_queue.put("__idle__")
            except Exception:
                # Silent crashes inside a run_coroutine_threadsafe future
                # are the classic reason voice "just stops working"
                # mid-session. Log + re-raise so the root cause is visible
                # in ~/.andrewcli/tray.log.
                voice_log.exception("voice listener crashed")
                raise
            finally:
                voice_log.info("voice listener task exited")

        self._voice_listener_future = asyncio.run_coroutine_threadsafe(
            _listener(), self._loop,
        )

    # -- voice busy/idle gate -------------------------------------------------

    def _refresh_voice_gate_sync(self) -> None:
        """Open/close the idle gate based on current user+agent state.

        Must be called on the bg asyncio loop (where the event lives).
        Use :meth:`_refresh_voice_gate` from other threads. If the gate
        transitions to closed, also cancels the currently-running
        ``listen_once`` task so the mic goes cold immediately instead
        of at the next loop iteration.
        """
        ev = self._voice_idle_event
        if ev is None:
            return
        should_be_idle = self._voice_user_enabled and not self._voice_agent_busy
        if should_be_idle:
            ev.set()
        else:
            ev.clear()
            task = self._current_listen_task
            if task is not None and not task.done():
                task.cancel()

    def _refresh_voice_gate(self) -> None:
        """Thread-safe wrapper around :meth:`_refresh_voice_gate_sync`."""
        if self._voice_idle_event is None:
            return
        self._loop.call_soon_threadsafe(self._refresh_voice_gate_sync)

    def _set_voice_busy(self, busy: bool) -> None:
        """Mark the agent busy/idle and refresh the gate.

        Called from the Qt main thread (submit/finish/error slots) and
        from the bg loop (event-dispatch wrapper). The gate is only
        open when ``user_enabled AND NOT agent_busy``.
        """
        self._voice_agent_busy = busy
        self._refresh_voice_gate()

    def _on_voice_toggle(self, enabled: bool) -> None:
        """Slot for the panel's mic toggle button.

        ``enabled`` is the new desired state (True = user wants STT on).
        When toggled off mid-utterance we cancel the active listen_once
        so the mic goes cold right away instead of finishing the 8 s
        recording cap.
        """
        self._voice_user_enabled = enabled
        self._refresh_voice_gate()
        # When toggling off, clear any stale "listening..." spinner that
        # a just-fired wake might have left behind.
        if not enabled:
            self._voice_prompt_queue.put("__idle__")

    # -- event bus callbacks (called from asyncio thread) ---------------------

    def _event_notify(self, event: Event) -> None:
        self._event_notify_queue.put(event)

    async def _event_dispatch(self, event: Event) -> None:
        if not event.message:
            return
        from src.core import server as bridge
        sid = getattr(event, "_bridge_sid", None)
        # Consume the sid for this iteration only — multi-iteration events
        # (LoopEvent, ProjectEvent) fire _event_dispatch repeatedly; each
        # subsequent call gets no bridge session so it only renders to the UI.
        if sid:
            event._bridge_sid = None
        self._voice_agent_busy = True
        self._refresh_voice_gate_sync()
        try:
            async for token in self.domain.generate_event(event.message):
                self._event_token_queue.put(token)
                if sid and isinstance(token, str):
                    bridge.put_token(sid, token)
            self._event_token_queue.put(None)
            if sid:
                bridge.finish(sid)
        except asyncio.CancelledError:
            self._event_token_queue.put(None)
            if sid:
                bridge.finish(sid, error="Event stopped")
            raise
        except Exception as e:
            if sid:
                bridge.finish(sid, error=str(e))
            raise
        finally:
            self._voice_agent_busy = False
            self._refresh_voice_gate_sync()

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
            else:
                self._panel.append_token(item)

        # Voice events from the bg STT loop: "__wake__" means the wake
        # word just fired (surface the panel + show a cue), anything
        # else is a final transcript to submit.
        while not self._voice_prompt_queue.empty():
            item = self._voice_prompt_queue.get_nowait()
            if item == "__wake__":
                if not self._panel.isVisible():
                    self._panel.toggle()
                self._panel.on_tool_status("🎙 listening...")
                continue
            if item == "__idle__":
                # Utterance ended with no transcript (silence after wake,
                # or Whisper VAD filtered it out). Clear the "listening..."
                # spinner so the panel doesn't look hung while we wait for
                # the next wake word. Skip if we're already streaming a
                # response - that spinner belongs to the LLM turn.
                if not self._panel._streaming:
                    self._panel._stop_spinner()
                    self._panel._label.setText(
                        "Andrew" if self._panel._response_md else "Ask Andrew"
                    )
                continue
            # Typed submits run through panel._on_submit which preps
            # the conversation state (user bubble, "Thinking..."
            # spinner). Voice bypasses that, so we must do it manually
            # or the panel stays stuck on the "listening..." spinner.
            self._panel.show_user_message(item)
            self.submit(item)

        # Bridge inbox: messages injected via the HTTP server.
        from src.core import server as bridge
        while not bridge.inbox.empty():
            try:
                sid, message = bridge.inbox.get_nowait()
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
        return load_domain(name)

import asyncio
import queue
import sys
from pathlib import Path
from src.tray.bootstrap import init; init()
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from src.core.event import Event
from src.core.llm import ToolEvent, RouteEvent, format_tool_status
from src.core.registry import available_domains, load_domain
from src.shared.config import Config
from src.tray.icon import create_tray
from src.tray.panel import ChatPanel
from src.tray.worker import StreamWorker, get_event_loop

class AndrewTrayApp:

    def __init__(self, voice_enabled: bool = False):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.app.setStyleSheet(self._load_stylesheet())

        self._config = Config()
        self._loop = get_event_loop()
        self._domain_name = self._config.domain
        self.domain = self._create_domain(self._domain_name)

        # Voice I/O. The bg asyncio loop in worker.get_event_loop() is
        # the same one the StreamWorker uses for domain.generate, so
        # STT and TTS share it and token teeing stays lock-free.
        self.voice_enabled = voice_enabled
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
        if voice_enabled:
            from src.voice import build_voice_io
            self.stt, self.tts = build_voice_io(self._config)

        self.tray = create_tray(self.app, self._toggle, self.app.quit)
        self.panel = ChatPanel()
        self.panel.set_domain_name(self._domain_name)
        self.panel.submitted.connect(self._on_submit)
        self.panel.stop_requested.connect(self._on_stop)
        self.panel.clear_requested.connect(self._on_clear)
        self.panel.domain_switch.connect(self._on_domain_switch)
        if voice_enabled:
            self.panel.set_voice_enabled(True)
            self.panel.voice_toggle.connect(self._on_voice_toggle)
        self._worker = None

        # Thread-safe queues bridging the asyncio event bus to the Qt main thread
        self._event_notify_queue: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self._event_token_queue: queue.SimpleQueue = queue.SimpleQueue()

        self._event_poll_timer = QTimer(self.app)
        self._event_poll_timer.setInterval(100)
        self._event_poll_timer.timeout.connect(self._poll_event_queues)
        self._event_poll_timer.start()

        self._start_event_bus()
        if self.voice_enabled:
            self._start_voice_listener()
        self.tray.show()

    def _start_event_bus(self):
        bus = self.domain.event_bus
        bus.notify = self._event_notify
        bus.dispatch = self._event_dispatch
        asyncio.run_coroutine_threadsafe(bus.start(), self._loop)

    # -- voice listener (bg asyncio thread) ----------------------------------

    def _start_voice_listener(self):
        """Run STT forever on the bg loop; push transcripts to Qt main.

        Each completed utterance goes on ``_voice_prompt_queue`` and is
        picked up by the 100-ms poll timer, which submits it through the
        normal ``_on_submit`` path - exactly as if the user had typed it.
        On wake-word detection we also push a ``"__wake__"`` sentinel so
        the Qt thread can surface a visual cue before the user starts
        speaking.
        """
        import logging
        log = logging.getLogger("src.tray.voice")

        def _on_wake():
            log.info("voice: wake fired, enqueuing __wake__ sentinel")
            self._voice_prompt_queue.put("__wake__")

        async def _listener():
            log.info("voice listener task started on bg loop")
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
                        log.info("voice: STT paused (waiting for gate)")
                        await self._voice_idle_event.wait()
                        log.info("voice: STT resumed")
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
                        log.info("voice: listen_once cancelled by gate close")
                        text = ""
                    finally:
                        self._current_listen_task = None
                    log.info("voice: listen_once returned (len=%d): %r",
                             len(text or ""), (text or "")[:80])
                    if text and text.strip():
                        self._voice_prompt_queue.put(text)
                        log.info("voice: transcript enqueued for Qt poller")
                    else:
                        self._voice_prompt_queue.put("__idle__")
            except Exception:
                # Silent crashes inside a run_coroutine_threadsafe future
                # are the classic reason voice "just stops working"
                # mid-session. Log + re-raise so the root cause is visible
                # in ~/.andrewcli/tray.log.
                log.exception("voice listener crashed")
                raise
            finally:
                log.info("voice listener task exited")

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
            task = getattr(self, "_current_listen_task", None)
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

    def _event_notify(self, event: Event):
        self._event_notify_queue.put(event)

    async def _event_dispatch(self, event: Event):
        if not event.message:
            return
        # Pause voice for the duration of the event turn so the wake
        # word can't retrigger mid-generation and TTS output from the
        # reply doesn't bleed back in through the mic. This runs on
        # the bg loop already, so we flip the state + refresh the gate
        # synchronously (which also cancels any active listen_once).
        self._voice_agent_busy = True
        self._refresh_voice_gate_sync()
        try:
            async for token in self.domain.generate_event(event.message):
                self._event_token_queue.put(token)
            self._event_token_queue.put(None)  # sentinel: stream done
        finally:
            self._voice_agent_busy = False
            self._refresh_voice_gate_sync()

    # -- Qt main thread: poll queues every 100 ms ----------------------------

    def _poll_event_queues(self):
        while not self._event_notify_queue.empty():
            event = self._event_notify_queue.get_nowait()
            self.tray.showMessage(
                "Andrew",
                f"Event triggered: {event.name}",
            )
            if event.message:
                self.panel.start_event_response(event.name)
                if not self.panel.isVisible():
                    self.panel.toggle()

        while not self._event_token_queue.empty():
            item = self._event_token_queue.get_nowait()
            if item is None:
                self.panel.on_stream_done()
            elif isinstance(item, (RouteEvent, ToolEvent)):
                status = format_tool_status(item)
                if status is not None:
                    self.panel.on_tool_status(status)
            else:
                self.panel.append_token(item)

        # Voice events from the bg STT loop: "__wake__" means the wake
        # word just fired (surface the panel + show a cue), anything
        # else is a final transcript to submit.
        while not self._voice_prompt_queue.empty():
            item = self._voice_prompt_queue.get_nowait()
            if item == "__wake__":
                if not self.panel.isVisible():
                    self.panel.toggle()
                self.panel.on_tool_status("🎙 listening...")
                continue
            if item == "__idle__":
                # Utterance ended with no transcript (silence after wake,
                # or Whisper VAD filtered it out). Clear the "listening..."
                # spinner so the panel doesn't look hung while we wait for
                # the next wake word. Skip if we're already streaming a
                # response - that spinner belongs to the LLM turn.
                if not self.panel._streaming:
                    self.panel._stop_spinner()
                    self.panel._label.setText(
                        "Andrew" if self.panel._response_md else "Ask Andrew"
                    )
                continue
            # Typed submits run through panel._on_submit which preps
            # the conversation state (user bubble, "Thinking..."
            # spinner). Voice bypasses that, so we must do it manually
            # or the panel stays stuck on the "listening..." spinner.
            self.panel.show_user_message(item)
            self._on_submit(item)

    # -- domain ---------------------------------------------------------------

    def _create_domain(self, name):
        future = asyncio.run_coroutine_threadsafe(
            self._async_create_domain(name), self._loop,
        )
        return future.result(timeout=10)

    async def _async_create_domain(self, name):
        return load_domain(name)

    def _load_stylesheet(self):
        return (Path(__file__).parent / "style.css").read_text()

    # -- tray / panel ---------------------------------------------------------

    def _toggle(self):
        self.panel.toggle()

    def _on_stop(self):
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

    def _on_clear(self):
        self._on_stop()
        self.domain.llm.memory.clear()

    def _on_domain_switch(self):
        domains = available_domains()
        if len(domains) <= 1:
            return
        try:
            idx = domains.index(self._domain_name)
            next_name = domains[(idx + 1) % len(domains)]
        except ValueError:
            next_name = domains[0]
        try:
            self._on_stop()
            self.domain.event_bus.stop()
            self.domain = self._create_domain(next_name)
            self._domain_name = next_name
            self.panel.set_domain_name(next_name)
            self._start_event_bus()
        except ValueError as exc:
            # Domain module failed to import. Keep the current domain
            # alive rather than leaving the tray in a half-broken state.
            self.tray.showMessage("Andrew", f"Domain switch failed: {exc}")
            self._start_event_bus()

    def _on_submit(self, message):
        self._on_stop()
        # Pause voice for the whole user turn (generation + any tool
        # calls + TTS playback). Resumed on finished/error. This lets
        # us ignore the wake word mid-turn and prevents self-wake from
        # the speaker bleed-through of our own TTS output.
        self._set_voice_busy(True)
        # Pass tts only when voice is on; worker tees tokens to speaker.
        self._worker = StreamWorker(message, self.domain, tts=self.tts)
        self._worker.token_received.connect(self.panel.append_token)
        self._worker.tool_status.connect(self.panel.on_tool_status)
        self._worker.finished.connect(self.panel.on_stream_done)
        self._worker.finished.connect(lambda: self._set_voice_busy(False))
        self._worker.error.connect(self.panel.on_error)
        self._worker.error.connect(lambda _e: self._set_voice_busy(False))
        self._worker.start()

    def run(self):
        sys.exit(self.app.exec())


def main(voice_enabled: bool = False):
    app = AndrewTrayApp(voice_enabled=voice_enabled)
    app.run()


if __name__ == "__main__":
    main()

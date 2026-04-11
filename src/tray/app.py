import asyncio
import importlib
import os
import queue
import sys
from pathlib import Path
from src.tray.bootstrap import init; init()
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from src.core.event import Event
from src.core.llm import ToolEvent, RouteEvent
from src.shared.config import Config
from src.tray.icon import create_tray
from src.tray.panel import ChatPanel
from src.tray.worker import StreamWorker, get_event_loop


def _load_domain(name):
    module = importlib.import_module(f"src.domains.{name}")
    cls = getattr(module, f"{name.capitalize()}Domain")
    return cls()

def _available_domains():
    domains = []
    for f in os.listdir("src/domains"):
        if f.endswith(".py") and f != "__init__.py":
            domains.append(f[:-3])
    return sorted(domains)

class AndrewTrayApp:

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.app.setStyleSheet(self._load_stylesheet())

        self._config = Config()
        self._loop = get_event_loop()
        self._domain_name = self._config.domain
        self.domain = self._create_domain(self._domain_name)

        self.tray = create_tray(self.app, self._toggle, self.app.quit)
        self.panel = ChatPanel()
        self.panel.set_domain_name(self._domain_name)
        self.panel.submitted.connect(self._on_submit)
        self.panel.stop_requested.connect(self._on_stop)
        self.panel.clear_requested.connect(self._on_clear)
        self.panel.domain_switch.connect(self._on_domain_switch)
        self._worker = None

        # Thread-safe queues bridging the asyncio event bus to the Qt main thread
        self._event_notify_queue: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self._event_token_queue: queue.SimpleQueue = queue.SimpleQueue()

        self._event_poll_timer = QTimer(self.app)
        self._event_poll_timer.setInterval(100)
        self._event_poll_timer.timeout.connect(self._poll_event_queues)
        self._event_poll_timer.start()

        self._start_event_bus()
        self.tray.show()

    def _start_event_bus(self):
        bus = self.domain.event_bus
        bus.notify = self._event_notify
        bus.dispatch = self._event_dispatch
        asyncio.run_coroutine_threadsafe(bus.start(), self._loop)

    # -- event bus callbacks (called from asyncio thread) ---------------------

    def _event_notify(self, event: Event):
        self._event_notify_queue.put(event)

    async def _event_dispatch(self, event: Event):
        if not event.message:
            return
        async for token in self.domain.generate_event(event.message):
            self._event_token_queue.put(token)
        self._event_token_queue.put(None)  # sentinel: stream done

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
            elif isinstance(item, RouteEvent):
                if item.tool_names:
                    self.panel.on_tool_status(f"Loading: {', '.join(item.tool_names)}")
            elif isinstance(item, ToolEvent):
                if item.tool_name:
                    first_val = str(next(iter(item.tool_args.values()), "")) if item.tool_args else ""
                    if len(first_val) > 60:
                        first_val = first_val[:57] + "..."
                    detail = f": {first_val}" if first_val else ""
                    self.panel.on_tool_status(f"Running {item.tool_name}{detail}")
                else:
                    self.panel.on_tool_status("Thinking...")
            else:
                self.panel.append_token(item)

    # -- domain ---------------------------------------------------------------

    def _create_domain(self, name):
        future = asyncio.run_coroutine_threadsafe(
            self._async_create_domain(name), self._loop,
        )
        return future.result(timeout=10)

    async def _async_create_domain(self, name):
        return _load_domain(name)

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

    def _on_clear(self):
        self._on_stop()
        self.domain.llm.memory.clear()

    def _on_domain_switch(self):
        domains = _available_domains()
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
            self._domain_name = next_name
            self.domain = self._create_domain(next_name)
            self.panel.set_domain_name(next_name)
            self._start_event_bus()
        except Exception:
            pass

    def _on_submit(self, message):
        self._on_stop()
        self._worker = StreamWorker(message, self.domain)
        self._worker.token_received.connect(self.panel.append_token)
        self._worker.tool_status.connect(self.panel.on_tool_status)
        self._worker.finished.connect(self.panel.on_stream_done)
        self._worker.error.connect(self.panel.on_error)
        self._worker.start()

    def run(self):
        sys.exit(self.app.exec())


def main():
    app = AndrewTrayApp()
    app.run()


if __name__ == "__main__":
    main()

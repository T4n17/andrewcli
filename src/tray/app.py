import asyncio
import importlib
import os
import sys
from pathlib import Path
from src.tray.bootstrap import init; init()
from PyQt6.QtWidgets import QApplication
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
        self.tray.show()

    def _create_domain(self, name):
        future = asyncio.run_coroutine_threadsafe(
            self._async_create_domain(name), self._loop,
        )
        return future.result(timeout=10)

    async def _async_create_domain(self, name):
        return _load_domain(name)

    def _load_stylesheet(self):
        return (Path(__file__).parent / "style.css").read_text()

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
            self._domain_name = next_name
            self.domain = self._create_domain(next_name)
            self.panel.set_domain_name(next_name)
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

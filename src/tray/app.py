"""Standalone tray shell on top of :class:`TrayController`.

This module's only job is the standalone-process concerns:

* construct the :class:`QApplication` and configure its global font,
* build the :class:`ChatPanel` as a top-level frameless window,
* build the :class:`QSystemTrayIcon` and wire its menu actions,
* surface event notifications as tray balloons,
* call ``app.exec()`` and propagate the exit code.

All chat-orchestration, voice gating, event-bus bridging, and
:class:`StreamWorker` lifecycle live in :class:`TrayController` so they
can be reused by other shells (e.g. a desktop-widget host) that embed
:class:`ChatPanel` as a child widget.
"""
import sys
from pathlib import Path
from src.tray.bootstrap import init; init()
from PyQt6.QtWidgets import QApplication

from src.shared.config import Config
from src.tray.controller import TrayController
from src.tray.icon import create_tray
from src.tray.panel import ChatPanel
# Re-exported for backwards compatibility with code that imports
# ``StreamWorker`` from ``src.tray.app``.
from src.tray.worker import StreamWorker  # noqa: F401


class AndrewTrayApp:
    """Thin shell: owns the QApplication + tray icon + ChatPanel window."""

    def __init__(self, voice_enabled: bool = False):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.app.setStyleSheet(self._load_stylesheet())
        self._configure_font()

        self._config = Config()
        self.panel = ChatPanel()
        self.tray = create_tray(self.app, self._toggle, self._quit)

        self.controller = TrayController(
            panel=self.panel,
            config=self._config,
            voice_enabled=voice_enabled,
            parent=self.app,
            on_event_notification=self._on_event_notification,
        )
        self.controller.start()
        self.tray.show()

    def _configure_font(self):
        # Application-wide font with an explicit emoji-font fallback
        # list. Qt's automatic fallback often skips color-emoji fonts
        # on Linux (it picks the first family that *exists* rather than
        # the first one that has the glyph), so the model's emoji bytes
        # render as tofu boxes. ``QFont.setFamilies`` makes Qt iterate
        # the list per-glyph: the first family supplies normal text,
        # the emoji families supply the pictographs.
        ui_font = self.app.font()
        ui_font.setFamilies([
            ui_font.family(),
            "Noto Color Emoji",
            "Noto Emoji",
            "Symbola",
            "sans-serif",
        ])
        self.app.setFont(ui_font)

    def _on_event_notification(self, event):
        """Standalone reaction to a background event firing.

        The controller calls this from the Qt main thread (inside its
        100 ms poll). We surface a tray balloon with the event name
        and, if the event wants to start an agent turn, raise the
        chat panel so the user sees the streamed response.
        """
        self.tray.showMessage("Andrew", f"Event triggered: {event.name}")
        if event.message and not self.panel.isVisible():
            self.panel.toggle()

    def _toggle(self):
        self.panel.toggle()

    def _quit(self):
        # Tear down the controller first (stops poll timer, voice
        # listener, worker, event bus) so background tasks can't fire
        # callbacks on a dying QApplication.
        self.controller.shutdown()
        self.app.quit()

    def _load_stylesheet(self):
        return (Path(__file__).parent / "style.css").read_text()

    def run(self):
        sys.exit(self.app.exec())


def main(voice_enabled: bool = False):
    app = AndrewTrayApp(voice_enabled=voice_enabled)
    app.run()


if __name__ == "__main__":
    main()

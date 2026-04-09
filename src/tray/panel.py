from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextBrowser, QLineEdit, QLabel, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from src.shared.config import Config

_DIR = Path(__file__).parent


class ChatPanel(QWidget):
    submitted = pyqtSignal(str)
    stop_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._config = Config()
        self._md_css = (_DIR / "md.css").read_text()

        self.setObjectName("ChatPanel")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowOpacity(self._config.tray_opacity)
        self._expanded = False
        self._streaming = False
        self._response_md = ""
        self._spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_idx = 0
        self._spinner_text = "Thinking..."

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(80)
        self._spinner_timer.timeout.connect(self._tick_spinner)

        self._build_ui()
        self._set_compact()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel("Ask Andrew")
        self._label.setObjectName("PanelLabel")
        header.addWidget(self._label)
        header.addStretch()

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("StopBtn")
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.hide()
        header.addWidget(self._stop_btn)

        header.addSpacing(6)

        self._toggle_btn = QPushButton("\u25BD")
        self._toggle_btn.setObjectName("CloseBtn")
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.clicked.connect(self._toggle_expand)
        self._toggle_btn.hide()
        header.addWidget(self._toggle_btn)

        header.addSpacing(4)

        self._close_btn = QPushButton("\u2715")
        self._close_btn.setObjectName("CloseBtn")
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.clicked.connect(self._hide)
        self._close_btn.hide()
        header.addWidget(self._close_btn)

        layout.addLayout(header)

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        self._browser.document().setDefaultStyleSheet(self._md_css)
        self._browser.hide()
        layout.addWidget(self._browser)

        self._entry = QLineEdit()
        self._entry.setObjectName("InputEntry")
        self._entry.setPlaceholderText("Type your message and press Enter...")
        self._entry.returnPressed.connect(self._on_submit)
        layout.addWidget(self._entry)

    # -- spinner --------------------------------------------------------------

    def _start_spinner(self, text="Thinking..."):
        self._spinner_text = text
        self._spinner_idx = 0
        self._tick_spinner()
        self._spinner_timer.start()

    def _stop_spinner(self):
        self._spinner_timer.stop()

    def _tick_spinner(self):
        frame = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
        self._label.setText(f"{frame} {self._spinner_text}")
        self._spinner_idx += 1

    # -- state transitions ---------------------------------------------------

    def _set_compact(self):
        self._expanded = False
        self._browser.hide()
        self._toggle_btn.setText("\u25B3")
        self._toggle_btn.show()
        self._close_btn.show()
        self._stop_btn.hide()
        if not self._streaming:
            self._stop_spinner()
            self._label.setText("Ask Andrew")
        self.setFixedSize(
            self._config.tray_width_compact,
            self._config.tray_height_compact,
        )

    def _set_expanded(self):
        self._expanded = True
        self._browser.show()
        self._toggle_btn.setText("\u25BD")
        self._toggle_btn.show()
        self._close_btn.show()
        if self._streaming:
            self._stop_btn.show()
        self.setFixedSize(
            self._config.tray_width_expanded,
            self._config.tray_height_expanded,
        )
        self._position()

    def _position(self):
        margin = 12
        screen = QApplication.primaryScreen().geometry()
        sw, sh = screen.width(), screen.height()
        w, h = self.width(), self.height()

        pos = self._config.tray_position
        v, hz = "top", "right"
        parts = pos.replace("-", " ").split()
        if len(parts) == 1:
            if parts[0] in ("top", "center", "bottom"):
                v = parts[0]
            else:
                hz = parts[0]
        elif len(parts) >= 2:
            v, hz = parts[0], parts[1]

        if hz == "left":
            x = margin
        elif hz == "right":
            x = sw - w - margin
        else:
            x = (sw - w) // 2

        if v == "top":
            y = margin
        elif v == "bottom":
            y = sh - h - margin
        else:
            y = (sh - h) // 2

        self.move(x, y)

    def _toggle_expand(self):
        if self._expanded:
            self._set_compact()
            self._position()
        else:
            self._set_expanded()
            self._browser.setMarkdown(self._response_md)
            sb = self._browser.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _hide(self):
        self.hide()

    # -- user actions ---------------------------------------------------------

    def _on_stop(self):
        self.stop_requested.emit()
        self._streaming = False
        self._stop_spinner()
        self._stop_btn.hide()
        self._label.setText("Andrew")
        self._entry.setPlaceholderText("Reply...")
        self._entry.setFocus()

    def _on_submit(self):
        text = self._entry.text().strip()
        if not text:
            return
        self._entry.clear()
        self._response_md = ""
        self._browser.setPlainText("")
        self._streaming = True
        self._start_spinner("Thinking...")
        self._stop_btn.show()
        self._set_expanded()
        self.submitted.emit(text)

    # -- slots for StreamWorker signals ---------------------------------------

    def append_token(self, token: str):
        if self._spinner_timer.isActive():
            self._stop_spinner()
            self._label.setText("Andrew")
        self._response_md += token
        if self.isVisible():
            self._browser.setMarkdown(self._response_md)
            sb = self._browser.verticalScrollBar()
            sb.setValue(sb.maximum())

    def on_tool_status(self, status: str):
        self._start_spinner(status)

    def on_stream_done(self):
        self._streaming = False
        self._stop_spinner()
        self._stop_btn.hide()
        self._label.setText("Andrew")
        if self.isVisible():
            self._entry.setPlaceholderText("Reply...")
            self._entry.setFocus()

    def on_error(self, text: str):
        self._response_md = f"**Error:** {text}"
        self._streaming = False
        self._stop_spinner()
        self._stop_btn.hide()
        self._label.setText("Error")
        if self.isVisible():
            self._browser.setMarkdown(self._response_md)
            self._entry.setFocus()

    # -- public interface -----------------------------------------------------

    def toggle(self):
        if self.isVisible():
            self._hide()
            return
        if self._response_md:
            self._set_expanded()
            self._browser.setMarkdown(self._response_md)
            sb = self._browser.verticalScrollBar()
            sb.setValue(sb.maximum())
        else:
            self._set_compact()
            self._position()
        self.show()
        self.activateWindow()
        self.raise_()
        self._entry.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._hide()
        else:
            super().keyPressEvent(event)

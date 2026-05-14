from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextBrowser, QLineEdit, QLabel, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

from src.shared.config import Config
from src.shared.paths import DATA_DIR

_DIR = Path(__file__).parent
_CONVO_FILE = DATA_DIR / "conversation.md"


class ChatPanel(QWidget):
    submitted = pyqtSignal(str)
    stop_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    domain_switch = pyqtSignal()
    # Emitted by ``_hide()`` when the panel was constructed with
    # ``embedded=True``. Lets a host widget decide what "hide" means
    # in its layout (collapse a slot, swap pages, ignore, ...) instead
    # of having ChatPanel call ``self.hide()`` on a child widget that
    # the host expects to manage itself.
    hide_requested = pyqtSignal()

    def __init__(self, parent=None, *, embedded: bool = False):
        super().__init__(parent)
        self._config = Config()
        self._md_css = (_DIR / "md.css").read_text()
        # When True, ChatPanel behaves as a child widget: no frameless
        # window flags, no top-level opacity, no screen-edge
        # positioning, and ``_hide()`` emits ``hide_requested`` instead
        # of calling ``self.hide()``. Default False keeps the
        # standalone tray's behavior byte-for-byte identical.
        self._embedded = embedded

        self.setObjectName("ChatPanel")
        if not embedded:
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

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(30)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._flush_render)
        self._render_cursor_pos = 0

        self._history: list[str] = []
        self._history_idx: int = 0
        self._history_saved: str = ""

        self._response_md = self._load_conversation()
        self._build_ui()
        self._set_compact()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        self._domain_btn = QPushButton("general")
        self._domain_btn.setObjectName("DomainBtn")
        self._domain_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._domain_btn.clicked.connect(lambda: self.domain_switch.emit())
        header.addWidget(self._domain_btn)

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

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("StopBtn")
        self._clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_btn.clicked.connect(self._on_clear)
        self._clear_btn.hide()
        header.addWidget(self._clear_btn)

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
        # The streaming render path (``cursor.insertText``) uses the
        # document's default font, not the CSS in md.css - so emoji
        # in mid-stream tokens render as tofu boxes unless we attach
        # the same fallback list here too. ``setMarkdown`` (called on
        # stream-done) goes through CSS and picks up md.css's family
        # list, so this covers the streaming half of the pipeline.
        browser_font = self._browser.font()
        browser_font.setFamilies([
            browser_font.family(),
            "Noto Color Emoji",
            "Noto Emoji",
            "Symbola",
            "sans-serif",
        ])
        self._browser.document().setDefaultFont(browser_font)
        self._browser.hide()
        layout.addWidget(self._browser)

        self._entry = QLineEdit()
        self._entry.setObjectName("InputEntry")
        self._entry.setPlaceholderText("Type your message and press Enter...")
        self._entry.returnPressed.connect(self._on_submit)
        self._entry.installEventFilter(self)
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
        self._clear_btn.hide()
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
        if self._response_md:
            self._clear_btn.show()
        self.setFixedSize(
            self._config.tray_width_expanded,
            self._config.tray_height_expanded,
        )
        self._position()

    def _position(self):
        # Embedded mode: the host widget owns layout/positioning, so
        # screen-relative placement would either fight the host's
        # geometry or move the host's whole window. Bail out.
        if self._embedded:
            return
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
        # Embedded mode: defer the hide policy to the host widget.
        # Calling ``self.hide()`` on a child widget often surprises the
        # host's layout (sudden gap, wrong focus) - emitting a signal
        # lets the host swap pages, collapse a slot, or ignore the
        # request entirely.
        if self._embedded:
            self.hide_requested.emit()
            return
        self.hide()

    # -- user actions ---------------------------------------------------------

    def _on_stop(self):
        self.stop_requested.emit()
        self._streaming = False
        self._stop_spinner()
        self._render_timer.stop()
        self._render_cursor_pos = len(self._response_md)
        if self.isVisible():
            self._browser.setMarkdown(self._response_md)
            sb = self._browser.verticalScrollBar()
            sb.setValue(sb.maximum())
        self._stop_btn.hide()
        self._label.setText("Andrew")
        self._entry.setPlaceholderText("Reply...")
        self._entry.setFocus()

    def _on_clear(self):
        self._response_md = ""
        self._render_cursor_pos = 0
        self._browser.setPlainText("")
        self._clear_btn.hide()
        self._save_conversation()
        self.clear_requested.emit()

    def _on_submit(self):
        text = self._entry.text().strip()
        if not text:
            return
        self._history.append(text)
        self._history_idx = len(self._history)
        self._history_saved = ""
        self._entry.clear()
        self.show_user_message(text)
        self.submitted.emit(text)

    def show_user_message(self, text: str):
        """Prepare the panel for a new user message from any source.

        Appends the user's line to the conversation, flips the panel
        into streaming state, and swaps the status spinner to
        "Thinking...". Callable from outside the panel so server-injected
        messages produce the same visual timeline as typed ones.
        """
        if self._response_md:
            self._response_md += "\n\n---\n\n"
        self._response_md += f"**You:** {text}\n\n**Andrew:** "
        self._browser.setMarkdown(self._response_md)
        sb = self._browser.verticalScrollBar()
        sb.setValue(sb.maximum())
        self._render_cursor_pos = len(self._response_md)
        self._streaming = True
        self._start_spinner("Thinking...")
        self._stop_btn.show()
        self._set_expanded()

    # -- slots for StreamWorker signals ---------------------------------------

    def append_token(self, token: str):
        if self._spinner_timer.isActive():
            self._stop_spinner()
            self._label.setText("Andrew")
        self._response_md += token
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _flush_render(self):
        if not self._response_md or not self.isVisible():
            return
        sb = self._browser.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 2
        self._browser.setMarkdown(self._response_md)
        self._render_cursor_pos = len(self._response_md)
        if at_bottom:
            sb.setValue(sb.maximum())

    def on_tool_status(self, status: str):
        self._start_spinner(status)

    def on_stream_done(self):
        self._streaming = False
        self._stop_spinner()
        self._render_timer.stop()
        self._render_cursor_pos = len(self._response_md)
        if self.isVisible():
            self._browser.setMarkdown(self._response_md)
            sb = self._browser.verticalScrollBar()
            sb.setValue(sb.maximum())
        self._stop_btn.hide()
        self._label.setText("Andrew")
        self._save_conversation()
        if self.isVisible():
            self._entry.setPlaceholderText("Reply...")
            self._entry.setFocus()

    def on_error(self, text: str):
        # Append the error to the existing conversation instead of wiping
        # it. The old behavior persisted a bare "**Error:** ..." markdown
        # file to disk on the next turn, destroying history.
        if self._response_md and not self._response_md.endswith("\n"):
            self._response_md += "\n\n"
        self._response_md += f"**Error:** {text}\n"
        self._streaming = False
        self._stop_spinner()
        self._render_timer.stop()
        self._render_cursor_pos = len(self._response_md)
        self._stop_btn.hide()
        self._label.setText("Error")
        if self.isVisible():
            self._browser.setMarkdown(self._response_md)
            sb = self._browser.verticalScrollBar()
            sb.setValue(sb.maximum())
            self._entry.setFocus()
        self._save_conversation()

    def start_event_response(self, event_name: str):
        """Prepare the panel to receive a streamed agent response from an event."""
        if self._response_md:
            self._response_md += "\n\n---\n\n"
        self._response_md += f"**◆ Event [{event_name}]:** "
        self._render_cursor_pos = len(self._response_md)
        self._streaming = True
        self._start_spinner(f"Event: {event_name}")
        self._stop_btn.show()
        if not self._expanded:
            self._set_expanded()
        else:
            self._browser.setMarkdown(self._response_md)

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

    # -- persistence ----------------------------------------------------------

    @staticmethod
    def _load_conversation():
        try:
            return _CONVO_FILE.read_text()
        except FileNotFoundError:
            return ""

    def _save_conversation(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _CONVO_FILE.write_text(self._response_md)

    def set_domain_name(self, name: str):
        self._domain_btn.setText(name)

    def eventFilter(self, obj, event):
        if obj is self._entry and event.type() == event.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Tab:
                self.domain_switch.emit()
                return True
            if key == Qt.Key.Key_Up and self._history:
                if self._history_idx == len(self._history):
                    self._history_saved = self._entry.text()
                self._history_idx = max(0, self._history_idx - 1)
                self._entry.setText(self._history[self._history_idx])
                self._entry.end(False)
                return True
            if key == Qt.Key.Key_Down and self._history:
                self._history_idx = min(len(self._history), self._history_idx + 1)
                if self._history_idx == len(self._history):
                    self._entry.setText(self._history_saved)
                else:
                    self._entry.setText(self._history[self._history_idx])
                self._entry.end(False)
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._hide()
        else:
            super().keyPressEvent(event)

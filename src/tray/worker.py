import asyncio
import threading

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.llm import ToolEvent, RouteEvent, format_tool_status


_loop = None
_loop_thread = None


def get_event_loop():
    global _loop, _loop_thread
    if _loop is None:
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
        _loop_thread.start()
    return _loop


class StreamWorker(QThread):
    token_received = pyqtSignal(str)
    tool_status = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, message, domain):
        super().__init__()
        self.message = message
        self.domain = domain
        self._future = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        if self._future:
            self._future.cancel()

    def run(self):
        loop = get_event_loop()
        self._future = asyncio.run_coroutine_threadsafe(self._stream(), loop)
        try:
            self._future.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))

    async def _stream(self):
        async for token in self.domain.generate(self.message):
            if self._cancelled:
                return
            if isinstance(token, (RouteEvent, ToolEvent)):
                status = format_tool_status(token)
                if status is not None:
                    self.tool_status.emit(status)
                continue
            self.token_received.emit(token)
        self.finished.emit()

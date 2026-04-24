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

    def __init__(self, message, domain, tts=None):
        super().__init__()
        self.message = message
        self.domain = domain
        # Optional TTS backend; when set, each text token is also fed
        # into its sentence-level streaming queue so the chat panel
        # and the speaker hear the same response simultaneously.
        self.tts = tts
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
        tts_queue = None
        tts_task = None
        if self.tts is not None:
            from src.voice import strip_markdown
            tts_queue = asyncio.Queue()

            async def _feed():
                while True:
                    tok = await tts_queue.get()
                    if tok is None:
                        return
                    yield tok

            # Render markdown *before* TTS so the speaker doesn't say
            # "asterisk asterisk bold asterisk asterisk"; strip_markdown
            # is a stateful char-level filter that removes ``*_~`#``
            # and elides ``(url)`` after a link ``]``.
            tts_task = asyncio.create_task(
                self.tts.speak_stream(strip_markdown(_feed()))
            )

        try:
            async for token in self.domain.generate(self.message):
                if self._cancelled:
                    return
                if isinstance(token, (RouteEvent, ToolEvent)):
                    status = format_tool_status(token)
                    if status is not None:
                        self.tool_status.emit(status)
                    continue
                self.token_received.emit(token)
                if tts_queue is not None:
                    await tts_queue.put(token)
        finally:
            if tts_queue is not None:
                await tts_queue.put(None)  # sentinel
                if self._cancelled and self.tts is not None:
                    # User hit Stop: kill the speaker mid-sentence.
                    try:
                        await self.tts.stop()
                    except Exception:
                        pass
                if tts_task is not None:
                    try:
                        await tts_task
                    except (asyncio.CancelledError, Exception):
                        pass
        self.finished.emit()

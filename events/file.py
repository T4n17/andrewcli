import asyncio
import os
from src.core.event import Event


class FileEvent(Event):
    """Fires when a watched file is modified."""

    name = "file"

    def __init__(self, path: str, poll_interval: float = 2.0, message: str = "File has been modified"):
        self.path = path
        self.poll_interval = poll_interval
        self.message = message
        # Description reflects the actual watched path so logs/UX are accurate.
        self.description = f"Fires when {path} is modified"
        self._last_mtime = self._mtime()

    def _mtime(self) -> float:
        try:
            return os.path.getmtime(self.path)
        except FileNotFoundError:
            return 0.0

    async def condition(self):
        while True:
            await asyncio.sleep(self.poll_interval)
            mtime = self._mtime()
            if mtime != self._last_mtime:
                self._last_mtime = mtime
                return  # condition met — EventBus will call trigger()

    async def trigger(self):
        pass  # notification and agent message are handled by EventBus

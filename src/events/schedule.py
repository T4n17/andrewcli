import asyncio
from datetime import datetime

from src.core.event import Event

_FMT = "%d-%m-%Y-%H-%M"


class ScheduleEvent(Event):
    """Fires a message once at a specific datetime, then stops.

    Usage: /schedule "<message>" dd-mm-yyyy-hh-mm
    Example: /schedule "Run skill 1" 10-10-2026-11-30
    """

    name = "schedule"

    def __init__(self, message: str = "", when: str = ""):
        if not when:
            raise ValueError('Usage: /schedule "<message>" dd-mm-yyyy-hh-mm')
        try:
            self.when = datetime.strptime(when, _FMT)
        except ValueError:
            raise ValueError(
                f"Invalid datetime {when!r}. Expected format: dd-mm-yyyy-hh-mm"
            )
        self.message = message
        self._fired = False
        self.description = (
            f"Scheduled {self.when.strftime('%d/%m/%Y %H:%M')}: {message[:50]}"
        )

    async def condition(self):
        if self._fired:
            raise asyncio.CancelledError
        delta = (self.when - datetime.now()).total_seconds()
        if delta > 0:
            await asyncio.sleep(delta)

    async def trigger(self):
        self._fired = True

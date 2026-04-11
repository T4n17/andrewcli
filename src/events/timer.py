import asyncio
from src.core.event import Event


class TimerEvent(Event):
    """Fires on a fixed interval and asks the agent for a brief update."""

    name = "timer"
    description = "Fires every N seconds"
    message = "Give me the actual date and time"

    def __init__(self, interval: float = 20.0):
        self.interval = interval

    async def condition(self):
        await asyncio.sleep(self.interval)

    async def trigger(self):
        pass  # notification and agent message are handled by EventBus

import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Awaitable


class Event(ABC):
    name: str
    description: str
    message: str = ""  # if set, sent to the agent after trigger fires

    @abstractmethod
    async def condition(self):
        """Await until the event should fire.

        Blocks until the triggering condition is satisfied; called again
        after each trigger, so it should naturally yield control via
        asyncio.sleep, asyncio.Event.wait, queue.get, etc.
        """

    @abstractmethod
    async def trigger(self):
        """Perform any side-effect when the condition fires.

        If the event only needs to message the agent, leave this as a no-op
        and set `message` instead.
        """


class EventBus:
    """Runs a list of Event instances as concurrent asyncio tasks.

    Set `notify` and `dispatch` before calling `start()` — these are
    injected by the app layer so that each surface (CLI, tray) can handle
    notification and rendering in its own way.

        bus.notify   = sync  (event: Event) -> None   — for UI notification
        bus.dispatch = async (event: Event) -> None   — for agent response

    Typical usage from Domain:
        self.event_bus = EventBus(self.events)

    Then in the app before starting:
        self.domain.event_bus.notify  = self._on_event_notify
        self.domain.event_bus.dispatch = self._on_event_dispatch
        asyncio.create_task(self.domain.event_bus.start())
    """

    def __init__(self, events: list[Event] = None):
        self._events: list[Event] = events or []
        self._tasks: list[asyncio.Task] = []
        self.notify: Callable[[Event], None] | None = None
        self.dispatch: Callable[[Event], Awaitable] | None = None

    async def start(self) -> None:
        """Run all events concurrently. Runs until cancelled."""
        if not self._events:
            return
        self._tasks = [
            asyncio.create_task(self._run(event), name=f"event:{event.name}")
            for event in self._events
        ]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def stop(self) -> None:
        """Cancel all running event tasks."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    async def _run(self, event: Event) -> None:
        while True:
            try:
                await event.condition()
                await event.trigger()
                if self.notify:
                    self.notify(event)
                if event.message and self.dispatch:
                    await self.dispatch(event)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[EventBus] error in event '{event.name}': {exc}")
                await asyncio.sleep(1)

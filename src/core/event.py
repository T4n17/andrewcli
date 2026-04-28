import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable, Awaitable

log = logging.getLogger(__name__)


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

    def add(self, event: Event) -> None:
        """Add and immediately start a new event on the running bus.

        Safe to call after start() — creates an independent asyncio task
        that is tracked in _tasks so stop() cancels it too.
        notify and dispatch must already be set before the event first fires.
        """
        self._events.append(event)
        task = asyncio.create_task(self._run(event), name=f"event:{event.name}")
        self._tasks.append(task)

    def remove(self, name: str) -> bool:
        """Cancel and remove the event with the given name.

        Returns True if found and cancelled, False if no such event is running.
        Task.cancel() is thread-safe; list mutation is safe here because only
        the owner thread (the one that calls add/remove/stop) ever mutates the
        lists — the bg asyncio loop only reads them.
        """
        for i, event in enumerate(self._events):
            if event.name == name:
                if i < len(self._tasks):
                    self._tasks[i].cancel()
                    self._tasks.pop(i)
                self._events.pop(i)
                return True
        return False

    def running(self) -> list[str]:
        """Return the names of all currently active (non-done) events."""
        return [
            event.name
            for event, task in zip(self._events, self._tasks)
            if not task.done()
        ]

    def stop(self) -> None:
        """Cancel all running event tasks."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        self._events.clear()

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
            except Exception:
                log.exception("error in event '%s'", event.name)
                await asyncio.sleep(1)

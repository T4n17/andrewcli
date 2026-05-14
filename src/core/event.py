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
    """Runs a set of Event instances as concurrent asyncio tasks.

    Events are independent of domains — the bus is created empty by
    ``Domain.__init__`` and populated at runtime through
    :py:meth:`add` (usually driven by user slash commands like
    ``/timer 30``).

    Each event instance gets a unique ID of the form ``name#N`` (e.g.
    ``loop#1``, ``loop#2``) so multiple instances of the same event type
    can run simultaneously. :py:meth:`remove` accepts either an exact
    instance ID or a bare name (which removes *all* instances of that
    type).

    Set `notify` and `dispatch` before calling `start()` — these are
    injected by the app layer so that each surface (CLI, tray) can handle
    notification and rendering in its own way.

        bus.notify   = sync  (event: Event) -> None   — for UI notification
        bus.dispatch = async (event: Event) -> None   — for agent response
    """

    def __init__(self):
        self._events: dict[str, Event] = {}         # instance_id -> Event
        self._tasks: dict[str, asyncio.Task] = {}   # instance_id -> Task
        self._counter = 0
        self.notify: Callable[[Event], None] | None = None
        self.dispatch: Callable[[Event], Awaitable] | None = None

    async def start(self) -> None:
        """No-op entry point kept for API compatibility.

        All event tasks are created and managed by :py:meth:`add`; there
        are no pre-seeded events to start here.
        """

    def add(self, event: Event) -> str:
        """Start a new event and return its unique instance ID.

        Safe to call after start() — creates an independent asyncio task
        that is tracked so :py:meth:`remove` / :py:meth:`stop` cancel it.
        notify and dispatch must already be set before the event first fires.
        """
        self._counter += 1
        instance_id = f"{event.name}#{self._counter}"
        event._instance_id = instance_id
        self._events[instance_id] = event
        task = asyncio.create_task(self._run(event), name=f"event:{instance_id}")
        self._tasks[instance_id] = task
        return instance_id

    def remove(self, key: str) -> bool:
        """Cancel and remove an event by instance ID or by name.

        If *key* matches an exact instance ID (e.g. ``loop#2``), only that
        instance is removed.  If *key* is a bare name (e.g. ``loop``), ALL
        running instances of that event type are removed.

        Returns True if at least one event was found and cancelled.
        """
        # Exact instance ID match
        if key in self._events:
            self._tasks[key].cancel()
            del self._tasks[key]
            del self._events[key]
            return True

        # Name match — remove all instances of this event type
        matches = [iid for iid, e in self._events.items() if e.name == key]
        if not matches:
            return False
        for iid in matches:
            self._tasks[iid].cancel()
            del self._tasks[iid]
            del self._events[iid]
        return True

    def running(self) -> list[str]:
        """Return instance IDs of all currently active (non-done) events."""
        return [
            iid for iid, task in self._tasks.items()
            if not task.done()
        ]

    def stop(self) -> None:
        """Cancel all running event tasks."""
        for task in self._tasks.values():
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
                log.exception(
                    "error in event '%s'",
                    getattr(event, "_instance_id", event.name),
                )
                await asyncio.sleep(1)

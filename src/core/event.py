import asyncio
import glob
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Callable, Awaitable, ClassVar

log = logging.getLogger(__name__)


class Event(ABC):
    name: str
    description: str
    message: str = ""        # if set, used as the user-turn trigger for dispatch
    system_message: str = "" # if set, injected as system-level instructions in the fresh event LLM
    required_tools: list[str] = []  # tool names that must be available regardless of routing

    def on_response(self, response: str) -> None:
        """Called after each dispatch with the model's full text output.

        Override to extract state from model text as a fallback when tool
        calls fail or the model outputs structured data as plain text.
        """

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
                if (event.message or event.system_message) and self.dispatch:
                    await self.dispatch(event)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception(
                    "error in event '%s'",
                    getattr(event, "_instance_id", event.name),
                )
                await asyncio.sleep(1)


class StatefulEvent(Event):
    """Base for events backed by a JSON state file with self-healing reconciliation.

    Lifecycle:
      Planning   — _snapshot is None; model writes the initial state file.
      Execution  — _snapshot is set; condition() reconciles the file each iteration.
      Done       — _summary_sent is True; next condition() raises CancelledError.

    Subclasses must implement five abstract methods:

    _parse() -> model | None
        Read and validate the state file. Return None if missing or invalid.

    _reconcile_raw(raw: dict) -> dict
        Called with a valid raw dict. Enforce invariants (immutable fields from
        the snapshot, monotonic counters, sticky flags). May update self state
        as a side effect. Return the corrected dict to write back to disk.

    _salvage_progress(text: str) -> None
        Called when the file exists but contains invalid JSON. Extract whatever
        progress can be recovered from the raw text and update in-memory state
        (e.g. self._snapshot["done_ids"]).

    _rebuild_raw() -> dict
        Return a clean, valid dict to write when the file is missing or corrupt.
        Should reflect current in-memory state.

    _compute_both() -> tuple[str, str]
        Return (system_message, user_message) for the current iteration.
        All prompt logic lives here. Called once per iteration; result is cached.

    Subclasses must also re-declare these class variables as their own:
        _session_files: set[str] = set()       # per-type file collision avoidance
        _state_file_default: str = "state.json" # triggers auto-suffix when unchanged
    """

    required_tools: list[str] = ["write_file"]
    _session_files: ClassVar[set[str]] = set()
    _state_file_default: ClassVar[str] = "state.json"

    # ------------------------------------------------------------------ init

    def _init_state_file(self, goal: str, state_file: str) -> None:
        """Initialise common state-file tracking. Call from subclass __init__."""
        self._state_file_arg = state_file
        self._use_instance_suffix = (
            bool(goal) and state_file == self._state_file_default
        )
        self.state_file = os.path.abspath(state_file)
        self._summary_sent: bool = False
        self._plan_sent: bool = False
        self._plan_poll_count: int = 0
        self._cache: tuple[str, str] | None = None
        self._snapshot: dict | None = None

    # ------------------------------------------------------------------ file I/O

    @staticmethod
    def _find_state_file(default_path: str) -> str | None:
        """Return a state-file path to resume from, or None if none found.

        Tries the exact path first, then numbered variants (e.g. state_1.json).
        Raises ValueError when multiple candidates exist.
        """
        if os.path.exists(default_path):
            return default_path
        base, ext = os.path.splitext(default_path)
        candidates = sorted(glob.glob(f"{base}_*{ext}"))
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        names = ", ".join(os.path.basename(c) for c in candidates)
        raise ValueError(
            f"Multiple state files found: {names}. "
            f"Pass the path explicitly to resume a specific one."
        )

    def _load_raw(self) -> dict | None:
        """Read the state file as a raw dict; return None if missing or invalid JSON."""
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write(self, raw: dict) -> None:
        """Write a dict to the state file as indented JSON."""
        with open(self.state_file, "w") as f:
            json.dump(raw, f, indent=2)

    # ------------------------------------------------------------------ reconciliation

    def _reconcile(self) -> None:
        """Enforce state-file invariants; rebuild automatically if the file is corrupt.

        Called at the start of every condition() so the model always reads
        valid, authoritative state on its next iteration.

        * Valid JSON  → _reconcile_raw() enforces invariants, result written back.
        * Missing / invalid JSON → _salvage_progress() recovers in-memory state
          from raw text, then _rebuild_raw() writes a clean replacement.
        """
        if self._snapshot is None:
            return
        raw = self._load_raw()
        if raw is None:
            try:
                with open(self.state_file) as f:
                    self._salvage_progress(f.read())
            except OSError:
                pass
            self._write(self._rebuild_raw())
            return
        self._write(self._reconcile_raw(raw))

    # ------------------------------------------------------------------ abstract

    @abstractmethod
    def _parse(self):
        """Parse the state file into a typed model. Return None if missing/invalid."""

    @abstractmethod
    def _reconcile_raw(self, raw: dict) -> dict:
        """Enforce invariants on a valid raw dict and return the corrected dict."""

    @abstractmethod
    def _salvage_progress(self, text: str) -> None:
        """Extract progress from invalid JSON text and update in-memory state."""

    @abstractmethod
    def _rebuild_raw(self) -> dict:
        """Return a clean dict to write when the state file is missing or corrupt."""

    @abstractmethod
    def _compute_both(self) -> tuple[str, str]:
        """Return (system_message, user_message) for the current iteration."""

    # ------------------------------------------------------------------ message caching

    def _ensure_cache(self) -> tuple[str, str]:
        if self._cache is None:
            self._cache = self._compute_both()
        return self._cache

    @property
    def system_message(self) -> str:  # type: ignore[override]
        return self._ensure_cache()[0]

    @property
    def message(self) -> str:  # type: ignore[override]
        return self._ensure_cache()[1]

    # ------------------------------------------------------------------ event interface

    async def condition(self) -> None:
        cls = type(self)
        if self._use_instance_suffix:
            base, ext = os.path.splitext(os.path.abspath(self._state_file_arg))
            n = 1
            while True:
                candidate = f"{base}_{n}{ext}"
                if not os.path.exists(candidate) and candidate not in cls._session_files:
                    break
                n += 1
            self.state_file = candidate
            cls._session_files.add(candidate)
            self._use_instance_suffix = False

        self._reconcile()
        self._cache = None

        if self._summary_sent:
            raise asyncio.CancelledError

        if self._plan_sent and self._snapshot is None:
            await asyncio.sleep(1)
            self._plan_poll_count += 1
            if self._plan_poll_count >= 10:
                self._plan_sent = False
                self._plan_poll_count = 0

    async def trigger(self) -> None:
        pass

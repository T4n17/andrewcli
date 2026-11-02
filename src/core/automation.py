"""Automations — self-scheduling runs of prompts, events, and workflows.

An automation is a YAML file in ``~/.config/andrewcli/automations/``
that pairs one or more **triggers** with a single **run** string::

    name: daily_brief                  # optional, defaults to the file name
    description: Morning news summary  # optional, shown in /automations
    enabled: true                      # optional, default true
    domain: general                    # optional str or list; default: every domain
    silent: false                      # optional, log to /status only
    tools: [google_search]             # optional, tools forced into prompt runs
    trigger:
      at: "08:30"
      days: weekdays
    run: "Summarize today's top headlines."

``run`` is interpreted exactly like typed input: text starting with
``/`` is a slash command (built-in, workflow, or event), anything else
is a prompt answered by an isolated, memory-free generation — the same
path background events use.

Supported triggers (a single mapping, or a list of them for "whichever
comes first"):

* ``startup``                       — once per session, when first armed
* ``{every: "30m"}``                — fixed interval (``90s``, ``2h``, ``1h30m``, seconds)
* ``{at: "08:30", days: weekdays}`` — wall-clock time(s), optional day filter
* ``{cron: "*/15 9-17 * * mon-fri"}`` — five-field cron expression
* ``{once: "2026-12-01 09:00"}``    — single datetime; skipped if already past

Each armed automation runs as an :class:`AutomationEvent` on the active
domain's :class:`~src.core.event.EventBus`, so it shares the event log
(``/status``), ``/stop``, notifications, and the ``busy_lock``
serialization with every other background job.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import yaml

from src.core.event import Event

# Upper bound for a single sleep while waiting for the next run. The
# monotonic clock asyncio uses does not advance during system suspend, so
# long sleeps are split up and re-checked against the wall clock.
_MAX_SLEEP = 30.0

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([smhd])")
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

_WEEKDAYS = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DAY_ALIASES = {
    "daily": "*", "everyday": "*", "*": "*",
    "weekdays": "mon-fri", "weekends": "sat,sun",
}


# ----------------------------------------------------------------------
# Schedules
# ----------------------------------------------------------------------


class CronSchedule:
    """Five-field cron expression: ``minute hour day-of-month month day-of-week``.

    Supports ``*``, lists (``1,15``), ranges (``9-17``), steps (``*/15``,
    ``0-30/10``), and three-letter month/weekday names. As in Vixie cron,
    when both day fields are restricted a day matches if *either* does.
    """

    _FIELDS = (
        (0, 59, {}),
        (0, 23, {}),
        (1, 31, {}),
        (1, 12, _MONTHS),
        (0, 7, _WEEKDAYS),
    )

    def __init__(self, expr: str):
        parts = str(expr).split()
        if len(parts) != 5:
            raise ValueError(f"cron expression {expr!r} must have 5 fields")
        minutes, hours, days, months, weekdays = (
            self._parse_field(part, lo, hi, names)
            for part, (lo, hi, names) in zip(parts, self._FIELDS)
        )
        self.expr = " ".join(parts)
        self.minutes, self.hours, self.days, self.months = minutes, hours, days, months
        self.weekdays = {d % 7 for d in weekdays}  # 7 is an alias for Sunday
        self._any_day = parts[2].startswith("*")
        self._any_weekday = parts[4].startswith("*")

    @staticmethod
    def _parse_field(text: str, lo: int, hi: int, names: dict) -> set[int]:
        def value(token: str) -> int:
            return names[token] if token in names else int(token)

        values: set[int] = set()
        for item in text.lower().split(","):
            span, has_step, step_text = item.partition("/")
            step = int(step_text) if has_step else 1
            if step < 1:
                raise ValueError(f"invalid cron step in {text!r}")
            if span == "*":
                start, end = lo, hi
            elif "-" in span:
                first, last = span.split("-", 1)
                start, end = value(first), value(last)
            else:
                start = value(span)
                end = hi if has_step else start
            if not lo <= start <= end <= hi:
                raise ValueError(f"cron field {text!r} out of range {lo}-{hi}")
            values.update(range(start, end + 1, step))
        return values

    def _day_matches(self, day: date) -> bool:
        in_month = day.day in self.days
        in_week = day.isoweekday() % 7 in self.weekdays
        if self._any_day:
            return in_week
        if self._any_weekday:
            return in_month
        return in_month or in_week

    def next_after(self, after: datetime) -> datetime | None:
        """Return the first matching minute strictly after *after*."""
        t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        # Five years covers leap-day-only expressions; anything rarer
        # (e.g. "0 0 31 2 *") can never fire.
        limit = after + timedelta(days=366 * 5)
        while t <= limit:
            if t.month not in self.months:
                t = (t.replace(day=1, hour=0, minute=0) + timedelta(days=32)).replace(day=1)
            elif not self._day_matches(t.date()):
                t = t.replace(hour=0, minute=0) + timedelta(days=1)
            elif t.hour not in self.hours:
                t = t.replace(minute=0) + timedelta(hours=1)
            elif t.minute not in self.minutes:
                t += timedelta(minutes=1)
            else:
                return t
        return None


class Trigger:
    """Base trigger: subclasses return the next run strictly after a moment."""

    def next_after(self, after: datetime) -> datetime | None:
        return None

    def describe(self) -> str:
        raise NotImplementedError


class StartupTrigger(Trigger):
    def describe(self) -> str:
        return "at startup"


class IntervalTrigger(Trigger):
    def __init__(self, seconds: float):
        self.seconds = seconds

    def next_after(self, after: datetime) -> datetime | None:
        return after + timedelta(seconds=self.seconds)

    def describe(self) -> str:
        return f"every {_format_duration(self.seconds)}"


class CronTrigger(Trigger):
    def __init__(self, schedule: CronSchedule, label: str = ""):
        self.schedule = schedule
        self.label = label or f"cron '{schedule.expr}'"

    def next_after(self, after: datetime) -> datetime | None:
        return self.schedule.next_after(after)

    def describe(self) -> str:
        return self.label


class OnceTrigger(Trigger):
    def __init__(self, when: datetime):
        self.when = when

    def next_after(self, after: datetime) -> datetime | None:
        return self.when if self.when > after else None

    def describe(self) -> str:
        return f"once at {self.when:%Y-%m-%d %H:%M}"


# ----------------------------------------------------------------------
# YAML value parsing
# ----------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)}{unit}"
    return f"{seconds:g}s"


def _parse_duration(value) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid duration {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower().replace(" ", "")
        parts = _DURATION_RE.findall(text)
        if not parts or "".join(n + u for n, u in parts) != text:
            raise ValueError(f"invalid duration {value!r} (use e.g. 90s, 30m, 2h, 1h30m)")
        seconds = sum(float(n) * _DURATION_UNITS[u] for n, u in parts)
    if seconds <= 0:
        raise ValueError(f"duration must be positive, got {value!r}")
    return seconds


def _parse_clock(value) -> time:
    # YAML 1.1 reads unquoted 08:30 as the base-60 integer 510.
    if isinstance(value, int) and not isinstance(value, bool):
        hour, minute = divmod(value, 60)
    else:
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value).strip())
        if not match:
            raise ValueError(f"invalid time {value!r} (expected HH:MM)")
        hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        raise ValueError(f"invalid time {value!r}")
    return time(hour, minute)


def _parse_days(value) -> str:
    """Translate a ``days`` value into a cron day-of-week field."""
    if value is None:
        return "*"
    items = value if isinstance(value, list) else str(value).split(",")
    fields = []
    for item in items:
        token = str(item).strip().lower()
        token = _DAY_ALIASES.get(token, token)
        if token == "*":
            return "*"
        for name in re.split(r"[-,]", token):
            if name not in _WEEKDAYS:
                raise ValueError(f"unknown day {item!r} (use mon..sun, weekdays, weekends)")
        fields.append(token)
    return ",".join(fields)


def _parse_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time())
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    try:
        # Same format the /schedule event accepts.
        return datetime.strptime(text, "%d-%m-%Y-%H-%M")
    except ValueError:
        raise ValueError(
            f"invalid datetime {value!r} (expected YYYY-MM-DD HH:MM)"
        ) from None


def parse_triggers(raw) -> list[Trigger]:
    """Build triggers from the ``trigger`` value of an automation file."""
    if raw is None:
        raise ValueError("missing 'trigger'")
    items = raw if isinstance(raw, list) else [raw]
    triggers: list[Trigger] = []
    for item in items:
        if item == "startup":
            triggers.append(StartupTrigger())
            continue
        if not isinstance(item, dict):
            raise ValueError(f"invalid trigger {item!r}")
        unknown = set(item) - {"startup", "every", "at", "days", "cron", "once"}
        if unknown:
            raise ValueError(f"unknown trigger key(s): {', '.join(sorted(unknown))}")
        if "days" in item and "at" not in item:
            raise ValueError("'days' requires 'at'")
        if item.get("startup"):
            triggers.append(StartupTrigger())
        if "every" in item:
            triggers.append(IntervalTrigger(_parse_duration(item["every"])))
        if "cron" in item:
            triggers.append(CronTrigger(CronSchedule(item["cron"])))
        if "once" in item:
            triggers.append(OnceTrigger(_parse_datetime(item["once"])))
        if "at" in item:
            days = _parse_days(item.get("days"))
            times = item["at"] if isinstance(item["at"], list) else [item["at"]]
            suffix = "daily" if days == "*" else f"on {days}"
            for clock in map(_parse_clock, times):
                triggers.append(CronTrigger(
                    CronSchedule(f"{clock.minute} {clock.hour} * * {days}"),
                    label=f"{suffix} at {clock:%H:%M}",
                ))
    if not triggers:
        raise ValueError("'trigger' defines no schedule")
    return triggers


# ----------------------------------------------------------------------
# Automation definition
# ----------------------------------------------------------------------


@dataclass
class Automation:
    name: str
    run: str
    triggers: list[Trigger]
    description: str = ""
    domains: tuple[str, ...] = ()
    enabled: bool = True
    silent: bool = False
    tools: list[str] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def from_file(cls, path: Path | str) -> "Automation":
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError("expected a YAML mapping")
        return cls.from_dict(data, default_name=path.stem, path=path)

    @classmethod
    def from_dict(cls, data: dict, default_name: str = "", path: Path | None = None) -> "Automation":
        unknown = set(data) - {
            "name", "description", "enabled", "domain", "silent", "tools", "trigger", "run",
        }
        if unknown:
            raise ValueError(f"unknown key(s): {', '.join(sorted(unknown))}")
        name = str(data.get("name") or default_name).strip()
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid name {name!r} (letters, digits, '_', '-', '.')")
        run = str(data.get("run") or "").strip()
        if not run:
            raise ValueError("missing 'run'")
        domain = data.get("domain") or ()
        tools = data.get("tools") or []
        return cls(
            name=name,
            run=run,
            triggers=parse_triggers(data.get("trigger")),
            description=str(data.get("description") or "").strip(),
            domains=tuple(domain) if isinstance(domain, list) else (str(domain),) if domain else (),
            enabled=bool(data.get("enabled", True)),
            silent=bool(data.get("silent", False)),
            tools=list(tools) if isinstance(tools, list) else [str(tools)],
            path=path,
        )

    @property
    def runs_at_startup(self) -> bool:
        return any(isinstance(t, StartupTrigger) for t in self.triggers)

    @property
    def is_scheduled(self) -> bool:
        """Whether any trigger can fire after startup."""
        return any(not isinstance(t, StartupTrigger) for t in self.triggers)

    def applies_to(self, domain: str) -> bool:
        return not self.domains or domain in self.domains

    def next_run(self, after: datetime) -> datetime | None:
        candidates = [t.next_after(after) for t in self.triggers]
        return min((c for c in candidates if c is not None), default=None)

    def describe_trigger(self) -> str:
        return ", ".join(t.describe() for t in self.triggers)


def load_automations(directory: Path) -> tuple[list[Automation], list[str]]:
    """Load every ``*.yaml``/``*.yml`` file in *directory*.

    Returns ``(automations, errors)``; a broken file is reported in
    *errors* instead of preventing the others from loading.
    """
    if not directory.is_dir():
        return [], []
    automations: list[Automation] = []
    errors: list[str] = []
    seen: set[str] = set()
    paths = sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])
    for path in paths:
        try:
            automation = Automation.from_file(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if automation.name in seen:
            errors.append(f"{path.name}: duplicate automation name '{automation.name}'")
            continue
        seen.add(automation.name)
        automations.append(automation)
    return automations, errors


# ----------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------


class AutomationEvent(Event):
    """Runs an :class:`Automation` on an :class:`~src.core.event.EventBus`.

    ``condition()`` sleeps until the next scheduled run; the bus then
    dispatches ``message`` (a prompt) or ``command`` (a slash command)
    like any other event. The instance is named after the automation, so
    ``/stop <name>`` and ``/status <name>#N`` work as usual.

    *run_now* fires immediately on arming (startup triggers, manual
    runs); *oneshot* stops after that first run.
    """

    def __init__(self, automation: Automation, *, run_now: bool = False, oneshot: bool = False):
        self.automation = automation
        self.name = automation.name
        self.description = automation.description or f"Automation: {automation.run[:60]}"
        self.silent = automation.silent
        self.required_tools = list(automation.tools)
        if automation.run.startswith("/"):
            self.command = automation.run
        else:
            self.message = automation.run
        self._run_now = run_now or oneshot
        self._oneshot = oneshot
        self._last = datetime.now()
        self.next_run: datetime | None = self._last if self._run_now else automation.next_run(self._last)

    async def condition(self):
        if self._run_now:
            self._run_now = False
        elif self._oneshot:
            raise asyncio.CancelledError
        else:
            target = self.next_run = self.automation.next_run(self._last)
            if target is None:
                raise asyncio.CancelledError
            while (remaining := (target - datetime.now()).total_seconds()) > 0:
                await asyncio.sleep(min(remaining, _MAX_SLEEP))
        self._last = datetime.now()

    async def trigger(self):
        pass

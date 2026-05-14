import asyncio
import glob
import json
import os

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from src.core.event import Event


class ProjectTask(BaseModel):
    """A single task in the project plan.

    Extra (non-schema) fields are *preserved* so the agent can attach
    custom per-task scratchpad data (notes, dependencies, sub-checks,
    observations) that the project machinery itself does not interpret.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
        str_strip_whitespace=True,
    )

    id: str
    title: str = ""
    done: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "done",
            "complete",
            "completed",
            "is_done",
            "finished",
        ),
    )

    @field_validator("id", mode="before")
    @classmethod
    def _str_id(cls, v):
        if v is None or v == "":
            raise ValueError("task.id is required")
        return str(v).strip()


class ProjectState(BaseModel):
    """Canonical schema for the `project_state.json` file.

    Tolerant on input — accepts alias keys for `done` flags and a single
    string for `constraints` — but always emits canonical field names on
    serialisation. Extra (non-schema) fields are *preserved* so the agent
    can use the state file as a scratchpad for top-level custom data
    (notes, accumulated logs, dependencies map, etc.) that the project
    machinery itself does not interpret.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
        str_strip_whitespace=True,
    )

    goal: str = ""
    constraints: list[str] = Field(default_factory=list)
    tasks: list[ProjectTask] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)

    @field_validator("constraints", "log", mode="before")
    @classmethod
    def _coerce_str_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [
                str(c).strip()
                for c in v
                if isinstance(c, (str, int, float)) and not isinstance(c, bool)
            ]
        return []


_PLAN_PROMPT = """\
Project goal: {goal}

## Step 1 — Extract constraints
Read the goal above. Find any explicit rules, limits, deadlines, or stop \
conditions (e.g. "only use Y", "stop when X", "notify me if Z"). Copy each \
one verbatim into the `constraints` list. If the goal has none, use `[]`.

## Step 2 — List the tasks
Break the goal into concrete, sequential tasks a developer can execute one \
at a time. Set every task to `"done": false` now. Only flip a task to \
`"done": true` after its real-world effect has actually happened. Never \
pre-mark tasks as done.

## Step 3 — Write the plan to `{state_file}`
Write valid JSON using this exact schema (all keys are required):

{{
  "goal": "{goal}",
  "constraints": ["constraint 1", "constraint 2"],
  "tasks": [
    {{"id": 1, "title": "first task", "done": false}},
    {{"id": 2, "title": "second task", "done": false}}
  ],
  "log": []
}}

No constraints? Use: `"constraints": []`
Initialize `log` as an empty array — the system will require you to \
append one entry per completed task. Do not pre-fill it.

You may add extra fields (notes, observations, dependencies) at the top \
level or inside task objects. These extra fields are your private scratchpad \
and will be preserved verbatim across all iterations. Only the schema fields \
listed above are interpreted by the system.

## Step 4 — Stop
Write the file, then stop immediately. \
The system will dispatch task 1 on the next iteration.\
"""

_TASK_PROMPT = """\
Project goal: {goal}
{constraints_block}\
{log_block}\
Progress: {done}/{total} tasks complete.

## Current state — do NOT read `{state_file}`, use this JSON directly:

```json
{canonical_json}
```

## Task list
{task_list_block}
## Your task: [{task_id}/{total}] {title}

Do this task and only this task. Do not work ahead on other tasks. \
Do not call any verification tool more than once.

## When the task is done, follow these steps IN ORDER then stop:

  1. Write the updated JSON to `{state_file}` with task {task_id} \
changed to `"done": true` AND one new entry appended to `log` — a \
single factual line describing what you did (file paths, key values, \
decisions, errors encountered). Be concrete, not vague.
  2. Write ONE sentence confirming what you did. Stop immediately after \
that sentence — do not add more text, do not repeat yourself.

## WARNING — the only way to advance
The ONLY way to move to the next task is to write `{state_file}` with \
task {task_id} set to `"done": true`. If you skip this write, this same \
task will repeat on the next iteration indefinitely.

## State file rules
  • Only change `done` from false to true. Never set it back to false.
  • Do NOT edit: `goal`, `constraints`, task `id`s, task `title`s, or \
task order.
  • Do NOT add or remove tasks.
  • Do NOT redo tasks already marked [x] — they are permanently done.
  • `log` is append-only — never remove or edit existing entries, only \
add a new one at the end.
  • Task not yet complete (e.g. waiting for a condition)? Leave `done` \
as false and describe what you observed. The loop will call you again.
  • You MAY add or update custom fields (notes, observed values, scratch \
data) at the top level or inside task objects. They are preserved \
verbatim every iteration as your private scratchpad.\
"""

_DONE_PROMPT = """\
Project goal: {goal}
{constraints_block}\
{log_block}\
## Project complete — all {total} tasks done

Final task list:
{task_list_block}
Write ONE short paragraph (2-4 sentences): what was built and whether every \
constraint above was honoured. Do not repeat yourself. Do not add extra \
sections or lists. Stop immediately after the paragraph — your turn is over.\
"""


class ProjectEvent(Event):
    """Drives the agent through a multi-step project until completion.

    Iteration 0 — planning:
        No state file exists yet. The agent is asked to write a task plan to
        `state_file` as JSON and start task 1.

    Iterations 1..N — execution:
        The agent is asked to complete the next pending task and mark it done
        in `state_file`.

    Final iteration:
        All tasks are done. The agent is asked for a completion summary, then
        the event stops.
    """

    name = "project"
    # Tracks state files claimed this session so parallel /project calls
    # don't race to the same slot even before any file is written.
    _session_files: set[str] = set()

    def __init__(self, goal: str = "", state_file: str = "project_state.json"):
        self._state_file_arg = state_file
        self._use_instance_suffix = bool(goal) and state_file == "project_state.json"
        self.state_file = os.path.abspath(state_file)
        self._summary_sent = False
        self._plan_sent = False
        self._current_message: str | None = None
        # Canonical snapshot of the immutable parts of the plan. Captured
        # the first time a planned state file is read; subsequent reads are
        # reconciled against it so the agent cannot drop the goal,
        # constraints, or task structure.
        self._snapshot: dict | None = None
        # Monotonic log floor — only grows, never shrinks, so accidental
        # truncation by the agent is silently undone on the next read.
        self._log_floor: list[str] = []

        if not goal:
            # Resume mode: find and load the right state file.
            resolved = self._find_state_file(self.state_file)
            if resolved is None:
                raise ValueError(
                    f"/project requires a goal when no state file exists. "
                    f"Usage: /project <goal>"
                )
            self.state_file = resolved
            raw = self._load_raw()
            goal = (raw or {}).get("goal", "")
            if not goal:
                raise ValueError(
                    f"State file {self.state_file!r} is missing a 'goal' field; "
                    f"cannot resume. Pass an explicit goal: /project <goal>"
                )

        self.goal = goal
        self.description = f"Project: {goal[:60]}"

    # ------------------------------------------------------------------ state

    @staticmethod
    def _find_state_file(default_path: str) -> str | None:
        """Return a state file path to resume from, or None if none found.

        Tries the exact path first, then scans for numbered variants
        (e.g. project_state_1.json). Raises ValueError when multiple exist.
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
            f"Multiple state files found: {names}\n"
            f"Specify which to resume, e.g.: "
            f"/project \"\" {os.path.basename(candidates[0])}"
        )

    def _load_raw(self) -> dict | None:
        """Read the state file verbatim as a dict, with no schema validation."""
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _parse(self) -> ProjectState | None:
        """Read and parse the state file via the canonical schema."""
        raw = self._load_raw()
        if raw is None:
            return None
        try:
            return ProjectState.model_validate(raw)
        except Exception:
            # Malformed payload — treat as if the file were missing so the
            # event replans rather than crashing.
            return None

    def _load(self) -> ProjectState | None:
        """Parse the state file and reconcile against the canonical snapshot.

        On the first successful read of a planned file (one that has a
        non-empty `tasks` list), we capture an immutable snapshot of the
        goal, constraints, and task `id`/`title` pairs. From then on every
        read rebuilds the model from the snapshot, importing only the
        `done` flags from disk. The agent's writes are therefore
        effectively append-only on `done` — any other corruption is
        silently undone. Done flags are also **monotonic**: once a task
        is observed as done, it stays done permanently.
        """
        parsed = self._parse()
        if parsed is None:
            return None

        if self._snapshot is None and parsed.tasks:
            self._snapshot = {
                "goal": parsed.goal or self.goal,
                "constraints": list(parsed.constraints),
                "task_titles": [(t.id, t.title) for t in parsed.tasks],
                # Monotonic set of task ids ever observed as done. Once a
                # task is completed it stays completed, even if the agent
                # later flips its `done` back to false.
                "done_ids": set(),
            }
            self.goal = self._snapshot["goal"]

        if self._snapshot is None:
            return parsed

        # Monotonic merge: any task seen as done on disk joins the set
        # permanently; backward flips are ignored.
        self._snapshot["done_ids"].update(t.id for t in parsed.tasks if t.done)
        done_ids = self._snapshot["done_ids"]

        # Monotonic log: take the longer of the in-memory floor and the disk
        # value so accidental truncation by the agent is silently undone.
        if len(parsed.log) > len(self._log_floor):
            self._log_floor = list(parsed.log)

        # Carry through any custom scratchpad fields, both top-level and
        # per-task.
        disk_tasks_by_id = {t.id: t for t in parsed.tasks}
        reconciled_tasks = []
        for tid, title in self._snapshot["task_titles"]:
            task_extras: dict = {}
            disk_task = disk_tasks_by_id.get(tid)
            if disk_task is not None:
                task_extras = dict(disk_task.__pydantic_extra__ or {})
            reconciled_tasks.append(
                ProjectTask(
                    id=tid, title=title, done=tid in done_ids, **task_extras
                )
            )
        state_extras = dict(parsed.__pydantic_extra__ or {})
        return ProjectState(
            goal=self._snapshot["goal"],
            constraints=list(self._snapshot["constraints"]),
            tasks=reconciled_tasks,
            log=list(self._log_floor),
            **state_extras,
        )

    def _all_done(self) -> bool:
        state = self._load()
        return state is not None and bool(state.tasks) and all(t.done for t in state.tasks)

    def _constraints_block(self, state: ProjectState | None) -> str:
        constraints = list(state.constraints) if state is not None else []
        if not constraints:
            return ""
        bullets = "\n".join(f"  • {c}" for c in constraints)
        return f"Constraints (must hold throughout):\n{bullets}\n"

    def _log_block(self, state: ProjectState | None) -> str:
        entries = list(state.log) if state is not None else []
        if not entries:
            return ""
        lines = "\n".join(f"  • {e}" for e in entries)
        return f"## History of completed tasks\n{lines}\n\n"

    def _task_list_block(self, state: ProjectState | None, current_id=None) -> str:
        tasks = list(state.tasks) if state is not None else []
        lines = []
        for t in tasks:
            mark = "[x]" if t.done else "[ ]"
            pointer = "  <- CURRENT" if t.id == current_id else ""
            lines.append(f"  {mark} {t.id}: {t.title}{pointer}")
        return "\n".join(lines) + "\n"

    # ---------------------------------------------------- dynamic message

    def _compute_message(self) -> str:
        state = self._load()

        if state is None or not state.tasks:
            if self._plan_sent:
                return ""
            self._plan_sent = True
            return _PLAN_PROMPT.format(goal=self.goal, state_file=self.state_file)

        tasks = state.tasks
        total = len(tasks)
        done = sum(1 for t in tasks if t.done)
        pending = [t for t in tasks if not t.done]
        constraints_block = self._constraints_block(state)
        log_block = self._log_block(state)

        if not pending:
            self._summary_sent = True
            return _DONE_PROMPT.format(
                goal=self.goal,
                total=total,
                constraints_block=constraints_block,
                log_block=log_block,
                task_list_block=self._task_list_block(state),
            )

        task = pending[0]
        return _TASK_PROMPT.format(
            goal=self.goal,
            state_file=self.state_file,
            done=done,
            total=total,
            task_id=task.id,
            title=task.title,
            constraints_block=constraints_block,
            log_block=log_block,
            canonical_json=state.model_dump_json(indent=2),
            task_list_block=self._task_list_block(state, current_id=task.id),
        )

    @property
    def message(self) -> str:
        if self._current_message is None:
            self._current_message = self._compute_message()
        return self._current_message

    @message.setter
    def message(self, value):
        pass

    # ---------------------------------------------------- event interface

    async def condition(self):
        if self._use_instance_suffix:
            base, ext = os.path.splitext(os.path.abspath(self._state_file_arg))
            n = 1
            while True:
                candidate = f"{base}_{n}{ext}"
                if not os.path.exists(candidate) and candidate not in ProjectEvent._session_files:
                    break
                n += 1
            self.state_file = candidate
            ProjectEvent._session_files.add(candidate)
            self._use_instance_suffix = False
        self._current_message = None  # invalidate per-iteration cache
        if self._summary_sent:
            raise asyncio.CancelledError
        if self._plan_sent and self._snapshot is None:
            await asyncio.sleep(1)

    async def trigger(self):
        pass

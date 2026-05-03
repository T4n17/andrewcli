import asyncio
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

    @field_validator("constraints", mode="before")
    @classmethod
    def _coerce_constraints(cls, v):
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

You are starting a new project. First, read the goal carefully and identify \
any explicit CONSTRAINTS, stop conditions, deadlines, or "do not" rules \
stated in it (e.g. "stop when X", "only use Y", "notify me if Z"). Copy \
them verbatim into the `constraints` array below — they will be re-injected \
into every subsequent iteration so you do not forget them.

Then break the goal into concrete, sequential tasks a developer can execute \
one at a time. A task must only be marked `done` once its real-world effect \
has actually happened: do NOT pre-mark monitoring or conditional tasks as \
done just because you wrote code for them.

Write the plan to `{state_file}` using this JSON schema as a MINIMUM \
(the keys below are mandatory):
{{
  "goal": "{goal}",
  "constraints": ["...", "..."],
  "tasks": [
    {{"id": 1, "title": "...", "done": false}},
    ...
  ]
}}

If the goal contains no explicit constraints, use an empty array: \
`"constraints": []`.

You MAY EXTEND this schema with additional top-level fields and \
additional fields inside each task object if you need to track custom \
data across iterations — notes, dependencies between tasks, observations, \
artefact paths, sub-step checklists, anything that helps you stay on \
track between turns. Any extra fields you add will be preserved \
verbatim in the canonical state shown to you each iteration; treat them \
as your private scratchpad. Only the fields listed above are \
interpreted by the project machinery, everything else is yours.

After writing the file, stop. The project driver will assign task 1 on \
the next iteration.\
"""

_TASK_PROMPT = """\
Project goal: {goal}
{constraints_block}\
Progress: {done}/{total} tasks complete.

Canonical state of `{state_file}` — do NOT read the file; this JSON \
is authoritative and already matches what the driver sees:

```json
{canonical_json}
```

Task list at a glance:
{task_list_block}
Current task ({task_id}/{total}): {title}

Complete this task and nothing more. Do not call any verification tool \
more than once — if you already confirmed the effect, proceed immediately \
to the steps below.

When the task's real-world effect is achieved, do the following IN ORDER \
and then stop:
  1. Write the updated JSON to `{state_file}` with task {task_id}'s \
`"done"` flipped to true.
  2. Write one short paragraph stating what you did.
  3. Make NO further tool calls. Your turn ends here. The loop will \
dispatch the next task on the following iteration.

CRITICAL — THE ONLY WAY TO ADVANCE: if you do not write `{state_file}` \
with task {task_id}'s `"done": true`, this exact task will be dispatched \
again on the next iteration. There is no other signal. Forgetting to \
write the file means repeating this task indefinitely.

STATE FILE RULES — the loop owns the schema fields:
  • You may only change `done` flags from false to true. Any edits to \
`goal`, `constraints`, task `id`s or `title`s, the task list itself, or \
any attempt to flip a `done` flag back to false, will be silently \
overwritten on the next iteration.
  • Do not delete tasks. Do not add tasks. Do not reorder them.
  • Do not redo a task already marked [x] above — it is permanently done.
  • If the current task's condition is not yet met (e.g. a monitored value \
hasn't crossed a threshold), leave `done` as false and explain what you \
observed — the loop will call you again.
  • You MAY add or update CUSTOM fields not listed in the schema, both \
at the top level and inside individual task objects (notes, observed \
values, dependency lists, scratch data). Custom fields will be preserved \
verbatim across iterations — use them as your private scratchpad.\
"""

_DONE_PROMPT = """\
Project goal: {goal}
{constraints_block}\
Final state of `{state_file}`:
{task_list_block}
All {total} tasks are marked done. Summarise what was built, confirm \
every constraint above was honoured, and confirm the project is \
complete.\
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

    def __init__(self, goal: str = "", state_file: str = "project_state.json"):
        self.state_file = os.path.abspath(state_file)
        self._summary_sent = False
        self._plan_sent = False
        self._current_message: str | None = None
        # Canonical snapshot of the immutable parts of the plan. Captured
        # the first time a planned state file is read; subsequent reads are
        # reconciled against it so the agent cannot drop the goal,
        # constraints, or task structure.
        self._snapshot: dict | None = None

        if not goal:
            # Resume mode: recover the goal from the existing state file.
            raw = self._load_raw()
            if raw is None:
                raise ValueError(
                    f"/project requires a goal when no state file exists at "
                    f"{state_file!r}. Usage: /project <goal>"
                )
            goal = raw.get("goal", "")
            if not goal:
                raise ValueError(
                    f"State file {state_file!r} is missing a 'goal' field; "
                    f"cannot resume. Pass an explicit goal: /project <goal>"
                )

        self.goal = goal
        self.description = f"Project: {goal[:60]}"

    # ------------------------------------------------------------------ state

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

        if not pending:
            self._summary_sent = True
            return _DONE_PROMPT.format(
                goal=self.goal,
                state_file=self.state_file,
                total=total,
                constraints_block=constraints_block,
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
        self._current_message = None  # invalidate per-iteration cache
        if self._summary_sent:
            raise asyncio.CancelledError
        if self._plan_sent and self._snapshot is None:
            await asyncio.sleep(1)

    async def trigger(self):
        pass

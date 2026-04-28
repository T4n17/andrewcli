import asyncio
import json
from src.core.event import Event


_PLAN_PROMPT = """\
Project goal: {goal}

You are starting a new project. Break this goal into a list of concrete, \
sequential tasks a developer can execute one at a time.

Write the plan to `{state_file}` using this exact JSON schema:
{{
  "goal": "{goal}",
  "tasks": [
    {{"id": 1, "title": "...", "done": false}},
    ...
  ]
}}

After writing the file, immediately start executing task 1.\
"""

_TASK_PROMPT = """\
Project goal: {goal}
Progress: {done}/{total} tasks complete.

Current task ({task_id}/{total}): {title}

Complete this task fully. When you are done, update `{state_file}` and set \
task {task_id} "done" to true. Do not move on to other tasks — the loop \
will assign the next one on the following iteration.\
"""

_DONE_PROMPT = """\
Project goal: {goal}

All {total} tasks in `{state_file}` are marked done. \
Summarise what was built and confirm the project is complete.\
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

    def __init__(self, goal: str, state_file: str = "project_state.json"):
        self.goal = goal
        self.state_file = state_file
        self.description = f"Project: {goal[:60]}"
        self._summary_sent = False

    # ------------------------------------------------------------------ state

    def _load(self) -> dict | None:
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _all_done(self) -> bool:
        state = self._load()
        return bool(state) and all(t["done"] for t in state.get("tasks", []))

    # ---------------------------------------------------- dynamic message

    @property
    def message(self) -> str:
        state = self._load()

        if state is None:
            return _PLAN_PROMPT.format(goal=self.goal, state_file=self.state_file)

        tasks = state.get("tasks", [])
        total = len(tasks)
        done = sum(1 for t in tasks if t["done"])
        pending = [t for t in tasks if not t["done"]]

        if not pending:
            self._summary_sent = True
            return _DONE_PROMPT.format(
                goal=self.goal, state_file=self.state_file, total=total
            )

        task = pending[0]
        return _TASK_PROMPT.format(
            goal=self.goal,
            state_file=self.state_file,
            done=done,
            total=total,
            task_id=task["id"],
            title=task["title"],
        )

    @message.setter
    def message(self, value):
        # Ignore external sets — message is always derived from state.
        pass

    # ---------------------------------------------------- event interface

    async def condition(self):
        # Stop after the completion summary has been dispatched.
        if self._summary_sent and self._all_done():
            raise asyncio.CancelledError

    async def trigger(self):
        pass

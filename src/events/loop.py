import asyncio
import json
import os
from typing import Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from src.core.event import Event


class LoopState(BaseModel):
    """Canonical schema for the `loop_state.json` file.

    Tolerant on input — accepts common alias keys (e.g. ``iterations_done``
    instead of ``iterations``) and coerces sloppy values (e.g. the string
    ``"unlimited"`` for ``max_iterations``) — but always emits canonical
    field names on serialisation. Extra (non-schema) fields are *preserved*
    so the agent can use the state file as a scratchpad for custom data
    that the loop machinery itself does not interpret.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        # Custom fields the agent adds (e.g. running totals, history,
        # retry counters) are preserved on the model and round-tripped
        # through the iteration prompt as a scratchpad.
        extra="allow",
        str_strip_whitespace=True,
    )

    goal: str = ""
    action: str = ""
    exit_criteria: list[str] = Field(default_factory=list)
    # ``None`` means uncapped — only an exit criterion can stop the loop.
    max_iterations: Optional[int] = None
    iterations: int = Field(
        default=0,
        validation_alias=AliasChoices(
            "iterations",
            "iterations_done",
            "iterations_completed",
            "iter_count",
            "iter",
            "iters",
            "current_iteration",
        ),
    )
    last_observation: str = ""
    terminated: bool = False
    termination_reason: str = ""

    @field_validator("max_iterations", mode="before")
    @classmethod
    def _coerce_max(cls, v):
        # Accept null, missing, 0, negatives, the string "unlimited",
        # "none", "null", and any non-numeric string as "uncapped".
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            n = int(v)
            return n if n > 0 else None
        if isinstance(v, str):
            try:
                n = int(v.strip())
                return n if n > 0 else None
            except ValueError:
                return None
        return None

    @field_validator("exit_criteria", mode="before")
    @classmethod
    def _coerce_exit(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [
                str(x).strip()
                for x in v
                if isinstance(x, (str, int, float)) and not isinstance(x, bool)
            ]
        return []


_PLAN_PROMPT = """\
Loop goal: {goal}

You are starting a LOOP — a single action that repeats every iteration \
until a stop condition is met. Read the goal carefully and identify:

  1. The ACTION to repeat each iteration. Phrase it as one concrete, \
self-contained step a developer can execute (e.g. "fetch the current oil \
price via google_search", "poll endpoint /status and parse the JSON").
  2. The EXIT CRITERIA — every distinct condition under which the loop \
must stop. Copy them verbatim from the goal where possible (e.g. \
"price drops below $100", "status == 'ready'", "5 consecutive failures", \
"24 hours have elapsed").

Write the loop spec to `{state_file}` using this JSON schema as a \
MINIMUM (the keys below are mandatory):
{{
  "goal": "{goal}",
  "action": "...",
  "exit_criteria": ["...", "..."],
  "max_iterations": {cap_value},
  "iterations": 0,
  "last_observation": "",
  "terminated": false,
  "termination_reason": ""
}}

{cap_explanation}

You MAY EXTEND this schema with additional top-level fields if you need \
to track custom data across iterations — running totals, observation \
history, retry counters, accumulated logs, anything that helps you \
reason between turns. Any extra fields you add will be preserved \
verbatim in the canonical state shown to you each iteration; treat them \
as your private scratchpad. Only the fields listed above are \
interpreted by the loop machinery, everything else is yours.

After writing the file, stop. The loop driver will start iteration 1 on \
the next run.\
"""

_ITER_PROMPT = """\
Loop goal: {goal}
{exit_block}\
{iter_header}

Current canonical state of `{state_file}` (authoritative — this is what \
the loop sees, regardless of what is on disk). The mandatory fields \
below MUST keep these exact names — inventing keys like `iterations_done` \
or values like the string `"unlimited"` will be ignored. Any other \
fields you see in the JSON below are custom scratchpad data you added \
in a previous iteration; they will be preserved verbatim, and you may \
add more or update them freely:

```json
{canonical_json}
```

Do NOT read `{state_file}` — the canonical JSON above is authoritative \
and already matches what the driver sees. Use it as the base for your write.

Now, in this single iteration:
  1. Perform the action exactly ONCE.
  2. Observe the result and record concrete facts (numbers, status, \
error messages — not opinions).
  3. Evaluate EVERY exit criterion above against your observation.
  4. Write the updated JSON to `{state_file}`:
       • increment `iterations` by 1
       • set `last_observation` to your factual record from step 2
       • if ANY exit criterion is met, set `terminated` to true and \
write the criterion verbatim into `termination_reason`
       • if no criterion is met, leave `terminated` as false — the loop \
will call you again
  5. Write a one-sentence observation summary, then STOP. Do not perform \
the action again. Do not start a second iteration — the loop schedules them.

LOOP RULES — the loop owns this file:
  • You may only update `iterations`, `last_observation`, `terminated`, \
and `termination_reason`. Edits to `goal`, `action`, `exit_criteria`, or \
`max_iterations` will be silently overwritten on the next iteration.
  • `iterations` is monotonic — never decrease it, never skip ahead.
  • `terminated` is sticky — once true, it cannot go back to false.
  • If you cannot evaluate the exit criteria (e.g. the action failed), \
record the failure in `last_observation`, leave `terminated` false, and \
the loop will retry on the next iteration.\
"""

_DONE_PROMPT = """\
Loop goal: {goal}
{exit_block}\
The loop has stopped after {iterations} iteration(s).

  termination reason : {termination_reason}
  last observation   : {last_observation}

Summarise what happened across the run, state which exit criterion \
triggered (or that the iteration cap was reached), and confirm the \
loop is complete.\
"""


class LoopEvent(Event):
    """Drives the agent through a 'do X until Y is met' loop.

    Iteration 0 — planning:
        No state file exists yet. The agent extracts the action and exit
        criteria from the goal and writes the initial loop spec.

    Iterations 1..N — execution:
        The agent performs the action once per iteration, records an
        observation, and flips `terminated` to true if any exit criterion
        is met.

    Final iteration:
        Either an exit criterion fired or `max_iterations` was reached.
        The agent is asked for a summary, then the event stops.

    State management mirrors `ProjectEvent`: an in-memory snapshot of the
    immutable fields (`goal`, `action`, `exit_criteria`, `max_iterations`)
    is captured on the first planned read, and subsequent reads are
    reconciled so the agent's writes are effectively append-only on the
    progress fields.
    """

    name = "loop"

    def __init__(
        self,
        goal: str = "",
        max_iterations: int = 0,
        state_file: str = "loop_state.json",
    ):
        """Create a LoopEvent.

        Args:
            goal: Natural-language description of the loop. If empty, the
                event resumes from `state_file` (which must exist and
                contain a `goal` field).
            max_iterations: Optional safety cap on the number of
                iterations. Defaults to ``0`` which means *uncapped* —
                the loop runs until an exit criterion fires. Any
                positive value enforces a hard ceiling regardless of
                what the planner wrote to disk; this user-supplied
                value always overrides any value already in the state
                file.
            state_file: Path to the JSON state file.
        """
        self.state_file = os.path.abspath(state_file)
        self._summary_sent = False
        self._plan_sent = False
        self._current_message: str | None = None
        # Canonical snapshot of immutable fields. Captured the first time
        # a planned state file is read; subsequent reads are reconciled
        # against it so the agent cannot drop the action, exit criteria,
        # or iteration cap.
        self._snapshot: dict | None = None
        # Monotonic floor for `iterations` — never decreases.
        self._iter_floor = 0
        # Sticky termination — once True, stays True.
        self._terminated = False
        self._termination_reason = ""
        # User-supplied iteration cap. 0 means "unset / uncapped"; any
        # positive value overrides anything found in the state file.
        self._user_max_iterations = max(0, int(max_iterations or 0))

        if not goal:
            # Resume mode: recover the goal from the existing state file.
            raw = self._load_raw()
            if raw is None:
                raise ValueError(
                    f"/loop requires a goal when no state file exists at "
                    f"{state_file!r}. Usage: /loop <goal>"
                )
            goal = raw.get("goal", "")
            if not goal:
                raise ValueError(
                    f"State file {state_file!r} is missing a 'goal' field; "
                    f"cannot resume. Pass an explicit goal: /loop <goal>"
                )

        self.goal = goal
        self.description = f"Loop: {goal[:60]}"

    # ------------------------------------------------------------------ state

    def _load_raw(self) -> dict | None:
        """Read the state file verbatim as a dict, with no schema validation."""
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _parse(self) -> LoopState | None:
        """Read and parse the state file via the canonical schema."""
        raw = self._load_raw()
        if raw is None:
            return None
        try:
            return LoopState.model_validate(raw)
        except Exception:
            # Malformed payload — treat as if the file were missing so the
            # loop replans rather than crashing.
            return None

    def _load(self) -> LoopState | None:
        """Parse the state file and reconcile against the canonical snapshot.

        On the first successful read of a planned file (one with an
        `action` field), we capture an immutable snapshot of the goal,
        action, exit criteria, and iteration cap. From then on every read
        rebuilds the model from the snapshot, importing only the progress
        fields from disk:
          • `iterations` is clamped to a monotonic floor (never decreases)
          • `terminated` is sticky (false→true only)
          • `last_observation` and `termination_reason` are free-form
        """
        parsed = self._parse()
        if parsed is None:
            return None

        if self._snapshot is None and parsed.action:
            # Precedence on the cap: user-passed > disk > uncapped (None).
            effective_max: Optional[int]
            if self._user_max_iterations > 0:
                effective_max = self._user_max_iterations
            else:
                effective_max = parsed.max_iterations  # already coerced
            self._snapshot = {
                "goal": parsed.goal or self.goal,
                "action": parsed.action,
                "exit_criteria": list(parsed.exit_criteria),
                "max_iterations": effective_max,  # int>0 or None
            }
            self.goal = self._snapshot["goal"]

        if self._snapshot is None:
            return parsed  # planning state, return as-parsed

        # Monotonic iterations: only forward.
        if parsed.iterations > self._iter_floor:
            self._iter_floor = parsed.iterations

        # Sticky termination + reason.
        if parsed.terminated:
            self._terminated = True
            if parsed.termination_reason and not self._termination_reason:
                self._termination_reason = parsed.termination_reason

        # Carry through any custom scratchpad fields the agent added.
        extras = dict(parsed.__pydantic_extra__ or {})
        return LoopState(
            goal=self._snapshot["goal"],
            action=self._snapshot["action"],
            exit_criteria=list(self._snapshot["exit_criteria"]),
            max_iterations=self._snapshot["max_iterations"],
            iterations=self._iter_floor,
            last_observation=parsed.last_observation,
            terminated=self._terminated,
            termination_reason=self._termination_reason,
            **extras,
        )

    def _exit_block(self, state: LoopState | None) -> str:
        criteria = list(state.exit_criteria) if state is not None else []
        if not criteria:
            return ""
        bullets = "\n".join(f"  • {c}" for c in criteria)
        return f"Exit criteria (stop when any of these holds):\n{bullets}\n"

    # ---------------------------------------------------- dynamic message

    def _compute_message(self) -> str:
        state = self._load()

        if state is None or not state.action:
            if self._plan_sent:
                return ""
            self._plan_sent = True
            if self._user_max_iterations > 0:
                cap_value = str(self._user_max_iterations)
                cap_explanation = (
                    f"`max_iterations` is fixed at {self._user_max_iterations} "
                    f"(set by the operator). Use this exact value — do not "
                    f"change it."
                )
            else:
                cap_value = "null"
                cap_explanation = (
                    "There is NO iteration cap. The loop runs until an "
                    "exit criterion is met. Set `max_iterations` to `null`."
                )
            return _PLAN_PROMPT.format(
                goal=self.goal,
                state_file=self.state_file,
                cap_value=cap_value,
                cap_explanation=cap_explanation,
            )

        exit_block = self._exit_block(state)
        iterations = state.iterations
        max_iter = state.max_iterations  # int>0 or None
        terminated = state.terminated
        capped = max_iter is not None and max_iter > 0

        if terminated or (capped and iterations >= max_iter):
            self._summary_sent = True
            reason = state.termination_reason or (
                "an exit criterion was met" if terminated
                else f"reached max_iterations ({max_iter}) without an exit criterion firing"
            )
            return _DONE_PROMPT.format(
                goal=self.goal,
                state_file=self.state_file,
                exit_block=exit_block,
                iterations=iterations,
                termination_reason=reason,
                last_observation=state.last_observation or "(none)",
            )

        if capped:
            iter_header = f"Iteration {iterations + 1} of up to {max_iter}."
        else:
            iter_header = (
                f"Iteration {iterations + 1} (no iteration cap — the loop "
                f"runs until an exit criterion fires)."
            )
        canonical_json = state.model_dump_json(indent=2)
        return _ITER_PROMPT.format(
            goal=self.goal,
            state_file=self.state_file,
            exit_block=exit_block,
            iter_header=iter_header,
            canonical_json=canonical_json,
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

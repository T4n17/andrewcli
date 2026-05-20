import json
import openai
import os
import asyncio
from typing import List
from src.core.memory import Memory
from src.core.tool import Tool
from src.core.skill import Skill


class ToolEvent:
    def __init__(self, tool_name=None, tool_args=None):
        self.tool_name = tool_name
        self.tool_args = tool_args


class ToolResultEvent:
    def __init__(self, tool_name: str, tool_args: dict, result: str):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.result = result


class RouteEvent:
    def __init__(self, tool_names: list[str]):
        self.tool_names = tool_names


def format_tool_status(event) -> str | None:
    """Format a RouteEvent/ToolEvent into a user-visible status string.

    Returns None when the event should not change the status line.
    Shared between the CLI renderer, the tray panel, and the server so
    every frontend displays identical routing/tool messages.
    """
    if isinstance(event, RouteEvent):
        if event.tool_names:
            return f"Loading: {', '.join(event.tool_names)}"
        return None
    if isinstance(event, ToolEvent):
        if event.tool_name:
            first_val = (
                str(next(iter(event.tool_args.values()), ""))
                if event.tool_args else ""
            )
            if len(first_val) > 60:
                first_val = first_val[:57] + "..."
            detail = f": {first_val}" if first_val else ""
            return f"Running {event.tool_name}{detail}"
        return "Thinking..."
    return None


class LLM:
    def __init__(self, api_base_url: str = None, model: str = None):
        self.api_base_url = api_base_url or os.getenv("API_BASE_URL", "http://localhost:8080/v1")
        self.model = model or os.getenv("MODEL", "qwen3.5:9B")
        # Memory summarization is background work that doesn't need the
        # same capacity as the main chat model. Point SUMMARY_MODEL at a
        # smaller model on the same server to cut background load.
        self.summary_model = os.getenv("SUMMARY_MODEL", self.model)
        self.client = openai.AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=os.getenv("OPENAI_API_KEY", "local"),
        )
        self.memory = Memory()

    def set_system_prompt(self, prompt: str):
        self.memory.add({"role": "system", "content": prompt})

    async def generate(self, prompt: str, tools: List[Tool] = None, skills: List[Skill] = None, max_rounds: int = 50):
        self.memory.add({"role": "user", "content": prompt})
        # Record where in the message list the current turn begins (the user
        # message we just appended). Used to strip prior-turn history when a
        # skill activates so the model only sees the current request.
        turn_start_idx = len(self.memory.messages) - 1

        skills_schemas = [s.to_openai_schema() for s in skills] if skills else None
        tool_schemas = [t.to_openai_schema() for t in tools] if tools else None

        all_schemas = (skills_schemas or []) + (tool_schemas or []) or None
        all_callables = (skills or []) + (tools or [])

        # When a skill activates we snapshot the pre-turn messages here and
        # restore them (plus only the final response) when the skill finishes,
        # so intermediate tool-call chains never leak into the long-term history.
        _saved_messages: list | None = None
        _skill_active = False

        # Everything below runs inside a try/finally so the turn-scoped
        # active-skill blocks in Memory are always cleared at turn end,
        # including on exception, early return, and generator aclose()
        # (which fires a GeneratorExit inside the `async for` yield).
        # Without this a skill activated mid-turn would leak into the
        # system prompt of the *next* user turn and bias its routing.
        try:
            last_content = ""
            for _ in range(max_rounds):
                kwargs = {"model": self.model, "messages": self.memory.get(), "stream": True}
                if all_schemas:
                    kwargs["tools"] = all_schemas
                try:
                    stream = await self.client.chat.completions.create(**kwargs)
                except openai.APIConnectionError:
                    yield "There was an error with the LLM endpoint. Check that the server is running and that API_BASE_URL is configured correctly."
                    return

                content = ""
                tool_calls_accum = {}

                async for chunk in stream:
                    delta = chunk.choices[0].delta

                    if delta.content:
                        content += delta.content
                        yield delta.content

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_accum:
                                tool_calls_accum[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_delta.id:
                                tool_calls_accum[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_accum[idx]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_accum[idx]["arguments"] += tc_delta.function.arguments

                if content:
                    last_content = content

                if not tool_calls_accum:
                    self.memory.add({"role": "assistant", "content": content})
                    if _skill_active and _saved_messages is not None:
                        # Skill finished: restore the conversation that existed
                        # before this turn and append only the final result so
                        # the intermediate skill tool chain is never persisted.
                        self.memory.messages = _saved_messages + [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": content},
                        ]
                        _saved_messages = None  # signal finally that restore is done
                    await self.memory.summarize_turn(self.client, self.summary_model)
                    return

                self.memory.add({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls_accum.values()
                    ],
                })

                for tc in tool_calls_accum.values():
                    try:
                        args = json.loads(tc["arguments"]) if tc.get("arguments") else {}
                    except json.JSONDecodeError:
                        args = {}
                    yield ToolEvent(tc["name"], args)
                    await asyncio.sleep(2)

                    # Skills deliver scripted instructions rather than real
                    # side effects, so we promote their body into the system
                    # prompt (higher authority than a tool response) and
                    # reply to the tool_call with a short acknowledgement.
                    # This makes the model treat the steps as binding
                    # system-level instructions instead of optional
                    # reference material buried in a tool response.
                    callable_obj = next(
                        (c for c in all_callables if c.name == tc["name"]),
                        None,
                    )
                    if isinstance(callable_obj, Skill):
                        if not _skill_active:
                            _skill_active = True
                            # Snapshot history before this turn and replace the
                            # conversation with only the current-turn messages so
                            # the model executes the skill without prior context.
                            _saved_messages = self.memory.messages[:turn_start_idx]
                            self.memory.messages = self.memory.messages[turn_start_idx:]
                        instructions = callable_obj.run(**args)
                        self.memory.add_active_skill(callable_obj.name, instructions)
                        result = (
                            f"Skill '{callable_obj.name}' activated. Follow the "
                            f"<skill:{callable_obj.name}> block in your system "
                            "instructions: execute each step in order by "
                            "calling the appropriate tools."
                        )
                    else:
                        # Tools can do blocking I/O (subprocess, HTTP, scrapers).
                        # Run them in a worker thread so the event loop keeps
                        # scheduling the spinner, SSE heartbeats, and other events.
                        result = await asyncio.to_thread(
                            self._execute_tool_call_from_dict, tc, all_callables,
                        )
                    self.memory.add({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    })
                    yield ToolResultEvent(tc["name"], args, str(result))

                yield ToolEvent()

            # Fell out of the max_rounds loop without a final text response.
            # Persist the last non-empty content we actually produced instead
            # of the (possibly empty) content from the final tool-only round.
            fallback = last_content or "(tool loop exceeded max rounds)"
            self.memory.add({"role": "assistant", "content": fallback})
            if _skill_active and _saved_messages is not None:
                self.memory.messages = _saved_messages + [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": fallback},
                ]
                _saved_messages = None  # signal finally that restore is done
            await self.memory.summarize_turn(self.client, self.summary_model)
        finally:
            # Turn scope ends here for every exit path (normal return,
            # max_rounds fall-through, exception, or async generator
            # aclose() from a cancelled stream). Drop all active-skill
            # blocks so they don't leak into the next user turn's
            # system prompt.
            self.memory.clear_active_skills()
            if _skill_active and _saved_messages is not None:
                # Generator was closed before the skill completed (user stopped
                # mid-execution). Restore the pre-skill conversation so the
                # history isn't left in the stripped current-turn-only state.
                self.memory.messages = _saved_messages

    def _execute_tool_call_from_dict(self, tool_call: dict, tools: list) -> str:
        func_name = tool_call["name"]
        try:
            arguments = json.loads(tool_call["arguments"]) if tool_call.get("arguments") else {}
        except json.JSONDecodeError:
            return f"[Tool Error] Invalid JSON arguments for '{func_name}': {tool_call.get('arguments')!r}"
        for tool in tools:
            if tool.name == func_name:
                return tool.run(**arguments)
        return f"Error: Tool '{func_name}' not found."


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
    def __init__(self):
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080/v1")
        self.model = os.getenv("MODEL", "qwen3.5:9B")
        # Memory summarization is background work that doesn't need the
        # same capacity as the main chat model. Point SUMMARY_MODEL at a
        # smaller model on the same server to cut background load.
        self.summary_model = os.getenv("SUMMARY_MODEL", self.model)
        self.client = openai.AsyncOpenAI(base_url=self.api_base_url)
        self.memory = Memory()

    def set_system_prompt(self, prompt: str):
        self.memory.add({"role": "system", "content": prompt})

    async def generate(self, prompt: str, tools: List[Tool] = None, skills: List[Skill] = None, max_rounds: int = 50):
        self.memory.add({"role": "user", "content": prompt})

        skills_schemas = [s.to_openai_schema() for s in skills] if skills else None
        tool_schemas = [t.to_openai_schema() for t in tools] if tools else None

        all_schemas = (skills_schemas or []) + (tool_schemas or []) or None
        all_callables = (skills or []) + (tools or [])

        last_content = ""
        for _ in range(max_rounds):
            kwargs = {"model": self.model, "messages": self.memory.get(), "stream": True}
            if all_schemas:
                kwargs["tools"] = all_schemas
            stream = await self.client.chat.completions.create(**kwargs)

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

            yield ToolEvent()

        # Fell out of the max_rounds loop without a final text response.
        # Persist the last non-empty content we actually produced instead
        # of the (possibly empty) content from the final tool-only round.
        self.memory.add({
            "role": "assistant",
            "content": last_content or "(tool loop exceeded max rounds)",
        })
        await self.memory.summarize_turn(self.client, self.summary_model)

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


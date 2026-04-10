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


class LLM:
    def __init__(self):
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080/v1")
        self.model = os.getenv("MODEL", "qwen3.5:9B")
        self.client = openai.AsyncOpenAI(base_url=self.api_base_url)
        self.memory = Memory()

    def set_system_prompt(self, prompt: str):
        self.memory.add({"role": "system", "content": prompt})

    async def generate(self, prompt: str, tools: List[Tool] = None, skills: List[Skill] = None, max_rounds: int = 50):
        self.memory.add({"role": "user", "content": prompt})

        skills_schemas = [s.to_openai_schema() for s in skills] if skills else None
        tool_schemas = [t.to_openai_schema() for t in tools] if tools else None

        all_schemas = (tool_schemas or []) + (skills_schemas or []) or None
        all_callables = (tools or []) + (skills or [])

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

            if not tool_calls_accum:
                self.memory.add({"role": "assistant", "content": content})
                await self.memory.summarize_turn(self.client, self.model)
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
                result = self._execute_tool_call_from_dict(tc, all_callables)
                self.memory.add({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                })

            yield ToolEvent()

        self.memory.add({"role": "assistant", "content": content})
        await self.memory.summarize_turn(self.client, self.model)

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


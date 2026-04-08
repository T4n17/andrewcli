import json
import openai
import os
from typing import List
from src.core.memory import Memory
from src.core.tool import Tool
from src.core.skill import Skill


class LLM:
    def __init__(self):
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080/v1")
        self.model = os.getenv("MODEL", "qwen3.5:9B")
        self.client = openai.OpenAI(base_url=self.api_base_url)
        self.memory = Memory()

    def set_system_prompt(self, prompt: str):
        self.memory.add({"role": "system", "content": prompt})

    def generate(self, prompt: str, tools: List[Tool] = None, skills: List[Skill] = None, max_rounds: int = 10) -> str:
        self.memory.add({"role": "user", "content": prompt})

        skills_schemas = [s.to_openai_schema() for s in skills] if skills else None
        tool_schemas = [t.to_openai_schema() for t in tools] if tools else None

        all_schemas = (tool_schemas or []) + (skills_schemas or []) or None
        all_callables = (tools or []) + (skills or [])

        for _ in range(max_rounds):
            kwargs = {"model": self.model, "messages": self.memory.get()}
            if all_schemas:
                kwargs["tools"] = all_schemas
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            if not message.tool_calls:
                self.memory.add({"role": "assistant", "content": message.content})
                return message.content

            self.memory.add({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tool_call in message.tool_calls:
                result = self._execute_tool_call(tool_call, all_callables)
                self.memory.add({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

        self.memory.add({"role": "assistant", "content": message.content})
        return message.content

    def _execute_tool_call(self, tool_call, tools: list) -> str:
        func_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        for tool in tools:
            if tool.name == func_name:
                return tool.execute(**arguments)
        return f"Error: Tool '{func_name}' not found."

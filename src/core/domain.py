from abc import ABC
from typing import List
from src.core.llm import LLM, RouteEvent
from src.core.router import ToolRouter
from src.core.event import EventBus


class Domain(ABC):
    system_prompt: str
    tools: List
    skills: List
    events: List = []

    def __init__(self):
        self.system_prompt = self.system_prompt
        self.tools = self.tools
        self.skills = self.skills
        self.events = self.events
        self.llm = LLM()
        self.llm.set_system_prompt(self.system_prompt)
        self.router = ToolRouter()
        self.event_bus = EventBus(self.events)

    async def generate_event(self, prompt: str):
        """One-shot generation for event dispatches.

        Routes to the right tools like generate() does, but uses a fresh LLM
        with no conversation context so event exchanges never pollute the
        conversation memory or affect routing for user queries.
        """
        tools, skills = await self.router.route(prompt, self.tools, self.skills)

        existing_names = {t.name for t in tools}
        required_names = {name for s in skills for name in s.required_tools}
        for tool in self.tools:
            if tool.name in required_names and tool.name not in existing_names:
                tools.append(tool)
                existing_names.add(tool.name)

        event_llm = LLM()
        event_llm.set_system_prompt(self.system_prompt)
        yield RouteEvent([item.name for item in tools + skills])
        async for token in event_llm.generate(prompt, tools, skills):
            yield token

    async def generate(self, prompt: str):
        tools, skills = await self.router.route(
            prompt, self.tools, self.skills,
            summary=self.llm.memory.summary,
            last_exchange=self.llm.memory.last_exchange,
        )

        # Pull in any tools declared as required by the selected skills
        existing_names = {t.name for t in tools}
        required_names = {name for s in skills for name in s.required_tools}
        for tool in self.tools:
            if tool.name in required_names and tool.name not in existing_names:
                tools.append(tool)
                existing_names.add(tool.name)

        yield RouteEvent([item.name for item in tools + skills])
        async for token in self.llm.generate(prompt, tools, skills):
            yield token

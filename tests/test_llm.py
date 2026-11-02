import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.core import memory as memory_module
from src.core.llm import LLM, ToolEvent, ToolResultEvent
from src.core.memory import Memory
from src.core.router import ToolRouter
from src.core.skill import Skill
from src.core.tool import Tool


def _chunk(content=None, tool_call=None):
    tool_calls = None
    if tool_call is not None:
        call_id, name, arguments = tool_call
        tool_calls = [SimpleNamespace(
            index=0, id=call_id, function=SimpleNamespace(name=name, arguments=arguments),
        )]
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _Stream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


class _Client:
    """Fake AsyncOpenAI: returns one scripted round per completion call."""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        round_ = self.rounds.pop(0)
        if isinstance(round_, Exception):
            raise round_
        if kwargs.get("stream"):
            return _Stream(round_)
        message = SimpleNamespace(content=round_)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Echo(Tool):
    name = "echo"
    description = "Echo text"
    execution_delay = 0.0

    def execute(self, text: str) -> str:
        return f"echo: {text}"


def _llm(rounds) -> LLM:
    llm = LLM(api_base_url="http://unused/v1", model="test", memory=Memory(enabled=False))
    llm.client = _Client(rounds)
    llm.set_system_prompt("Base prompt")
    return llm


class LLMToolLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_round_trip(self):
        llm = _llm([
            [_chunk(tool_call=("call-1", "echo", json.dumps({"text": "hi"})))],
            [_chunk("all "), _chunk("done")],
        ])

        tokens = [t async for t in llm.generate("say hi", tools=[_Echo()])]

        tool_events = [t for t in tokens if isinstance(t, ToolEvent) and t.tool_name]
        results = [t for t in tokens if isinstance(t, ToolResultEvent)]
        self.assertEqual([(t.tool_name, t.tool_args) for t in tool_events], [("echo", {"text": "hi"})])
        self.assertEqual(results[0].result, "echo: hi")
        self.assertEqual("".join(t for t in tokens if isinstance(t, str)), "all done")
        second_request = llm.client.requests[1]["messages"]
        self.assertEqual(second_request[-1], {"role": "tool", "tool_call_id": "call-1", "content": "echo: hi"})
        # The turn is trimmed to the final exchange afterwards.
        self.assertEqual(
            llm.memory.messages,
            [{"role": "user", "content": "say hi"}, {"role": "assistant", "content": "all done"}],
        )

    async def test_malformed_arguments_do_not_crash(self):
        llm = _llm([
            [_chunk(tool_call=("call-1", "echo", "{not json"))],
            [_chunk("recovered")],
        ])

        tokens = [t async for t in llm.generate("go", tools=[_Echo()])]

        result = next(t for t in tokens if isinstance(t, ToolResultEvent)).result
        self.assertIn("[Tool Error] Invalid JSON arguments", result)
        self.assertEqual(tokens[-1], "recovered")

    async def test_skill_is_promoted_to_system_prompt_and_cleared(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brief.md"
            path.write_text("---\nname: brief\ndescription: Brief\n---\nStep one.\n")
            skill = Skill(path)
        llm = _llm([
            [_chunk(tool_call=("call-1", "brief", "{}"))],
            [_chunk("briefed")],
        ])
        llm.memory.messages = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier answer"},
        ]

        tokens = [t async for t in llm.generate("brief me", tools=[_Echo()], skills=[skill])]

        skill_round = llm.client.requests[1]["messages"]
        self.assertIn("<skill:brief>", skill_round[0]["content"])
        self.assertIn("Step one.", skill_round[0]["content"])
        # Prior history is hidden while the skill runs, then restored.
        self.assertNotIn("earlier", json.dumps(skill_round[1:]))
        self.assertEqual(tokens[-1], "briefed")
        self.assertEqual(llm.memory._active_skills, [])
        self.assertNotIn("<skill:", llm.memory.get()[0]["content"])


class RouterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tools = [_Echo(), SimpleNamespace(name="read_file", description="Read a file")]
        self.skills = [SimpleNamespace(name="brief", description="Write a brief")]

    async def test_selects_named_tools_and_skills(self):
        router = ToolRouter(_Client(['Sure: ["brief", "echo"]']), "test")
        tools, skills = await router.route("brief me", self.tools, self.skills)
        self.assertEqual([t.name for t in tools], ["echo"])
        self.assertEqual([s.name for s in skills], ["brief"])

    async def test_falls_back_to_full_catalog(self):
        for reply in ("no json here", RuntimeError("endpoint down")):
            with self.subTest(reply=reply):
                router = ToolRouter(_Client([reply]), "test")
                tools, skills = await router.route("anything", self.tools, self.skills)
                self.assertEqual((tools, skills), (self.tools, self.skills))


class MemorySummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_turn_is_appended_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            memory_file = Path(directory) / "memory.json"
            with mock.patch.object(memory_module, "MEMORY_FILE", memory_file), \
                    mock.patch.object(memory_module, "DATA_DIR", Path(directory)):
                memory = Memory(enabled=True)
                memory.min_summary_chars = 200
                memory.add({"role": "user", "content": "hi"})
                memory.add({"role": "assistant", "content": "hello"})
                await memory.summarize_turn(client=None, model="unused")
                reloaded = Memory(enabled=True)

        self.assertEqual(memory.summary, "user: hi\nassistant: hello")
        self.assertEqual(reloaded.summary, memory.summary)
        self.assertIn("<memory>", memory.get()[0]["content"])


if __name__ == "__main__":
    unittest.main()

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.core import memory as memory_module
from src.core.domain import Domain
from src.core.memory import Memory
from src.core.registry import Registry
from src.core.workflow import Workflow


class IsolatedMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_memory_never_reads_or_writes_the_summary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            memory_file = Path(directory) / "memory.json"
            memory_file.write_text('{"version": 2, "summary": "user conversation"}')
            with mock.patch.object(memory_module, "MEMORY_FILE", memory_file), \
                    mock.patch.object(memory_module, "DATA_DIR", Path(directory)):
                persistent = Memory(enabled=True)
                isolated = Memory(enabled=False)
                isolated.add({"role": "user", "content": "event prompt " * 40})
                isolated.add({"role": "assistant", "content": "event answer " * 40})
                await isolated.summarize_turn(client=None, model="unused")
                on_disk = memory_file.read_text()

        self.assertEqual(persistent.summary, "user conversation")
        self.assertEqual(isolated.summary, "")
        self.assertNotIn("event", on_disk)
        self.assertEqual(len(isolated.messages), 2)


class RegistryReloadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.package = f"reload_pkg_{id(self)}"
        tools = self.root / self.package / "tools"
        tools.mkdir(parents=True)
        (self.root / self.package / "__init__.py").write_text("")
        (tools / "__init__.py").write_text("")
        self.tool_file = tools / "echo.py"
        self._write_tool("first")
        events = self.root / "events"
        events.mkdir()
        self.events_dir = events
        sys.path.insert(0, str(self.root))

    def tearDown(self):
        sys.path.remove(str(self.root))
        for name in [m for m in sys.modules if m.startswith(self.package)]:
            del sys.modules[name]
        self._tmp.cleanup()

    def _write_tool(self, description: str, bump: int = 0) -> None:
        self.tool_file.write_text(
            "from src.core.tool import Tool\n\n"
            "class Echo(Tool):\n"
            "    name = 'echo'\n"
            f"    description = {description!r}\n\n"
            "    def execute(self) -> str:\n"
            "        return 'ok'\n"
        )
        if bump:
            stat = self.tool_file.stat()
            os.utime(self.tool_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + bump))

    def test_edited_tool_module_is_reimported(self):
        registry = Registry(self.root / "domains", self.events_dir, self.root / "automations")
        first = registry.tools(f"{self.package}.tools")
        unchanged = registry.tools(f"{self.package}.tools")
        self._write_tool("second", bump=2_000_000_000)
        edited = registry.tools(f"{self.package}.tools")

        self.assertEqual(first[0].description, "first")
        self.assertEqual(unchanged[0].description, "first")
        self.assertEqual(edited[0].description, "second")

    def test_broken_event_module_is_logged_not_hidden(self):
        (self.events_dir / "broken_evt.py").write_text("")
        registry = Registry(
            self.root / "domains", self.events_dir, self.root / "automations",
            builtin_events_dir=self.root / "no_builtins",
        )
        with mock.patch("src.core.registry.importlib.import_module", side_effect=RuntimeError("boom")), \
                self.assertLogs("src.core.registry", level="ERROR") as logs:
            self.assertEqual(registry.events(), {})
        self.assertIn("broken_evt.py", logs.output[0])


class EventDiscoveryTests(unittest.TestCase):
    def test_builtins_are_discovered_and_user_events_override_them(self):
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events"
            events.mkdir()
            module = f"user_timer_{id(self)}"
            (events / f"{module}.py").write_text(
                "from src.events.timer import TimerEvent\n\n"
                "class MyTimer(TimerEvent):\n"
                "    name = 'timer'\n"
            )
            registry = Registry(Path(directory) / "domains", events, Path(directory) / "automations")
            sys.path.insert(0, directory)
            try:
                with mock.patch.dict(sys.modules):
                    sys.modules["events"] = __import__("types").ModuleType("events")
                    sys.modules["events"].__path__ = [str(events)]
                    found = registry.events()
            finally:
                sys.path.remove(directory)

        self.assertTrue({"loop", "project", "monitor", "file", "schedule"} <= set(found))
        self.assertEqual(found["timer"].__name__, "MyTimer")

    def test_migration_removes_identical_copies_once(self):
        from src.shared import paths

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)
            builtins = config / "builtins"
            events = config / "events"
            builtins.mkdir()
            events.mkdir()
            (builtins / "timer.py").write_text("same")
            (builtins / "monitor.py").write_text("from src.events.file import FileEvent")
            (builtins / "loop.py").write_text("new version")
            (events / "timer.py").write_text("same")
            (events / "monitor.py").write_text("from events.file import FileEvent")
            (events / "loop.py").write_text("old version")
            (events / "mine.py").write_text("custom")
            with mock.patch.multiple(
                paths, CONFIG_DIR=config, EVENTS_DIR=events, BUILTIN_EVENTS_DIR=builtins,
            ), mock.patch("sys.stderr") as stderr:
                paths._migrate_bundled_event_copies()
                (events / "timer.py").write_text("same")
                paths._migrate_bundled_event_copies()
            remaining = sorted(p.name for p in events.iterdir())

        self.assertEqual(remaining, ["loop.py", "mine.py", "timer.py"])
        reported = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("loop.py differs", reported)
        self.assertEqual(reported.count("differs"), 1)


class GenerateEventToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_tools_limits_tools_and_drops_skills(self):
        import asyncio

        seen = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                self.api_base_url, self.model = "u", "m"

            def set_system_prompt(self, prompt):
                pass

            async def generate(self, prompt, tools, skills):
                seen.update(tools=[t.name for t in tools], skills=skills)
                yield "ok"

        domain = Domain.__new__(Domain)
        domain.busy_lock = asyncio.Lock()
        domain.routing_enabled = True  # must not be consulted
        domain.router = None
        domain.system_prompt = ""
        domain.llm = FakeLLM()
        domain.tools = [SimpleNamespace(name="read_file"), SimpleNamespace(name="execute_command")]
        domain.skills = [SimpleNamespace(name="brief", required_tools=["execute_command"])]
        with mock.patch("src.core.domain.LLM", FakeLLM):
            [_ async for _ in domain.generate_event("p", only_tools=["read_file"])]
            self.assertEqual(seen, {"tools": ["read_file"], "skills": []})
            [_ async for _ in domain.generate_event("p", only_tools=[])]
            self.assertEqual(seen, {"tools": [], "skills": []})


class WorkflowConfigurationTests(unittest.TestCase):
    def test_workflows_keep_tools_after_reconfiguration(self):
        class Probe(Workflow):
            name = "probe"
            description = "probe"

            async def run(self):
                yield ""

        domain = Domain.__new__(Domain)
        domain.name = "general"
        domain.tools = [SimpleNamespace(name="read_file")]
        domain.llm = SimpleNamespace(client="client-b", model="model-b")
        with mock.patch("src.core.domain.registry.workflows", return_value=[Probe()]):
            domain.load_workflows()

        (workflow,) = domain.workflows
        self.assertEqual((workflow._client, workflow._model), ("client-b", "model-b"))
        self.assertEqual(workflow._tools, domain.tools)


if __name__ == "__main__":
    unittest.main()

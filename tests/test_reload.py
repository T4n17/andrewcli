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
        registry = Registry(self.root / "domains", self.events_dir, self.root / "automations")
        with mock.patch("src.core.registry.importlib.import_module", side_effect=RuntimeError("boom")), \
                self.assertLogs("src.core.registry", level="ERROR") as logs:
            self.assertEqual(registry.events(), {})
        self.assertIn("broken_evt.py", logs.output[0])


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

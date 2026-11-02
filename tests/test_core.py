import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from src.core.andrew import AndrewCore
from src.core.event import Event
from src.core.registry import Registry
from src.events.loop import LoopEvent


class _Domain:
    def __init__(self, name):
        self.name = name
        self.busy_lock = asyncio.Lock()
        self.closed = False

    def run_workflow(self, cmd):
        if cmd != "/wf":
            return None

        async def gen():
            yield "workflow output"

        return gen()

    def close(self):
        self.closed = True


class _Wait(Event):
    name = "wait"
    description = "waits"

    def __init__(self, message: str = ""):
        self.message = message

    async def condition(self):
        await asyncio.sleep(3600)

    async def trigger(self):
        pass


class RunCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.automations_dir = root / "automations"
        self.automations_dir.mkdir()
        self.registry = Registry(root / "domains", root / "events", self.automations_dir)
        self.registry.events = lambda: {"wait": _Wait, "loop": LoopEvent}
        self._patch = mock.patch("src.core.andrew.registry", self.registry)
        self._patch.start()
        self.core = AndrewCore()
        self.core.domain = _Domain("general")

    def tearDown(self):
        self.core.bus.stop()
        self._patch.stop()
        self._tmp.cleanup()

    async def test_resolution_order_and_results(self):
        builtin = self.core.run_command("/help")
        workflow = self.core.run_command("  /wf  ")
        unknown = self.core.run_command("/nope")

        self.assertIn("Built-in commands", builtin.text)
        self.assertEqual([t async for t in workflow.stream], ["workflow output"])
        self.assertTrue(unknown.error)
        self.assertEqual(self.core.bus.running(), [])

    async def test_bridge_session_held_only_by_dispatching_events(self):
        quiet = self.core.run_command("/wait", bridge_sid="s1")
        talking = self.core.run_command('/wait "hello"', bridge_sid="s2")
        local = self.core.run_command('/wait "hello"')

        self.assertFalse(quiet.holds_session)
        self.assertTrue(talking.holds_session)
        self.assertFalse(local.holds_session)
        bridged = [getattr(e, "_bridge_sid", None) for e in self.core.bus.events().values()]
        self.assertEqual(bridged, [None, "s2", None])

    async def test_dynamic_message_is_not_evaluated_when_started(self):
        state = Path(self._tmp.name) / "loop_state.json"
        result = self.core.run_command(f'/loop "Poll until ready" 0 {state}', bridge_sid="s1")

        (event,) = self.core.bus.events().values()
        self.assertTrue(result.holds_session)
        self.assertIsNone(event._cache)  # message property untouched until the bus fires

    async def test_events_survive_domain_switch_and_automations_rearm(self):
        (self.automations_dir / "general_only.yaml").write_text(
            yaml.safe_dump({"trigger": {"every": "1h"}, "run": "hi", "domain": "general"})
        )
        first = self.core.domain
        armed, _ = self.core.set_domain(first)
        self.assertEqual([i.split("#")[0] for i in armed], ["general_only"])
        self.core.run_command('/wait "hello"')

        second = _Domain("coding")
        armed, _ = self.core.set_domain(second)

        self.assertEqual(armed, [])
        self.assertEqual([e.name for e in self.core.bus.events().values()], ["wait"])
        self.assertTrue(first.closed)
        self.assertIs(second.busy_lock, first.busy_lock)
        self.assertIs(self.core.domain, second)

    async def test_armed_automations_are_listed_apart_from_running_events(self):
        (self.automations_dir / "hourly.yaml").write_text(
            yaml.safe_dump({"trigger": {"every": "1h"}, "run": "hi"})
        )
        self.core.set_domain(self.core.domain)
        self.core.run_command('/wait "hello"')

        listing = self.core.handle_slash("/events")

        self.assertIn("Running events: wait#2\n", listing)
        self.assertRegex(listing, r"Armed automations: hourly \(next run \d\d:\d\d\) — see /automations")

    async def test_only_automations_armed(self):
        (self.automations_dir / "hourly.yaml").write_text(
            yaml.safe_dump({"trigger": {"every": "1h"}, "run": "hi"})
        )
        self.core.set_domain(self.core.domain)

        listing = self.core.handle_slash("/events")

        self.assertIn("No events currently running.\nArmed automations: hourly", listing)


class NotifyTests(unittest.IsolatedAsyncioTestCase):
    async def test_pushed_message_is_shown_without_the_agent(self):
        from fastapi.testclient import TestClient
        from src.core.server import Server

        bridge = Server()
        response = TestClient(bridge.app).post(
            "/notify", json={"source": "wud watcher", "message": "Drafts ready:\n- post 1"},
        )
        core = AndrewCore()
        core.domain = _Domain("general")
        outputs = []
        core._on_event_output = lambda iid, desc, text: outputs.append((iid, desc, text))
        with mock.patch("src.core.andrew.server", bridge):
            pump = asyncio.create_task(core.pump_notices())
            for _ in range(50):
                await asyncio.sleep(0.05)
                if outputs:
                    break
            pump.cancel()
        core.bus.stop()

        self.assertEqual(response.status_code, 202)
        (instance_id, description, text) = outputs[0]
        self.assertTrue(instance_id.startswith("wud-watcher#"))
        self.assertEqual(description, "Message from wud watcher")
        self.assertEqual(text, "Drafts ready:\n- post 1")

    def test_echo_builtin(self):
        core = AndrewCore()
        self.assertEqual(core.handle_slash("/echo  hello there "), "hello there")
        self.assertIsNone(core.handle_slash("/echoes"))


if __name__ == "__main__":
    unittest.main()

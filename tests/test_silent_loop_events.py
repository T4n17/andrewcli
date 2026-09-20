import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from src.core.andrew import AndrewCore
from src.core.event import EventBus
from src.defaults.events.loop import LoopEvent


class EventBusLoopTests(unittest.IsolatedAsyncioTestCase):
    async def _run_until_dispatch(self, event):
        notifications = []
        dispatches = []
        bus = EventBus()
        bus.notify = notifications.append

        async def dispatch(current):
            dispatches.append((current.message, current.system_message, current.silent))
            raise asyncio.CancelledError

        bus.dispatch = dispatch
        await bus._run(event)
        return notifications, dispatches

    async def test_planning_is_silent_before_notification(self):
        with tempfile.TemporaryDirectory() as directory:
            event = LoopEvent(
                goal="Poll until ready",
                state_file=str(Path(directory) / "loop_state.json"),
            )
            notifications, dispatches = await self._run_until_dispatch(event)

        self.assertEqual(notifications, [])
        self.assertEqual(len(dispatches), 1)
        self.assertTrue(dispatches[0][2])

    async def test_iteration_is_visible_before_notification(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "loop_state.json"
            state_file.write_text(json.dumps({
                "goal": "Poll until ready",
                "action": "Poll once",
                "exit_criteria": ["status is ready"],
                "max_iterations": None,
                "iterations": 0,
                "last_observation": "",
                "terminated": False,
                "termination_reason": "",
            }))
            event = LoopEvent(state_file=str(state_file))
            notifications, dispatches = await self._run_until_dispatch(event)

        self.assertEqual(notifications, [event])
        self.assertEqual(len(dispatches), 1)
        self.assertFalse(dispatches[0][2])

    async def test_completed_loop_stops_without_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "loop_state.json"
            state_file.write_text(json.dumps({
                "goal": "Poll until ready",
                "action": "Poll once",
                "exit_criteria": ["status is ready"],
                "max_iterations": None,
                "iterations": 1,
                "last_observation": "status is ready",
                "terminated": True,
                "termination_reason": "status is ready",
            }))
            event = LoopEvent(state_file=str(state_file))
            notifications = []
            dispatches = []
            bus = EventBus()
            bus.notify = notifications.append

            async def dispatch(current):
                dispatches.append(current)

            bus.dispatch = dispatch
            await bus._run(event)

        self.assertEqual(dispatches, [])
        self.assertTrue(event._summary_sent)


class SilentDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_silent_dispatch_does_not_emit_surface_hooks(self):
        class Domain:
            async def generate_event(self, message, system_message, required_tools):
                yield "hidden output"

        event = type("SilentEvent", (), {
            "name": "silent",
            "description": "Silent event",
            "message": "run",
            "system_message": "",
            "required_tools": [],
            "silent": True,
        })()
        core = AndrewCore()
        core.domain = Domain()
        tokens = []
        outputs = []
        completions = []
        core._on_event_token = lambda instance_id, token: tokens.append(token)
        core._on_event_output = lambda instance_id, description, response: outputs.append(response)
        core._on_event_done = completions.append

        await core._event_dispatch(event)

        self.assertEqual(tokens, [])
        self.assertEqual(outputs, [])
        self.assertEqual(completions, [])
        self.assertEqual(core._event_log["silent"], ["hidden output"])


if __name__ == "__main__":
    unittest.main()

import asyncio
import inspect
import os
import select
import sys
import termios
import tty

from src.core.andrew import AndrewCore
from src.core.registry import registry
from src.core.server import server
from src.shared.config import Config
from src.cli.renderer import StreamRenderer


class AndrewCLI(AndrewCore):

    def __init__(self):
        super().__init__()
        self.config = Config()
        self.history = []
        self.renderer = StreamRenderer()
        self.domain_name = self.config.domain
        self.domain = registry.load_domain(self.domain_name)
        self._current_prompt: str = ""
        self._current_buf: list[str] = []

    # ------------------------------------------------------------------
    # AndrewCore hooks
    # ------------------------------------------------------------------

    def _on_event_output(self, instance_id: str, description: str, response: str) -> None:
        header = f"\n\033[33m◆ Event [{instance_id}]: {description}\033[0m"
        body = f"\nAndrew: {response}" if response else ""
        self._bg_print(f"{header}{body}")

    # ------------------------------------------------------------------
    # Terminal I/O
    # ------------------------------------------------------------------

    def _cycle_domain(self):
        domains = registry.domains()
        if len(domains) <= 1:
            return
        try:
            idx = domains.index(self.domain_name)
            next_name = domains[(idx + 1) % len(domains)]
        except ValueError:
            next_name = domains[0]
        try:
            new_domain = registry.load_domain(next_name)
        except ValueError:
            return
        self.domain = new_domain
        self.domain_name = next_name

    async def _read_input(self, prompt):
        self._current_prompt = prompt
        self._current_buf = []

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        sys.stdout.write(prompt)
        sys.stdout.flush()

        buf = []
        history_idx = len(self.history)
        saved_buf = []

        try:
            while True:
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    ch = os.read(fd, 1)

                    if ch in (b'\r', b'\n'):
                        sys.stdout.write('\n')
                        sys.stdout.flush()
                        line = ''.join(buf)
                        if line.strip():
                            self.history.append(line)
                        self._current_prompt = ""
                        self._current_buf = []
                        return line

                    elif ch == b'\x09':  # TAB — cycle domain
                        old_name = self.domain_name
                        self._cycle_domain()
                        if self.domain_name != old_name:
                            sys.stdout.write(f'\r\033[K\033[90mSwitched to domain: {self.domain_name}\033[0m\n')
                            prompt = f"[{self.domain_name}] Ask: "
                            self._current_prompt = prompt
                            sys.stdout.write(f'{prompt}{"".join(buf)}')
                            sys.stdout.flush()

                    elif ch == b'\x1b':  # escape sequence
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            seq1 = os.read(fd, 1)
                            if seq1 == b'[' and select.select([sys.stdin], [], [], 0.05)[0]:
                                seq2 = os.read(fd, 1)
                                if seq2 == b'A' and history_idx > 0:  # UP
                                    if history_idx == len(self.history):
                                        saved_buf = buf[:]
                                    history_idx -= 1
                                    buf = list(self.history[history_idx])
                                    self._current_buf = buf[:]
                                    sys.stdout.write(f'\r\033[K{prompt}{"".join(buf)}')
                                    sys.stdout.flush()
                                elif seq2 == b'B' and history_idx < len(self.history):  # DOWN
                                    history_idx += 1
                                    buf = saved_buf[:] if history_idx == len(self.history) else list(self.history[history_idx])
                                    self._current_buf = buf[:]
                                    sys.stdout.write(f'\r\033[K{prompt}{"".join(buf)}')
                                    sys.stdout.flush()

                    elif ch in (b'\x7f', b'\x08'):  # Backspace
                        if buf:
                            buf.pop()
                            self._current_buf = buf[:]
                            sys.stdout.write('\b \b')
                            sys.stdout.flush()

                    elif ch >= b' ':  # Printable character
                        char = ch.decode('utf-8', errors='replace')
                        buf.append(char)
                        self._current_buf = buf[:]
                        sys.stdout.write(char)
                        sys.stdout.flush()
                else:
                    await asyncio.sleep(0.01)
        finally:
            self._current_prompt = ""
            self._current_buf = []
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _bg_print(self, text: str) -> None:
        """Atomically clear the prompt, print text, then restore the prompt."""
        restore = f"{self._current_prompt}{''.join(self._current_buf)}"
        sys.stdout.write(f"\r\033[K{text}\n{restore}")
        sys.stdout.flush()

    def _event_notify(self, event):
        if not event.message:
            instance_id = getattr(event, "_instance_id", event.name)
            self._bg_print(f"\n\033[33m◆ Event [{instance_id}]: {event.description}\033[0m")

    async def _stream_response(self, prompt: str):
        await self.renderer.render(self.domain.generate(prompt))

    async def _await_bridge(self) -> tuple[str, str]:
        while True:
            try:
                return server.inbox.get_nowait()
            except Exception:
                await asyncio.sleep(0.05)

    async def _stream_response_bridged(self, sid: str, message: str) -> None:
        async def _teed():
            async for token in self.domain.generate(message):
                if isinstance(token, str):
                    server.put_token(sid, token)
                yield token

        try:
            await self.renderer.render(_teed())
        except Exception as e:
            server.finish(sid, error=str(e))
            raise
        else:
            server.finish(sid)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        print("Andrew is running...")
        self.domain.event_bus.notify = self._event_notify
        self.domain.event_bus.dispatch = self._event_dispatch
        asyncio.create_task(self.domain.event_bus.start())
        while True:
            prompt_str = f"[{self.domain_name}] Ask: "
            stdin_task = asyncio.create_task(self._read_input(prompt_str))
            bridge_task = asyncio.create_task(self._await_bridge())

            done, pending = await asyncio.wait(
                {stdin_task, bridge_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            if bridge_task in done:
                sid, user_input = bridge_task.result()
                sys.stdout.write(f"\r\033[K\033[36m[server] {user_input}\033[0m\n")
                sys.stdout.flush()
            else:
                sid = None
                user_input = stdin_task.result()

            if not user_input.strip():
                if sid:
                    server.finish(sid)
                continue

            if user_input.startswith("/"):
                cmd = user_input.strip()
                bus = self.domain.event_bus
                started_event = None
                response = self.handle_slash(cmd, bus)
                if response is None:
                    try:
                        started_event = registry.parse_slash_command(user_input)
                    except ValueError as exc:
                        response = str(exc)
                        started_event = None
                    if started_event is not None:
                        msg_descriptor = inspect.getattr_static(type(started_event), 'message', None)
                        has_message = isinstance(msg_descriptor, property) or bool(started_event.message)
                        if sid and has_message:
                            started_event._bridge_sid = sid
                        instance_id = bus.add(started_event)
                        response = f"✓ Event '{started_event.name}' started [{instance_id}]"
                    else:
                        response = f"Unknown command: {user_input}\n" + registry.list_commands(bus.running())
                sys.stdout.write(f"\033[32m{response}\033[0m\n")
                sys.stdout.flush()
                if sid:
                    server.put_token(sid, response)
                    if not getattr(started_event, '_bridge_sid', None):
                        server.finish(sid)
                continue

            if sid:
                await self._stream_response_bridged(sid, user_input)
            else:
                await self._stream_response(user_input)


if __name__ == "__main__":
    try:
        asyncio.run(AndrewCLI().run())
    except KeyboardInterrupt:
        print("\nAndrew stopped. Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

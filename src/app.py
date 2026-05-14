from src.core.registry import registry
from src.core.server import server
from src.shared.config import Config
from src.ui.renderer import StreamRenderer
import asyncio
import os
import select
import sys
import termios
import tty


class AndrewCLI:

    def __init__(self):
        self.config = Config()
        self.history = []
        self.renderer = StreamRenderer()
        self.domain_name = self.config.domain
        self.domain = registry.load_domain(self.domain_name)

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
                        return line

                    elif ch == b'\x09':  # TAB — cycle domain
                        old_name = self.domain_name
                        self._cycle_domain()
                        if self.domain_name != old_name:
                            sys.stdout.write(f'\r\033[K\033[90mSwitched to domain: {self.domain_name}\033[0m\n')
                            prompt = f"[{self.domain_name}] Ask: "
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
                                    sys.stdout.write(f'\r\033[K{prompt}{"".join(buf)}')
                                    sys.stdout.flush()
                                elif seq2 == b'B' and history_idx < len(self.history):  # DOWN
                                    history_idx += 1
                                    buf = saved_buf[:] if history_idx == len(self.history) else list(self.history[history_idx])
                                    sys.stdout.write(f'\r\033[K{prompt}{"".join(buf)}')
                                    sys.stdout.flush()

                    elif ch in (b'\x7f', b'\x08'):  # Backspace
                        if buf:
                            buf.pop()
                            sys.stdout.write('\b \b')
                            sys.stdout.flush()

                    elif ch >= b' ':  # Printable character
                        char = ch.decode('utf-8', errors='replace')
                        buf.append(char)
                        sys.stdout.write(char)
                        sys.stdout.flush()
                else:
                    await asyncio.sleep(0.01)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    async def _stream_response(self, prompt: str):
        await self.renderer.render(self.domain.generate(prompt))

    def _event_notify(self, event):
        # Only print immediately if no dispatch will follow (no interleaving risk)
        if not event.message:
            sys.stdout.write(f"\n\033[33m◆ Event [{event.name}]: {event.description}\033[0m\n")
            sys.stdout.flush()

    async def _event_dispatch(self, event):
        sys.stdout.write(f"\n\033[33m◆ Event [{event.name}]: {event.description}\033[0m\n")
        sys.stdout.flush()
        sid = getattr(event, "_bridge_sid", None)
        if sid:
            event._bridge_sid = None  # consume once; multi-iteration events reuse no session

        async def _teed():
            async for token in self.domain.generate_event(event.message):
                if isinstance(token, str) and sid:
                    server.put_token(sid, token)
                yield token

        try:
            await self.renderer.render(_teed())
        except asyncio.CancelledError:
            if sid:
                server.finish(sid, error="Event stopped")
            raise
        except Exception as e:
            if sid:
                server.finish(sid, error=str(e))
            raise
        else:
            if sid:
                server.finish(sid)

    async def _await_bridge(self) -> tuple[str, str]:
        """Block until a message arrives in the server's inbox."""
        while True:
            try:
                return server.inbox.get_nowait()
            except Exception:
                await asyncio.sleep(0.05)

    async def _stream_response_bridged(self, sid: str, message: str) -> None:
        """Like _stream_response but also captures tokens into the bridge session."""

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

    async def _get_next_prompt(self, prompt_str: str) -> str:
        return await self._read_input(prompt_str)

    async def run(self):
        print("Andrew is running...")
        self.domain.event_bus.notify = self._event_notify
        self.domain.event_bus.dispatch = self._event_dispatch
        asyncio.create_task(self.domain.event_bus.start())
        while True:
            prompt_str = f"[{self.domain_name}] Ask: "
            stdin_task = asyncio.create_task(self._get_next_prompt(prompt_str))
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
                # Clear any partial typed input, then show the injected message.
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
                if cmd == "/events":
                    response = registry.list_commands(bus.running())
                elif cmd.startswith("/stop"):
                    parts = cmd.split(None, 1)
                    if len(parts) == 1:
                        running = bus.running()
                        response = ("Running: " + ", ".join(running)) if running else "No events running."
                    elif bus.remove(parts[1]):
                        response = f"✓ Event '{parts[1]}' stopped"
                    else:
                        response = f"No running event named '{parts[1]}'"
                else:
                    import inspect
                    started_event = registry.parse_slash_command(user_input)
                    if started_event is not None:
                        msg_descriptor = inspect.getattr_static(type(started_event), 'message', None)
                        has_message = isinstance(msg_descriptor, property) or bool(started_event.message)
                        if sid and has_message:
                            started_event._bridge_sid = sid
                        bus.add(started_event)
                        response = f"✓ Event '{started_event.name}' started"
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

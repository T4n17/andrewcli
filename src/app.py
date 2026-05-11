from src.core.registry import available_domains, load_domain, parse_slash_command, list_commands
from src.shared.config import Config
from src.ui.renderer import StreamRenderer
import asyncio
import os
import select
import sys
import termios
import tty


class AndrewCLI:

    def __init__(self, voice_enabled: bool = False):
        self.config = Config()
        self.history = []
        self.renderer = StreamRenderer()
        self.domain_name = self.config.domain
        self.domain = load_domain(self.domain_name)

        # Voice I/O is optional and additive. When enabled the CLI accepts
        # both typed prompts and wake-word-triggered spoken prompts; each
        # typed or spoken response also streams through TTS. Constructed
        # eagerly on purpose: failing fast with a clear error beats a
        # mysterious silence later on.
        self.voice_enabled = voice_enabled
        self.stt = None
        self.tts = None
        if voice_enabled:
            from src.voice import build_voice_io
            self.stt, self.tts = build_voice_io(self.config)
        # Single-producer prompt channel fed by stdin and (optionally) STT.
        self._prompts: asyncio.Queue[tuple[str, str]] | None = None

    def _cycle_domain(self):
        domains = available_domains()
        if len(domains) <= 1:
            return
        try:
            idx = domains.index(self.domain_name)
            next_name = domains[(idx + 1) % len(domains)]
        except ValueError:
            next_name = domains[0]
        try:
            new_domain = load_domain(next_name)
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
        # Serialization with events is handled by Domain.busy_lock.
        # When voice is on we tee the token stream: the renderer
        # consumes one branch (stdout) and the TTS consumes the other
        # (speaker). The domain iterator must only be iterated once, so
        # the tee lives inside an async generator fed from the single
        # source.
        if not self.voice_enabled:
            await self.renderer.render(self.domain.generate(prompt))
            return

        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def teed():
            try:
                async for item in self.domain.generate(prompt):
                    if isinstance(item, str):
                        await tts_queue.put(item)
                    yield item
            finally:
                # Ensure the TTS consumer terminates even if the
                # renderer raises or the producer is cancelled.
                await tts_queue.put(None)

        async def tts_feed():
            while True:
                tok = await tts_queue.get()
                if tok is None:
                    return
                yield tok

        # Strip markdown punctuation before TTS so the speaker doesn't
        # read ``**bold**`` as "asterisk asterisk bold asterisk asterisk"
        # (and drops URL text after ``[link]`` closes).
        from src.voice import strip_markdown
        tts_task = asyncio.create_task(
            self.tts.speak_stream(strip_markdown(tts_feed()))
        )
        try:
            await self.renderer.render(teed())
        finally:
            # Wait for any trailing audio to flush before accepting the
            # next turn so the final sentence isn't cut off.
            try:
                await tts_task
            except asyncio.CancelledError:
                pass

    def _event_notify(self, event):
        # Only print immediately if no dispatch will follow (no interleaving risk)
        if not event.message:
            sys.stdout.write(f"\n\033[33m◆ Event [{event.name}]: {event.description}\033[0m\n")
            sys.stdout.flush()

    async def _event_dispatch(self, event):
        sys.stdout.write(f"\n\033[33m◆ Event [{event.name}]: {event.description}\033[0m\n")
        sys.stdout.flush()
        from src.core import server as bridge
        sid = getattr(event, "_bridge_sid", None)
        if sid:
            event._bridge_sid = None  # consume once; multi-iteration events reuse no session

        async def _teed():
            async for token in self.domain.generate_event(event.message):
                if isinstance(token, str) and sid:
                    bridge.put_token(sid, token)
                yield token

        try:
            await self.renderer.render(_teed())
        except asyncio.CancelledError:
            if sid:
                bridge.finish(sid, error="Event stopped")
            raise
        except Exception as e:
            if sid:
                bridge.finish(sid, error=str(e))
            raise
        else:
            if sid:
                bridge.finish(sid)

    async def _await_bridge(self) -> tuple[str, str]:
        """Block until a message arrives in the bridge inbox."""
        from src.core import server as bridge
        while True:
            try:
                return bridge.inbox.get_nowait()
            except Exception:
                await asyncio.sleep(0.05)

    async def _stream_response_bridged(self, sid: str, message: str) -> None:
        """Like _stream_response but also captures tokens into the bridge session."""
        from src.core import server as bridge

        async def _teed():
            async for token in self.domain.generate(message):
                if isinstance(token, str):
                    bridge.put_token(sid, token)
                yield token

        try:
            await self.renderer.render(_teed())
        except Exception as e:
            bridge.finish(sid, error=str(e))
            raise
        else:
            bridge.finish(sid)

    async def _get_next_prompt(self, prompt_str: str) -> str:
        """Return the next user prompt, whether typed or spoken.

        With voice off this is just ``_read_input``. With voice on we
        race the stdin reader against an :meth:`SpeechToText.listen_once`
        call; whichever produces first wins and the other is cancelled.
        If voice wins but returns an empty transcript (wake word fired
        but no speech / unintelligible), we silently re-arm both
        producers rather than surfacing a confusing blank prompt.
        """
        if not self.voice_enabled:
            return await self._read_input(prompt_str)

        # Only print the prompt on the first pass; silent re-arms after
        # empty-transcript cycles must not reprint it (would produce
        # `[general] Ask: [general] Ask: ...` on the same line).
        first_pass = True

        def _on_wake():
            # Visual cue: overwrite the current prompt line with a
            # "listening" indicator so the user knows the wake word was
            # heard and we're now recording their actual request.
            sys.stdout.write(
                f"\r\033[K{prompt_str}\033[35m🎙 listening...\033[0m"
            )
            sys.stdout.flush()

        while True:
            read_prompt = prompt_str if first_pass else ""
            first_pass = False
            typed_task = asyncio.create_task(self._read_input(read_prompt))
            voice_task = asyncio.create_task(self.stt.listen_once(on_wake=_on_wake))
            try:
                done, _ = await asyncio.wait(
                    {typed_task, voice_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                typed_task.cancel()
                voice_task.cancel()
                raise

            if typed_task in done:
                voice_task.cancel()
                try:
                    await voice_task
                except (asyncio.CancelledError, Exception):
                    pass
                return typed_task.result()

            # Voice won. Stop the terminal reader so termios is restored.
            typed_task.cancel()
            try:
                await typed_task
            except (asyncio.CancelledError, Exception):
                pass

            spoken = (voice_task.result() or "").strip()
            if not spoken:
                # Empty transcript: wake word misfired or no speech.
                # Silently re-arm without disturbing the prompt line.
                continue

            # Rewrite the prompt with the transcribed text so the
            # terminal log reads the same as a typed-in prompt.
            sys.stdout.write(f"\r\033[K{prompt_str}\033[35m🎙 {spoken}\033[0m\n")
            sys.stdout.flush()
            return spoken

    async def run(self):
        print(
            "Andrew is running"
            + (f" (voice on - say '{self.stt.wake_word}' or just type)"
               if self.voice_enabled else "")
            + "..."
        )
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
                    from src.core import server as bridge
                    bridge.finish(sid)
                continue

            if user_input.startswith("/"):
                cmd = user_input.strip()
                bus = self.domain.event_bus
                started_event = None
                if cmd == "/events":
                    response = list_commands(bus.running())
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
                    started_event = parse_slash_command(user_input)
                    if started_event is not None:
                        msg_descriptor = inspect.getattr_static(type(started_event), 'message', None)
                        has_message = isinstance(msg_descriptor, property) or bool(started_event.message)
                        if sid and has_message:
                            started_event._bridge_sid = sid
                        bus.add(started_event)
                        response = f"✓ Event '{started_event.name}' started"
                    else:
                        response = f"Unknown command: {user_input}\n" + list_commands(bus.running())
                sys.stdout.write(f"\033[32m{response}\033[0m\n")
                sys.stdout.flush()
                if sid:
                    from src.core import server as bridge
                    bridge.put_token(sid, response)
                    if not getattr(started_event, '_bridge_sid', None):
                        bridge.finish(sid)
                continue

            if sid:
                await self._stream_response_bridged(sid, user_input)
            else:
                await self._stream_response(user_input)


if __name__ == "__main__":
    try:
        andrew = AndrewCLI()
        asyncio.run(andrew.run())
    except KeyboardInterrupt:
        print("\nAndrew stopped. Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

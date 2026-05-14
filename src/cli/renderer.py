import asyncio
import os
import sys
import termios
import tty

from src.core.llm import ToolEvent, RouteEvent, format_tool_status
from src.cli.animations import Spinner
from src.cli.filter import ThinkFilter


class StreamRenderer:
    def __init__(self):
        self.spinner = Spinner()

    async def render(self, token_stream):
        self.spinner.status = "Thinking..."
        self.spinner.start()
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        think_filter = ThinkFilter()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()

        async def _consume():
            first = True
            async for token in token_stream:
                if isinstance(token, (RouteEvent, ToolEvent)):
                    status = format_tool_status(token)
                    if isinstance(token, ToolEvent):
                        sys.stdout.write("\r\033[K")
                    if status is not None:
                        self.spinner.status = status
                        self.spinner.restart()
                    sys.stdout.flush()
                    continue

                if self.spinner.is_running:
                    self.spinner.stop()
                    sys.stdout.write("\r\033[K")
                    if first:
                        sys.stdout.write("Andrew: ")
                        first = False
                    sys.stdout.flush()

                segments = think_filter.process(token)
                for text, is_thinking in segments:
                    if is_thinking:
                        sys.stdout.write("\033[2;3m")
                    # Stream text at token granularity rather than
                    # per-character — the LLM already paces delivery,
                    # and a per-char sleep made long responses crawl.
                    sys.stdout.write(text)
                    if is_thinking:
                        sys.stdout.write("\033[0m")
                    sys.stdout.flush()

            if self.spinner.is_running:
                self.spinner.stop()
                sys.stdout.write("\r\033[K")

        consume_task = asyncio.create_task(_consume())

        def _on_stdin():
            try:
                ch = os.read(fd, 1)
                if ch == b'\x1b':
                    stop.set()
                    consume_task.cancel()
            except OSError:
                pass

        loop.add_reader(fd, _on_stdin)

        try:
            await consume_task
            print()
        except asyncio.CancelledError:
            if stop.is_set():
                if self.spinner.is_running:
                    self.spinner.stop()
                    sys.stdout.write("\r\033[K")
                sys.stdout.write("\033[0m [stopped]")
                sys.stdout.flush()
                print()
            else:
                raise
        finally:
            loop.remove_reader(fd)
            await token_stream.aclose()
            if self.spinner.is_running:
                self.spinner.stop()
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            termios.tcflush(fd, termios.TCIFLUSH)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

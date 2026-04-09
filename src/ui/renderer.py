import asyncio
import os
import select
import sys
import termios
import tty

from src.core.llm import ToolEvent
from src.ui.animations import Spinner
from src.ui.filter import ThinkFilter


class StreamRenderer:
    def __init__(self):
        self.spinner = Spinner()

    def _check_esc(self, fd):
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = os.read(fd, 1)
            if ch == b'\x1b':
                return True
        return False

    async def render(self, token_stream):
        self.spinner.status = "Thinking..."
        self.spinner.start()
        cancelled = False
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        think_filter = ThinkFilter()

        try:
            first = True
            async for token in token_stream:
                if cancelled:
                    continue

                if self._check_esc(fd):
                    cancelled = True
                    if self.spinner.is_running:
                        self.spinner.stop()
                        sys.stdout.write("\r\033[K")
                    sys.stdout.write("\033[0m [stopped]")
                    sys.stdout.flush()
                    continue

                if isinstance(token, ToolEvent):
                    sys.stdout.write("\r\033[K")
                    if token.tool_name:
                        detail = ""
                        if token.tool_args:
                            first_val = str(next(iter(token.tool_args.values()), ""))
                            if len(first_val) > 60:
                                first_val = first_val[:57] + "..."
                            if first_val:
                                detail = f": {first_val}"
                        self.spinner.status = f"Running {token.tool_name}{detail}"
                    else:
                        self.spinner.status = "Thinking..."
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
                    for char in text:
                        sys.stdout.write(char)
                        sys.stdout.flush()
                        await asyncio.sleep(0.02)
                    if is_thinking:
                        sys.stdout.write("\033[0m")
                        sys.stdout.flush()

            if self.spinner.is_running:
                self.spinner.stop()
                sys.stdout.write("\r\033[K")
            print()
        finally:
            termios.tcflush(fd, termios.TCIFLUSH)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

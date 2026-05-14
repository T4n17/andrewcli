import asyncio
import itertools
import shutil
import sys


class Spinner:
    def __init__(self):
        self.status = "Thinking..."
        self._task = None
        self._frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    async def _animate(self):
        for frame in itertools.cycle(self._frames):
            width = shutil.get_terminal_size((80, 20)).columns
            line = f"{frame} {self.status}"
            if len(line) > width - 1:
                line = line[:width - 4] + "..."
            sys.stdout.write(f"\r\033[K\033[36m{line}\033[0m")
            sys.stdout.flush()
            await asyncio.sleep(0.08)

    def start(self):
        self._task = asyncio.create_task(self._animate())

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()

    @property
    def is_running(self):
        return self._task is not None and not self._task.done()

    def restart(self):
        if not self.is_running:
            self.start()

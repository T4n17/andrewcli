from src.shared.config import Config
from src.ui.renderer import StreamRenderer
import asyncio
import importlib
import os
import select
import sys
import termios
import tty
import argparse


class AndrewCLI:

    def __init__(self):
        self.config = Config()
        self.history = []
        self.renderer = StreamRenderer()
        self.domain = self._load_domain()

    def _load_domain(self, domain_name=None):
        try:
            self.domain_name = domain_name or self.config.domain
            module = importlib.import_module(f"src.domains.{self.domain_name}")
            class_name = f"{self.domain_name.capitalize()}Domain"
            domain_class = getattr(module, class_name)
            return domain_class()
        except KeyError:
            raise ValueError("Domain not found in config")
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Could not load domain '{self.domain_name}': {e}")

    def _get_available_domains(self):
        domains = []
        for f in os.listdir('src/domains'):
            if f.endswith('.py') and f != '__init__.py':
                domains.append(f[:-3])
        return sorted(domains)

    def _cycle_domain(self):
        domains = self._get_available_domains()
        if len(domains) <= 1:
            return
        try:
            idx = domains.index(self.domain_name)
            next_name = domains[(idx + 1) % len(domains)]
        except ValueError:
            next_name = domains[0]
        try:
            self.domain = self.load_domain(next_name)
        except ValueError:
            pass

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

    async def run(self):
        print(f"Andrew is running...")
        while True:
            prompt = f"[{self.domain_name}] Ask: "
            user_input = await self._read_input(prompt)
            if not user_input.strip():
                continue
            await self._stream_response(user_input)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--gui", type=bool, default=False)
        args = parser.parse_args()
        if args.gui:
            # TODO: Implement GUI mode
            pass
        andrew = AndrewCLI()
        asyncio.run(andrew.run())
    except KeyboardInterrupt:
        print("\nAndrew stopped. Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

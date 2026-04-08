import asyncio
import importlib
import itertools
import sys
import yaml


class AndrewCLI:

    def __init__(self):
        self.config = self._load_config()
        self.domain = self._load_domain()

    def _load_config(self):
        try:
            with open("config.yaml", "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {"domain": "general"}

    def _load_domain(self):
        try:
            self.domain_name = self.config["domain"]
            module = importlib.import_module(f"src.domains.{self.domain_name}")
            class_name = f"{self.domain_name.capitalize()}Domain"
            domain_class = getattr(module, class_name)
            return domain_class()
        except KeyError:
            raise ValueError("Domain not found in config")
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Could not load domain '{self.domain_name}': {e}")

    async def _spinner(self):
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        for frame in itertools.cycle(frames):
            sys.stdout.write(f"\r\033[36m{frame} Thinking...\033[0m")
            sys.stdout.flush()
            await asyncio.sleep(0.08)

    async def _stream_response(self, prompt: str):
        spinner_task = asyncio.create_task(self._spinner())

        first = True
        async for token in self.domain.generate(prompt):
            if first:
                spinner_task.cancel()
                sys.stdout.write("\r\033[K")
                sys.stdout.write("Andrew: ")
                sys.stdout.flush()
                first = False
            for char in token:
                sys.stdout.write(char)
                sys.stdout.flush()
                await asyncio.sleep(0.02)

        if first:
            spinner_task.cancel()
            sys.stdout.write("\r\033[K")
        print()

    async def run(self):
        print(f"Andrew is running... (Domain: {self.domain_name})")
        loop = asyncio.get_event_loop()
        while True:
            user_input = await loop.run_in_executor(None, input, "Ask: ")
            await self._stream_response(user_input)


if __name__ == "__main__":
    andrew = AndrewCLI()
    asyncio.run(andrew.run())

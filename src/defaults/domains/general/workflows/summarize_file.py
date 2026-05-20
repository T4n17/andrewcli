from pathlib import Path

from src.core.workflow import Workflow


class SummarizeFile(Workflow):
    """Summarize a file: read it deterministically, then run one LLM inference."""

    name = "summarize"
    description = "Read a file and summarize its contents"

    async def run(self, path: str):
        try:
            content = Path(path).read_text()
        except FileNotFoundError:
            yield f"Error: file not found — {path}"
            return
        except Exception as exc:
            yield f"Error reading file: {exc}"
            return

        word_count = len(content.split())
        yield f"Read `{path}` ({word_count} words). Summarizing...\n\n"

        async for token in self.infer(
            [{"role": "user", "content": f"File: {path}\n\n{content}"}],
            system=(
                "You are a concise technical summarizer. "
                "Produce a clear, structured summary of the file's purpose and key contents. "
                "Use bullet points for the main points."
            ),
        ):
            yield token

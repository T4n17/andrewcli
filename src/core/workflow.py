import openai
from abc import ABC, abstractmethod


class Workflow(ABC):
    """A Workflow is a user-invoked pipeline that runs outside the LLM loop.

    Workflows are triggered directly via slash commands (``/name [args]``)
    and are never called by the LLM. Unlike :class:`~src.core.tool.Tool`
    and :class:`~src.core.skill.Skill`, they own their own execution flow:
    deterministic steps run as plain Python, and :meth:`infer` is available
    for single-shot LLM calls on dynamic parts.

    Subclasses define ``name``, ``description``, and implement
    :meth:`run` as an async generator that yields ``str`` tokens.  That
    contract makes every workflow directly consumable by the existing
    :class:`~src.cli.renderer.StreamRenderer` with no extra plumbing.

    Workflow files live in a domain's ``workflows/`` folder and are
    auto-discovered by :py:meth:`src.core.registry.Registry.workflows`.

    Example::

        class Summarize(Workflow):
            name = "summarize"
            description = "Summarize a file with a single LLM call"

            async def run(self, path: str):
                content = Path(path).read_text()
                yield "Summarizing...\\n"
                async for token in self.infer(
                    [{"role": "user", "content": content}],
                    system="Produce a concise summary.",
                ):
                    yield token
    """

    name: str
    description: str

    def __init__(self):
        self._client: openai.AsyncOpenAI | None = None
        self._model: str | None = None

    def configure(self, client: openai.AsyncOpenAI, model: str) -> None:
        """Inject the domain LLM client. Called by Domain after instantiation."""
        self._client = client
        self._model = model

    @abstractmethod
    async def run(self, **kwargs):
        """Implement the workflow as an async generator that yields str tokens."""

    async def infer(self, messages: list[dict], system: str = ""):
        """Stream a single LLM inference, yielding str tokens.

        Use this inside :meth:`run` for dynamic steps that need the model.
        The call is one-shot — it never enters the agent tool-call loop.
        """
        if self._client is None:
            raise RuntimeError(
                f"Workflow '{self.name}' is not configured — "
                "Domain must call configure() before run() is invoked."
            )
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

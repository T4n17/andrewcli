"""Glue layer: run the full voice loop on top of a Domain.

Pipeline per turn:

    1. SpeechToText.listen_once()      # blocks until wake + user speech
    2. domain.generate(prompt)          # same Domain used by CLI/tray/server
    3. TextToSpeech.speak_stream(...)   # play tokens as sentences complete

The console still mirrors the LLM's streamed text so you can see what
the agent is saying in addition to hearing it.

Why a class-based session?
    - Matches the rest of the codebase (AndrewCLI, AndrewTray, ...).
    - Isolates lifecycle (stop, cleanup) in one place.
    - Lets tray/future surfaces reuse the same loop.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from src.core.domain import Domain
from src.core.llm import RouteEvent, ToolEvent
from src.shared.config import Config

log = logging.getLogger(__name__)


class VoiceSession:
    """Wake-word-triggered voice I/O driving a Domain."""

    def __init__(self, domain: Domain, config: Config | None = None):
        cfg = config or Config()
        if not cfg.voice_enabled:
            log.warning(
                "voice.enabled is false in config.yaml; starting voice "
                "session anyway because --voice was passed explicitly."
            )

        # Local import avoids a circular dependency (session is imported
        # from :mod:`src.voice.__init__`).
        from src.voice import build_voice_io

        self.domain = domain
        self.stt, self.tts = build_voice_io(cfg)
        self._stop = False

    async def run(self) -> None:
        """Main loop. Ctrl-C stops it cleanly."""
        sys.stdout.write(
            f"\n\033[36mVoice mode ready. Say '{self.stt.wake_word}' "
            "followed by your request.\033[0m\n"
        )
        sys.stdout.flush()

        while not self._stop:
            prompt = await self.stt.listen_once()
            if not prompt:
                continue

            sys.stdout.write(f"\n\033[35mYou:\033[0m {prompt}\n\033[35mAndrew:\033[0m ")
            sys.stdout.flush()

            # Bridge domain.generate() (which yields RouteEvent/ToolEvent/str)
            # to an async iterator of str tokens for TTS, while echoing to the
            # terminal. Both TTS and console see the same stream.
            # Pre-render markdown before TTS so the speaker doesn't
            # pronounce asterisks / backticks / link URLs.
            from src.voice import strip_markdown
            await self.tts.speak_stream(
                strip_markdown(self._token_stream(prompt))
            )
            sys.stdout.write("\n")
            sys.stdout.flush()

    async def _token_stream(self, prompt: str):
        async for item in self.domain.generate(prompt):
            if isinstance(item, RouteEvent):
                log.debug("routed to: %s", item.tool_names)
                continue
            if isinstance(item, ToolEvent):
                log.debug("tool event: %s %s", item.tool_name, item.tool_args)
                continue
            # Plain string token: echo to console AND feed to TTS.
            if isinstance(item, str):
                sys.stdout.write(item)
                sys.stdout.flush()
                yield item

    def stop(self) -> None:
        self._stop = True
        self.stt.stop()

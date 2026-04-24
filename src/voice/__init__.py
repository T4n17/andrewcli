"""Voice I/O: wake-word STT and streaming TTS.

See :mod:`src.voice.stt` and :mod:`src.voice.tts` for the two public
classes. Both lazy-import their heavy optional dependencies
(``faster-whisper``, ``openwakeword``, ``piper-tts``, ``sounddevice``)
inside ``__init__``, so importing this package is always cheap.

:func:`build_voice_io` is the canonical way to construct the pair
given a :class:`~src.shared.config.Config`; it's used by the CLI
(``AndrewCLI``), tray (``AndrewTrayApp``) and ``VoiceSession`` so the
same ``voice.*`` config keys produce the same behavior everywhere.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from src.voice.sanitize import strip_markdown
from src.voice.session import VoiceSession
from src.voice.stt import SpeechToText
from src.voice.tts import TextToSpeech
from src.voice.tts_edge import EdgeTTS

if TYPE_CHECKING:
    from src.shared.config import Config

__all__ = [
    "SpeechToText", "TextToSpeech", "EdgeTTS", "VoiceSession",
    "build_voice_io", "strip_markdown",
]

# Public type alias for the TTS backend: either Piper or Edge.
TTSBackend = Union[TextToSpeech, EdgeTTS]


def build_voice_io(config: "Config | None" = None) -> tuple[SpeechToText, TTSBackend]:
    """Construct the ``(stt, tts)`` pair from config.

    Single source of truth for voice instantiation. Reads:

    - ``voice.wake_word``, ``voice.wake_threshold`` (STT activation)
    - ``voice.stt_model``, ``voice.stt_language`` (faster-whisper)
    - ``voice.tts_engine`` (``"piper"`` or ``"edge"``)
    - ``voice.tts_voice``, ``voice.tts_speed``
    - ``voice.input_device`` / ``voice.output_device``
    """
    from src.shared.config import Config
    cfg = config or Config()

    stt = SpeechToText(
        whisper_model=cfg.voice_stt_model,
        language=cfg.voice_stt_language,
        wake_word=cfg.voice_wake_word,
        wake_threshold=cfg.voice_wake_threshold,
        input_device=cfg.voice_input_device,
    )

    engine = (cfg.voice_tts_engine or "piper").lower()
    if engine == "edge":
        tts: TTSBackend = EdgeTTS(
            voice=cfg.voice_tts_voice,
            speed=cfg.voice_tts_speed,
            output_device=cfg.voice_output_device,
        )
    elif engine == "piper":
        tts = TextToSpeech(
            voice=cfg.voice_tts_voice,
            speed=cfg.voice_tts_speed,
            output_device=cfg.voice_output_device,
        )
    else:
        raise ValueError(
            f"Unknown voice.tts_engine={engine!r}; use 'piper' or 'edge'."
        )
    return stt, tts

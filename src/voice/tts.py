"""Text-to-Speech via piper (ONNX VITS).

Piper voices are tiny (~25 MB each), ONNX-based, near-realtime on CPU,
and cover 30+ languages. Models live on Hugging Face at
``rhasspy/piper-voices``; we fetch them on first use via
``huggingface_hub``.

Voice names follow piper's own scheme: ``<lang>_<region>-<speaker>-<quality>``
(e.g. ``en_US-amy-medium``, ``it_IT-riccardo-x_low``). The repo layout is
``<lang>/<lang_region>/<speaker>/<quality>/<voice_name>.onnx[.json]``.

Usage::

    tts = TextToSpeech(voice="en_US-amy-medium")
    await tts.speak("Hello world")

    # Streaming from an LLM token generator:
    async def tokens():
        yield "Hello "; yield "world, "; yield "how are you?"
    await tts.speak_stream(tokens())

Interruption: call :meth:`stop` to cancel the current playback (useful
when the wake word fires while the agent is talking).

Optional dep: if ``piper-tts``, ``sounddevice``, or ``huggingface_hub``
is missing, :class:`TextToSpeech` raises ``ImportError`` at construction.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import AsyncIterable

log = logging.getLogger(__name__)


# Sentence boundary detector for streaming playback. We break on any of
# .!? followed by whitespace/EOL so we can start synthesizing a sentence
# as soon as it's complete without waiting for the whole response.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class TextToSpeech:
    """Piper-based TTS with streaming synthesis and interruptible playback."""

    # Default voice; English, medium quality, ~60 MB.
    DEFAULT_VOICE = "en_US-amy-medium"

    # HuggingFace repo hosting all piper voices.
    HF_REPO = "rhasspy/piper-voices"

    # Local cache for downloaded voice files. Parallels the whisper cache.
    CACHE_DIR = Path.home() / ".cache" / "andrewcli" / "piper"

    def __init__(
        self,
        *,
        voice: str | None = None,
        output_device: int | str | None = None,
        speed: float = 1.0,
    ):
        # Lazy-import optional deps.
        try:
            import numpy as np
            import sounddevice as sd
            from piper.voice import PiperVoice
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "TextToSpeech requires the voice extras. Install with: "
                "pip install piper-tts sounddevice huggingface_hub"
            ) from exc

        self._np = np
        self._sd = sd
        self._hf_hub_download = hf_hub_download

        self.voice_name = voice or self.DEFAULT_VOICE
        self._speed = speed
        self._output_device = output_device

        t0 = time.monotonic()
        model_path, config_path = self._ensure_voice_files(self.voice_name)
        self._voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        self._sample_rate = int(self._voice.config.sample_rate)
        log.info(
            "loaded Piper voice %r (sr=%d Hz) in %.2fs",
            self.voice_name, self._sample_rate, time.monotonic() - t0,
        )

        # Track the current playback task so stop() can cancel it.
        self._play_task: asyncio.Task | None = None

        # Output-side sample-rate negotiation (resolved lazily on first
        # playback, see _resolve_output_rate). Piper voices are typically
        # 22050 Hz which many ALSA hw: devices reject outright; we pick
        # a rate the device accepts and upsample int16 PCM in Python.
        self._out_sr: int | None = None
        self._out_resample = None

    # ---- public API -------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Synthesize `text` and play it to the default output device."""
        text = text.strip()
        if not text:
            return
        await self._cancel_current()
        self._play_task = asyncio.create_task(self._speak_chunks([text]))
        try:
            await self._play_task
        except asyncio.CancelledError:
            pass

    async def speak_stream(self, token_iter: AsyncIterable[str]) -> None:
        """Buffer streamed tokens into sentences and play them as each completes.

        Gives far lower time-to-first-audio than waiting for the full
        response: as soon as the first sentence ends we start synthesizing
        and playing, while later tokens continue streaming in.
        """
        await self._cancel_current()

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _producer():
            buf = ""
            async for token in token_iter:
                if not isinstance(token, str):
                    continue
                buf += token
                # Emit every complete sentence as soon as we see one.
                while True:
                    match = _SENTENCE_RE.search(buf)
                    if not match:
                        break
                    end = match.end()
                    sentence = buf[:end].strip()
                    buf = buf[end:]
                    if sentence:
                        await queue.put(sentence)
            tail = buf.strip()
            if tail:
                await queue.put(tail)
            await queue.put(None)  # sentinel

        async def _consumer():
            while True:
                sentence = await queue.get()
                if sentence is None:
                    return
                await self._speak_chunks([sentence])

        # asyncio.gather() already returns a cancellable Future; wrapping
        # it in create_task() is a TypeError ("a coroutine was expected").
        self._play_task = asyncio.gather(_producer(), _consumer())
        try:
            await self._play_task
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Cancel any in-flight playback (e.g. on wake-word barge-in)."""
        await self._cancel_current()
        # sounddevice.stop() is sync and affects global stream state; call it
        # to flush any audio already queued in PortAudio's ring buffer.
        try:
            self._sd.stop()
        except Exception:
            pass

    # ---- internal ---------------------------------------------------------

    async def _cancel_current(self) -> None:
        task = self._play_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._play_task = None

    async def _speak_chunks(self, texts: list[str]) -> None:
        """Synthesize each text and stream its PCM into an output stream.

        Piper's ``synthesize_stream_raw`` yields int16 PCM bytes in small
        chunks as they're decoded; we pipe them straight to PortAudio so
        playback starts within ~100 ms of the call.
        """
        loop = asyncio.get_running_loop()
        if self._out_sr is None:
            self._resolve_output_rate()
        out_sr = self._out_sr
        resample = self._out_resample

        # Piper >=1.3 replaced synthesize_stream_raw with synthesize()
        # yielding AudioChunk objects. Each chunk already exposes a
        # numpy int16 array, so no frombuffer() step is needed.
        from piper.config import SynthesisConfig

        syn_cfg = SynthesisConfig(
            length_scale=1.0 / max(self._speed, 0.1),
        )

        def _blocking_synthesize_and_play():
            with self._sd.OutputStream(
                samplerate=out_sr,
                channels=1,
                dtype="int16",
                device=self._output_device,
            ) as stream:
                for text in texts:
                    for chunk in self._voice.synthesize(text, syn_config=syn_cfg):
                        samples = chunk.audio_int16_array
                        if resample is not None:
                            samples = resample(samples)
                        stream.write(samples)

        try:
            await loop.run_in_executor(None, _blocking_synthesize_and_play)
        except Exception:
            log.exception("piper playback failed")

    def _resolve_output_rate(self) -> None:
        """Pick an output sample rate the device accepts.

        Tries the voice's native rate first; on rejection, falls back to
        the device's default rate and builds a linear-interp resampler.
        Mirrors :meth:`SpeechToText._pick_sample_rate` for the output path.
        """
        try:
            dev = self._sd.query_devices(self._output_device, "output")
            native_default = int(dev["default_samplerate"])
        except Exception:
            native_default = 48000  # reasonable guess if query fails

        for sr in (self._sample_rate, native_default, 48000, 44100):
            try:
                self._sd.check_output_settings(
                    device=self._output_device,
                    channels=1, dtype="int16", samplerate=sr,
                )
            except Exception:
                continue

            if sr == self._sample_rate:
                self._out_sr = sr
                self._out_resample = None
                return

            # Linear-interp int16 resampler. Upsampling 22050 -> 44100/48000
            # is benign (no aliasing), and Piper's output is already
            # band-limited below ~10 kHz by the VITS decoder, so quality
            # stays indistinguishable from a proper polyphase filter.
            src_sr = self._sample_rate
            np_ = self._np
            ratio = sr / src_sr

            def _resample(samples, ratio=ratio, np_=np_):
                if samples.size == 0:
                    return samples
                out_n = int(round(samples.size * ratio))
                xp_in = np_.arange(samples.size, dtype=np_.float32)
                xp_out = np_.linspace(
                    0, samples.size - 1, out_n, dtype=np_.float32
                )
                return np_.interp(xp_out, xp_in, samples).astype(np_.int16)

            self._out_sr = sr
            self._out_resample = _resample
            log.info(
                "output device doesn't accept %d Hz; playing at %d Hz "
                "(upsampling in Python)",
                self._sample_rate, sr,
            )
            return

        # Last-ditch: let PortAudio try anyway; it'll raise a clear error.
        self._out_sr = self._sample_rate
        self._out_resample = None

    # ---- voice file management -------------------------------------------

    def _ensure_voice_files(self, voice_name: str) -> tuple[Path, Path]:
        """Return (model_path, config_path), downloading from HF if needed.

        Piper voice-name convention: ``<lang>_<region>-<speaker>-<quality>``
        e.g. ``en_US-amy-medium`` -> ``en/en_US/amy/medium/<voice>.onnx``.
        """
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = self.CACHE_DIR / f"{voice_name}.onnx"
        config_path = self.CACHE_DIR / f"{voice_name}.onnx.json"

        if model_path.exists() and config_path.exists():
            return model_path, config_path

        try:
            lang_region, speaker, quality = voice_name.split("-")
            lang = lang_region.split("_")[0]
        except ValueError as exc:
            raise ValueError(
                f"Voice name {voice_name!r} is not in the expected "
                "'<lang>_<region>-<speaker>-<quality>' form, e.g. "
                "'en_US-amy-medium' or 'it_IT-riccardo-x_low'."
            ) from exc

        log.info("downloading Piper voice %r from HuggingFace...", voice_name)
        base = f"{lang}/{lang_region}/{speaker}/{quality}"
        for relpath, local in (
            (f"{base}/{voice_name}.onnx", model_path),
            (f"{base}/{voice_name}.onnx.json", config_path),
        ):
            fetched = self._hf_hub_download(
                repo_id=self.HF_REPO,
                filename=relpath,
                cache_dir=str(self.CACHE_DIR / "_hf_cache"),
            )
            # hf_hub_download returns a symlink/path into its internal cache
            # layout; mirror the file to our flat cache for a stable path.
            Path(local).write_bytes(Path(fetched).read_bytes())

        return model_path, config_path

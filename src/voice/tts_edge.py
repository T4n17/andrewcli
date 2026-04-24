"""Edge TTS backend - Microsoft Azure Neural TTS via the free Edge endpoint.

Provides the same public API as :class:`src.voice.tts.TextToSpeech`
(``speak``, ``speak_stream``, ``stop``) so ``VoiceSession`` can swap
backends by config alone.

Why this backend
----------------
Piper's 22050 Hz VITS-small models are fast but robotic. Azure's
Neural TTS is currently state-of-the-art commercial quality and
Microsoft exposes it for free via the Edge browser's "read aloud"
backend, which ``edge-tts`` speaks to directly. The free endpoint
returns MP3 over WebSocket; we decode it with ``miniaudio`` (single
pip wheel, no ffmpeg dependency) and play via the same sounddevice
pipeline Piper used, including the output sample-rate negotiator.

Tradeoffs
---------
* Requires an internet connection.
* ~300-500 ms first-audio latency (network round-trip + Azure warmup).
  For longer responses this is hidden by our sentence-level streaming.
* Technically the free Edge endpoint is meant for the Edge browser;
  fine for personal use, consider Azure Cognitive Services for
  anything commercial.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from typing import AsyncIterable

log = logging.getLogger(__name__)

# Same sentence splitter as the Piper backend so behavior is identical.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class EdgeTTS:
    """Streaming TTS using Microsoft Edge's free Neural TTS endpoint."""

    # Well-known Italian + English neural voices. Users pick by name via
    # ``tts_voice`` in ``config.yaml``; any voice listed by
    # ``edge-tts --list-voices`` works.
    DEFAULT_VOICE = "it-IT-IsabellaNeural"

    def __init__(
        self,
        *,
        voice: str | None = None,
        output_device: int | str | None = None,
        speed: float = 1.0,
    ):
        try:
            import numpy as np
            import sounddevice as sd
            import edge_tts
            import miniaudio
        except ImportError as exc:
            raise ImportError(
                "EdgeTTS requires extra packages. Install with: "
                "pip install edge-tts miniaudio sounddevice"
            ) from exc

        self._np = np
        self._sd = sd
        self._edge_tts = edge_tts
        self._miniaudio = miniaudio

        self.voice_name = voice or self.DEFAULT_VOICE
        self._speed = speed
        self._output_device = output_device

        # Edge-TTS `rate` is a percentage string like "+10%" or "-20%".
        # length_scale=1.0/speed in Piper maps inversely to rate here.
        pct = int(round((self._speed - 1.0) * 100))
        self._rate = f"{pct:+d}%" if pct != 0 else "+0%"

        log.info("EdgeTTS ready; voice=%r rate=%s", self.voice_name, self._rate)

        # Track the current playback task so stop() can cancel it.
        self._play_task: asyncio.Future | None = None

        # Output sample rate is resolved lazily on first playback so
        # we can query the device. Azure returns 24 kHz mono by default.
        self._out_sr: int | None = None
        self._out_resample = None
        self._edge_sr = 24000  # audio-24khz-48kbitrate-mono-mp3

    # ---- public API -------------------------------------------------------

    async def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        await self._cancel_current()
        self._play_task = asyncio.ensure_future(self._speak_chunks([text]))
        try:
            await self._play_task
        except asyncio.CancelledError:
            pass

    async def speak_stream(self, token_iter: AsyncIterable[str]) -> None:
        """Mirror of PiperTTS.speak_stream: sentence-level streaming."""
        await self._cancel_current()

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _producer():
            buf = ""
            async for token in token_iter:
                if not isinstance(token, str):
                    continue
                buf += token
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
            await queue.put(None)

        async def _consumer():
            while True:
                sentence = await queue.get()
                if sentence is None:
                    return
                await self._speak_chunks([sentence])

        self._play_task = asyncio.gather(_producer(), _consumer())
        try:
            await self._play_task
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        await self._cancel_current()
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
        """Synthesize each sentence via Edge, decode MP3, play PCM."""
        loop = asyncio.get_running_loop()
        if self._out_sr is None:
            self._resolve_output_rate()
        out_sr = self._out_sr
        resample = self._out_resample

        for text in texts:
            t0 = time.monotonic()
            mp3 = await self._synthesize_mp3(text)
            synth_ms = int((time.monotonic() - t0) * 1000)

            # Decode MP3 -> int16 mono PCM at the Azure native 24 kHz.
            pcm = await loop.run_in_executor(
                None, self._decode_mp3_to_pcm, mp3,
            )
            if resample is not None:
                pcm = resample(pcm)

            log.debug(
                "edge-tts synth=%dms mp3=%dKB pcm=%d samples",
                synth_ms, len(mp3) // 1024, pcm.size,
            )

            # Play blocking in a worker thread so we don't stall the loop.
            def _play(pcm=pcm, out_sr=out_sr):
                with self._sd.OutputStream(
                    samplerate=out_sr,
                    channels=1,
                    dtype="int16",
                    device=self._output_device,
                ) as stream:
                    stream.write(pcm)

            try:
                await loop.run_in_executor(None, _play)
            except Exception:
                log.exception("edge-tts playback failed")

    async def _synthesize_mp3(self, text: str) -> bytes:
        """Stream Edge TTS for one sentence, collect MP3 bytes."""
        communicate = self._edge_tts.Communicate(
            text, voice=self.voice_name, rate=self._rate,
        )
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    def _decode_mp3_to_pcm(self, mp3_bytes: bytes):
        """Decode MP3 into an int16 mono numpy array."""
        ma = self._miniaudio
        decoded = ma.decode(
            mp3_bytes,
            output_format=ma.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=self._edge_sr,
        )
        # ``decoded.samples`` is an ``array.array('h', ...)``.
        return self._np.asarray(decoded.samples, dtype=self._np.int16)

    def _resolve_output_rate(self) -> None:
        """Pick an output sample rate the device accepts.

        Copy of :meth:`src.voice.tts.TextToSpeech._resolve_output_rate`
        parameterized on Azure's 24 kHz native rate instead of Piper's
        22.05 kHz. Kept duplicated here (rather than abstracted into a
        shared helper) because the two backends are intentionally
        independent and may diverge as they add format-specific knobs.
        """
        src_sr = self._edge_sr
        try:
            dev = self._sd.query_devices(self._output_device, "output")
            native_default = int(dev["default_samplerate"])
        except Exception:
            native_default = 48000

        for sr in (src_sr, native_default, 48000, 44100):
            try:
                self._sd.check_output_settings(
                    device=self._output_device,
                    channels=1, dtype="int16", samplerate=sr,
                )
            except Exception:
                continue

            if sr == src_sr:
                self._out_sr = sr
                self._out_resample = None
                return

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
                "(upsampling in Python)", src_sr, sr,
            )
            return

        self._out_sr = src_sr
        self._out_resample = None

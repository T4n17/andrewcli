"""Speech-to-Text with wake-word activation.

All-local, CPU-friendly, ONNX-backed stack:

* ``openwakeword`` - tiny ONNX wake-word detector. Ships pre-trained
  models for ``hey_jarvis``, ``alexa``, ``hey_mycroft``, ``hey_rhasspy``.
* Energy-based VAD - detects end-of-utterance by RMS drop. No extra
  model, a single tunable threshold; good enough for quiet rooms.
* ``faster-whisper`` - CTranslate2 Whisper. Multilingual (99 langs,
  auto-detect), ``int8`` quantization on CPU.
* ``sounddevice`` - PortAudio wrapper for mic capture.

Flow (``listen_once``):

    1. Open a 16 kHz mono InputStream, stream frames into an asyncio queue.
    2. Score every frame with ``openwakeword``; if any model > threshold
       -> wake detected, proceed.
    3. Record frames until either:
         - ``silence_timeout_ms`` of consecutive silent frames, or
         - ``max_utterance_ms`` reached (safety cap).
    4. Hand the concatenated audio to ``faster-whisper`` (in an executor,
       since transcribe() is blocking).
    5. Return the transcript string.

Usage::

    stt = SpeechToText(language="auto")
    async for text in stt.listen_forever():
        print("you said:", text)

Optional dep: if ``faster-whisper``, ``openwakeword``, or ``sounddevice``
is missing, :class:`SpeechToText` raises ``ImportError`` at construction;
callers should catch it and disable voice mode.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncGenerator

log = logging.getLogger(__name__)


class SpeechToText:
    """Wake-word-activated speech-to-text using openwakeword + faster-whisper."""

    # ---- audio / framing --------------------------------------------------

    SAMPLE_RATE = 16_000          # openwakeword + whisper both want 16 kHz
    FRAME_DURATION_MS = 80        # 1280 samples/frame; matches openwakeword expected chunk
    FRAME_SIZE = SAMPLE_RATE * FRAME_DURATION_MS // 1000

    # ---- behavior ---------------------------------------------------------

    # Prefer our bundled custom "hey_andrew" model (shipped as
    # src/voice/hey_andrew.onnx). _load_wake_model resolves bare
    # names against both ``BUILTIN_WAKE_WORDS`` and the bundled
    # directory, so this default is picked up transparently.
    DEFAULT_WAKE_WORD = "hey_andrew"
    DEFAULT_WAKE_THRESHOLD = 0.5      # openwakeword score > 0.5 -> detected
    DEFAULT_WHISPER_MODEL = "small"   # tiny/base/small/medium/large-v3

    # Built-in openwakeword models that ship with the package. Anything
    # outside this set is treated as either a filesystem path to a custom
    # ``.tflite`` / ``.onnx`` model, or (if neither) an unknown name that
    # falls back to ``DEFAULT_WAKE_WORD`` with a clear warning.
    BUILTIN_WAKE_WORDS = frozenset({
        "alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy",
        "timer", "weather",
    })

    # Energy-based VAD thresholds (RMS of a frame normalized to [-1, 1]).
    # These are empirically tuned; expose via constructor if needed.
    # Absolute RMS floor. A frame below this is always silence. Kept
    # low so quiet-mic setups still work; noisy mics rely on the
    # adaptive peak-relative cutoff in :meth:`_record_until_silence`.
    DEFAULT_SILENCE_RMS = 0.008
    # Trailing silence duration that marks end-of-utterance. Shorter
    # feels snappier; too short cuts people off mid-pause. 800 ms is a
    # common sweet spot for voice assistants.
    DEFAULT_SILENCE_TIMEOUT_MS = 800
    # Safety cap per utterance. Hot mics where ambient noise > 35% of
    # speech peak never hit the trailing-silence condition; the cap
    # guarantees progress. 8 s covers the vast majority of spoken
    # commands while still feeling responsive when the cap fires.
    DEFAULT_MAX_UTTERANCE_MS = 8_000

    # ---- init -------------------------------------------------------------

    def __init__(
        self,
        *,
        whisper_model: str | None = None,
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        language: str | None = None,      # None/"auto" -> auto-detect
        wake_word: str | None = None,
        wake_threshold: float | None = None,
        silence_rms: float | None = None,
        silence_timeout_ms: int | None = None,
        max_utterance_ms: int | None = None,
        input_device: int | str | None = None,  # sounddevice device id
    ):
        # Lazy-import optional deps so the module file stays importable
        # even when voice extras aren't installed.
        try:
            import numpy as np
            import sounddevice as sd
            from faster_whisper import WhisperModel
            from openwakeword.model import Model as WakeModel
        except ImportError as exc:
            raise ImportError(
                "SpeechToText requires the voice extras. Install with: "
                "pip install faster-whisper openwakeword sounddevice"
            ) from exc

        self._np = np
        self._sd = sd

        # Load models up front so the first listen() is fast. The first
        # run downloads ~500 MB (for the default `small` model) from
        # HuggingFace into ~/.cache/huggingface/, which can take a while;
        # hf_hub prints its own tqdm progress bar to stderr.
        whisper_name = whisper_model or self.DEFAULT_WHISPER_MODEL
        log.info(
            "loading faster-whisper %r (downloads model on first run, "
            "cached to ~/.cache/huggingface/)...",
            whisper_name,
        )
        t0 = time.monotonic()
        # cpu_threads lets CTranslate2 parallelize matmuls; default in
        # faster-whisper is 4, which leaves most modern laptops idle.
        # num_workers stays at 1: we only transcribe one utterance at a
        # time so extra workers would just duplicate model copies in RAM.
        self._whisper = WhisperModel(
            whisper_name,
            device=whisper_device,
            compute_type=whisper_compute_type,
            cpu_threads=os.cpu_count() or 0,
        )
        log.info(
            "loaded Whisper %r on %s/%s in %.2fs",
            whisper_model or self.DEFAULT_WHISPER_MODEL,
            whisper_device, whisper_compute_type, time.monotonic() - t0,
        )

        self.wake_word = wake_word or self.DEFAULT_WAKE_WORD
        t0 = time.monotonic()
        self._wake_model = self._load_wake_model(WakeModel, self.wake_word)
        log.info(
            "loaded wake-word model %r in %.2fs",
            self.wake_word, time.monotonic() - t0,
        )

        # Behavior knobs
        self._language = (
            None if language in (None, "", "auto") else language
        )
        self._wake_threshold = (
            wake_threshold if wake_threshold is not None
            else self.DEFAULT_WAKE_THRESHOLD
        )
        self._silence_rms = (
            silence_rms if silence_rms is not None
            else self.DEFAULT_SILENCE_RMS
        )
        self._silence_frames_limit = (
            (silence_timeout_ms or self.DEFAULT_SILENCE_TIMEOUT_MS)
            // self.FRAME_DURATION_MS
        )
        self._max_frames = (
            (max_utterance_ms or self.DEFAULT_MAX_UTTERANCE_MS)
            // self.FRAME_DURATION_MS
        )
        self._input_device = input_device

        self._stop = False

    # ---- public API -------------------------------------------------------

    def stop(self) -> None:
        """Signal listen_forever() to exit after the current iteration."""
        self._stop = True

    async def listen_forever(self, on_wake=None) -> AsyncGenerator[str, None]:
        """Yield a transcript each time the wake word fires and speech ends.

        See :meth:`listen_once` for the ``on_wake`` contract.
        """
        self._stop = False
        while not self._stop:
            text = await self.listen_once(on_wake=on_wake)
            if text:
                yield text

    async def listen_once(self, on_wake=None) -> str:
        """Wait for the wake word, record until silence, return transcript.

        Returns an empty string if stop() is called before transcription or
        if Whisper found no speech.

        ``on_wake`` is an optional zero-arg callable fired the moment the
        wake word crosses the threshold, before recording starts. Use it
        to give the user visual or audible feedback (terminal cue, panel
        indicator, chime). May be sync or async; exceptions are logged
        and swallowed so a broken callback doesn't break the session.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        # Pick a sample rate the device actually supports, then resample to
        # 16 kHz ourselves. ALSA hw: devices often reject 16 kHz outright
        # (``PaErrorCode -9997: Invalid sample rate``) while happily doing
        # 44.1/48 kHz; pulse/pipewire does resample but isn't always the
        # selected device. This path works regardless of backend.
        stream_sr, native_block, resample = self._pick_sample_rate()

        def _audio_callback(indata, frames, time_info, status):
            # Called by PortAudio in a non-asyncio thread.
            if status:
                log.debug("sounddevice status: %s", status)
            block = indata[:, 0]
            if resample is not None:
                block = resample(block)
            loop.call_soon_threadsafe(queue.put_nowait, block.copy())

        with self._sd.InputStream(
            samplerate=stream_sr,
            channels=1,
            dtype="float32",
            blocksize=native_block,
            callback=_audio_callback,
            device=self._input_device,
        ):
            # Reset openwakeword buffers so an old trigger doesn't carry over.
            self._wake_model.reset()

            if not await self._wait_for_wake(queue):
                return ""
            log.info("wake word %r detected, recording...", self.wake_word)

            # Fire the user-supplied feedback hook (terminal cue, chime...)
            # before recording so the user knows their "hey X" was heard.
            if on_wake is not None:
                try:
                    result = on_wake()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    log.exception("on_wake callback failed")

            t0 = time.monotonic()
            audio = await self._record_until_silence(queue)
            rec_dur = time.monotonic() - t0
            log.info(
                "recording finished after %.2fs (audio=%s samples)",
                rec_dur, "None" if audio is None else str(audio.size),
            )

        if audio is None or audio.size == 0:
            return ""

        # Whisper transcribe() is blocking; run off the event loop.
        log.info("submitting %d samples (%.1fs) to Whisper...",
                 audio.size, audio.size / self.SAMPLE_RATE)
        t0 = time.monotonic()
        text = await loop.run_in_executor(None, self._transcribe, audio)
        log.info("whisper returned after %.2fs: %r",
                 time.monotonic() - t0, text[:80])
        return text

    # ---- internal: phases -------------------------------------------------

    async def _wait_for_wake(self, queue: asyncio.Queue) -> bool:
        """Consume frames until a wake-word score crosses the threshold.

        Also learns the ambient noise floor (rolling max RMS over the
        last ~2 s) which :meth:`_record_until_silence` uses as a
        baseline for adaptive end-of-utterance detection. This is
        critical on noisy mics where the ambient RMS is orders of
        magnitude above the default silence threshold.
        """
        HEARTBEAT_FRAMES = max(1, 2000 // self.FRAME_DURATION_MS)
        frame_count = 0
        max_score = 0.0
        rolling_max_rms = 0.0

        while not self._stop:
            frame = await queue.get()
            # openwakeword expects int16 PCM 16 kHz; we captured float32 in [-1,1].
            frame_i16 = (frame * 32767).astype(self._np.int16)
            scores = self._wake_model.predict(frame_i16)

            score = max(scores.values()) if scores else 0.0
            rms = float(self._np.sqrt(self._np.mean(frame ** 2)))
            max_score = max(max_score, score)
            rolling_max_rms = max(rolling_max_rms, rms)

            if score > self._wake_threshold:
                log.debug("wake score %.2f > %.2f threshold",
                          score, self._wake_threshold)
                return True

            frame_count += 1
            if frame_count % HEARTBEAT_FRAMES == 0:
                log.debug(
                    "listening... max wake score=%.2f, max mic rms=%.4f",
                    max_score, rolling_max_rms,
                )
                max_score = 0.0
                rolling_max_rms = 0.0
        return False

    async def _record_until_silence(self, queue: asyncio.Queue):
        """Collect frames until trailing silence, or until the hard cap.

        Design:

        1. **Minimum recording floor** (``MIN_RECORDING_MS``, 1500 ms).
           Silence detection is disabled for the first 1.5 s
           unconditionally. This is enough for the user to finish
           their wake word, pause briefly, and start their real
           request. Without this floor, the peak-relative silence
           check can terminate the recording within 1 s on quiet
           post-wake lulls - the exact failure mode observed with
           earlier revisions of this method.
        2. **Peak-relative trailing silence.** After the floor, a
           frame is "silent" iff ``rms < max(silence_rms, 0.35 *
           peak_rms)``. ``silence_timeout_ms`` of consecutive silent
           frames ends the recording.
        3. **Hard cap** (``max_utterance_ms``, 8 s). On hot mics
           where ambient RMS is a large fraction of speech peak,
           trailing silence never triggers; the cap guarantees
           bounded latency. Whisper's bundled Silero VAD
           (``vad_filter=True``) trims the leading/trailing noise
           from the handed-off buffer.

        This approach makes no assumption about absolute speech
        level. It works equally well for quiet speakers, whispered
        requests, hot mics with auto-gain, and quiet-room mics with
        near-zero ambient.
        """
        MIN_RECORDING_MS = 1500
        min_recording_frames = MIN_RECORDING_MS // self.FRAME_DURATION_MS

        frames = []
        silent = 0
        peak_rms = 0.0

        while len(frames) < self._max_frames:
            frame = await queue.get()
            frames.append(frame)

            rms = float(self._np.sqrt(self._np.mean(frame ** 2)))
            if rms > peak_rms:
                peak_rms = rms

            # Floor: always capture at least MIN_RECORDING_MS before
            # any silence check. No onset gate - we trust that the
            # wake word already filtered out the misfire case.
            if len(frames) < min_recording_frames:
                continue

            rel_cut = max(self._silence_rms, 0.35 * peak_rms)
            if rms < rel_cut:
                silent += 1
                if silent >= self._silence_frames_limit:
                    break
            else:
                silent = 0

            if self._stop:
                break

        if not frames:
            return None
        return self._np.concatenate(frames)

    # ---- sample-rate negotiation ------------------------------------------

    def _pick_sample_rate(self):
        """Return ``(stream_sr, native_block, resample_fn)``.

        * ``stream_sr`` is the rate we'll ask PortAudio for.
        * ``native_block`` is how many native-rate samples correspond to
          one downstream 16 kHz frame (so one callback produces one
          openwakeword frame).
        * ``resample_fn`` is ``None`` when ``stream_sr == 16000``, else
          a callable that maps ``native_block`` samples to
          :attr:`FRAME_SIZE` 16 kHz samples.

        Strategy: try the device's advertised default rate first (always
        accepted by that backend), then common fallbacks. We resample to
        16 kHz in Python because wake-word and Whisper both require it
        and many ALSA hw devices refuse 16 kHz outright.
        """
        dev = self._sd.query_devices(self._input_device, "input")
        native_default = int(dev["default_samplerate"])

        candidates = []
        for sr in (self.SAMPLE_RATE, native_default, 48000, 44100, 22050):
            if sr and sr not in candidates:
                candidates.append(sr)

        last_err = None
        for sr in candidates:
            try:
                self._sd.check_input_settings(
                    device=self._input_device,
                    channels=1, dtype="float32", samplerate=sr,
                )
            except Exception as exc:  # PortAudioError or ValueError
                last_err = exc
                continue

            if sr == self.SAMPLE_RATE:
                return sr, self.FRAME_SIZE, None

            # Build a simple linear-interpolation resampler. It's good
            # enough for speech features (wake word / Whisper both
            # lowpass via mel extraction). Closures keep the arrays hot.
            native_block = int(round(self.FRAME_SIZE * sr / self.SAMPLE_RATE))
            xp_in = self._np.linspace(0.0, 1.0, native_block, endpoint=False)
            xp_out = self._np.linspace(0.0, 1.0, self.FRAME_SIZE, endpoint=False)
            np_ = self._np

            def _resample(block, xp_in=xp_in, xp_out=xp_out, np_=np_):
                return np_.interp(xp_out, xp_in, block).astype(np_.float32)

            log.info(
                "device sr %d Hz not directly supported at 16 kHz; "
                "capturing at %d Hz and resampling to %d Hz",
                native_default, sr, self.SAMPLE_RATE,
            )
            return sr, native_block, _resample

        raise RuntimeError(
            f"No usable input sample rate on device {self._input_device!r}. "
            f"Tried {candidates}. Last error: {last_err}"
        )

    # ---- wake-word loading -----------------------------------------------

    @classmethod
    def _load_wake_model(cls, WakeModel, wake_word: str):
        """Load a built-in or custom wake-word model with a safe fallback.

        Accepts any of:
            * a built-in name listed in ``BUILTIN_WAKE_WORDS``
            * a filesystem path to a ``.tflite`` or ``.onnx`` model
              (trained offline via the openwakeword pipeline)

        Anything else - e.g. a hopeful ``"hey_andrew"`` when no such
        model exists - logs a warning and falls back to
        :attr:`DEFAULT_WAKE_WORD` so the app still runs.

        On first use, the openwakeword pip package does NOT ship the
        actual ``.tflite`` model weights - they must be fetched from
        GitHub releases via ``openwakeword.utils.download_models``.
        We call it on demand so users don't have to remember.
        """
        from pathlib import Path

        # Force ONNX backend. Reason: openwakeword's default is tflite,
        # which relies on tflite_runtime compiled against NumPy 1.x and
        # crashes hard ("_ARRAY_API not found") on NumPy >= 2. The ONNX
        # path goes through onnxruntime (already pulled in by fastembed
        # / faster-whisper) and is fully NumPy-2-compatible.
        backend = "onnx"

        if wake_word in cls.BUILTIN_WAKE_WORDS:
            cls._ensure_builtin_models_downloaded(wake_word)
            return WakeModel(
                wakeword_models=[wake_word], inference_framework=backend,
            )

        # Bundled custom models shipped inside the repo (src/voice/<name>.onnx
        # or .tflite). Lets users reference them by bare name in
        # config.yaml without caring where this package lives on disk.
        # Prefer .onnx since we force the ONNX inference backend; the
        # .tflite is kept alongside so the user's training artefacts
        # stay complete, but loading it would silently switch backends.
        bundled_dir = Path(__file__).parent
        for ext in (".onnx", ".tflite"):
            bundled = bundled_dir / f"{wake_word}{ext}"
            if bundled.is_file():
                log.info("loading bundled wake-word model %s", bundled)
                return WakeModel(
                    wakeword_models=[str(bundled)],
                    inference_framework=backend,
                )

        # Custom model path
        p = Path(wake_word).expanduser()
        if p.is_file() and p.suffix.lower() in (".tflite", ".onnx"):
            log.info("loading custom wake-word model from %s", p)
            return WakeModel(
                wakeword_models=[str(p)], inference_framework=backend,
            )

        log.warning(
            "wake word %r is not a built-in (%s) and not a valid model "
            "file; falling back to %r. To train a custom wake word, see "
            "https://github.com/dscripka/openWakeWord#training-new-models "
            "and point `voice.wake_word` at the resulting .tflite/.onnx file.",
            wake_word,
            ", ".join(sorted(cls.BUILTIN_WAKE_WORDS)),
            cls.DEFAULT_WAKE_WORD,
        )
        cls._ensure_builtin_models_downloaded(cls.DEFAULT_WAKE_WORD)
        return WakeModel(
            wakeword_models=[cls.DEFAULT_WAKE_WORD],
            inference_framework=backend,
        )

    @staticmethod
    def _ensure_builtin_models_downloaded(wake_word: str) -> None:
        """Download openwakeword's built-in .tflite files if missing.

        The pip package ships Python code only; weights live in the
        ``openwakeword/resources/models/`` directory and are fetched
        lazily by ``openwakeword.utils.download_models``. The common
        symptom of skipping this step is the cryptic
        ``ValueError: Could not open '.../hey_jarvis_v0.1.tflite'``.
        """
        import os
        from openwakeword import utils as ow_utils

        # resources/models sits next to the openwakeword package
        import openwakeword
        models_dir = os.path.join(
            os.path.dirname(openwakeword.__file__), "resources", "models"
        )
        # Any .tflite file present means a previous run already downloaded
        # everything; the package ships them as a bundle so an all-or-nothing
        # check is good enough.
        if os.path.isdir(models_dir) and any(
            f.endswith(".tflite") for f in os.listdir(models_dir)
        ):
            return

        log.info(
            "openwakeword models missing; downloading built-in wake-word "
            "weights (one-time, ~15 MB) for %r ...", wake_word,
        )
        # download_models() with no args fetches the full built-in bundle,
        # which includes all BUILTIN_WAKE_WORDS; that's fine because it's
        # small and avoids future calls.
        ow_utils.download_models()

    # ---- internal: transcribe --------------------------------------------

    def _transcribe(self, audio) -> str:
        """Run Whisper on a float32 mono 16 kHz array. Blocking."""
        t0 = time.monotonic()
        segments, info = self._whisper.transcribe(
            audio,
            language=self._language,
            vad_filter=True,          # drop silent/noise segments
            beam_size=1,              # faster; quality loss is minor for short clips
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text for seg in segments).strip()
        log.debug(
            "whisper transcribe took %.2fs lang=%s conf=%.2f text=%r",
            time.monotonic() - t0,
            info.language, info.language_probability, text[:80],
        )
        return text

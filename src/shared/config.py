import yaml

from src.shared.paths import CONFIG_FILE


class Config:
    """Singleton: the config file is read once per process."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"config.yaml not found at {CONFIG_FILE}") from exc

        self.domain = config.get("domain", "general")
        self.execute_bash_automatically = config.get("execute_bash_automatically", False)
        self.tray_width_compact = config.get("tray_width_compact", 600)
        self.tray_height_compact = config.get("tray_height_compact", 80)
        self.tray_width_expanded = config.get("tray_width_expanded", 900)
        self.tray_height_expanded = config.get("tray_height_expanded", 600)
        self.tray_platform = config.get("tray_platform", "")
        self.tray_position = config.get("tray_position", "top-right")
        self.tray_opacity = self._parse_opacity(config.get("tray_opacity", "100%"))

        # Routing backend: "embed" uses fastembed-based cosine similarity
        # (fast, local, deterministic), "llm" uses the main LLM as a
        # classifier (slow but can reason about ambiguous intent).
        self.router_backend = config.get("router_backend", "embed")
        self.router_threshold = float(config.get("router_threshold", 0.40))

        # Voice (src/voice/). All keys are optional; heavy deps are only
        # imported when SpeechToText / TextToSpeech are actually
        # constructed, so leaving voice disabled costs nothing.
        voice = config.get("voice", {}) or {}
        self.voice_enabled = bool(voice.get("enabled", False))
        self.voice_wake_word = voice.get("wake_word", "hey_jarvis")
        self.voice_wake_threshold = float(voice.get("wake_threshold", 0.5))
        self.voice_stt_model = voice.get("stt_model", "small")
        self.voice_stt_language = voice.get("stt_language", "auto")
        self.voice_tts_engine = voice.get("tts_engine", "piper")  # "piper" | "edge"
        self.voice_tts_voice = voice.get("tts_voice", "en_US-amy-medium")
        self.voice_tts_speed = float(voice.get("tts_speed", 1.0))
        self.voice_input_device = voice.get("input_device")   # None = default
        self.voice_output_device = voice.get("output_device")  # None = default

    @staticmethod
    def _parse_opacity(value):
        s = str(value).strip().rstrip("%")
        try:
            return max(0.0, min(1.0, float(s) / 100))
        except ValueError:
            return 1.0

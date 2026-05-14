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

        # Optional global LLM defaults. A per-domain ``config.yaml`` can
        # override these on a key-by-key basis; if neither layer sets a
        # value the Domain class falls back to its own built-in default.
        if "api_base_url" in config:
            self.api_base_url = config.get("api_base_url")
        if "model" in config:
            self.model = config.get("model")
        if "routing_enabled" in config:
            self.routing_enabled = bool(config.get("routing_enabled"))
        # Rolling memory (src/core/memory.py). Disabling switches off
        # both the persistent ``memory.json`` summary and the in-prompt
        # ``<memory>`` block; messages are still trimmed every turn so
        # the LLM context stays bounded.
        memory = config.get("memory", {}) or {}
        self.memory_enabled = bool(memory.get("enabled", True))
        self.memory_min_summary_chars = int(memory.get("min_summary_chars", 200))

        # FastAPI bridge (src/core/server.py). When enabled, the CLI
        # and tray auto-start the HTTP server in a background thread so
        # external clients can submit prompts via /chat. The explicit
        # ``andrewcli --server`` mode always starts regardless of this
        # flag — it's the user's intent, not a side effect.
        server = config.get("server", {}) or {}
        self.server_enabled = bool(server.get("enabled", True))

        self.tray_width_compact = config.get("tray_width_compact", 600)
        self.tray_height_compact = config.get("tray_height_compact", 80)
        self.tray_width_expanded = config.get("tray_width_expanded", 900)
        self.tray_height_expanded = config.get("tray_height_expanded", 600)
        self.tray_platform = config.get("tray_platform", "")
        self.tray_position = config.get("tray_position", "top-right")
        self.tray_opacity = self._parse_opacity(config.get("tray_opacity", "100%"))

    @staticmethod
    def _parse_opacity(value):
        s = str(value).strip().rstrip("%")
        try:
            return max(0.0, min(1.0, float(s) / 100))
        except ValueError:
            return 1.0

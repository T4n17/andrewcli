import yaml

class Config:
    def __init__(self):
        self._load()
        
    def _load(self):
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
                self.domain = config.get("domain", "general")
                self.execute_bash_automatically = config.get("execute_bash_automatically", False)
                self.tray_width_compact = config.get("tray_width_compact", 600)
                self.tray_height_compact = config.get("tray_height_compact", 80)
                self.tray_width_expanded = config.get("tray_width_expanded", 900)
                self.tray_height_expanded = config.get("tray_height_expanded", 600)
                self.tray_platform = config.get("tray_platform", "")
                self.tray_position = config.get("tray_position", "top-right")
                self.tray_opacity = self._parse_opacity(config.get("tray_opacity", "100%"))
        except FileNotFoundError:
            raise FileNotFoundError("config.yaml not found")

    @staticmethod
    def _parse_opacity(value):
        s = str(value).strip().rstrip("%")
        try:
            return max(0.0, min(1.0, float(s) / 100))
        except ValueError:
            return 1.0

import os
import yaml

from src.shared.paths import CONFIG_FILE


def init():
    try:
        with open(CONFIG_FILE) as f:
            platform = (yaml.safe_load(f) or {}).get("tray_platform", "")
        if platform:
            os.environ["QT_QPA_PLATFORM"] = platform
    except FileNotFoundError:
        pass

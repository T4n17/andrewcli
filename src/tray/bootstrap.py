import os
import yaml


def init():
    try:
        with open("config.yaml") as f:
            platform = yaml.safe_load(f).get("tray_platform", "")
        if platform:
            os.environ["QT_QPA_PLATFORM"] = platform
    except FileNotFoundError:
        pass

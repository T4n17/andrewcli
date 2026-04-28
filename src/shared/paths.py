"""Centralized filesystem paths.

All modules resolve paths from PROJECT_ROOT instead of relying on the
current working directory, so the package works when launched from any
directory (systemd, scripts, tray via subprocess, etc.).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
SKILLS_DIR = PROJECT_ROOT / "src" / "skills" / "skills_files"
DOMAINS_DIR = PROJECT_ROOT / "src" / "domains"
EVENTS_DIR  = PROJECT_ROOT / "src" / "events"
DATA_DIR = Path.home() / ".andrewcli" / "data"

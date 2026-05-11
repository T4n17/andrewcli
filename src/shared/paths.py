"""Centralized filesystem paths.

Two anchors are captured at import time so every module resolves paths
against fixed roots, regardless of the process' later cwd:

* ``PROJECT_ROOT`` — the AndrewCLI installation (where ``config.yaml``
  and the ``src/`` package live). Used for code/data shipped with the
  package: domains, events, the config file itself.
* ``LAUNCH_DIR`` — the directory the user was in when they ran
  ``andrewcli``. All *per-project* runtime state (memory, tray log,
  event state files) lives under ``LAUNCH_DIR / ".andrewcli"`` — the
  same pattern Claude Code and OpenCode use, so different projects
  keep independent conversation histories.

Subprocesses (notably the detached tray) honor the parent's launch
directory via the ``ANDREW_LAUNCH_DIR`` environment variable so a
background daemon can never accidentally drift to a different project
root.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
DOMAINS_DIR = PROJECT_ROOT / "domains"
EVENTS_DIR  = PROJECT_ROOT / "events"

# Resolved once, at first import. The ``ANDREW_LAUNCH_DIR`` override
# exists so subprocesses (tray, event runners) inherit the original
# launch directory even after they start — critical for the tray,
# which is spawned detached via subprocess.Popen.
LAUNCH_DIR = Path(os.environ.get("ANDREW_LAUNCH_DIR") or Path.cwd()).resolve()

# Per-project data directory (memory, tray log, event state). Lives
# alongside the user's project, not in $HOME.
DATA_DIR = LAUNCH_DIR / ".andrewcli" / "data"

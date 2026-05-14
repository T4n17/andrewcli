"""Centralized filesystem paths.

Three anchors are captured at import time so every module resolves paths
against fixed roots, regardless of the process' later cwd:

* ``PROJECT_ROOT`` — the AndrewCLI installation (where the ``src/``
  package lives). Used to locate code shipped with the package.
* ``CONFIG_DIR`` — the user's runtime configuration directory at
  ``~/.config/andrewcli/``. Hosts the global ``config.yaml`` plus the
  user-customizable ``domains/`` and ``events/`` trees. Created and
  seeded from the bundled defaults on first import.
* ``LAUNCH_DIR`` — the directory the user was in when they ran
  ``andrewcli``. All *per-project* runtime state (memory, tray log,
  event state files) lives under ``LAUNCH_DIR / ".andrewcli"`` — the
  same pattern Claude Code and OpenCode use, so different projects
  keep independent conversation histories.

Subprocesses (notably the detached tray) honor the parent's launch
directory via the ``ANDREW_LAUNCH_DIR`` environment variable so a
background daemon can never accidentally drift to a different project
root.

To make ``import domains.<name>`` and ``import events.<name>`` work
against the user's runtime directory, ``CONFIG_DIR`` is inserted at
the front of ``sys.path`` at import time. The bundled defaults (under
``src/defaults/``) are *not* on ``sys.path``; they are only used as
the source of the first-run seeding copy.
"""
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Shipped defaults — the read-only template used to seed CONFIG_DIR on
# first run. Lives inside the ``src`` package so it travels with any
# pip install (regular or editable).
DEFAULTS_DIR = Path(__file__).resolve().parent.parent / "defaults"

# User-facing runtime configuration directory. Anything in here can be
# freely edited; AndrewCLI never overwrites existing files once the
# initial seeding has happened.
CONFIG_DIR  = Path.home() / ".config" / "andrewcli"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
DOMAINS_DIR = CONFIG_DIR / "domains"
EVENTS_DIR  = CONFIG_DIR / "events"

# Resolved once, at first import. The ``ANDREW_LAUNCH_DIR`` override
# exists so subprocesses (tray, event runners) inherit the original
# launch directory even after they start — critical for the tray,
# which is spawned detached via subprocess.Popen.
LAUNCH_DIR = Path(os.environ.get("ANDREW_LAUNCH_DIR") or Path.cwd()).resolve()

# Per-project data directory (memory, tray log, event state). Lives
# alongside the user's project, not in $HOME.
DATA_DIR = LAUNCH_DIR / ".andrewcli" / "data"


def _seed_user_config() -> None:
    """Copy bundled defaults into ``CONFIG_DIR`` if not already present.

    Each top-level item (``config.yaml``, ``domains/``, ``events/``) is
    seeded independently, so a user who has customized one of them but
    deleted another will get the missing piece restored without losing
    their edits.
    """
    if not DEFAULTS_DIR.is_dir():
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for item in DEFAULTS_DIR.iterdir():
        target = CONFIG_DIR / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _register_user_config_on_path() -> None:
    """Make ``domains.<name>`` and ``events.<name>`` importable.

    The user's runtime config directory holds Python sub-packages
    (``domains/`` and ``events/``) that are dynamically discovered by
    the registry. Adding it to ``sys.path[0]`` lets the standard import
    machinery handle them, including cross-imports (e.g. an event that
    extends ``events.file.FileEvent``).
    """
    p = str(CONFIG_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


# Seed + register on first import so every consumer (CLI, tray, server,
# events subprocess) sees a fully provisioned config tree without
# needing to call any setup code explicitly.
_seed_user_config()
_register_user_config_on_path()

"""Discover and instantiate domains.

Centralizes the domain loading that was previously duplicated in
src/app.py, src/server.py and src/tray/app.py.
"""
import importlib

from src.shared.paths import DOMAINS_DIR


def available_domains() -> list[str]:
    """Return the sorted list of domain module stems under src/domains/."""
    if not DOMAINS_DIR.is_dir():
        return []
    return sorted(
        p.stem for p in DOMAINS_DIR.glob("*.py")
        if p.is_file() and p.stem != "__init__"
    )


def load_domain(name: str):
    """Import src.domains.<name> and instantiate the <Name>Domain class."""
    try:
        module = importlib.import_module(f"src.domains.{name}")
        cls = getattr(module, f"{name.capitalize()}Domain")
    except (ModuleNotFoundError, AttributeError) as exc:
        raise ValueError(f"Could not load domain '{name}': {exc}") from exc
    return cls()

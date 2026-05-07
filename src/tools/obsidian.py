import os

from src.shared.config import Config
from src.core.tool import Tool

Obsidian_path = "/home/tani/Images/Ubuntu/Obsidian Vault"

_IGNORED_DIRS = {".obsidian", ".trash", ".git", ".obsidian.trash"}


def _resolve(note_path: str) -> str:
    if not note_path:
        raise ValueError("note_path must not be empty")
    path = note_path.strip()
    if not path.lower().endswith(".md"):
        path += ".md"
    if os.path.isabs(path):
        full = os.path.normpath(path)
    else:
        full = os.path.normpath(os.path.join(Obsidian_path, path))
    vault = os.path.normpath(Obsidian_path)
    if os.path.commonpath([full, vault]) != vault:
        raise ValueError(f"note_path escapes the Obsidian vault: {note_path}")
    return full


def _iter_notes(subfolder: str = ""):
    vault = os.path.normpath(Obsidian_path)
    root = vault
    if subfolder:
        root = os.path.normpath(os.path.join(vault, subfolder))
        if os.path.commonpath([root, vault]) != vault:
            raise ValueError(f"subfolder escapes the Obsidian vault: {subfolder}")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Folder not found: {subfolder or '<vault root>'}")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.lower().endswith(".md"):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, vault)
                yield rel, full


class ObsidianWrite(Tool):
    name: str = "obsidian_write"
    description: str = "Write content to an Obsidian note (path relative to the vault, .md is added automatically)."

    def execute(self, note_path: str, content: str) -> str:
        full = _resolve(note_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Note {note_path} written successfully."


class ObsidianRead(Tool):
    name: str = "obsidian_read"
    description: str = "Read content from an Obsidian note (path relative to the vault, .md is added automatically)."

    def execute(self, note_path: str) -> str:
        full = _resolve(note_path)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"Note not found: {note_path}")
        with open(full, "r", encoding="utf-8") as f:
            return f.read()

class ObsidianSearch(Tool):
    name: str = "obsidian_search"
    description: str = "Search the Obsidian vault for notes whose filename or contents match a case-insensitive query. Returns matching note paths with line snippets."

    def execute(self, query: str, max_results: int = 20) -> str:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        needle = query.strip().lower()
        max_line_hits = 10
        max_preview_lines = 40
        max_preview_chars = 2000
        results: list[str] = []
        for rel, full in _iter_notes():
            if len(results) >= max_results:
                break
            name_hit = needle in rel.lower()
            line_hits: list[str] = []
            extra_hits = 0
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                continue
            for i, line in enumerate(content.splitlines(), start=1):
                if needle in line.lower():
                    if len(line_hits) < max_line_hits:
                        snippet = line.strip()
                        if len(snippet) > 200:
                            snippet = snippet[:200] + "..."
                        line_hits.append(f"  L{i}: {snippet}")
                    else:
                        extra_hits += 1
            if not (name_hit or line_hits):
                continue
            tag = " [filename match]" if name_hit else ""
            block = [rel + tag]
            if line_hits:
                block.extend(line_hits)
                if extra_hits:
                    block.append(f"  ... ({extra_hits} more match(es) in this note)")
            if name_hit:
                preview_lines = [
                    ln for ln in content.splitlines() if ln.strip()
                ][:max_preview_lines]
                preview = "\n".join(preview_lines)
                if len(preview) > max_preview_chars:
                    preview = preview[:max_preview_chars] + "\n... (truncated)"
                if preview:
                    block.append("  --- content ---")
                    block.extend(f"  {ln}" for ln in preview.splitlines())
            results.append("\n".join(block))
        if not results:
            return f"No matches for '{query}'."
        return f"Found {len(results)} match(es) for '{query}':\n\n" + "\n\n".join(results)


class ObsidianList(Tool):
    name: str = "obsidian_list"
    description: str = "List Obsidian notes in the vault, optionally restricted to a subfolder (relative to the vault root)."

    def execute(self, subfolder: str = "", max_results: int = 200) -> str:
        notes = []
        for rel, _ in _iter_notes(subfolder):
            notes.append(rel)
            if len(notes) >= max_results:
                break
        notes.sort()
        if not notes:
            return "No notes found."
        scope = subfolder or "<vault root>"
        return f"{len(notes)} note(s) under {scope}:\n" + "\n".join(notes)


class ObsidianAppend(Tool):
    name: str = "obsidian_append"
    description: str = "Append content to an Obsidian note, creating it if it does not exist (path relative to the vault, .md is added automatically)."

    def execute(self, note_path: str, content: str) -> str:
        full = _resolve(note_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        existed = os.path.isfile(full)
        prefix = ""
        if existed:
            with open(full, "rb") as f:
                try:
                    f.seek(-1, os.SEEK_END)
                    last = f.read(1)
                    if last and last not in (b"\n", b"\r"):
                        prefix = "\n"
                except OSError:
                    pass
        with open(full, "a", encoding="utf-8") as f:
            f.write(prefix + content)
        action = "appended to" if existed else "created"
        return f"Note {note_path} {action} successfully."

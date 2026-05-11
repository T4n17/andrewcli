import re
from pathlib import Path

from src.core.tool import Tool


# Match standard --- frontmatter --- body format with any surrounding whitespace.
_FRONTMATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


class Skill(Tool):
    """A Skill is a Tool whose ``execute()`` returns scripted instructions.

    A skill is defined entirely by a markdown file with a YAML-style
    frontmatter describing its ``name``, ``description``, and the
    ``tools`` it relies on. Skill files live inside each domain's
    ``skills/`` folder and are auto-discovered by
    :func:`src.core.registry.available_skills`.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        try:
            content = path.read_text()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Skill file not found: {path}") from exc

        match = _FRONTMATTER_RE.match(content)
        if not match:
            raise ValueError(
                f"Skill file '{path}' is missing valid frontmatter "
                "(expected '---\\n<meta>\\n---\\n<body>')"
            )
        frontmatter, body = match.group(1), match.group(2)

        meta: dict[str, str] = {}
        for line in frontmatter.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()

        for required_key in ("name", "description"):
            if required_key not in meta:
                raise ValueError(
                    f"Skill '{path}' is missing required frontmatter key '{required_key}'"
                )

        self.path = path
        self.name = meta["name"]
        self.description = meta["description"]
        self.instructions = body.strip()

        tools_val = meta.get("tools", "").strip()
        if tools_val.startswith("[") and tools_val.endswith("]"):
            tools_val = tools_val[1:-1]
        self.required_tools: list[str] = [
            t.strip() for t in tools_val.split(",") if t.strip()
        ]

    def execute(self) -> str:
        return (
            "[SKILL INSTRUCTIONS] You MUST now execute the following steps one by one "
            "using the available tools. Do NOT summarize or skip any step. "
            "Call the appropriate tools to carry out each action.\n\n"
            + self.instructions
        )

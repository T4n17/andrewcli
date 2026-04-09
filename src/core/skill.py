import inspect
from src.core.tool import Tool

TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

class Skill(Tool):
    skill_file: str

    def __init__(self):
        with open("src/skills/skills_files/" + self.skill_file, "r") as f:
            content = f.read()
        frontmatter = content.split("---")[1]
        meta = {}
        for line in frontmatter.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        self.name = meta["name"]
        self.description = meta["description"]
        self.instructions = content.split("---")[2].strip()
    
    def execute(self) -> str:
        return (
            "[SKILL INSTRUCTIONS] You MUST now execute the following steps one by one "
            "using the available tools. Do NOT summarize or skip any step. "
            "Call the appropriate tools to carry out each action.\n\n"
            + self.instructions
        )

    def to_openai_schema(self) -> dict:
        sig = inspect.signature(self.execute)
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "args", "kwargs"):
                continue
            json_type = TYPE_MAP.get(param.annotation, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

from src.core.tool import Tool
from src.shared.paths import SKILLS_DIR


class SkillCompiler(Tool):
    name: str = "compile_skill"
    description: str = "Use this tool only when create_new_skill skill is used Args: name: The name of the skill description: The description of the skill tools: The tools needed to execute the skill"

    def execute(self, name: str, description: str, tools: str) -> str:
        template = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"tools: [{tools}]\n"
            f"---\n\n"
            f"# Instruction to execute\n"
        )
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = SKILLS_DIR / f"{name}.md"
        file_path.write_text(template)
        return f"Skill '{name}' compiled successfully at {file_path}. Proceed with filling the instructions"

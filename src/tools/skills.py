from src.core.tool import Tool
import os

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
        file_path = os.path.join("src", "skills", "skills_files", f"{name}.md")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(template)
        return f"Skill '{name}' compiled successfully at {file_path}. Proceed with filling the instructions"

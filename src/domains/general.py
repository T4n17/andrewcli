from src.core.domain import Domain
from src.tools.common import WriteFile, ReadFile, ExecuteCommand, GetCurrentDate
from src.tools.skills import SkillCompiler
from src.skills.myskills import Example, CreateSkill

class GeneralDomain(Domain):
    routing_enabled: bool = False
    system_prompt: str = (
        "You are a helpful assistant with access to tools. "
        "When a task can be accomplished with a tool, always call the tool — "
        "do not explain how the user could do it themselves. "
        "Only respond in plain text when no tool is needed."
    )
    tools: list = [
        WriteFile(),
        ReadFile(),
        ExecuteCommand(),
        GetCurrentDate(),
        SkillCompiler()
    ]
    skills: list = [
        Example(),
        CreateSkill()
    ]
    events: list = [
    ]

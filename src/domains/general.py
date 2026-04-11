from src.core.domain import Domain
from src.tools.common import WriteFile, ReadFile, ExecuteCommand, GetCurrentDate
from src.skills.myskills import Example

class GeneralDomain(Domain):
    system_prompt: str = "You are a helpful assistant."
    tools: list = [
        WriteFile(),
        ReadFile(),
        ExecuteCommand(),
        GetCurrentDate()
    ]
    skills: list = [
        Example
    ]

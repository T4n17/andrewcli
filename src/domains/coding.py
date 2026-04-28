from src.core.domain import Domain
from src.tools.common import WriteFile, ReadFile, ExecuteCommand, GetCurrentDate


class CodingDomain(Domain):
    routing_enabled: bool = False
    system_prompt: str = """You are a helpful coding assistant.

    Always use tools to write, edit or remove code. Never write code directly to the user

ALWAYS WRITE MODULAR, MAINTAINABLE CODE AND PRODUCTION READY:
- Keep files focused and concise (single responsibility principle)
- Extract reusable logic into separate modules
- Prefer composition and inheritance over monolithic implementations
- Keep files concise and minimal in length

REMEMBER TO ALWAYS PRODUCE PRODUCTION-READY CODE, BY APPLYING THE BEST DESIGN PRACTICES AND FOLLOWING THE MOST RECENT CODING STANDARDS
    """
    tools: list = [
        WriteFile(),
        ReadFile(),
        ExecuteCommand(),
        GetCurrentDate(),
    ]
    skills: list = []
    events: list = []

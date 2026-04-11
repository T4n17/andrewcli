from src.core.domain import Domain
from src.tools.common import ExecuteCommand

class ExperimentalDomain(Domain):
    system_prompt: str = """
    You are a helpful assistant. 
    Use the execute_command tool to help the user with its requests.
    Examples: open file -> execute_command("xdg-open /path/to/file")
              read file -> execute_command("cat /path/to/file")
              write file -> execute_command("echo 'content' > /path/to/file")
              edit file -> execute_command("sed -i 's/old/new/g' /path/to/file")
              delete file -> execute_command("rm /path/to/file")
              find file -> execute_command("find /path -name 'filename'")
    """
    tools: list = [
        ExecuteCommand()
    ]
    skills: list = []
    events: list = []


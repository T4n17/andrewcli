from src.core.domain import Domain
from src.core.tool import Tool
from src.core.skill import Skill

class WriteFile(Tool):
    name: str = "write_file"
    description: str = "Write content to a file."
    
    def execute(self, file_path: str, content: str) -> str:
        with open(file_path, "w") as f:
            f.write(content)
        return f"File {file_path} written successfully."

class ReadFile(Tool):
    name: str = "read_file"
    description: str = "Read content from a file."
    
    def execute(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            return f.read()

class ExecuteCommand(Tool):
    name: str = "execute_command"
    description: str = "Execute a bash shell command."
    
    def execute(self, command: str) -> str:
        import subprocess
        if input(str(command) + "\nAre you sure you want to execute this command? (y/n): ") != "y":
            return "Command cancelled."
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print("Command output:", result.stdout)
        return result.stdout

class Example(Skill):
    skill_file: str = "example.md"


class GeneralDomain(Domain):
    system_prompt: str = "You are a helpful assistant."
    tools: list = [
        WriteFile(),
        ReadFile(),
        ExecuteCommand()
    ]
    skills: list = [
        Example()
    ]
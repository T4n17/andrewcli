from src.shared.config import Config
from datetime import datetime
from src.core.tool import Tool

class GetCurrentDate(Tool):
    name: str = "get_current_date"
    description: str = "Get current datetime"
    
    def execute(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    def __init__(self):
        self.execute_bash_automatically = Config().execute_bash_automatically
    
    def execute(self, command: str) -> str:
        import subprocess
        if not self.execute_bash_automatically:
            if input(str(command) + "\nAre you sure you want to execute this command? (y/n): ") != "y":
                return "Command cancelled."
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stderr:
            return ("Command error:", result.stderr)
        if result.stdout != "":
            return ("Command output:", result.stdout)
        return "Command executed successfully."
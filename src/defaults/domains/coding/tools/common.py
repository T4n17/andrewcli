from src.shared.paths import LAUNCH_DIR
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

    def execute(self, command: str, timeout: int = 10) -> str:
        import subprocess
        # Always spawn in the directory the user launched andrewcli
        # from, not the interpreter's current cwd. That way a command
        # like ``ls`` or ``git status`` always reflects the project
        # the user opened andrewcli in.
        proc = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(LAUNCH_DIR),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Intentionally leave the process running so long commands
            # can keep executing in the background while the agent
            # continues with the rest of the workflow.
            return f"Command still running in background (pid {proc.pid})."

        # Preserve both streams: many commands write useful info to both.
        parts = []
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        if not parts:
            return "Command executed successfully."
        return "\n".join(parts)

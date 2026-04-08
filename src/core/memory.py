import json

class Memory:
    def __init__(self, mem_file: str = None):
        self.messages = []
        if mem_file is not None:
            self._load_mem_file(mem_file)
    
    def _load_mem_file(self, mem_file: str):
        with open(mem_file, "r") as f:
            self.messages = json.load(f)
    
    def add(self, message: dict):
        self.messages.append(message)
    
    def get(self) -> list:
        return self.messages
    
    def clear(self):
        self.messages = []
    
    def __str__(self):
        return str(self.get())
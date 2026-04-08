from abc import ABC, abstractmethod
from typing import List
from src.core.llm import LLM

class Domain(ABC):
    system_prompt: str
    tools: List
    skills: List
    
    def __init__(self):
        self.system_prompt = self.system_prompt
        self.tools = self.tools
        self.skills = self.skills
        self.llm = LLM()

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt, self.tools, self.skills)

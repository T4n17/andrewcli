import inspect
from abc import ABC, abstractmethod

TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class Tool(ABC):
    name: str
    description: str
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def to_openai_schema(self) -> dict:
        sig = inspect.signature(self.execute)
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "args", "kwargs"):
                continue
            json_type = TYPE_MAP.get(param.annotation, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
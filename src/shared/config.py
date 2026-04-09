import yaml

class Config:
    def __init__(self):
        self._load()
        
    def _load(self):
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
                self.domain = config.get("domain", "general")
                self.execute_bash_automatically = config.get("execute_bash_automatically", False)
        except FileNotFoundError:
            raise FileNotFoundError("config.yaml not found")

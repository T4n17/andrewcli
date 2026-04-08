import importlib
import yaml

class AndrewCLI:
    def __init__(self):
        self.config = self._load_config()
        self.domain = self._load_domain()
    
    def _load_config(self):
        try:
            with open("config.yaml", "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {"domain": "general"}

    def _load_domain(self):
        try:
            self.domain_name = self.config["domain"]
            module = importlib.import_module(f"src.domains.{self.domain_name}")
            class_name = f"{self.domain_name.capitalize()}Domain"
            domain_class = getattr(module, class_name)
            return domain_class()
        except KeyError:
            raise ValueError("Domain not found in config")
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Could not load domain '{self.domain_name}': {e}")

    def run(self):
        print(f"Andrew is running... (Domain: {self.domain_name})")
        while True:
            user_input = input("Ask: ")
            response = self.domain.generate(user_input)
            print(f"Andrew: {response}")

if __name__ == "__main__":
    andrew = AndrewCLI()
    andrew.run()

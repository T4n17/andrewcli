from src.core.domain import Domain


class GeneralDomain(Domain):
    api_base_url: str = "http://localhost:8080/v1"
    model: str = "qwen-35b-q5"
    routing_enabled: bool = True

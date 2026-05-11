from src.core.domain import Domain


class CodingDomain(Domain):
    api_base_url: str = "http://localhost:8081/v1"
    model: str = "qwen-35b-q5"
    routing_enabled: bool = False

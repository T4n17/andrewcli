import json
import re
import os
import openai


class ToolRouter:
    def __init__(self):
        api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080/v1")
        self.model = os.getenv("MODEL", "qwen3.5:9B")
        self.client = openai.AsyncOpenAI(base_url=api_base_url)

    async def route(self, prompt: str, tools: list, skills: list, summary: str = "", last_exchange: str = "") -> tuple[list, list]:
        all_items = tools + skills
        if len(all_items) <= 1:
            return tools, skills

        catalog = "\n".join(f"- {item.name}: {item.description}" for item in all_items)

        context_parts = []
        if summary:
            context_parts.append(f"Conversation summary:\n{summary}")
        if last_exchange:
            context_parts.append(f"Last exchange:\n{last_exchange}")
        context_block = ("\n\n".join(context_parts) + "\n\n") if context_parts else ""

        routing_prompt = (
            f"{context_block}"
            f'User request: "{prompt}"\n\n'
            f"Available tools/skills:\n{catalog}\n\n"
            'Which of these are needed to fulfill the request? '
            'Reply with a JSON array of names only, e.g. ["name1", "name2"]. '
            "Return [] if none are needed (pure conversation)."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": routing_prompt}],
                stream=False,
                temperature=0.0,
                max_tokens=60,
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                needed = set(json.loads(match.group()))
                return (
                    [t for t in tools if t.name in needed],
                    [s for s in skills if s.name in needed],
                )
        except Exception:
            pass

        # Fallback: return everything
        return tools, skills

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

        skill_names = {s.name for s in skills}
        skills_catalog = "\n".join(f"- [SKILL] {item.name}: {item.description}" for item in skills)
        tools_catalog = "\n".join(f"- [TOOL] {item.name}: {item.description}" for item in tools)
        catalog = skills_catalog + ("\n" if skills_catalog and tools_catalog else "") + tools_catalog

        context_parts = []
        if summary:
            context_parts.append(f"Conversation summary:\n{summary}")
        if last_exchange:
            context_parts.append(f"Last exchange:\n{last_exchange}")
        context_block = ("\n\n".join(context_parts) + "\n\n") if context_parts else ""

        routing_prompt = (
            f"{context_block}"
            f'User request: "{prompt}"\n\n'
            f"Available skills and tools:\n{catalog}\n\n"
            "IMPORTANT: Always prefer [SKILL] items over [TOOL] items. "
            "If a skill can handle the request, choose it instead of individual tools. "
            "Only select individual tools when no skill matches.\n\n"
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
                matched_skills = [s for s in skills if s.name in needed]
                if matched_skills:
                    return [], matched_skills
                return (
                    [t for t in tools if t.name in needed],
                    [],
                )
        except Exception:
            pass

        # Fallback: return everything
        return tools, skills

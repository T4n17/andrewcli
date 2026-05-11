"""LLM-based tool and skill routing.

:class:`ToolRouter` sends the tool/skill catalog plus the user prompt
to the main chat model and parses the JSON array it replies with.
Costs 0.5-2 s per turn but handles ambiguous intent well. On any
failure it returns the full catalog so the LLM never ends up with
no tools.
"""
from __future__ import annotations

import json
import logging
import os
import re

import openai

log = logging.getLogger(__name__)


class ToolRouter:
    """LLM-based tool/skill router.

    Sends a short classification prompt listing the full catalog to the
    chat model and parses the JSON array in the response.
    """

    def __init__(self, api_base_url: str = None, model: str = None):
        api_base_url = api_base_url or os.getenv("API_BASE_URL", "http://localhost:8080/v1")
        self.model = model or os.getenv("MODEL", "qwen3.5:9B")
        self.client = openai.AsyncOpenAI(base_url=api_base_url, api_key=os.getenv("OPENAI_API_KEY", "local"))

    async def route(
        self,
        prompt: str,
        tools: list,
        skills: list,
        summary: str = "",
        last_exchange: str = "",
    ) -> tuple[list, list]:
        all_items = tools + skills
        if len(all_items) <= 1:
            return tools, skills

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
                # Return both matched skills AND matched tools. Previously,
                # tools were dropped whenever a skill matched, which
                # silently discarded tools the LLM explicitly asked for.
                # Domain.generate still merges in each skill's required_tools.
                matched_skills = [s for s in skills if s.name in needed]
                matched_tools = [t for t in tools if t.name in needed]
                return matched_tools, matched_skills
        except Exception as exc:
            log.warning("routing failed, falling back to full catalog: %s", exc)

        # Fallback: return everything
        return tools, skills

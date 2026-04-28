import asyncio
import json
import logging
import time

from src.shared.paths import DATA_DIR

log = logging.getLogger(__name__)

MEMORY_FILE = DATA_DIR / "memory.json"

# Skip the LLM merge call when the turn produced less than this much
# combined text. Short exchanges (greetings, one-word answers) aren't
# worth a full summarization request - we just append them verbatim.
# ~200 chars ≈ ~40 tokens, which matches the "short turn" heuristic.
MIN_SUMMARY_CHARS = 200

MERGE_SYSTEM_PROMPT = (
    "You are a memory summarizer. Merge the existing summary and new conversation "
    "excerpt into a single concise summary (max ~300 words). Keep facts, decisions, "
    "code written, tools used, and user preferences."
)


class Memory:
    def __init__(self):
        self.system_prompt = None
        self.messages = []
        self.summary = self._load_summary()
        self.last_exchange = ""
        self._trimmed = False
        self._merge_task = None
        # Skills that were activated during the current turn. Each entry
        # is (name, instructions). Rendered into the system message by
        # :meth:`get` under a ``<skill:...>`` tag so the model sees the
        # body at system-level priority instead of as a tool response.
        # Cleared by :meth:`clear_active_skills` at turn end.
        self._active_skills: list[tuple[str, str]] = []

    def add(self, message: dict):
        if message.get("role") == "system":
            self.system_prompt = message["content"]
        else:
            self.messages.append(message)

    def add_active_skill(self, name: str, instructions: str) -> None:
        """Promote a skill's body into the system prompt for this turn.

        Called from the LLM loop the moment a skill tool is invoked.
        Replaces any existing entry with the same name so re-invocations
        in the same turn don't stack duplicate blocks.
        """
        self._active_skills = [
            (n, i) for (n, i) in self._active_skills if n != name
        ]
        self._active_skills.append((name, instructions))

    def clear_active_skills(self) -> None:
        """Drop all turn-scoped skill annotations. Called at turn end."""
        self._active_skills = []

    def get(self) -> list:
        result = []
        sys_content = self.system_prompt or ""
        if self.summary:
            sys_content += f"\n\n<memory>\n{self.summary}\n</memory>"
        if self._trimmed:
            sys_content += (
                "\n\n<context>Earlier conversation was summarized in the "
                "memory blocks above. Full history has been cleared.</context>"
            )
        # Active-skill blocks go last inside the system message so they
        # sit closest to the user's request in the context window -
        # models attend more strongly to later parts of long system
        # prompts. Each skill is wrapped in a named tag so the model
        # can distinguish concurrent skills if more than one is active.
        for name, instructions in self._active_skills:
            sys_content += (
                f"\n\n<skill:{name}>\n"
                "The user's request has triggered this skill. Execute the "
                "steps below verbatim by calling the appropriate tools. "
                "Do not summarize, skip, or paraphrase any step.\n\n"
                f"{instructions}\n"
                f"</skill:{name}>"
            )
        result.append({"role": "system", "content": sys_content})
        result.extend(self.messages)
        return result

    async def summarize_turn(self, client, model: str):
        if self._merge_task and not self._merge_task.done():
            await self._merge_task

        excerpt = self._extract_excerpt(1500)
        if not excerpt:
            return

        self._trim_messages()

        # Bonus optimization: for short turns (greetings, one-liners,
        # quick confirmations) the LLM merge is overkill. Append the
        # excerpt to the summary verbatim with a rolling window.
        if len(excerpt) < MIN_SUMMARY_CHARS:
            combined = f"{self.summary}\n{excerpt}" if self.summary else excerpt
            self.summary = combined[-2000:]
            self._save_summary()
            log.debug("summarize_turn: short turn (%d chars), merged inline", len(excerpt))
            return

        if not self.summary:
            self.summary = excerpt[:2000]
            self._save_summary()
        else:
            self._merge_task = asyncio.create_task(
                self._background_merge(client, model, excerpt)
            )

    async def _background_merge(self, client, model: str, excerpt: str):
        t0 = time.monotonic()
        self.summary = await self._merge_summary(client, model, self.summary, excerpt)
        self._save_summary()
        log.debug(
            "background summary merge (model=%s) took %.2fs",
            model, time.monotonic() - t0,
        )

    def _extract_excerpt(self, max_chars: int) -> str:
        lines = []
        for msg in self.messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if role in ("user", "assistant") and content:
                lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        return text[-max_chars:]

    async def _merge_summary(self, client, model: str, existing: str, new_excerpt: str) -> str:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"<existing_summary>{existing}</existing_summary>"
                            f"\n\n<new_conversation>{new_excerpt}</new_conversation>"
                        ),
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception:
            combined = existing + "\n" + new_excerpt
            return combined[-2000:]

    def _trim_messages(self):
        last_user_idx = next(
            (i for i in range(len(self.messages) - 1, -1, -1) if self.messages[i].get("role") == "user"),
            None,
        )
        last_assistant_idx = next(
            (i for i in range(len(self.messages) - 1, -1, -1)
             if self.messages[i].get("role") == "assistant" and self.messages[i].get("content")),
            None,
        )
        user_content = self.messages[last_user_idx].get("content", "") if last_user_idx is not None else ""
        assistant_content = self.messages[last_assistant_idx].get("content", "") if last_assistant_idx is not None else ""
        if user_content or assistant_content:
            self.last_exchange = f"user: {user_content}\nassistant: {assistant_content}".strip()
        kept = sorted(i for i in [last_user_idx, last_assistant_idx] if i is not None)
        self.messages = [self.messages[i] for i in kept]
        self._trimmed = True

    def _load_summary(self) -> str:
        try:
            data = json.loads(MEMORY_FILE.read_text())
            if data.get("version") == 2:
                return data.get("summary", "")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return ""

    def _save_summary(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(
            json.dumps({"version": 2, "summary": self.summary}, indent=2)
        )

    def rollback_turn(self):
        """Discard messages from a stopped turn.
        Saves the user request and any partial assistant response to last_exchange
        so the router can resolve follow-up references on the next turn."""
        last_user = next(
            (m["content"] for m in reversed(self.messages) if m.get("role") == "user"),
            "",
        )
        last_assistant = next(
            (m["content"] for m in reversed(self.messages) if m.get("role") == "assistant" and m.get("content")),
            "",
        )
        if last_user:
            parts = [f"user: {last_user}"]
            if last_assistant:
                parts.append(f"assistant (partial, stopped by user): {last_assistant}")
            self.last_exchange = "\n".join(parts)
        self.messages = []
        self._active_skills = []

    def clear(self):
        self.messages = []
        self._trimmed = False
        self._active_skills = []

    def __str__(self):
        return str(self.get())
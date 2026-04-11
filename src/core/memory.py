import asyncio
import json
import os

MEMORY_DIR = os.path.expanduser("~/.andrewcli/data")
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.json")

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

    def add(self, message: dict):
        if message.get("role") == "system":
            self.system_prompt = message["content"]
        else:
            self.messages.append(message)

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

        if not self.summary:
            self.summary = excerpt[:2000]
            self._save_summary()
        else:
            self._merge_task = asyncio.create_task(
                self._background_merge(client, model, excerpt)
            )

    async def _background_merge(self, client, model: str, excerpt: str):
        self.summary = await self._merge_summary(client, model, self.summary, excerpt)
        self._save_summary()

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
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
                if data.get("version") == 2:
                    return data.get("summary", "")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return ""

    def _save_summary(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        with open(MEMORY_FILE, "w") as f:
            json.dump({"version": 2, "summary": self.summary}, f, indent=2)

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

    def clear(self):
        self.messages = []
        self._trimmed = False

    def __str__(self):
        return str(self.get())
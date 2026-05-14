# Memory & Routing

AndrewCLI fights context bloat with two mechanisms that run on every turn: **rolling memory** and a **context-aware router**.

---

## Rolling Memory

Local models degrade fast as context grows: reasoning gets muddier, tool calls go off the rails, and token budgets hit the ceiling. The rolling memory keeps the effective prompt size bounded regardless of conversation length.

### How it works

After each completed turn:

1. The last 1500 characters of conversation are extracted as an excerpt.
2. **Short-turn fast path** — if the excerpt is under `memory.min_summary_chars` chars (default `200`; greetings, one-line confirmations, `"Ciao"` / `"grazie"`), it is appended inline with a rolling 2000-char window and **no LLM call is made at all**.
3. If no summary exists yet, the excerpt is saved as-is (capped at 2000 chars).
4. If a summary exists and the turn is long enough, it is **merged with the excerpt** via a background LLM call that produces a single `~300-word` summary — facts, decisions, code written, tools used, user preferences.
5. Messages are **trimmed to just the last exchange** (last user message + last assistant response). The summary is injected into the system prompt inside `<memory>` tags.
6. The merge runs as a **fire-and-forget `asyncio` task** — the next prompt is ready immediately. Sequential merges are serialized with a lock to prevent race conditions.

The model always has prior context, but the message history never grows beyond one exchange.

### Dedicated summary model

Summarization is background work that doesn't need the same capacity as the main chat model. Set the `SUMMARY_MODEL` env var (e.g. `qwen2.5-0.5b-instruct`) to route background merges to a smaller model on the same server. Defaults to `MODEL` when unset.

### Storage

`$LAUNCH_DIR/.andrewcli/data/memory.json` — the data directory lives inside the folder you launched `andrewcli` from (same pattern as Claude Code / OpenCode), so different projects keep independent memory. Override by exporting `ANDREW_LAUNCH_DIR` before launching.

### Disabling memory

Set `memory.enabled: false` in `~/.config/andrewcli/config.yaml` to turn the rolling summary off entirely. Messages are still trimmed each turn so the LLM context stays bounded, but no summary is generated, persisted, or injected. Toggling back to `true` resumes from any previously saved `memory.json`.

---

## Context-Aware Router

Before each generation the router selects a minimal tool/skill set. The generation call receives **only the selected schemas**, not the full catalog. A skill that requires specific tools can declare them in its YAML frontmatter — those are injected even if the router didn't select them. Any routing failure falls back to the full catalog so the LLM always has its tools.

### How it works

`ToolRouter` (`src/core/router.py`) is an **LLM-as-classifier**: it sends the prompt, the current memory context (`summary` + `last_exchange`), and the full tool/skill catalog to the chat model and parses the JSON array it replies with. Costs 0.5–2 s per turn but handles ambiguous intent and natural-language follow-ups well.

A domain can opt out of routing entirely by setting `routing_enabled: false` in its `config.yaml` — useful when the full toolset is always relevant (e.g. the coding domain).

### Configuration

```yaml
# ~/.config/andrewcli/config.yaml
memory:
  enabled: true             # set false to disable rolling summary entirely
  min_summary_chars: 200    # turns shorter than this skip the LLM merge
```

```yaml
# ~/.config/andrewcli/domains/<name>/config.yaml
routing_enabled: false      # expose every tool every turn (per-domain override)
```

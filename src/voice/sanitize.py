"""Markdown -> speakable-text filter for the TTS tee path.

LLMs happily emit `**bold**`, `*italic*`, `` `code` ``, `# heading`, and
`[text](url)`. Piped straight into TTS, the speaker pronounces every
asterisk as "asterisco" / "asterisk", every backtick as "backtick", and
reads URLs character by character. Users want the *rendered* version,
so we strip the punctuation before feeding TTS while still streaming
token-by-token (no buffering the whole reply).

Approach is intentionally simple and stateful: a character-level
finite state machine over the incoming token stream. Only markers we
can safely drop without context-free ambiguity are stripped:

* ``*`` ``_`` ``~`` ``#`` ``` ` ``` - always dropped; these are
  never meaningful in spoken prose.
* ``[text](url)`` - keep ``text``, drop the ``(url)`` portion. The
  parser tracks the two-char transition ``] -> (`` across token
  boundaries so streaming works.
* ```` ``` ```` fenced code blocks - the backticks themselves are
  dropped by the rule above; the code contents are still spoken.
  Callers that want silent code blocks can add a block-level filter
  on top.

List bullets (``- ``, ``* ``) and headings (``# ``) are intentionally
preserved structurally: the dash/hash chars are stripped but the
space/word after them remains, so the sentence reads naturally.
"""
from __future__ import annotations

from typing import AsyncIterable, AsyncIterator


# Characters we always drop. Kept as a frozenset for O(1) `in` checks
# inside the per-character hot path.
_STRIP_CHARS = frozenset("*_~`#")


async def strip_markdown(
    token_iter: AsyncIterable[str],
) -> AsyncIterator[str]:
    """Yield the input stream with markdown punctuation removed.

    Stateful across tokens so ``]`` in one chunk and ``(`` in the
    next still triggers link-URL suppression.
    """
    prev_was_bracket_close = False
    in_link_url = False

    async for token in token_iter:
        out = []
        for ch in token:
            if in_link_url:
                # Swallow everything until the matching ')'.
                if ch == ")":
                    in_link_url = False
                continue

            if prev_was_bracket_close and ch == "(":
                # Transition into a URL; suppress the '(' itself and
                # everything up to the closing ')'.
                prev_was_bracket_close = False
                in_link_url = True
                continue
            prev_was_bracket_close = (ch == "]")

            if ch in _STRIP_CHARS:
                continue
            # Strip the brackets around link text while keeping the
            # text itself. '[hello](url)' -> 'hello'.
            if ch in "[]":
                continue
            out.append(ch)

        if out:
            yield "".join(out)

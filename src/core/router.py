"""Tool and skill routing.

Two interchangeable backends that share the same ``route()`` signature:

* :class:`ToolRouter` - classic LLM-based router. Sends the tool/skill
  catalog to the main LLM and parses the JSON array it replies with.
  Flexible and handles ambiguous intent well; costs 0.5-2 s per turn.

* :class:`EmbeddingRouter` - sentence-embedding router powered by
  ``fastembed``. Precomputes an embedding for each tool/skill and picks
  matches by cosine similarity against the user query. 10-30 ms per
  call (sub-ms cached), deterministic, no LLM round-trip.

Backend selection happens in :func:`src.core.domain._make_router`,
driven by the ``router_backend`` config key. Both classes fall back to
returning the full catalog on unexpected errors so the LLM still sees
every tool.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import warnings
from collections import OrderedDict

import openai

try:
    import numpy as np
except ImportError:  # numpy is a transitive dep of fastembed
    np = None

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


class EmbeddingRouter:
    """Cosine-similarity router backed by a local sentence-embedding model.

    Drop-in replacement for :class:`ToolRouter` sharing the same
    ``route()`` signature. Uses ``fastembed`` so the model runs on CPU
    via ONNX Runtime with no PyTorch dependency.

    Model loading and the query-embedding LRU cache are both held on
    the class (process-wide), so multiple Domain instances or
    per-request routers on the server share one loaded model.

    On any runtime error (missing fastembed, model download failure,
    broken catalog) :meth:`route` returns the full catalog, matching
    :class:`ToolRouter`'s defensive behavior.
    """

    # Multilingual, ~278M params, ~420MB. Trained with paraphrase mining
    # on 50+ languages; scores meaningfully higher than MiniLM on
    # cross-lingual pairs (tested IT <-> EN). ~15 ms per embed on CPU.
    # Override via env var ROUTER_EMBED_MODEL or the constructor arg.
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Cosine threshold, calibrated on the real catalog with this model:
    #   unrelated chat ("Ciao", "grazie"):   ~0.20-0.35
    #   real cross-lingual query:            ~0.40-0.95
    # Lower = more recall, higher = more precision.
    DEFAULT_THRESHOLD = 0.40

    # Skill tie-break bonus: matches the LLM router's "prefer SKILL" rule.
    SKILL_BIAS = 0.03

    # LRU cap for recently-embedded queries. Retries and near-duplicates
    # hit this in sub-millisecond time.
    QUERY_CACHE_SIZE = 32

    # Class-level (process-wide) model cache. Keyed by fastembed model
    # name so two routers using the same model share one loaded instance.
    # Locked because warm-up threads race with route() calls.
    _model_cache: dict[str, object] = {}
    _model_cache_lock = threading.Lock()

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
    ):
        if np is None:
            raise ImportError(
                "EmbeddingRouter requires numpy and fastembed. "
                "Install with: pip install fastembed"
            )
        self.model_name = model_name or os.getenv(
            "ROUTER_EMBED_MODEL", self.DEFAULT_MODEL,
        )
        self.threshold = (
            threshold if threshold is not None else self.DEFAULT_THRESHOLD
        )

        # Catalog state: re-embedded only when the (tools, skills) set
        # changes, keyed by (tool-name-tuple, skill-name-tuple).
        self._catalog_key: tuple | None = None
        self._tool_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._skill_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)

        # Per-instance query cache; lock guards concurrent access from
        # the warm-up thread and route() on the event loop.
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_lock = threading.Lock()

        # Lazily populated on first use or by warm().
        self._model: object | None = None

    # -- model loading (classmethod so the cache is shared) -------------

    @classmethod
    def _load_model(cls, name: str):
        """Load (once) and return the fastembed TextEmbedding for `name`."""
        with cls._model_cache_lock:
            model = cls._model_cache.get(name)
            if model is None:
                from fastembed import TextEmbedding  # lazy: optional dep

                t0 = time.monotonic()
                # fastembed 0.8 emits a UserWarning for this
                # sentence-transformers/paraphrase-* family because it
                # switched default pooling from CLS to mean pooling.
                # Mean pooling is actually the correct default for this
                # family - the warning is historical/informational and
                # safe to silence.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*mean pooling.*",
                        category=UserWarning,
                    )
                    model = TextEmbedding(model_name=name)
                cls._model_cache[name] = model
                log.info(
                    "loaded embedding model %r in %.2fs",
                    name, time.monotonic() - t0,
                )
            return model

    def _ensure_model(self):
        if self._model is None:
            self._model = self._load_model(self.model_name)
        return self._model

    # -- warm-up helper (called from Domain.__init__) -------------------

    def warm(self, tools: list, skills: list) -> None:
        """Load the model and pre-embed the catalog in a daemon thread.

        Called once by Domain.__init__ so the first real route() call
        hits a fully warm cache. Failures are logged but do not crash
        the caller - the next route() will retry or fall back.
        """
        def _warmup():
            try:
                self._ensure_model()
                self._ensure_catalog(tools, skills)
            except Exception:
                log.exception("embedding router warm-up failed")

        threading.Thread(target=_warmup, daemon=True).start()

    # -- embedding plumbing ---------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = list(model.embed(texts))
        return np.asarray(vectors, dtype=np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        with self._query_cache_lock:
            cached = self._query_cache.get(text)
            if cached is not None:
                self._query_cache.move_to_end(text)
                return cached
        vec = self._embed([text])[0]
        with self._query_cache_lock:
            self._query_cache[text] = vec
            if len(self._query_cache) > self.QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return vec

    def _ensure_catalog(self, tools: list, skills: list) -> None:
        key = (
            tuple(t.name for t in tools),
            tuple(s.name for s in skills),
        )
        if key == self._catalog_key:
            return
        tool_texts = [f"{t.name}: {t.description}" for t in tools]
        skill_texts = [f"{s.name}: {s.description}" for s in skills]
        self._tool_matrix = (
            self._embed(tool_texts) if tool_texts
            else np.empty((0, 0), dtype=np.float32)
        )
        self._skill_matrix = (
            self._embed(skill_texts) if skill_texts
            else np.empty((0, 0), dtype=np.float32)
        )
        self._catalog_key = key

    @staticmethod
    def _cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.empty(0, dtype=np.float32)
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        # Guard against zero-norm vectors (shouldn't happen, cheap check).
        denom = np.where(denom == 0, 1e-9, denom)
        return (matrix @ query) / denom

    # -- public routing (matches ToolRouter.route signature) ------------

    async def route(
        self,
        prompt: str,
        tools: list,
        skills: list,
        summary: str = "",
        last_exchange: str = "",
    ) -> tuple[list, list]:
        # Matches the LLM router's fast path: 0 or 1 item => nothing to decide.
        if len(tools) + len(skills) <= 1:
            return tools, skills

        try:
            self._ensure_catalog(tools, skills)
        except Exception as exc:
            log.warning(
                "embedding catalog init failed, returning full catalog: %s", exc,
            )
            return tools, skills

        # Include recent context so follow-ups that rely on the previous
        # turn ("do it again", "same thing for X") still route correctly.
        query_parts: list[str] = []
        if last_exchange:
            query_parts.append(last_exchange[-400:])
        query_parts.append(prompt)
        query = "\n".join(query_parts)

        t0 = time.monotonic()
        try:
            q_vec = self._embed_query(query)
            tool_scores = (
                self._cosine(self._tool_matrix, q_vec) if tools
                else np.empty(0)
            )
            skill_scores = (
                self._cosine(self._skill_matrix, q_vec) + self.SKILL_BIAS
                if skills else np.empty(0)
            )
        except Exception as exc:
            log.warning(
                "embedding routing failed, returning full catalog: %s", exc,
            )
            return tools, skills

        matched_tools = [
            tools[i] for i, score in enumerate(tool_scores)
            if score >= self.threshold
        ]
        matched_skills = [
            skills[i] for i, score in enumerate(skill_scores)
            if score >= self.threshold
        ]

        if log.isEnabledFor(logging.DEBUG):
            elapsed_ms = (time.monotonic() - t0) * 1000
            top_tool = (
                (tools[int(np.argmax(tool_scores))].name, float(np.max(tool_scores)))
                if tool_scores.size else None
            )
            top_skill = (
                (skills[int(np.argmax(skill_scores))].name, float(np.max(skill_scores)))
                if skill_scores.size else None
            )
            log.debug(
                "embed route took %.1fms query=%r top_tool=%s top_skill=%s "
                "matched_tools=%s matched_skills=%s",
                elapsed_ms, prompt[:60], top_tool, top_skill,
                [t.name for t in matched_tools],
                [s.name for s in matched_skills],
            )

        return matched_tools, matched_skills

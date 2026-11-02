import asyncio
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from types import SimpleNamespace

from src.core.andrew import AndrewCore
from src.core.llm import LLM
from src.core.memory import Memory
from src.core.rag import (
    KnowledgeBase,
    _Chunk,
    _Document,
    _table_heading,
    _table_rows,
)


def _knowledge(root: Path, cache_path: Path, **kwargs) -> KnowledgeBase:
    kwargs.setdefault("watch", False)
    return KnowledgeBase(root, cache_path=cache_path, **kwargs)


class KnowledgeBaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_returns_relevant_chunks_with_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledgebase"
            root.mkdir()
            (root / "rockets.txt").write_text(
                "The orbital launch sequence starts with propellant loading."
            )
            (root / "gardening.txt").write_text(
                "Tomatoes need regular watering and direct sunlight."
            )
            knowledge = _knowledge(root, Path(directory) / "index.sqlite3")
            knowledge.start()
            await knowledge._refresh_task

            result = knowledge.search_evidence("How does the orbital launch sequence start?")
            unrelated = knowledge.search_evidence("bake sourdough")
            knowledge.stop()

        self.assertIn("[source: rockets.txt — rockets]", result)
        self.assertIn("propellant loading", result)
        self.assertNotIn("Tomatoes", result)
        self.assertEqual(unrelated, "No matching knowledge-base evidence was found.")

    def test_dense_search_recovers_lexically_unmatched_chunk(self):
        class Embedder:
            def encode(self, texts, **kwargs):
                return np.asarray([
                    [1.0, 0.0] if "automobile" in text or "vehicle" in text else [0.0, 1.0]
                    for text in texts
                ], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledgebase"
            root.mkdir()
            (root / "vehicles.txt").write_text("vehicle service handbook")
            (root / "cooking.txt").write_text("bread baking recipe")
            knowledge = _knowledge(
                root, Path(directory) / "cache" / "index.sqlite3", embedding_model="test",
            )
            knowledge._embedder = Embedder()
            knowledge._refresh_sync()
            result = knowledge.search_evidence("automobile upkeep", mode="semantic", limit=1)
            metrics = knowledge.metrics()
            knowledge.stop()

        self.assertIn("vehicle service handbook", result)
        self.assertNotIn("bread", result)
        self.assertIn("Agentic RAG calls: 1", metrics)

    async def test_tools_are_hidden_until_initial_index_finishes(self):
        gate = threading.Event()

        def slow_extractor(path, root):
            gate.wait(timeout=2)
            return _Document(
                str(path.relative_to(root)),
                (_Chunk(path.name, path.stem, None, path.read_text()),),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledgebase"
            root.mkdir()
            (root / "manual.txt").write_text("Emergency shutdown procedure.")
            knowledge = _knowledge(
                root, Path(directory) / "index.sqlite3", extractor=slow_extractor,
            )
            knowledge.start()
            await asyncio.sleep(0.01)

            during_index = knowledge.agent_tools()
            gate.set()
            await knowledge._refresh_task
            after_index = knowledge.agent_tools()
            knowledge.stop()

        self.assertEqual(during_index, [])
        self.assertIn("search_knowledge", {tool.name for tool in after_index})

    async def test_reindex_replaces_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledgebase"
            root.mkdir()
            document = root / "manual.txt"
            document.write_text("The access code is amber.")
            knowledge = _knowledge(root, Path(directory) / "index.sqlite3")
            knowledge.start()
            await knowledge._refresh_task
            self.assertIn("amber", knowledge.search_evidence("access code"))

            document.write_text("The access code is cobalt now.")
            self.assertTrue(knowledge.request_refresh())
            await asyncio.sleep(0.5)
            await knowledge._refresh_task
            result = knowledge.search_evidence("access code")
            knowledge.stop()

        self.assertIn("cobalt", result)
        self.assertNotIn("amber", result)

    async def test_watcher_indexes_new_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledgebase"
            root.mkdir()
            knowledge = _knowledge(root, Path(directory) / "index.sqlite3", watch=True)
            knowledge.start()
            await knowledge._refresh_task
            (root / "new.txt").write_text("Newly indexed procedure.")

            for _ in range(30):
                await asyncio.sleep(0.1)
                if knowledge._store.count_documents():
                    break
            result = knowledge.search_evidence("newly indexed")
            knowledge.stop()

        self.assertIn("[source: new.txt — new]", result)

    async def test_markdown_headings_and_tables_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "vehicle.md"
            document.write_text(
                "# Vehicle Guide\n\n## Launch Sequence\n\n"
                "| Step | Action |\n|---|---|\n| 1 | Load propellant |\n"
            )
            knowledge = _knowledge(root, root / "index.sqlite3")
            indexed = await asyncio.to_thread(knowledge._extract_document, document, root)
            knowledge.stop()

        self.assertTrue(any("Launch Sequence" in chunk.section for chunk in indexed.chunks))
        self.assertTrue(any("Load propellant" in chunk.text for chunk in indexed.chunks))

    def test_agent_tools_search_filter_neighbors_and_share_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledgebase"
            root.mkdir()
            knowledge = KnowledgeBase(
                root,
                cache_path=Path(directory) / "index.sqlite3",
                watch=False,
            )
            knowledge._store.sync_document(
                "alpha.txt",
                (1, 1),
                [
                    ("Forecast", 1, "Year 3", "Project Alpha"),
                    ("Forecast", 2, "Net profit: EUR 780", "Project Alpha"),
                ],
                None,
            )
            knowledge._store.sync_document(
                "beta.txt",
                (1, 1),
                [("Forecast", 1, "Net profit: EUR 999", "Project Beta")],
                None,
            )
            knowledge._ready = True
            tools = {tool.name: tool for tool in knowledge.agent_tools()}
            search = tools["search_knowledge"]
            context = tools["get_knowledge_context"]
            calculator = tools["calculate_knowledge"]

            result = search.execute("net profit", source="alpha.txt", limit=2)
            chunk_id = int(result.split("[chunk: ", 1)[1].split("]", 1)[0])
            neighbors = context.execute(chunk_id, before=1, after=1)
            search.execute("tax", source="alpha.txt")
            exhausted = context.execute(chunk_id)
            calculation = calculator.execute("(780 - 700) / 700 * 100")
            rejected = calculator.run(expression="2 ** 8")
            metrics = knowledge.metrics()
            knowledge.stop()

        self.assertIn("EUR 780", result)
        self.assertNotIn("EUR 999", result)
        self.assertIn("Year 3", neighbors)
        self.assertIn("budget exhausted", exhausted)
        self.assertIn("11.428571", calculation)
        self.assertIn("[Tool Error]", rejected)
        self.assertIn("Agentic RAG calls: 3", metrics)
        self.assertEqual(search.execution_delay, 0.0)
        self.assertTrue(search.turn_scoped)

    def test_agent_tools_hidden_for_empty_knowledge_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledgebase"
            root.mkdir()
            knowledge = KnowledgeBase(
                root,
                cache_path=Path(directory) / "index.sqlite3",
                watch=False,
            )
            knowledge._ready = True
            empty_tools = knowledge.agent_tools()
            knowledge._store.sync_document(
                "alpha.txt",
                (1, 1),
                [("Forecast", 1, "Net profit: EUR 780", "Project Alpha")],
                None,
            )
            tools = knowledge.agent_tools()
            knowledge.stop()

        self.assertEqual(empty_tools, [])
        self.assertIn("search_knowledge", {tool.name for tool in tools})

    def test_sqlite_cache_reuses_chunks_and_embeddings(self):
        class Embedder:
            def encode(self, texts, **kwargs):
                return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledgebase"
            root.mkdir()
            document_path = root / "manual.txt"
            document_path.write_text("Cached maintenance procedure.")
            cache_path = Path(directory) / "cache" / "index.sqlite3"
            calls = []

            def extractor(path, base):
                calls.append(path)
                source = str(path.relative_to(base))
                return _Document(source, (
                    _Chunk(source, "Manual", None, path.read_text(), "Document: Manual"),
                ))

            first = KnowledgeBase(
                root,
                watch=False,
                cache_path=cache_path,
                embedding_model="test",
                extractor=extractor,
            )
            first._embedder = Embedder()
            first._refresh_sync()
            self.assertEqual(len(calls), 1)

            def fail_extractor(path, base):
                raise AssertionError("unchanged cached document was reparsed")

            second = KnowledgeBase(
                root,
                watch=False,
                cache_path=cache_path,
                embedding_model="test",
                extractor=fail_extractor,
            )
            second._embedder = Embedder()
            second._refresh_sync()
            document_count = second._store.count_documents()
            chunk_count = second._store.count_chunks()
            dense_hits = second._store.dense_search(
                np.array([1.0, 0.0], dtype=np.float32), 1
            )

        self.assertEqual(document_count, 1)
        self.assertEqual(chunk_count, 1)
        self.assertEqual(len(dense_hits), 1)

    def test_embedding_model_change_reuses_chunks_and_reembeds(self):
        class Embedder:
            def __init__(self, vector):
                self.vector = vector

            def encode(self, texts, **kwargs):
                return np.array([self.vector for _ in texts], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledgebase"
            root.mkdir()
            document_path = root / "manual.txt"
            document_path.write_text("Reusable extracted content.")
            cache_path = Path(directory) / "cache" / "index.sqlite3"
            calls = []

            def extractor(path, base):
                calls.append(path)
                source = str(path.relative_to(base))
                return _Document(source, (_Chunk(source, "Manual", None, path.read_text()),))

            first = KnowledgeBase(
                root,
                watch=False,
                cache_path=cache_path,
                embedding_model="model-a",
                extractor=extractor,
            )
            first._embedder = Embedder([1.0, 0.0])
            first._refresh_sync()

            def fail_extractor(path, base):
                raise AssertionError("model change should not reparse documents")

            second = KnowledgeBase(
                root,
                watch=False,
                cache_path=cache_path,
                embedding_model="model-b",
                extractor=fail_extractor,
            )
            second._embedder = Embedder([0.0, 1.0])
            second._refresh_sync()
            hits = second._store.dense_search(
                np.array([0.0, 1.0], dtype=np.float32),
                1,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0][1], 1.0)

    def test_changed_file_invalidates_sqlite_cache(self):
        class Embedder:
            def encode(self, texts, **kwargs):
                return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledgebase"
            root.mkdir()
            document_path = root / "manual.txt"
            document_path.write_text("First version.")
            cache_path = Path(directory) / "cache" / "index.sqlite3"
            calls = []

            def extractor(path, base):
                calls.append(path.read_text())
                source = str(path.relative_to(base))
                return _Document(source, (_Chunk(source, "Manual", None, path.read_text()),))

            first = KnowledgeBase(
                root,
                watch=False,
                cache_path=cache_path,
                embedding_model="test",
                extractor=extractor,
            )
            first._embedder = Embedder()
            first._refresh_sync()
            document_path.write_text("Second version with a different size.")

            second = KnowledgeBase(
                root,
                watch=False,
                cache_path=cache_path,
                embedding_model="test",
                extractor=extractor,
            )
            second._embedder = Embedder()
            second._refresh_sync()

        self.assertEqual(calls, ["First version.", "Second version with a different size."])

    def test_table_rows_preserve_key_value_relationships(self):
        rows = _table_rows(
            "| Field | Year 2 | Year 3 |\n"
            "|---|---|---|\n"
            "| Net profit | 700 | 780 |\n"
        )

        self.assertTrue(any("Net profit: 700 | 780" in row for row in rows))
        self.assertTrue(any("Year 2: 700" in row for row in rows))
        self.assertTrue(any("Year 3: 780" in row for row in rows))

    def test_repeated_table_title_becomes_structural_heading(self):
        markdown = (
            "| Quarterly Performance Report - Metric | Quarterly Performance Report - Value |\n"
            "|---|---|\n"
            "| Revenue | 10 |\n"
        )

        self.assertEqual(_table_heading(markdown), "Quarterly Performance Report")

    async def test_pptx_slides_are_discovered_and_extracted(self):
        from pptx import Presentation
        from pptx.util import Inches

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "forecast.pptx"
            presentation = Presentation()
            slide = presentation.slides.add_slide(presentation.slide_layouts[1])
            slide.shapes.title.text = "AulaLayer Financial Forecast"
            slide.placeholders[1].text = "Year 3 net profit is EUR 780,000."
            table = slide.shapes.add_table(
                2, 2, Inches(1), Inches(4), Inches(6), Inches(1)
            ).table
            table.cell(0, 0).text = "Metric"
            table.cell(0, 1).text = "Year 3"
            table.cell(1, 0).text = "Revenue"
            table.cell(1, 1).text = "1,250,000"
            presentation.save(document)
            knowledge = _knowledge(root, Path(directory) / "index.sqlite3")
            discovered = document in knowledge._files()
            indexed = await asyncio.to_thread(knowledge._extract_document, document, root)

        self.assertTrue(discovered)
        self.assertEqual(indexed.source, "forecast.pptx")
        self.assertTrue(any("AulaLayer" in chunk.section for chunk in indexed.chunks))
        self.assertTrue(any("780,000" in chunk.text for chunk in indexed.chunks))
        self.assertTrue(any("Revenue" in chunk.text for chunk in indexed.chunks))
        self.assertTrue(any("Year 3: 1,250,000" in chunk.text for chunk in indexed.chunks))



class RAGCommandTests(unittest.TestCase):
    def test_status_and_reindex_commands(self):
        class RAG:
            def __init__(self):
                self.reindexed = False

            def status(self):
                return "RAG [ready]"

            def metrics(self):
                return "RAG metrics"

            def request_refresh(self, force=False):
                self.reindexed = force
                return True

        rag = RAG()
        core = AndrewCore()
        core.domain = SimpleNamespace(rag=rag)

        self.assertEqual(core.handle_slash("/rag status", None), "RAG [ready]")
        self.assertEqual(core.handle_slash("/rag metrics", None), "RAG metrics")
        self.assertEqual(
            core.handle_slash("/rag reindex", None),
            "✓ Knowledge base reindex started.",
        )
        self.assertTrue(rag.reindexed)


class TemporaryContextTests(unittest.TestCase):
    def test_turn_scoped_tool_evidence_is_removed_from_memory(self):
        llm = LLM.__new__(LLM)
        llm.memory = Memory.__new__(Memory)
        llm.memory.messages = [
            {"role": "user", "content": "Find the value"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "rag-call",
                        "function": {"name": "search_knowledge", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "rag-call",
                "content": "Large retrieved evidence",
            },
            {"role": "assistant", "content": "The cited final answer"},
        ]
        tool = SimpleNamespace(name="search_knowledge", turn_scoped=True)

        llm._strip_turn_scoped_tools(0, [tool])

        self.assertEqual(
            llm.memory.messages,
            [
                {"role": "user", "content": "Find the value"},
                {"role": "assistant", "content": "The cited final answer"},
            ],
        )

    def test_last_user_prompt_excludes_assistant_content(self):
        memory = Memory.__new__(Memory)
        memory.messages = [
            {"role": "user", "content": "Question about AulaLayer"},
            {"role": "assistant", "content": "An unsupported inferred value"},
        ]

        self.assertEqual(memory.last_user_prompt, "Question about AulaLayer")

    def test_knowledge_context_is_not_stored_in_memory(self):
        memory = Memory.__new__(Memory)
        memory.system_prompt = "Base prompt"
        memory.messages = [{"role": "user", "content": "Question"}]
        memory.summary = ""
        memory.last_exchange = ""
        memory._trimmed = False
        memory._active_skills = []

        messages = memory.get("<knowledge>Temporary fact</knowledge>")

        self.assertIn("Temporary fact", messages[0]["content"])
        self.assertNotIn("Temporary fact", str(memory.messages))


if __name__ == "__main__":
    unittest.main()

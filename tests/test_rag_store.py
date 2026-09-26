import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.core.rag_store import RAGStore


class RAGStoreTests(unittest.TestCase):
    def test_incremental_document_update_preserves_chunk_id_and_fts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RAGStore(
                Path(directory) / "index.sqlite3",
                index_version="test",
                embedding_model="test-model",
                ann_threshold=100,
            )
            store.sync_document(
                "manual.txt",
                (1, 10),
                [("Guide", None, "alpha procedure", "Document: manual")],
                np.array([[1.0, 0.0]], dtype=np.float32),
            )
            first = store.lexical_search("alpha", 10)
            store.sync_document(
                "manual.txt",
                (2, 9),
                [("Guide", None, "beta procedure", "Document: manual")],
                np.array([[0.0, 1.0]], dtype=np.float32),
            )
            second = store.lexical_search("beta", 10)

            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertEqual(first[0][0], second[0][0])
            self.assertEqual(store.lexical_search("alpha", 10), [])
            self.assertEqual(store.count_documents(), 1)
            self.assertEqual(store.count_chunks(), 1)

    def test_filtered_search_and_neighbor_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RAGStore(
                Path(directory) / "index.sqlite3",
                index_version="test",
                embedding_model="test-model",
                ann_threshold=100,
            )
            store.sync_document(
                "alpha.txt",
                (1, 10),
                [
                    ("Forecast", 1, "Year headings", "Project Alpha"),
                    ("Forecast", 2, "Net profit 780", "Project Alpha"),
                    ("Notes", 3, "Tax assumptions", "Project Alpha"),
                ],
                np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
            store.sync_document(
                "beta.txt",
                (1, 10),
                [("Forecast", 1, "Net profit 999", "Project Beta")],
                np.array([[1.0, 0.0]], dtype=np.float32),
            )

            lexical = store.lexical_search("net profit", 10, source="alpha")
            dense = store.dense_search(
                np.array([1.0, 0.0], dtype=np.float32),
                10,
                source="alpha",
                section="Forecast",
            )
            neighbors = store.fetch_neighbors(lexical[0][0], before=1, after=1)

            self.assertEqual(len(lexical), 1)
            self.assertEqual(len(dense), 2)
            self.assertTrue(all(chunk.source == "alpha.txt" for chunk in neighbors))
            self.assertEqual([chunk.page for chunk in neighbors], [1, 2, 3])

    def test_hnsw_shard_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite3"
            store = RAGStore(
                path,
                index_version="test",
                embedding_model="test-model",
                ann_threshold=1,
                ann_shard_size=10_000,
            )
            affected = store.sync_document(
                "manual.txt",
                (1, 10),
                [
                    ("Guide", None, "engine maintenance", "Vehicle manual"),
                    ("Guide", None, "bread recipe", "Cooking manual"),
                ],
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
            store.refresh_ann(affected)
            first = store.dense_search(np.array([1.0, 0.0], dtype=np.float32), 1)
            store.close()

            restored = RAGStore(
                path,
                index_version="test",
                embedding_model="test-model",
                ann_threshold=1,
                ann_shard_size=10_000,
            )
            second = restored.dense_search(
                np.array([1.0, 0.0], dtype=np.float32), 1
            )

            self.assertEqual(first[0][0], second[0][0])
            self.assertTrue(restored.stats()["ann_active"])
            self.assertEqual(restored.stats()["ann_shards"], 1)

    def test_removed_document_is_removed_from_hnsw_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RAGStore(
                Path(directory) / "index.sqlite3",
                index_version="test",
                embedding_model="test-model",
                ann_threshold=1,
                ann_shard_size=10_000,
            )
            affected = set()
            affected.update(store.sync_document(
                "target.txt",
                (1, 10),
                [("Guide", None, "target", "Target")],
                np.array([[1.0, 0.0]], dtype=np.float32),
            ))
            affected.update(store.sync_document(
                "other.txt",
                (1, 10),
                [("Guide", None, "other", "Other")],
                np.array([[0.0, 1.0]], dtype=np.float32),
            ))
            store.refresh_ann(affected)
            target_id = store.lexical_search("target", 1)[0][0]
            affected = store.remove_sources({"target.txt"})
            store.refresh_ann(affected)
            hits = store.dense_search(np.array([1.0, 0.0], dtype=np.float32), 2)

            self.assertNotIn(target_id, [chunk_id for chunk_id, _ in hits])
            self.assertEqual(store.count_chunks(), 1)


if __name__ == "__main__":
    unittest.main()

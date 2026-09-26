"""Persistent scalable storage and search backend for domain knowledge bases."""
from __future__ import annotations

import heapq
import itertools
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from usearch.index import Index

log = logging.getLogger(__name__)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class StoredChunk:
    id: int
    source: str
    section: str
    page: int | None
    text: str
    context: str


class RAGStore:
    def __init__(
        self,
        path: Path,
        *,
        index_version: str,
        embedding_model: str,
        ann_threshold: int = 50_000,
        ann_shard_size: int = 100_000,
        ann_connectivity: int = 16,
        ann_expansion_search: int = 64,
    ):
        self.path = path.resolve()
        self.ann_path = self.path.with_suffix("")
        self.index_version = index_version
        self.embedding_model = embedding_model
        self.ann_threshold = max(1, ann_threshold)
        self.ann_shard_size = max(10_000, ann_shard_size)
        self.ann_connectivity = max(2, ann_connectivity)
        self.ann_expansion_search = max(10, ann_expansion_search)
        self._ann_views: dict[int, Index] = {}
        self._ann_view_revisions: dict[int, int] = {}
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _meta(connection: sqlite3.Connection) -> dict[str, str]:
        return dict(connection.execute("SELECT key, value FROM meta"))

    @staticmethod
    def _set_meta(connection: sqlite3.Connection, key: str, value) -> None:
        connection.execute(
            """INSERT INTO meta(key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (key, str(value)),
        )

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            existing_meta = self._meta(connection)
            if existing_meta and existing_meta.get("index_version") != self.index_version:
                connection.executescript(
                    """DROP TRIGGER IF EXISTS chunks_ai;
                    DROP TRIGGER IF EXISTS chunks_ad;
                    DROP TRIGGER IF EXISTS chunks_au;
                    DROP TABLE IF EXISTS chunks_fts;
                    DROP TABLE IF EXISTS chunks;
                    DROP TABLE IF EXISTS documents;
                    DROP TABLE IF EXISTS ann_shards;
                    DROP TABLE IF EXISTS ann_revisions;
                    DELETE FROM meta;"""
                )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                    source TEXT PRIMARY KEY,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    page INTEGER,
                    text TEXT NOT NULL,
                    context TEXT NOT NULL,
                    embedding BLOB,
                    UNIQUE(source, position),
                    FOREIGN KEY(source) REFERENCES documents(source) ON DELETE CASCADE
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS ann_shards (
                    shard INTEGER PRIMARY KEY,
                    chunk_count INTEGER NOT NULL,
                    revision INTEGER NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS ann_revisions (
                    shard INTEGER PRIMARY KEY,
                    revision INTEGER NOT NULL
                )"""
            )
            connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    source, section, context, text,
                    content='chunks', content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                )"""
            )
            connection.executescript(
                """CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, source, section, context, text)
                    VALUES (new.id, new.source, new.section, new.context, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, source, section, context, text)
                    VALUES ('delete', old.id, old.source, old.section, old.context, old.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_au
                AFTER UPDATE OF source, section, context, text ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, source, section, context, text)
                    VALUES ('delete', old.id, old.source, old.section, old.context, old.text);
                    INSERT INTO chunks_fts(rowid, source, section, context, text)
                    VALUES (new.id, new.source, new.section, new.context, new.text);
                END;"""
            )
            meta = self._meta(connection)
            if meta.get("index_version") != self.index_version:
                connection.execute("DELETE FROM chunks")
                connection.execute("DELETE FROM documents")
                connection.execute("DELETE FROM ann_shards")
                connection.execute("DELETE FROM ann_revisions")
                connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
                connection.execute("DELETE FROM meta")
                self._set_meta(connection, "index_version", self.index_version)
                self._set_meta(connection, "embedding_model", "")
                self._set_meta(connection, "content_generation", 0)
                self._ann_views.clear()
                self._ann_view_revisions.clear()
                for path in self.path.parent.glob(f"{self.ann_path.name}-*.usearch"):
                    path.unlink(missing_ok=True)

    def close(self) -> None:
        self._ann_views.clear()
        self._ann_view_revisions.clear()

    def document_fingerprints(self) -> dict[str, tuple[int, int]]:
        with self._connect() as connection:
            return {
                source: (int(mtime_ns), int(size))
                for source, mtime_ns, size in connection.execute(
                    "SELECT source, mtime_ns, size FROM documents"
                )
            }

    def embedding_model_matches(self) -> bool:
        with self._connect() as connection:
            return self._meta(connection).get("embedding_model", "") == self.embedding_model

    def count_chunks(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def count_documents(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def sync_document(
        self,
        source: str,
        fingerprint: tuple[int, int],
        chunks: list[tuple[str, int | None, str, str]],
        embeddings: np.ndarray | None,
    ) -> set[int]:
        with self._connect() as connection:
            existing = {
                int(position): int(chunk_id)
                for chunk_id, position in connection.execute(
                    "SELECT id, position FROM chunks WHERE source = ?",
                    (source,),
                )
            }
            connection.execute(
                """INSERT INTO documents(source, mtime_ns, size) VALUES (?, ?, ?)
                   ON CONFLICT(source) DO UPDATE SET
                     mtime_ns = excluded.mtime_ns, size = excluded.size""",
                (source, *fingerprint),
            )
            affected_ids = set(existing.values())
            for position, (section, page, text, context) in enumerate(chunks):
                blob = None
                if embeddings is not None and position < len(embeddings):
                    blob = np.asarray(embeddings[position], dtype="<f4").tobytes()
                if position in existing:
                    chunk_id = existing[position]
                    connection.execute(
                        """UPDATE chunks SET section = ?, page = ?, text = ?,
                           context = ?, embedding = ? WHERE id = ?""",
                        (section, page, text, context, blob, chunk_id),
                    )
                else:
                    cursor = connection.execute(
                        """INSERT INTO chunks(
                            source, position, section, page, text, context, embedding
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (source, position, section, page, text, context, blob),
                    )
                    chunk_id = int(cursor.lastrowid)
                affected_ids.add(chunk_id)
            connection.execute(
                "DELETE FROM chunks WHERE source = ? AND position >= ?",
                (source, len(chunks)),
            )
            self._bump_generation(connection)
            shards = self._invalidate_shards(connection, affected_ids)
        return shards

    def remove_sources(self, sources: set[str]) -> set[int]:
        if not sources:
            return set()
        with self._connect() as connection:
            placeholders = ",".join("?" for _ in sources)
            ids = {
                int(row[0]) for row in connection.execute(
                    f"SELECT id FROM chunks WHERE source IN ({placeholders})",
                    tuple(sources),
                )
            }
            connection.executemany(
                "DELETE FROM documents WHERE source = ?",
                ((source,) for source in sources),
            )
            self._bump_generation(connection)
            shards = self._invalidate_shards(connection, ids)
        return shards

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM ann_shards")
            connection.execute("DELETE FROM ann_revisions")
            self._set_meta(connection, "embedding_model", "")
            self._bump_generation(connection)
        self._ann_views.clear()
        self._ann_view_revisions.clear()
        for path in self.path.parent.glob(f"{self.ann_path.name}-*.usearch"):
            path.unlink(missing_ok=True)

    def _bump_generation(self, connection: sqlite3.Connection) -> int:
        meta = self._meta(connection)
        generation = int(meta.get("content_generation", "0")) + 1
        self._set_meta(connection, "content_generation", generation)
        return generation

    def _invalidate_shards(
        self,
        connection: sqlite3.Connection,
        chunk_ids: set[int],
    ) -> set[int]:
        shards = {(chunk_id - 1) // self.ann_shard_size for chunk_id in chunk_ids}
        connection.executemany(
            """INSERT INTO ann_revisions(shard, revision) VALUES (?, 1)
               ON CONFLICT(shard) DO UPDATE SET revision = revision + 1""",
            ((shard,) for shard in shards),
        )
        connection.executemany(
            "DELETE FROM ann_shards WHERE shard = ?",
            ((shard,) for shard in shards),
        )
        for shard in shards:
            self._ann_views.pop(shard, None)
            self._ann_view_revisions.pop(shard, None)
        return shards

    def chunks_for_embedding(self, batch_size: int = 256):
        offset_id = 0
        while True:
            with self._connect() as connection:
                rows = connection.execute(
                    """SELECT id, context, text FROM chunks
                       WHERE id > ? ORDER BY id LIMIT ?""",
                    (offset_id, batch_size),
                ).fetchall()
            if not rows:
                break
            yield rows
            offset_id = int(rows[-1][0])

    def update_embeddings(self, values: list[tuple[int, np.ndarray]]) -> None:
        if not values:
            return
        with self._connect() as connection:
            connection.executemany(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                (
                    (np.asarray(vector, dtype="<f4").tobytes(), int(chunk_id))
                    for chunk_id, vector in values
                ),
            )

    def finish_embedding_update(self) -> None:
        with self._connect() as connection:
            self._set_meta(connection, "embedding_model", self.embedding_model)
            connection.execute("DELETE FROM ann_shards")
            connection.execute("DELETE FROM ann_revisions")
            self._bump_generation(connection)
        self._ann_views.clear()
        self._ann_view_revisions.clear()

    def lexical_search(
        self,
        query: str,
        limit: int,
        source: str = "",
        section: str = "",
    ) -> list[tuple[int, float]]:
        words = list(dict.fromkeys(_WORD_RE.findall(query.casefold())))[:32]
        if not words:
            return []
        expression = " OR ".join(f'"{word.replace(chr(34), chr(34) * 2)}"' for word in words)
        conditions = ["chunks_fts MATCH ?"]
        params: list = [expression]
        if source:
            conditions.append("c.source LIKE ? COLLATE NOCASE")
            params.append(f"%{source}%")
        if section:
            conditions.append("c.section LIKE ? COLLATE NOCASE")
            params.append(f"%{section}%")
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT chunks_fts.rowid,
                           bm25(chunks_fts, 1.2, 2.5, 2.0, 1.0)
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    WHERE {' AND '.join(conditions)}
                    ORDER BY bm25(chunks_fts, 1.2, 2.5, 2.0, 1.0)
                    LIMIT ?""",
                params,
            ).fetchall()
        return [(int(rowid), float(score)) for rowid, score in rows]

    def fetch_chunks(self, ids: list[int]) -> list[StoredChunk]:
        if not ids:
            return []
        result: dict[int, StoredChunk] = {}
        with self._connect() as connection:
            for start in range(0, len(ids), 900):
                batch = ids[start:start + 900]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""SELECT id, source, section, page, text, context
                        FROM chunks WHERE id IN ({placeholders})""",
                    batch,
                ).fetchall()
                for row in rows:
                    chunk = StoredChunk(*row)
                    result[chunk.id] = chunk
        return [result[chunk_id] for chunk_id in ids if chunk_id in result]

    def fetch_neighbors(
        self,
        chunk_id: int,
        before: int = 2,
        after: int = 2,
    ) -> list[StoredChunk]:
        with self._connect() as connection:
            location = connection.execute(
                "SELECT source, position FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if location is None:
                return []
            source, position = location
            rows = connection.execute(
                """SELECT id, source, section, page, text, context
                   FROM chunks
                   WHERE source = ? AND position BETWEEN ? AND ?
                   ORDER BY position""",
                (
                    source,
                    max(0, int(position) - max(0, before)),
                    int(position) + max(0, after),
                ),
            ).fetchall()
        return [StoredChunk(*row) for row in rows]

    def _generation(self) -> int:
        with self._connect() as connection:
            meta = self._meta(connection)
        return int(meta.get("content_generation", "0"))

    def _iter_embeddings(
        self,
        batch_size: int = 4096,
        *,
        minimum_id: int = 0,
        maximum_id: int | None = None,
        source: str = "",
        section: str = "",
    ):
        offset_id = minimum_id - 1
        while True:
            conditions = ["id > ?", "embedding IS NOT NULL"]
            params: list = [offset_id]
            if maximum_id is not None:
                conditions.append("id < ?")
                params.append(maximum_id)
            if source:
                conditions.append("source LIKE ? COLLATE NOCASE")
                params.append(f"%{source}%")
            if section:
                conditions.append("section LIKE ? COLLATE NOCASE")
                params.append(f"%{section}%")
            params.append(batch_size)
            with self._connect() as connection:
                rows = connection.execute(
                    f"""SELECT id, embedding FROM chunks
                        WHERE {' AND '.join(conditions)}
                        ORDER BY id LIMIT ?""",
                    params,
                ).fetchall()
            if not rows:
                break
            ids = np.asarray([row[0] for row in rows], dtype=np.uint64)
            vectors = np.vstack([
                np.frombuffer(row[1], dtype="<f4") for row in rows
            ]).astype(np.float32, copy=False)
            yield ids, vectors
            offset_id = int(rows[-1][0])

    def _shard_path(self, shard: int) -> Path:
        return self.path.parent / f"{self.ann_path.name}-{shard:06d}.usearch"

    def refresh_ann(self, shards: set[int] | None = None) -> None:
        if self.count_chunks() < self.ann_threshold:
            with self._connect() as connection:
                connection.execute("DELETE FROM ann_shards")
            self._ann_views.clear()
            self._ann_view_revisions.clear()
            for path in self.path.parent.glob(f"{self.ann_path.name}-*.usearch"):
                path.unlink(missing_ok=True)
            return
        with self._connect() as connection:
            current = {
                int(row[0]) for row in connection.execute(
                    """SELECT DISTINCT CAST((id - 1) / ? AS INTEGER)
                       FROM chunks WHERE embedding IS NOT NULL""",
                    (self.ann_shard_size,),
                )
            }
            ready = {
                int(row[0]) for row in connection.execute(
                    "SELECT shard FROM ann_shards"
                )
            }
        stale = ready - current
        if stale:
            with self._connect() as connection:
                connection.executemany(
                    "DELETE FROM ann_shards WHERE shard = ?",
                    ((shard,) for shard in stale),
                )
            for shard in stale:
                self._shard_path(shard).unlink(missing_ok=True)
                self._ann_views.pop(shard, None)
                self._ann_view_revisions.pop(shard, None)
        targets = (current - ready) if shards is None else (current & shards)
        for shard in sorted(targets):
            self._rebuild_shard(shard)

    def _rebuild_shard(self, shard: int) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT revision FROM ann_revisions WHERE shard = ?",
                (shard,),
            ).fetchone()
        revision = int(row[0]) if row else 0
        minimum_id = shard * self.ann_shard_size + 1
        maximum_id = minimum_id + self.ann_shard_size
        batches = self._iter_embeddings(
            minimum_id=minimum_id,
            maximum_id=maximum_id,
        )
        first = next(batches, None)
        path = self._shard_path(shard)
        if first is None:
            path.unlink(missing_ok=True)
            with self._connect() as connection:
                connection.execute("DELETE FROM ann_shards WHERE shard = ?", (shard,))
            self._ann_views.pop(shard, None)
            self._ann_view_revisions.pop(shard, None)
            return
        index = Index(
            ndim=first[1].shape[1],
            metric="cos",
            dtype="f16",
            connectivity=self.ann_connectivity,
            expansion_add=max(64, self.ann_expansion_search),
            expansion_search=self.ann_expansion_search,
        )
        count = 0
        for ids, vectors in itertools.chain((first,), batches):
            index.add(ids, vectors)
            count += len(ids)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        index.save(temporary)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision FROM ann_revisions WHERE shard = ?",
                (shard,),
            ).fetchone()
            current_revision = int(row[0]) if row else 0
            if current_revision != revision:
                connection.rollback()
                temporary.unlink(missing_ok=True)
                return
            os.replace(temporary, path)
            connection.execute(
                """INSERT INTO ann_shards(shard, chunk_count, revision)
                   VALUES (?, ?, ?)
                   ON CONFLICT(shard) DO UPDATE SET
                     chunk_count = excluded.chunk_count,
                     revision = excluded.revision""",
                (shard, count, revision),
            )
            connection.commit()
        finally:
            connection.close()
        self._ann_views.pop(shard, None)
        self._ann_view_revisions.pop(shard, None)

    def _ann_view(self, shard: int, revision: int) -> Index | None:
        if (
            shard in self._ann_views
            and self._ann_view_revisions.get(shard) == revision
        ):
            return self._ann_views[shard]
        self._ann_views.pop(shard, None)
        self._ann_view_revisions.pop(shard, None)
        path = self._shard_path(shard)
        if not path.is_file():
            return None
        try:
            index = Index.restore(path, view=True)
            index.expansion_search = self.ann_expansion_search
            self._ann_views[shard] = index
            self._ann_view_revisions[shard] = revision
            return index
        except Exception:
            log.exception("failed to restore ANN shard %s", path)
            return None

    def dense_search(
        self,
        vector: np.ndarray,
        limit: int,
        source: str = "",
        section: str = "",
    ) -> list[tuple[int, float]]:
        count = self.count_chunks()
        if not count:
            return []
        query = np.asarray(vector, dtype=np.float32)
        if count >= self.ann_threshold and not source and not section:
            self.refresh_ann()
            with self._connect() as connection:
                shards = connection.execute(
                    "SELECT shard, revision FROM ann_shards ORDER BY shard"
                ).fetchall()
            best: list[tuple[float, int]] = []
            for shard, revision in shards:
                index = self._ann_view(int(shard), int(revision))
                if index is None:
                    continue
                matches = index.search(query, min(limit, len(index)))
                for key, distance in zip(matches.keys, matches.distances):
                    item = (1.0 - float(distance), int(key))
                    if len(best) < limit:
                        heapq.heappush(best, item)
                    elif item[0] > best[0][0]:
                        heapq.heapreplace(best, item)
            return [
                (chunk_id, score)
                for score, chunk_id in sorted(best, reverse=True)
            ]

        best = []
        for ids, vectors in self._iter_embeddings(source=source, section=section):
            if vectors.shape[1] != query.size:
                continue
            scores = vectors @ query
            for chunk_id, score in zip(ids, scores):
                item = (float(score), int(chunk_id))
                if len(best) < limit:
                    heapq.heappush(best, item)
                elif item[0] > best[0][0]:
                    heapq.heapreplace(best, item)
        return [
            (chunk_id, score)
            for score, chunk_id in sorted(best, reverse=True)
        ]

    def stats(self) -> dict:
        chunks = self.count_chunks()
        with self._connect() as connection:
            ready_shards = int(
                connection.execute("SELECT COUNT(*) FROM ann_shards").fetchone()[0]
            )
            sections = int(connection.execute(
                "SELECT COUNT(*) FROM (SELECT 1 FROM chunks GROUP BY source, section)"
            ).fetchone()[0])
        return {
            "documents": self.count_documents(),
            "sections": sections,
            "chunks": chunks,
            "ann_active": chunks >= self.ann_threshold,
            "ann_shards": ready_shards,
            "content_generation": self._generation(),
        }

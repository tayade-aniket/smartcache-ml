"""
Semantic and Exact-Match Caching Engine for SmartCache ML.
Implements persistent SQLite backing, LRU eviction, TTL expiration,
namespace isolation, and separate embedding/lookup latency tracking.
"""

import time
import sqlite3
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from smartcache.config import settings
from smartcache.embeddings import encode_query, compute_matrix_cosine_similarities


@dataclass
class CacheLookupResult:
    hit: bool
    response: Optional[str] = None
    similarity_score: Optional[float] = None
    matched_query: Optional[str] = None
    cache_id: Optional[int] = None
    embedding_latency: float = 0.0
    search_latency: float = 0.0
    total_lookup_latency: float = 0.0


class ExactMatchCache:
    """
    Baseline exact string-match cache.
    Useful for comparison against semantic matching.
    """

    def __init__(self, ttl_seconds: int = settings.cache_ttl_seconds):
        self.ttl = ttl_seconds
        # Structure: {(namespace, normalized_query): (response, timestamp)}
        self._store: Dict[Tuple[str, str], Tuple[str, float]] = {}
        self.hits = 0
        self.misses = 0

    def lookup(self, query: str, namespace: str = "default") -> CacheLookupResult:
        start_time = time.perf_counter()
        key = (namespace, query.strip().lower())
        now = time.time()

        if key in self._store:
            response, timestamp = self._store[key]
            if now - timestamp <= self.ttl:
                self.hits += 1
                search_latency = time.perf_counter() - start_time
                return CacheLookupResult(
                    hit=True,
                    response=response,
                    similarity_score=1.0,
                    matched_query=query,
                    embedding_latency=0.0,
                    search_latency=search_latency,
                    total_lookup_latency=search_latency,
                )
            else:
                del self._store[key]

        self.misses += 1
        search_latency = time.perf_counter() - start_time
        return CacheLookupResult(
            hit=False,
            search_latency=search_latency,
            total_lookup_latency=search_latency,
        )

    def store(
        self,
        query: str,
        response: str,
        namespace: str = "default",
        is_cacheable: bool = True
    ) -> bool:
        if not is_cacheable:
            return False
        key = (namespace, query.strip().lower())
        self._store[key] = (response, time.time())
        return True

    def clear(self):
        self._store.clear()
        self.hits = 0
        self.misses = 0


class SemanticCache:
    """
    Persistent semantic cache using SQLite for durable storage
    and vectorized NumPy matrix operations for similarity search.
    """

    def __init__(
        self,
        db_path: Path = settings.cache_db_path,
        similarity_threshold: float = settings.similarity_threshold,
        max_cache_size: int = settings.max_cache_size,
        ttl_seconds: int = settings.cache_ttl_seconds,
    ):
        self.db_path = Path(db_path)
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds

        # Metrics counters
        self.hit_count = 0
        self.miss_count = 0

        # In-memory synchronization arrays for high-speed similarity search
        self._ids: List[int] = []
        self._namespaces: List[str] = []
        self._queries: List[str] = []
        self._responses: List[str] = []
        self._embeddings: List[np.ndarray] = []

        self._init_db()
        self._sync_from_db()

    def _get_connection(self):
        """Context manager to ensure SQLite connection is closed promptly."""
        from contextlib import contextmanager

        @contextmanager
        def _managed_conn():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

        return _managed_conn()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed_at REAL NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ns ON cache_entries(namespace);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp ON cache_entries(expires_at);")
            conn.commit()

    def _sync_from_db(self):
        """Load valid active cache entries from SQLite into memory."""
        now = time.time()
        self._ids.clear()
        self._namespaces.clear()
        self._queries.clear()
        self._responses.clear()
        self._embeddings.clear()

        with self._get_connection() as conn:
            # Delete expired entries
            conn.execute("DELETE FROM cache_entries WHERE expires_at < ?", (now,))
            conn.commit()

            cursor = conn.execute(
                "SELECT id, namespace, query_text, response_text, embedding FROM cache_entries ORDER BY id ASC"
            )
            for row in cursor.fetchall():
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                self._ids.append(row["id"])
                self._namespaces.append(row["namespace"])
                self._queries.append(row["query_text"])
                self._responses.append(row["response_text"])
                self._embeddings.append(emb)

    def lookup(
        self,
        query: str,
        namespace: str = "default",
        threshold_override: Optional[float] = None
    ) -> CacheLookupResult:
        """
        Search for a semantically similar entry within the specified namespace.
        Measures embedding generation and vector search time independently.
        """
        threshold = threshold_override if threshold_override is not None else self.similarity_threshold

        # 1. Measure Embedding Latency
        t_embed_start = time.perf_counter()
        query_vec = encode_query(query, normalize=True)
        embedding_latency = time.perf_counter() - t_embed_start

        # 2. Measure Vector Search Latency
        t_search_start = time.perf_counter()
        total_items = len(self._ids)

        if total_items == 0:
            self.miss_count += 1
            search_latency = time.perf_counter() - t_search_start
            return CacheLookupResult(
                hit=False,
                embedding_latency=embedding_latency,
                search_latency=search_latency,
                total_lookup_latency=embedding_latency + search_latency
            )

        # Filter by namespace
        ns_indices = [i for i, ns in enumerate(self._namespaces) if ns == namespace]
        if not ns_indices:
            self.miss_count += 1
            search_latency = time.perf_counter() - t_search_start
            return CacheLookupResult(
                hit=False,
                embedding_latency=embedding_latency,
                search_latency=search_latency,
                total_lookup_latency=embedding_latency + search_latency
            )

        sub_matrix = np.array([self._embeddings[i] for i in ns_indices], dtype=np.float32)
        sim_scores = compute_matrix_cosine_similarities(query_vec, sub_matrix)

        best_idx_in_sub = int(np.argmax(sim_scores))
        best_score = float(sim_scores[best_idx_in_sub])
        best_global_idx = ns_indices[best_idx_in_sub]

        search_latency = time.perf_counter() - t_search_start
        total_latency = embedding_latency + search_latency

        if best_score >= threshold:
            # Hit! Update access stats in DB
            self.hit_count += 1
            entry_id = self._ids[best_global_idx]
            self._update_access(entry_id)

            return CacheLookupResult(
                hit=True,
                response=self._responses[best_global_idx],
                similarity_score=best_score,
                matched_query=self._queries[best_global_idx],
                cache_id=entry_id,
                embedding_latency=embedding_latency,
                search_latency=search_latency,
                total_lookup_latency=total_latency
            )

        self.miss_count += 1
        return CacheLookupResult(
            hit=False,
            similarity_score=best_score,
            matched_query=self._queries[best_global_idx],
            embedding_latency=embedding_latency,
            search_latency=search_latency,
            total_lookup_latency=total_latency
        )

    def _update_access(self, entry_id: int):
        """Update access count and last_accessed_at timestamp in SQLite."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE cache_entries SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
                    (time.time(), entry_id)
                )
                conn.commit()
        except Exception:
            pass

    def store(
        self,
        query: str,
        response: str,
        embedding: Optional[np.ndarray] = None,
        namespace: str = "default",
        is_cacheable: bool = True,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Store a new query-response pair persistently in SQLite.
        Enforces maximum cache size using LRU eviction.
        """
        if not is_cacheable:
            return False

        if embedding is None:
            embedding = encode_query(query, normalize=True)

        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        expires_at = now + ttl

        # Evict oldest accessed entry if at capacity
        if len(self._ids) >= self.max_cache_size:
            self._evict_lru()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO cache_entries
                (namespace, query_text, response_text, embedding, created_at, expires_at, access_count, last_accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (namespace, query, response, embedding.tobytes(), now, expires_at, now)
            )
            new_id = cursor.lastrowid
            conn.commit()

        # Update in-memory structures
        self._ids.append(new_id)
        self._namespaces.append(namespace)
        self._queries.append(query)
        self._responses.append(response)
        self._embeddings.append(embedding)

        return True

    def _evict_lru(self):
        """Evict the least recently accessed item."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id FROM cache_entries ORDER BY last_accessed_at ASC LIMIT 1")
            row = cursor.fetchone()
            if row:
                evict_id = row["id"]
                conn.execute("DELETE FROM cache_entries WHERE id = ?", (evict_id,))
                conn.commit()
                if evict_id in self._ids:
                    idx = self._ids.index(evict_id)
                    del self._ids[idx]
                    del self._namespaces[idx]
                    del self._queries[idx]
                    del self._responses[idx]
                    del self._embeddings[idx]

    def invalidate(self, namespace: Optional[str] = None):
        """Invalidate all cache entries or entries belonging to a given namespace."""
        with self._get_connection() as conn:
            if namespace:
                conn.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))
            else:
                conn.execute("DELETE FROM cache_entries")
            conn.commit()
        self._sync_from_db()
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health and usage statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total) if total > 0 else 0.0
        return {
            "total_cached_entries": len(self._ids),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "similarity_threshold": self.similarity_threshold,
            "max_cache_size": self.max_cache_size,
            "ttl_seconds": self.ttl_seconds,
            "db_path": str(self.db_path),
        }

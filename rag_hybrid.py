"""
Hybrid retrieval extension for rag_pipeline.py

Adds on top of the base RAG class:
  - FAISS vector backend ("flat" = exact search, "hnsw" = approximate, faster at scale)
  - BM25 keyword search (rank_bm25)
  - Hybrid retrieval via Reciprocal Rank Fusion (RRF) of the vector and BM25 rankings
  - Optional Voyage reranking as a final stage
  - Query-embedding cache (saves API calls, especially during evaluation)

Install:  pip install faiss-cpu rank-bm25
Keep this file in the same folder as rag_pipeline.py.
"""

import json
import re
import time
from dataclasses import asdict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from rag_pipeline import RERANK_MODEL, RAG, Chunk, build_chunks, load_documents

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer (keeps identifiers like snake_case_names intact)."""
    return _TOKEN_RE.findall(text.lower())


class HybridRAG(RAG):
    def __init__(self, index_dir: str = "rag_index_hybrid", rrf_k: int = 60, rerank_delay: float = 0.0):
        """
        rrf_k:        RRF smoothing constant (60 is the common default).
        rerank_delay: seconds to sleep after each rerank call (useful on rate-limited plans).
        """
        super().__init__(index_dir)
        self.rrf_k = rrf_k
        self.rerank_delay = rerank_delay
        self.index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None
        self._qcache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Index construction / persistence
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_faiss(emb: np.ndarray, index_type: str) -> faiss.Index:
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        dim = emb.shape[1]
        if index_type == "flat":
            # Exact search. Inner product == cosine because vectors are unit length.
            index = faiss.IndexFlatIP(dim)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
        else:
            raise ValueError("index_type must be 'flat' or 'hnsw'")
        index.add(emb)
        return index

    def build_index(self, docs_dir: str, index_type: str = "flat") -> None:
        docs = load_documents(docs_dir)
        if not docs:
            raise SystemExit(f"No supported documents found in {docs_dir}")
        self.chunks = build_chunks(docs)
        print(f"Loaded {len(docs)} documents -> {len(self.chunks)} chunks")

        emb = self._embed([c.text for c in self.chunks], input_type="document")
        self.index = self._make_faiss(emb, index_type)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "vectors.faiss"))
        with open(self.index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)
        print(f"Saved {index_type} FAISS index + chunks to {self.index_dir}")

    def load_index(self) -> None:
        self.index = faiss.read_index(str(self.index_dir / "vectors.faiss"))
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = 64
        with open(self.index_dir / "chunks.json", encoding="utf-8") as f:
            self.chunks = [Chunk(**c) for c in json.load(f)]
        # BM25 is cheap to rebuild, so it is not persisted.
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    # ------------------------------------------------------------------ #
    # Query embedding cache
    # ------------------------------------------------------------------ #
    def _embed_query(self, query: str) -> np.ndarray:
        if query not in self._qcache:
            self._qcache[query] = self._embed([query], input_type="query")[0]
        return self._qcache[query]

    def warm_query_cache(self, queries: list[str]) -> None:
        """Embed many queries in batched API calls and cache them."""
        missing = [q for q in dict.fromkeys(queries) if q not in self._qcache]
        if missing:
            vectors = self._embed(missing, input_type="query")
            for q, v in zip(missing, vectors):
                self._qcache[q] = v

    # ------------------------------------------------------------------ #
    # Individual retrievers (return [(chunk_index, score), ...], best first)
    # ------------------------------------------------------------------ #
    def _vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self._embed_query(query).astype(np.float32)[None, :]
        scores, ids = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    @staticmethod
    def _rrf(rankings: list[list[tuple[int, float]]], weights: list[float], k: int) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: score(d) = sum_i weight_i / (k + rank_i(d))."""
        fused: dict[int, float] = {}
        for ranking, weight in zip(rankings, weights):
            for rank, (idx, _) in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + weight / (k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])

    def _rerank(self, query: str, docs: list[str], top_k: int):
        for attempt in range(5):
            try:
                res = self.voyage.rerank(
                    query=query, documents=docs, model=RERANK_MODEL, top_k=top_k)
                if self.rerank_delay:
                    time.sleep(self.rerank_delay)
                return res.results
            except Exception as exc:
                if attempt == 4:
                    raise
                wait = 3 * 2**attempt
                print(f"Rerank error ({exc}); retrying in {wait}s...")
                time.sleep(wait)

    # ------------------------------------------------------------------ #
    # Public retrieval API (overrides RAG.retrieve; answer() uses it automatically)
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        candidates: int = 20,
        top_k: int = 5,
        mode: str = "hybrid",       # "vector" | "bm25" | "hybrid"
        rerank: bool = True,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[tuple[Chunk, float]]:
        if mode == "vector":
            fused = self._vector_search(query, candidates)
        elif mode == "bm25":
            fused = self._bm25_search(query, candidates)
        elif mode == "hybrid":
            depth = candidates * 2  # look deeper in each list so fusion has material to work with
            fused = self._rrf(
                [self._vector_search(query, depth),
                 self._bm25_search(query, depth)],
                [vector_weight, bm25_weight],
                self.rrf_k,
            )[:candidates]
        else:
            raise ValueError("mode must be 'vector', 'bm25' or 'hybrid'")

        if not fused:
            return []

        if rerank:
            cand_chunks = [self.chunks[i] for i, _ in fused]
            results = self._rerank(query, [c.text for c in cand_chunks], top_k)
            return [(cand_chunks[r.index], r.relevance_score) for r in results]

        return [(self.chunks[i], score) for i, score in fused[:top_k]]

    # ------------------------------------------------------------------ #
    # Debug helper
    # ------------------------------------------------------------------ #
    def compare_modes(self, query: str, top_k: int = 5, rerank: bool = False) -> None:
        for mode in ("vector", "bm25", "hybrid"):
            print(f"\n=== {mode.upper()} (rerank={rerank}) ===")
            for rank, (chunk, score) in enumerate(self.retrieve(query, top_k=top_k, mode=mode, rerank=rerank), 1):
                preview = chunk.text[:100].replace("\n", " ")
                print(
                    f"{rank}. [{score:.4f}] {chunk.source}::{chunk.chunk_id}  {preview}...")

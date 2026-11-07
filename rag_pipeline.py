"""
Minimal end-to-end RAG over local documents.

Stack
  - Embeddings + reranking: Voyage AI
  - Vector store:           NumPy matrix persisted to disk (swap for FAISS/pgvector at scale)
  - Generation:             Claude via the Anthropic API

Setup
  pip install voyageai anthropic numpy pypdf
  export VOYAGE_API_KEY="..."
  export ANTHROPIC_API_KEY="..."

Usage
  python rag_pipeline.py index --docs ./docs --index ./rag_index
  python rag_pipeline.py ask   --index ./rag_index "How does X work?"
  python rag_pipeline.py chat  --index ./rag_index
"""

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
import numpy as np
import voyageai

# Check the Voyage / Anthropic docs for current model names.
EMBED_MODEL = "voyage-3.5"
RERANK_MODEL = "rerank-2.5"
LLM_MODEL = "claude-sonnet-5-5"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(docs_dir: str) -> dict[str, str]:
    docs = {}
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = read_file(path).strip()
            if text:
                docs[str(path)] = text
    return docs


# --------------------------------------------------------------------------- #
# 2. Chunking (paragraph-aware, with overlap)
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush():
        if buf.strip():
            chunks.append(buf.strip())

    for para in paragraphs:
        # Hard-split paragraphs that are larger than a chunk.
        if len(para) > size:
            pieces = [para[i: i + size]
                      for i in range(0, len(para), size - overlap)]
        else:
            pieces = [para]

        for piece in pieces:
            if len(buf) + len(piece) + 2 <= size:
                buf = f"{buf}\n\n{piece}" if buf else piece
            else:
                flush()
                tail = buf[-overlap:] if buf else ""
                if tail and len(tail) + len(piece) + 2 <= size:
                    buf = f"{tail}\n\n{piece}"
                else:
                    buf = piece
    flush()
    return chunks


def build_chunks(docs: dict[str, str]) -> list[Chunk]:
    chunks = []
    for source, text in docs.items():
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(Chunk(text=piece, source=source, chunk_id=i))
    return chunks


# --------------------------------------------------------------------------- #
# 3. RAG engine
# --------------------------------------------------------------------------- #
class RAG:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.claude = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"])
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    # ---- embedding helpers ------------------------------------------------ #
    def _embed(self, texts: list[str], input_type: str, batch_size: int = 128) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            for attempt in range(5):
                try:
                    res = self.voyage.embed(
                        batch,
                        model=EMBED_MODEL,
                        input_type=input_type,
                        truncation=True,
                    )
                    vectors.extend(res.embeddings)
                    break
                except Exception as exc:  # rate limits, transient errors
                    if attempt == 4:
                        raise
                    wait = 2**attempt
                    print(f"Embed error ({exc}); retrying in {wait}s...")
                    time.sleep(wait)
        arr = np.asarray(vectors, dtype=np.float32)
        # Voyage embeddings are normalized already; this is a cheap safeguard.
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr

    # ---- indexing --------------------------------------------------------- #
    def build_index(self, docs_dir: str) -> None:
        docs = load_documents(docs_dir)
        if not docs:
            raise SystemExit(f"No supported documents found in {docs_dir}")
        self.chunks = build_chunks(docs)
        print(f"Loaded {len(docs)} documents -> {len(self.chunks)} chunks")

        self.embeddings = self._embed(
            [c.text for c in self.chunks], input_type="document")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "embeddings.npy", self.embeddings)
        with open(self.index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)
        print(f"Index saved to {self.index_dir}")

    def load_index(self) -> None:
        self.embeddings = np.load(self.index_dir / "embeddings.npy")
        with open(self.index_dir / "chunks.json", encoding="utf-8") as f:
            self.chunks = [Chunk(**c) for c in json.load(f)]

    # ---- retrieval -------------------------------------------------------- #
    def retrieve(self, query: str, candidates: int = 20, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q = self._embed([query], input_type="query")[0]
        scores = self.embeddings @ q  # cosine similarity (unit vectors)
        top_idx = np.argsort(-scores)[:candidates]
        candidate_chunks = [self.chunks[i] for i in top_idx]

        # Second stage: cross-encoder rerank for precision.
        reranked = self.voyage.rerank(
            query=query,
            documents=[c.text for c in candidate_chunks],
            model=RERANK_MODEL,
            top_k=top_k,
        )
        return [(candidate_chunks[r.index], r.relevance_score) for r in reranked.results]

    # ---- generation ------------------------------------------------------- #
    def answer(self, query: str, history: list[dict] | None = None) -> tuple[str, list[tuple[Chunk, float]]]:
        hits = self.retrieve(query)
        context = "\n\n".join(
            f"[{i}] (source: {c.source})\n{c.text}" for i, (c, _) in enumerate(hits, 1)
        )

        system = (
            "You answer questions using only the provided context. "
            "Cite supporting passages inline like [1] or [2]. "
            "If the context does not contain the answer, say you could not find it "
            "in the documents instead of guessing."
        )
        user_msg = f"<context>\n{context}\n</context>\n\nQuestion: {query}"

        messages = (history or []) + [{"role": "user", "content": user_msg}]
        resp = self.claude.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, hits


# --------------------------------------------------------------------------- #
# 4. CLI
# --------------------------------------------------------------------------- #
def print_answer(text: str, hits: list[tuple[Chunk, float]]) -> None:
    print(f"\n{text}\n")
    print("Sources:")
    for i, (chunk, score) in enumerate(hits, 1):
        print(
            f"  [{i}] {chunk.source} (chunk {chunk.chunk_id}, relevance {score:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG over local docs with Voyage AI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Chunk, embed and store documents")
    p_index.add_argument("--docs", required=True)
    p_index.add_argument("--index", default="./rag_index")

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("--index", default="./rag_index")
    p_ask.add_argument("question")

    p_chat = sub.add_parser("chat", help="Interactive Q&A loop")
    p_chat.add_argument("--index", default="./rag_index")

    args = parser.parse_args()
    rag = RAG(args.index)

    if args.cmd == "index":
        rag.build_index(args.docs)
    elif args.cmd == "ask":
        rag.load_index()
        text, hits = rag.answer(args.question)
        print_answer(text, hits)
    elif args.cmd == "chat":
        rag.load_index()
        print("Type a question (or 'exit').")
        while True:
            q = input("\n> ").strip()
            if q.lower() in {"exit", "quit", ""}:
                break
            text, hits = rag.answer(q)
            print_answer(text, hits)


if __name__ == "__main__":
    main()

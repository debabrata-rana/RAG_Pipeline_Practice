# RAG Pipeline Practice

A lightweight retrieval-augmented generation (RAG) project for indexing local documents and answering questions from them using Voyage AI embeddings, reranking, and Anthropic Claude for generation.

This repo demonstrates both a simple vector-search pipeline and a more advanced hybrid retrieval setup that combines semantic search with BM25 keyword matching.

## What is included

- `rag_pipeline.py`: base end-to-end RAG pipeline
- `rag_hybrid.py`: hybrid retrieval extension with FAISS + BM25 + reranking
- `rag_eval.py`: synthetic evaluation dataset generation and retrieval metrics
- `run_rag_pipeline.ipynb`: notebook walkthrough for building and querying indexes
- `docs/Travel_guide_for_India.pdf`: sample document used for local RAG experiments
- `eval_set.jsonl`: generated evaluation examples

## Project workflow

The base pipeline does the following:

1. Loads supported files (`.txt`, `.md`, `.pdf`) from a directory
2. Splits text into paragraph-aware chunks with overlap
3. Embeds chunks with Voyage AI
4. Stores chunk metadata and embeddings on disk
5. Uses cosine similarity to retrieve relevant chunks for a query
6. Reranks candidate chunks using Voyage's reranker
7. Sends the retrieved context to Claude to answer using only that context

The hybrid version adds:

- FAISS vector search
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF) to combine both signals
- Query embedding caching to reduce repeated API calls
- Retrieval comparison utilities for debugging and benchmarking

## Setup

Install the required libraries:

```bash
pip install voyageai anthropic numpy pypdf
```

For the hybrid pipeline and evaluation tooling:

```bash
pip install faiss-cpu rank-bm25 pandas
```

Set your API keys:

```bash
export VOYAGE_API_KEY="your_voyage_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

## Basic RAG usage

Build an index from local documents:

```bash
python rag_pipeline.py index --docs ./docs --index ./rag_index
```

Ask a single question:

```bash
python rag_pipeline.py ask --index ./rag_index "How does the document describe travel in India?"
```

Run the interactive chat loop:

```bash
python rag_pipeline.py chat --index ./rag_index
```

## Hybrid retrieval usage

The hybrid retrieval implementation is designed to live alongside the base pipeline and uses `rag_index_hybrid` as the default folder.

```python
from rag_hybrid import HybridRAG

rag = HybridRAG(index_dir="rag_index_hybrid")
rag.build_index("./docs", index_type="flat")

answer, hits = rag.answer("Which months are best to visit India?")
print(answer)
```

You can also compare retrieval modes:

```python
rag.compare_modes("Which months are the best to visit India?", top_k=5, rerank=False)
```

## Evaluation

The repository includes a lightweight evaluation harness for measuring retrieval quality.

Create a synthetic evaluation set:

```python
from rag_eval import generate_eval_set

examples = generate_eval_set(rag, n=40, path="eval_set.jsonl")
```

Evaluate multiple retrieval configurations:

```python
from rag_eval import evaluate, load_eval_set

eval_set = load_eval_set("eval_set.jsonl")
summary, details = evaluate(rag, eval_set, ks=(1, 3, 5, 10))
print(summary.round(3))
```

The evaluation reports metrics such as:

- hit@k
- recall@k
- precision@k
- ndcg@k
- MRR (mean reciprocal rank)
- latency in milliseconds

## Notes

- The index files are stored on disk as JSON and NumPy/FAISS artifacts for reuse across runs.
- Document chunking uses paragraph-aware splitting with overlap to preserve context.
- The answer-generation step explicitly avoids guessing when the answer is missing from the indexed context.
- This project is intended as a practice and experimentation repo for learning how local-document RAG systems are built and evaluated.

## Example repository layout

```text
RAG_Pipeline_Practice/
├── rag_pipeline.py
├── rag_hybrid.py
├── rag_eval.py
├── run_rag_pipeline.ipynb
├── docs/
│   └── Travel_guide_for_India.pdf
├── rag_index/
├── rag_index_hybrid/
├── eval_set.jsonl
├── .env
├── .gitignore
└── README.md
```

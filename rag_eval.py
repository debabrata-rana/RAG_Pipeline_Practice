"""
Retrieval evaluation harness for HybridRAG.

Metrics (per query, averaged over the eval set), computed at several cutoffs k:
  hit@k        1 if any relevant chunk is in the top k
  recall@k     relevant chunks retrieved in top k / total relevant chunks in the corpus
  precision@k  relevant chunks in top k / k
  ndcg@k       rank-aware gain with binary relevance
  mrr          mean reciprocal rank of the first relevant chunk (within the largest k)

Eval-set format: a list of dicts (or a JSONL file, one dict per line) with "query" plus
at least one way of saying what counts as relevant:
  "relevant_chunks":  ["<source>::<chunk_id>", ...]   exact chunks (what generate_eval_set produces)
  "relevant_sources": ["handbook.pdf", ...]           any chunk from these files (matched by file name)
  "evidence":         ["30 days", ...]                any chunk containing one of these strings
                                                      (case-insensitive; survives re-chunking)

With doc-level "relevant_sources", many chunks count as relevant, so recall@k and
precision@k get small; focus on hit@k, mrr and ndcg@k in that case.

Install:  pip install pandas
"""

import json
import math
import random
import re
import time
from pathlib import Path

import pandas as pd

from rag_pipeline import LLM_MODEL


# --------------------------------------------------------------------------- #
# Relevance helpers
# --------------------------------------------------------------------------- #
def chunk_key(chunk) -> str:
    return f"{chunk.source}::{chunk.chunk_id}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def relevant_keys(example: dict, chunks: list) -> set[str]:
    """All chunk keys in the corpus that count as relevant for this example."""
    rel_chunks = set(example.get("relevant_chunks", []))
    rel_names = {Path(s).name for s in example.get("relevant_sources", [])}
    evidence = [_norm(e) for e in example.get("evidence", [])]

    keys = set()
    for c in chunks:
        k = chunk_key(c)
        if (
            k in rel_chunks
            or Path(c.source).name in rel_names
            or (evidence and any(e in _norm(c.text) for e in evidence))
        ):
            keys.add(k)
    return keys


# --------------------------------------------------------------------------- #
# Eval-set creation and I/O
# --------------------------------------------------------------------------- #
QGEN_PROMPT = """Below is a passage from a document. Write ONE question that a user might ask \
and that this passage answers.

Rules:
- The question must be self-contained (never say "this passage" or "the text").
- Paraphrase: avoid copying rare keywords or exact phrases from the passage.
- It should be answerable from the passage alone.

Respond with only JSON: {{"question": "..."}}

<passage>
{passage}
</passage>"""


def generate_eval_set(rag, n: int = 30, seed: int = 42, min_chars: int = 300, path: str | None = None) -> list[dict]:
    """Synthetic eval set: sample chunks and have Claude write a question for each.

    Caveat: synthetic questions are a quick proxy, not a substitute for real user
    queries. Review a sample by hand, and add real queries when you have them.
    """
    rng = random.Random(seed)
    pool = [c for c in rag.chunks if len(c.text) >= min_chars]
    sample = rng.sample(pool, min(n, len(pool)))

    examples = []
    for chunk in sample:
        resp = rag.claude.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": QGEN_PROMPT.format(passage=chunk.text)}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            question = json.loads(raw)["question"].strip()
        except (json.JSONDecodeError, KeyError, AttributeError):
            print(f"Skipping unparseable response for {chunk_key(chunk)}")
            continue
        examples.append({"query": question, "relevant_chunks": [chunk_key(chunk)]})

    print(f"Generated {len(examples)} eval examples")
    if path:
        save_eval_set(examples, path)
    return examples


def save_eval_set(examples: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved eval set to {path}")


def load_eval_set(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def score_query(retrieved: list[str], relevant: set[str], ks: tuple[int, ...]) -> dict:
    out = {}
    for k in ks:
        top = retrieved[:k]
        hits = sum(1 for key in top if key in relevant)
        dcg = sum(1 / math.log2(i + 2) for i, key in enumerate(top) if key in relevant)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        out[f"hit@{k}"] = float(hits > 0)
        out[f"recall@{k}"] = hits / len(relevant)
        out[f"precision@{k}"] = hits / k
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

    first = next((i + 1 for i, key in enumerate(retrieved[: max(ks)]) if key in relevant), None)
    out["mrr"] = 1 / first if first else 0.0
    out["first_hit_rank"] = first if first else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def default_configs(candidates: int = 30) -> dict[str, dict]:
    """Retrieval configurations to compare. Keys are labels; values are kwargs for rag.retrieve()."""
    return {
        "vector": dict(mode="vector", rerank=False, candidates=candidates),
        "bm25": dict(mode="bm25", rerank=False, candidates=candidates),
        "hybrid": dict(mode="hybrid", rerank=False, candidates=candidates),
        "vector+rerank": dict(mode="vector", rerank=True, candidates=candidates),
        "hybrid+rerank": dict(mode="hybrid", rerank=True, candidates=candidates),
    }


def evaluate(rag, eval_set: list[dict], configs: dict[str, dict] | None = None, ks: tuple[int, ...] = (1, 3, 5, 10)):
    """
    Returns (summary_df, details).
      summary_df: one row per config, mean metric values plus avg latency
      details:    {config_name: per-query DataFrame}, useful for inspecting failures

    Latency excludes query-embedding time because queries are embedded up front and cached.
    """
    configs = configs or default_configs()
    max_k = max(ks)

    # Precompute relevance sets, dropping examples that match nothing in the current index.
    prepared = []
    for ex in eval_set:
        rel = relevant_keys(ex, rag.chunks)
        if not rel:
            print(f"Warning: no relevant chunks found in index for query: {ex['query'][:80]!r} (skipped)")
            continue
        prepared.append((ex, rel))
    if not prepared:
        raise ValueError("No usable eval examples.")

    rag.warm_query_cache([ex["query"] for ex, _ in prepared])

    rows, details = [], {}
    for name, cfg in configs.items():
        per_query, latencies = [], []
        for ex, rel in prepared:
            t0 = time.perf_counter()
            hits = rag.retrieve(ex["query"], top_k=max_k, **cfg)
            latencies.append((time.perf_counter() - t0) * 1000)
            keys = [chunk_key(c) for c, _ in hits]
            per_query.append({"query": ex["query"], **score_query(keys, rel, ks)})

        df = pd.DataFrame(per_query)
        summary = df.drop(columns=["query", "first_hit_rank"]).mean()
        summary["latency_ms"] = sum(latencies) / len(latencies)
        rows.append(summary.rename(name))
        details[name] = df
        print(f"Finished {name}")

    summary_df = pd.DataFrame(rows)
    print(f"\nEvaluated {len(prepared)} queries")
    return summary_df, details

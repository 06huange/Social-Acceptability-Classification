#!/usr/bin/env python3
import argparse
import heapq
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_DB = Path("experiments/gpt_rag_socialchem101/database_local/socialchem101_rag_project_matched.jsonl")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in", "is",
    "it", "my", "of", "on", "or", "our", "that", "the", "their", "this", "to", "was", "were",
    "with", "you", "your", "situation", "scenario", "culture", "category", "norm",
}


def tokenize(text):
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text or "")
        if token.lower() not in STOPWORDS
    ]


def score_chunk(query_tf, category, chunk):
    text = " ".join([
        chunk.get("rot", ""),
        chunk.get("action", ""),
        chunk.get("source_situation", ""),
        " ".join(chunk.get("project_category_hints", [])),
        " ".join(chunk.get("norm_families", [])),
        " ".join(chunk.get("moral_foundations", [])),
    ])
    doc_tf = Counter(tokenize(text))
    score = 0.0
    for token, q_count in query_tf.items():
        if token in doc_tf:
            score += q_count * min(doc_tf[token], 4)

    if category:
        if category in chunk.get("project_category_hints", []):
            score += 8.0
        if category in chunk.get("norm_families", []):
            score += 4.0

    signal = chunk.get("acceptability_signal", "")
    if signal in {"strongly not acceptable", "strongly acceptable"}:
        score += 0.5
    elif signal == "unknown":
        score -= 1.0
    elif signal:
        score += 0.25
    if chunk.get("annotation_count", 1) > 1:
        score += 0.25
    return score


def retrieve(db_path, query, category, top_k):
    query_tf = Counter(tokenize(query))
    heap = []
    seen = 0
    with db_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            score = score_chunk(query_tf, category, chunk)
            if score <= 0:
                continue
            seen += 1
            item = (score, chunk["chunk_id"], chunk)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    return seen, sorted(heap, reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Stream-retrieve chunks from the Social-Chem-101 RAG corpus.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--culture", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    query = f"scenario: {args.scenario} culture: {args.culture} category: {args.category}"
    seen, results = retrieve(args.db, query, args.category, args.top_k)
    for score, _, chunk in results:
        print(json.dumps({
            "score": round(score, 3),
            "chunk_id": chunk["chunk_id"],
            "acceptability_signal": chunk.get("acceptability_signal", ""),
            "project_category_hints": chunk.get("project_category_hints", []),
            "norm_families": chunk.get("norm_families", []),
            "rot": chunk.get("rot", ""),
            "action": chunk.get("action", ""),
            "retrieval_text": chunk.get("retrieval_text", ""),
        }, ensure_ascii=False))
    if not results:
        print(json.dumps({"matches_scored": seen, "results": []}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


DEFAULT_DB = Path("experiments/gpt_rag_dataset/results/rag_database/rag_norm_database_no_leak.jsonl")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "with", "when", "whether", "usually", "often",
    "generally", "more", "less", "likely", "action", "acceptable", "unacceptable", "context", "social",
}


def tokenize(text):
    return [
        token.lower().replace("-", " ")
        for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", text)
        if token.lower() not in STOPWORDS
    ]


def load_chunks(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_chunks(query, chunks):
    docs = [tokenize(chunk["retrieval_text"] + " " + " ".join(chunk.get("keywords", []))) for chunk in chunks]
    df = Counter()
    for doc in docs:
        df.update(set(doc))

    n_docs = len(docs)
    query_tf = Counter(tokenize(query))
    results = []
    for chunk, doc_tokens in zip(chunks, docs):
        doc_tf = Counter(doc_tokens)
        score = 0.0
        for token, q_count in query_tf.items():
            if token not in doc_tf:
                continue
            idf = math.log((1 + n_docs) / (1 + df[token])) + 1
            score += q_count * doc_tf[token] * idf
        if score > 0:
            results.append((score, chunk))
    return sorted(results, key=lambda item: item[0], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Retrieve social/cultural norm chunks from the local RAG database.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--culture", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chunks = load_chunks(Path(args.db))
    query = f"scenario: {args.scenario} culture: {args.culture} category: {args.category}"
    for score, chunk in score_chunks(query, chunks)[: args.top_k]:
        title = chunk.get("title") or chunk.get("scenario_pattern") or chunk.get("culture_region") or chunk["chunk_id"]
        print(json.dumps({
            "score": round(score, 3),
            "chunk_id": chunk["chunk_id"],
            "title": title,
            "chunk_type": chunk.get("chunk_type", ""),
            "category": chunk.get("category", ""),
            "norm_statement": chunk.get("norm_statement", chunk.get("orientation", "")),
            "retrieval_text": chunk["retrieval_text"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()

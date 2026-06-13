#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


DEFAULT_DATASET = Path("data/processed/CS263_dataset_with_predictions.csv")
DEFAULT_RAG_DB = Path("experiments/gpt_rag_dataset/results/rag_database/rag_norm_database_no_leak.jsonl")
DEFAULT_OUT_DIR = Path("experiments/gpt_rag_dataset/results/rag_experiment")
LABELS = {"acceptable", "not acceptable", "context-dependent"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in", "is", "it",
    "my", "of", "on", "or", "our", "that", "the", "their", "this", "to", "was", "were", "with",
    "when", "whether", "usually", "often", "generally", "more", "less", "likely", "action",
    "acceptable", "unacceptable", "context", "social",
}


def tokenize(text):
    return [
        token.lower().replace("-", " ")
        for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", text)
        if token.lower() not in STOPWORDS
    ]


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(chunks):
    docs = [tokenize(chunk["retrieval_text"] + " " + " ".join(chunk.get("keywords", []))) for chunk in chunks]
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    return docs, df


def retrieve(query, chunks, docs, df, top_k):
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
    return sorted(results, key=lambda item: item[0], reverse=True)[:top_k]


def chunk_title(chunk):
    return chunk.get("title") or chunk.get("scenario_pattern") or chunk.get("culture_region") or chunk["chunk_id"]


def build_prompt(
    situation,
    retrieved,
    culture="",
    cultural_context="",
    decomposition_prompt=False,
    strict_context_dependent=False,
    rating_prompt=False,
):
    snippets = []
    for i, (score, chunk) in enumerate(retrieved, start=1):
        snippets.append(
            f"[Retrieved norm {i} | {chunk['chunk_id']} | {chunk_title(chunk)} | score={score:.3f}]\n"
            f"{chunk['retrieval_text']}"
        )
    retrieved_text = "\n\n".join(snippets) if snippets else "No relevant norms retrieved."
    if rating_prompt:
        system = (
            "You are a social acceptability classifier.\n\n"
            "Rate the social acceptability of the user's situation from 1 to 10:\n"
            "1 = clearly socially unacceptable\n"
            "10 = clearly socially acceptable\n"
            "5 = genuinely ambiguous, mixed, or highly dependent on missing context\n\n"
            "Use only the situation, cultural/social context, and retrieved norm context included in the user message. "
            "Do not infer hidden cultural context beyond those inputs.\n\n"
            "After assigning a score, map it to exactly one label using this threshold:\n"
            "1-3 = not acceptable\n"
            "4-7 = context-dependent\n"
            "8-10 = acceptable\n\n"
            "Return only JSON in this format: "
            '{"score": 5, "label": "context-dependent", "reason": "..."}'
        )
    elif decomposition_prompt:
        if strict_context_dependent:
            ambiguity_rules = (
                "5. Do not choose context-dependent just because additional details could hypothetically matter.\n"
                "6. Choose context-dependent only if one of these is true:\n"
                "   a. the provided cultural/social context explicitly says expectations vary, depend, may change, are ambiguous, or are conditional; or\n"
                "   b. the retrieved norms provide both plausible acceptable and not acceptable conditions for this same scenario pattern; and\n"
                "   c. the scenario lacks the specific variable needed to choose between those conditions.\n"
                "7. If the cultural/social context gives a clear general judgment, choose acceptable or not acceptable even when minor details are missing.\n"
                "8. Missing variables must be central and norm-relevant, not merely imaginable edge cases.\n\n"
            )
        else:
            ambiguity_rules = (
                "5. A missing variable is important only if plausible values of it could change the judgment under the given cultural/social context.\n"
                "6. Do not choose context-dependent merely because some details are missing; ordinary missing background details are not enough.\n"
                "7. Choose context-dependent when the cultural/social context or retrieved norms indicate that acceptability varies, depends on conditions, "
                "or is ambiguous, and the missing variable is central enough to plausibly flip the judgment.\n"
                "8. Choose acceptable or not acceptable when the provided cultural/social context gives a clear general judgment, even if minor details are missing.\n\n"
            )
        system = (
            "You are a social acceptability classifier.\n\n"
            "Classify the user's situation into exactly one label:\n\n"
            "acceptable\n"
            "not acceptable\n"
            "context-dependent\n\n"
            "Use only the situation, cultural/social context, and retrieved norm context included in the user message. "
            "Do not infer hidden cultural context beyond those inputs.\n\n"
            "Decision procedure:\n"
            "1. Identify the main social/cultural norm involved.\n"
            "2. Treat the provided cultural/social context description as the primary evidence. If it gives a clear general judgment "
            "such as generally accepted, commonly respected, expected, rude, disrespectful, intrusive, or not justified, do not override "
            "that with hypothetical missing details.\n"
            "3. Identify decision-relevant variables, such as relationship closeness, hierarchy/status, public vs private setting, "
            "tone, consent/permission, age/status difference, reason/necessity, frequency, severity/harm, prior agreement, "
            "resource constraints, and cultural formality.\n"
            "4. Determine whether any important decision-relevant variable is missing.\n"
            f"{ambiguity_rules}"
            "Return only JSON in this format: "
            '{"norm": "...", "missing_decision_variables": ["..."], '
            '"could_missing_info_flip_label": true, "label": "..."}'
        )
    else:
        system = (
            "You are a social acceptability classifier.\n\n"
            "Classify the user's situation into exactly one label:\n\n"
            "acceptable\n"
            "not acceptable\n"
            "context-dependent\n\n"
            "Use only the situation and the retrieved norm context included in the user message. "
            "Do not infer hidden cultural context beyond those inputs.\n"
            'Return only JSON in this format: {"label": "..."}'
        )
    context_block = ""
    if culture or cultural_context:
        context_block = (
            f"Cultural/social context label:\n{culture}\n\n"
            f"Cultural/social context description:\n{cultural_context}\n\n"
        )
    user = (
        f"Situation:\n{situation}\n\n"
        f"{context_block}"
        f"Retrieved norm context:\n{retrieved_text}\n\n"
        "Return only the JSON label."
    )
    return system, user


def parse_label(text):
    try:
        data = json.loads(text)
        label = str(data.get("label", "")).strip().lower()
        if label in LABELS:
            return label, data
    except json.JSONDecodeError:
        pass

    lowered = text.strip().lower()
    for label in LABELS:
        if lowered == label or f'"{label}"' in lowered:
            return label, {"label": label, "raw": text}
    return "parse_error", {"raw": text}


def call_openai(api_key, model, system, user, max_tokens=40, max_retries=4):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(max_retries):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenAI API error {error.code}: {body[:500]}") from error
        except urllib.error.URLError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("OpenAI API call failed after retries")


def load_existing_trace(path):
    if not path.exists():
        return {}
    done = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            done[item["id"]] = item
    return done


def main():
    parser = argparse.ArgumentParser(description="Run GPT-4o mini scenario+RAG social acceptability experiment.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--rag-db", default=str(DEFAULT_RAG_DB))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--disable-rag",
        action="store_true",
        help="Do not retrieve or include RAG chunks; useful for direct scenario/context baselines.",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include culture and cultural_context in retrieval query and GPT prompt.",
    )
    parser.add_argument(
        "--decomposition-prompt",
        action="store_true",
        help="Ask the model to identify norms and missing decision variables before labeling.",
    )
    parser.add_argument(
        "--strict-context-dependent",
        action="store_true",
        help="Use stricter context-dependent criteria with the decomposition prompt.",
    )
    parser.add_argument(
        "--rating-prompt",
        action="store_true",
        help="Ask the model for a 1-10 acceptability score and threshold it into a label.",
    )
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests.")
    parser.add_argument("--rerun-parse-errors", action="store_true", help="Do not reuse cached parse_error rows.")
    parser.add_argument("--api-key-stdin", action="store_true", help="Read API key from stdin instead of environment.")
    args = parser.parse_args()

    api_key = sys.stdin.readline().strip() if args.api_key_stdin else os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key-stdin.")

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context_name = "scenario_context" if args.include_context else "scenario"
    rag_name = "no_rag" if args.disable_rag else "rag"
    if args.rating_prompt:
        run_name = f"gpt4o_mini_{context_name}_{rag_name}_rating"
        prediction_column = f"gpt_{context_name}_{rag_name}_rating"
    elif args.decomposition_prompt and args.strict_context_dependent:
        run_name = "gpt4o_mini_scenario_context_rag_v2_decomposition_strict"
        prediction_column = "gpt_scenario_context_rag_v2_decomposition_strict"
    elif args.decomposition_prompt:
        run_name = "gpt4o_mini_scenario_context_rag_v2_decomposition"
        prediction_column = "gpt_scenario_context_rag_v2_decomposition"
    elif args.include_context:
        run_name = "gpt4o_mini_scenario_context_rag"
        prediction_column = "gpt_scenario_context_rag"
    else:
        run_name = "gpt4o_mini_scenario_rag"
        prediction_column = "gpt_scenario_rag"
    trace_path = out_dir / f"{run_name}_trace.jsonl"
    augmented_path = out_dir / f"CS263_dataset_with_predictions_{prediction_column}.csv"
    metrics_path = out_dir / f"{run_name}_metrics.json"

    with dataset_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    if args.limit:
        rows = rows[: args.limit]

    chunks = [] if args.disable_rag else load_jsonl(Path(args.rag_db))
    docs, df = ([], Counter()) if args.disable_rag else build_index(chunks)
    existing = load_existing_trace(trace_path)

    with trace_path.open("a", encoding="utf-8") as trace:
        for index, row in enumerate(rows, start=1):
            if row["id"] in existing and not (args.rerun_parse_errors and existing[row["id"]].get("predicted_label") == "parse_error"):
                print(f"[{index}/{len(rows)}] {row['id']} cached -> {existing[row['id']]['predicted_label']}")
                continue

            if args.include_context:
                query = (
                    f"scenario: {row['situation']} "
                    f"culture/social context: {row['culture']} "
                    f"context description: {row['cultural_context']}"
                )
            else:
                query = row["situation"]
            retrieved = [] if args.disable_rag else retrieve(query, chunks, docs, df, args.top_k)
            system, user = build_prompt(
                row["situation"],
                retrieved,
                culture=row["culture"] if args.include_context else "",
                cultural_context=row["cultural_context"] if args.include_context else "",
                decomposition_prompt=args.decomposition_prompt,
                strict_context_dependent=args.strict_context_dependent,
                rating_prompt=args.rating_prompt,
            )
            api_response = call_openai(
                api_key,
                args.model,
                system,
                user,
                max_tokens=220 if args.decomposition_prompt or args.rating_prompt else 40,
            )
            content = api_response["choices"][0]["message"]["content"]
            label, parsed = parse_label(content)

            item = {
                "id": row["id"],
                "row_index": index,
                "situation": row["situation"],
                "gold_label": row["label"],
                "retrieval_query": query,
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "score": round(score, 6),
                        "chunk_id": chunk["chunk_id"],
                        "chunk_type": chunk.get("chunk_type", ""),
                        "source_id": chunk.get("source_id", ""),
                        "source_ids": chunk.get("source_ids", []),
                        "title": chunk_title(chunk),
                        "category": chunk.get("category", ""),
                        "culture_region": chunk.get("culture_region", ""),
                        "cultures_compared": chunk.get("cultures_compared", []),
                        "norm_type": chunk.get("norm_type", ""),
                        "norm_statement": chunk.get("norm_statement", chunk.get("orientation", "")),
                        "retrieval_text": chunk["retrieval_text"],
                    }
                    for rank, (score, chunk) in enumerate(retrieved, start=1)
                ],
                "prompt_system": system,
                "prompt_user": user,
                "raw_response": content,
                "parsed_response": parsed,
                "predicted_label": label,
                "model": args.model,
            }
            trace.write(json.dumps(item, ensure_ascii=False) + "\n")
            trace.flush()
            existing[row["id"]] = item
            print(f"[{index}/{len(rows)}] {row['id']} -> {label} (gold: {row['label']})")
            time.sleep(args.sleep)

    predictions = load_existing_trace(trace_path)
    augmented_fields = fieldnames + [prediction_column]
    with augmented_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=augmented_fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out[prediction_column] = predictions.get(row["id"], {}).get("predicted_label", "")
            writer.writerow(out)

    evaluated = [item for item in predictions.values() if item["id"] in {row["id"] for row in rows}]
    correct = sum(1 for item in evaluated if item["predicted_label"] == item["gold_label"])
    total = len(evaluated)
    metrics = {
        "model": args.model,
        "setup": (
            "scenario + culture + cultural_context + retrieved norms + 1-10 acceptability rating prompt"
            if args.rating_prompt and not args.disable_rag and args.include_context
            else "scenario + retrieved norms + 1-10 acceptability rating prompt"
            if args.rating_prompt and not args.disable_rag
            else "scenario + culture + cultural_context + 1-10 acceptability rating prompt"
            if args.rating_prompt and args.disable_rag and args.include_context
            else "scenario + 1-10 acceptability rating prompt"
            if args.rating_prompt and args.disable_rag
            else
            "scenario + culture + cultural_context + retrieved norms + strict ambiguity-aware decomposition prompt"
            if args.decomposition_prompt and args.strict_context_dependent
            else "scenario + culture + cultural_context + retrieved norms + ambiguity-aware decomposition prompt"
            if args.decomposition_prompt
            else
            "scenario + culture + cultural_context + retrieved norms; retrieval query uses situation, culture, and cultural_context"
            if args.include_context
            else "scenario + retrieved norms; retrieval query uses situation only; no dataset cultural_context in prompt"
        ),
        "prediction_column": prediction_column,
        "top_k": args.top_k,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "trace_path": str(trace_path.resolve()),
        "augmented_dataset_path": str(augmented_path.resolve()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

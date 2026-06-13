#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DATASET = Path("data/processed/CS263_dataset_with_predictions.csv")
DEFAULT_RAG_DB = Path("experiments/gpt_rag_socialchem101/database_local/socialchem101_rag_project_matched.jsonl")
DEFAULT_OUT_DIR = Path("experiments/gpt_rag_socialchem101/results/rag_experiment_socialchem101_reasoning_gpt4o")
LABELS = {"acceptable", "not acceptable", "context-dependent"}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in", "is",
    "it", "my", "of", "on", "or", "our", "that", "the", "their", "this", "to", "was", "were",
    "with", "you", "your", "situation", "scenario", "culture", "category", "norm", "social",
    "acceptable", "acceptability", "context", "dependent",
}


SYSTEM_PROMPT = """You are a careful social acceptability classifier for a research experiment.

Classify the situation into exactly one label:
- acceptable
- not acceptable
- context-dependent

You will receive:
1. A scenario.
2. A culture/social-context label from the dataset.
3. A culture/social-context description from the dataset.
4. Retrieved Social-Chem-101 rule-of-thumb chunks.

Important evidence rules:
- Treat the dataset culture/social-context description as the primary culture-specific evidence.
- Treat Social-Chem-101 retrieved norms as broad social and moral norm evidence, not as country-specific cultural evidence unless a retrieved chunk explicitly names a country/culture.
- Do not use the gold label, confidence score, or prior model predictions.
- Do not invent hidden facts. If a needed fact is missing, list it as a missing decision variable.
- If the retrieved norms are irrelevant, contradictory, too generic, or culturally mismatched, say so in the output instead of forcing them to support the answer.
- Prefer context-dependent only when the given context or retrieved norms show that the judgment genuinely depends on missing central details, or when the culture-specific evidence and broad social norms point in different plausible directions.

You must expose intermediate RAG-grounded findings before the final label. Each intermediate finding should be a short normalized norm such as "keeping promises is good", "stealing is bad", "public embarrassment is discouraged", "asking private questions requires closeness or consent", or "respecting workplace hierarchy is expected".

Return only valid JSON. Do not include markdown, prose outside JSON, or trailing comments."""


USER_TEMPLATE = """Scenario:
{scenario}

Culture/social-context label:
{culture}

Culture/social-context description:
{cultural_context}

Retrieved Social-Chem-101 norms:
{retrieved_norms}

Task:
Use the retrieved norms to extract intermediate findings, evaluate whether they apply to the scenario, then classify the scenario.

Return JSON with exactly this schema:
{{
  "rag_intermediate_findings": [
    {{
      "finding": "short normalized norm claim",
      "source_chunk_id": "chunk id from retrieved norm",
      "support_from_rag": "direct | indirect | weak | irrelevant",
      "applies_to_scenario": true,
      "polarity": "supports_acceptable | supports_not_acceptable | supports_context_dependent | background_only",
      "note": "brief explanation of why this norm does or does not apply"
    }}
  ],
  "rag_quality": {{
    "overall_relevance": "high | medium | low",
    "has_conflicting_norms": true,
    "has_culture_specific_evidence": true,
    "limitations": ["brief limitation"]
  }},
  "culture_context_interpretation": {{
    "primary_cultural_norm": "brief summary of what the dataset culture/context says",
    "points_toward": "acceptable | not acceptable | context-dependent | unclear",
    "note": "brief explanation"
  }},
  "missing_decision_variables": [
    "relationship closeness, consent, public/private setting, hierarchy, tone, frequency, severity, prior agreement, or other central missing variable"
  ],
  "decision": {{
    "label": "acceptable | not acceptable | context-dependent",
    "confidence": "high | medium | low",
    "reason": "one concise paragraph connecting scenario, culture/context, and RAG findings"
  }}
}}

Additional constraints:
- Include at most 5 rag_intermediate_findings.
- Use only retrieved chunk IDs that appear in the input.
- If no retrieved norm is useful, return an empty rag_intermediate_findings list and explain the limitation in rag_quality.limitations.
- The final label must be exactly one of: acceptable, not acceptable, context-dependent."""


def tokenize(text):
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text or "")
        if token.lower() not in STOPWORDS
    ]


def parse_situation(interaction):
    match = re.search(r"User:\s*(.*?)(?:\\n|\n|$)", interaction or "")
    return match.group(1).strip() if match else (interaction or "").strip()


def normalize_label(value):
    label = str(value or "").strip().lower()
    if label == "unacceptable":
        return "not acceptable"
    return label if label in LABELS else "parse_error"


def load_chunks(path):
    chunks = []
    docs = []
    df = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            text = " ".join([
                chunk.get("rot", ""),
                chunk.get("action", ""),
                chunk.get("source_situation", ""),
                " ".join(chunk.get("project_category_hints", [])),
                " ".join(chunk.get("norm_families", [])),
                " ".join(chunk.get("moral_foundations", [])),
                chunk.get("acceptability_signal", ""),
            ])
            tokens = tokenize(text)
            chunks.append(chunk)
            docs.append(Counter(tokens))
            df.update(set(tokens))
    return chunks, docs, df


def retrieve(row, chunks, docs, df, top_k):
    query = " ".join([
        row["situation"],
        row["category"],
        row["culture"],
        row["cultural_context"],
    ])
    query_tf = Counter(tokenize(query))
    n_docs = len(chunks)
    heap = []
    category = row.get("category", "")
    for index, (chunk, doc_tf) in enumerate(zip(chunks, docs)):
        score = 0.0
        for token, q_count in query_tf.items():
            if token not in doc_tf:
                continue
            idf = math.log((1 + n_docs) / (1 + df[token])) + 1
            score += q_count * min(doc_tf[token], 4) * idf

        if category in chunk.get("project_category_hints", []):
            score += 8.0
        if category in chunk.get("norm_families", []):
            score += 4.0

        signal = chunk.get("acceptability_signal", "")
        if signal == "unknown":
            score -= 1.0
        elif signal in {"strongly not acceptable", "strongly acceptable"}:
            score += 0.5
        elif signal:
            score += 0.25

        if score <= 0:
            continue
        item = (score, index, chunk)
        if len(heap) < top_k:
            heapq_push(heap, item)
        elif item > heap[0]:
            heapq_replace(heap, item)
    return sorted(heap, reverse=True)


def heapq_push(heap, item):
    import heapq

    heapq.heappush(heap, item)


def heapq_replace(heap, item):
    import heapq

    heapq.heapreplace(heap, item)


def compact_chunk(rank, score, chunk):
    return {
        "rank": rank,
        "score": round(score, 6),
        "chunk_id": chunk["chunk_id"],
        "rot": chunk.get("rot", ""),
        "action": chunk.get("action", ""),
        "acceptability_signal": chunk.get("acceptability_signal", ""),
        "moral_foundations": chunk.get("moral_foundations", []),
        "norm_families": chunk.get("norm_families", []),
        "project_category_hints": chunk.get("project_category_hints", []),
        "context_dependent_cues": chunk.get("context_dependent_cues", []),
        "source_situation": chunk.get("source_situation", ""),
    }


def format_retrieved(retrieved):
    if not retrieved:
        return "No relevant Social-Chem-101 norms retrieved."
    blocks = []
    for rank, (score, _, chunk) in enumerate(retrieved, start=1):
        blocks.append(
            f"[Retrieved norm {rank} | {chunk['chunk_id']} | score={score:.3f}]\n"
            f"Rule of thumb: {chunk.get('rot', '')}\n"
            f"Action: {chunk.get('action', '')}\n"
            f"Acceptability signal: {chunk.get('acceptability_signal', '')}\n"
            f"Moral foundations: {', '.join(chunk.get('moral_foundations', [])) or 'none'}\n"
            f"Norm families: {', '.join(chunk.get('norm_families', [])) or 'none'}\n"
            f"Project category hints: {', '.join(chunk.get('project_category_hints', [])) or 'none'}\n"
            f"Context-dependent cues: {'; '.join(chunk.get('context_dependent_cues', [])) or 'none'}\n"
            f"Source situation: {chunk.get('source_situation', '')}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(row, retrieved):
    return USER_TEMPLATE.format(
        scenario=row["situation"],
        culture=row["culture"],
        cultural_context=row["cultural_context"],
        retrieved_norms=format_retrieved(retrieved),
    )


def call_openai(api_key, model, user, max_tokens, max_retries=5):
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    context = None
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()
    for attempt in range(max_retries):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90, context=context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenAI API error {error.code}: {body[:800]}") from error
        except urllib.error.URLError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("OpenAI API call failed after retries")


def parse_response(content):
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return "parse_error", {"raw": content}
    label = normalize_label(data.get("decision", {}).get("label", data.get("label", "")))
    return label, data


def load_existing_trace(path):
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rows[item["id"]] = item
    return rows


def write_outputs(rows, trace_items, out_dir, model, top_k, prediction_column):
    trace_by_id = {item["id"]: item for item in trace_items}
    augmented_path = out_dir / f"CS263_dataset_with_predictions_{prediction_column}.csv"
    trace_path = out_dir / f"{prediction_column}_trace.jsonl"
    metrics_path = out_dir / f"{prediction_column}_metrics.json"
    findings_path = out_dir / f"{prediction_column}_rag_findings.csv"

    fieldnames = list(rows[0].keys()) + [prediction_column]
    with augmented_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out[prediction_column] = trace_by_id.get(row["id"], {}).get("predicted_label", "")
            writer.writerow(out)

    finding_fields = [
        "id", "gold_label", "predicted_label", "finding_index", "finding",
        "source_chunk_id", "support_from_rag", "applies_to_scenario", "polarity", "note",
    ]
    with findings_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=finding_fields)
        writer.writeheader()
        for item in trace_items:
            findings = item.get("parsed_response", {}).get("rag_intermediate_findings", [])
            for index, finding in enumerate(findings, start=1):
                writer.writerow({
                    "id": item["id"],
                    "gold_label": item["gold_label"],
                    "predicted_label": item["predicted_label"],
                    "finding_index": index,
                    "finding": finding.get("finding", ""),
                    "source_chunk_id": finding.get("source_chunk_id", ""),
                    "support_from_rag": finding.get("support_from_rag", ""),
                    "applies_to_scenario": finding.get("applies_to_scenario", ""),
                    "polarity": finding.get("polarity", ""),
                    "note": finding.get("note", ""),
                })

    evaluated = [item for item in trace_items if item["predicted_label"] in LABELS]
    correct = sum(item["predicted_label"] == item["gold_label"] for item in evaluated)
    by_label = {}
    for label in sorted(LABELS):
        subset = [item for item in evaluated if item["gold_label"] == label]
        by_label[label] = {
            "correct": sum(item["predicted_label"] == item["gold_label"] for item in subset),
            "total": len(subset),
            "accuracy": (
                sum(item["predicted_label"] == item["gold_label"] for item in subset) / len(subset)
                if subset else None
            ),
        }

    confusion = {gold: {pred: 0 for pred in sorted(LABELS)} for gold in sorted(LABELS)}
    for item in evaluated:
        confusion[item["gold_label"]][item["predicted_label"]] += 1

    rag_relevance = Counter(
        item.get("parsed_response", {}).get("rag_quality", {}).get("overall_relevance", "missing")
        for item in evaluated
    )
    confidence = Counter(
        item.get("parsed_response", {}).get("decision", {}).get("confidence", "missing")
        for item in evaluated
    )

    metrics = {
        "model": model,
        "setup": "scenario + culture column + cultural_context column + Social-Chem-101 RAG + exposed RAG intermediate findings",
        "prediction_column": prediction_column,
        "top_k": top_k,
        "total": len(evaluated),
        "correct": correct,
        "accuracy": correct / len(evaluated) if evaluated else None,
        "by_gold_label": by_label,
        "confusion_matrix_gold_to_predicted": confusion,
        "rag_relevance_counts": dict(rag_relevance),
        "decision_confidence_counts": dict(confidence),
        "trace_path": str(trace_path.resolve()),
        "augmented_dataset_path": str(augmented_path.resolve()),
        "findings_path": str(findings_path.resolve()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run GPT-4o Social-Chem-101 RAG reasoning experiment.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rag-db", type=Path, default=DEFAULT_RAG_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rerun-parse-errors", action="store_true")
    parser.add_argument("--api-key-stdin", action="store_true")
    args = parser.parse_args()

    api_key = sys.stdin.readline().strip() if args.api_key_stdin else os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key-stdin.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_column = "gpt4o_socialchem101_rag_reasoning"
    trace_path = args.out_dir / f"{prediction_column}_trace.jsonl"

    with args.dataset.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.setdefault("situation", parse_situation(row.get("interaction", "")))
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loading Social-Chem-101 RAG database: {args.rag_db}", flush=True)
    chunks, docs, df = load_chunks(args.rag_db)
    print(f"Loaded {len(chunks)} chunks", flush=True)

    existing = load_existing_trace(trace_path)
    with trace_path.open("a", encoding="utf-8") as trace:
        for index, row in enumerate(rows, start=1):
            cached = existing.get(row["id"])
            if cached and not (args.rerun_parse_errors and cached.get("predicted_label") == "parse_error"):
                print(f"[{index}/{len(rows)}] {row['id']} cached -> {cached['predicted_label']}", flush=True)
                continue

            retrieved = retrieve(row, chunks, docs, df, args.top_k)
            user_prompt = build_user_prompt(row, retrieved)
            api_response = call_openai(api_key, args.model, user_prompt, args.max_tokens)
            content = api_response["choices"][0]["message"]["content"]
            label, parsed = parse_response(content)

            item = {
                "id": row["id"],
                "row_index": index,
                "category": row["category"],
                "situation": row["situation"],
                "culture": row["culture"],
                "cultural_context": row["cultural_context"],
                "gold_label": normalize_label(row["label"]),
                "retrieved_chunks": [
                    compact_chunk(rank, score, chunk)
                    for rank, (score, _, chunk) in enumerate(retrieved, start=1)
                ],
                "prompt_system": SYSTEM_PROMPT,
                "prompt_user": user_prompt,
                "raw_response": content,
                "parsed_response": parsed,
                "predicted_label": label,
                "model": args.model,
            }
            trace.write(json.dumps(item, ensure_ascii=False) + "\n")
            trace.flush()
            existing[row["id"]] = item
            print(f"[{index}/{len(rows)}] {row['id']} -> {label} (gold: {row['label']})", flush=True)
            time.sleep(args.sleep)

    trace_items = list(load_existing_trace(trace_path).values())
    row_ids = {row["id"] for row in rows}
    trace_items = [item for item in trace_items if item["id"] in row_ids]
    metrics = write_outputs(rows, trace_items, args.out_dir, args.model, args.top_k, prediction_column)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

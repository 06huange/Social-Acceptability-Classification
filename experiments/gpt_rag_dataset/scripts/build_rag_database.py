#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path


SOURCE = Path("data/processed/CS263_dataset_with_predictions.csv")
OUT_DIR = Path("experiments/gpt_rag_dataset/results/rag_database")


CATEGORY_VARIABLES = {
    "family": ["actor", "family member", "age/status difference", "public/private setting", "autonomy vs obligation", "tone"],
    "relationship": ["actor", "partner/friend", "privacy", "trust", "consent", "public/private setting", "closeness"],
    "workplace": ["employee", "coworker/manager", "hierarchy", "professional role", "public/private setting", "policy expectations"],
    "communication": ["speaker", "listener", "directness", "face-saving", "tone", "public/private setting"],
    "social_etiquette": ["guest/host", "relationship closeness", "public/private setting", "reciprocity", "punctuality", "respect"],
    "privacy": ["actor", "target", "personal information", "consent", "relationship closeness", "public/private setting"],
    "friendship": ["friend", "closeness", "reciprocity", "support expectation", "honesty", "boundary"],
    "morality_fairness": ["actor", "affected party", "fairness", "honesty", "harm", "rule expectation"],
    "care_harm": ["actor", "vulnerable person", "harm risk", "care duty", "available alternatives", "urgency"],
    "morality_loyalty": ["group member", "group", "loyalty expectation", "honesty", "conflict of interest", "relationship"],
    "authority": ["subordinate", "authority figure", "hierarchy", "respect", "public/private setting", "tone"],
    "fairness": ["actor", "recipient", "equity", "rule consistency", "resource distribution", "transparency"],
    "commitment": ["actor", "recipient", "promise/obligation", "reliability", "relationship", "reason for breaking commitment"],
}


KEYWORD_HINTS = {
    "family": ["family", "parents", "relatives", "elders", "marriage", "support"],
    "relationship": ["partner", "dating", "trust", "privacy", "jealousy", "family introduction"],
    "workplace": ["workplace", "manager", "boss", "coworker", "feedback", "overtime"],
    "communication": ["directness", "tone", "feedback", "politeness", "harmony", "face"],
    "social_etiquette": ["etiquette", "host", "guest", "punctuality", "gift", "phone", "attention"],
    "privacy": ["privacy", "consent", "personal information", "phone", "photo", "public"],
    "friendship": ["friend", "support", "reciprocity", "boundaries", "honesty"],
    "morality_fairness": ["fairness", "honesty", "cheating", "credit", "rules"],
    "care_harm": ["care", "harm", "safety", "help", "vulnerability"],
    "morality_loyalty": ["loyalty", "group", "family", "team", "betrayal"],
    "authority": ["authority", "respect", "hierarchy", "teacher", "elder", "manager"],
    "fairness": ["fairness", "equity", "rules", "distribution", "transparency"],
    "commitment": ["commitment", "promise", "reliability", "agreement", "obligation"],
}


def parse_culture(value):
    parts = [p.strip() for p in value.split(",", 1)]
    region = parts[0]
    norm = parts[1] if len(parts) > 1 else ""
    norm = re.sub(r"\s+norm$", "", norm.strip())
    return region, norm


def title_case_norm(norm):
    if not norm:
        return "Social Norm"
    return norm.replace("-", " ").title()


def extract_keywords(row, region, norm):
    text = " ".join([row["category"], row["culture"], row["cultural_context"], norm]).lower()
    words = set(KEYWORD_HINTS.get(row["category"], []))
    words.update([region.lower(), norm.replace("-", " ").lower()])
    for token in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text):
        if token not in {"acceptable", "acceptability", "depends", "generally", "commonly", "usually", "important", "depending"}:
            words.add(token.replace("-", " "))
    return sorted(words)[:18]


def make_chunk(row, include_example=False):
    region, norm = parse_culture(row["culture"])
    category = row["category"]
    variables = CATEGORY_VARIABLES.get(category, ["actor", "target", "relationship", "setting", "norm", "possible harm"])
    norm_title = title_case_norm(norm)

    norm_statement = row["cultural_context"].strip()
    ambiguity = (
        "Key ambiguity factors include relationship closeness, tone, consent, public vs private setting, "
        "relative status, and whether the action is routine, invited, necessary, or repeated."
    )
    acceptable_when = (
        "The action is more likely to be acceptable when it respects the relevant relationship, minimizes embarrassment "
        "or harm, follows local expectations, and is communicated with appropriate consent or explanation."
    )
    unacceptable_when = (
        "The action is more likely to be unacceptable when it violates privacy, respect, hierarchy, reciprocity, "
        "commitment, or care expectations, especially in public or with a vulnerable/lower-power target."
    )

    retrieval_text = (
        f"Culture or social context: {region}. Norm domain: {norm_title}. "
        f"Situation category: {category.replace('_', ' ')}. Norm: {norm_statement} "
        f"Relevant latent variables: {', '.join(variables)}. "
        f"Acceptable when: {acceptable_when} "
        f"Unacceptable when: {unacceptable_when} "
        f"Ambiguity factors: {ambiguity}"
    )

    chunk = {
        "chunk_id": f"RAG_{row['id']}",
        "source_id": row["id"],
        "category": category,
        "culture_region": region,
        "norm_type": norm,
        "title": f"{region}: {norm_title}",
        "norm_statement": norm_statement,
        "latent_variables": variables,
        "acceptable_when": acceptable_when,
        "unacceptable_when": unacceptable_when,
        "ambiguity_factors": ambiguity,
        "keywords": extract_keywords(row, region, norm),
        "retrieval_text": retrieval_text,
    }
    if include_example:
        chunk["example_situation"] = row["situation"]
    return chunk


def write_jsonl(path, chunks):
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def write_csv(path, chunks):
    fields = [
        "chunk_id",
        "source_id",
        "category",
        "culture_region",
        "norm_type",
        "title",
        "norm_statement",
        "latent_variables",
        "acceptable_when",
        "unacceptable_when",
        "ambiguity_factors",
        "keywords",
        "retrieval_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for chunk in chunks:
            row = dict(chunk)
            row["latent_variables"] = "; ".join(row["latent_variables"])
            row["keywords"] = "; ".join(row["keywords"])
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(chunks):
    def counts(key):
        out = {}
        for chunk in chunks:
            out[chunk[key]] = out.get(chunk[key], 0) + 1
        return dict(sorted(out.items(), key=lambda item: (-item[1], item[0])))

    return {
        "chunk_count": len(chunks),
        "category_counts": counts("category"),
        "culture_region_counts": counts("culture_region"),
        "notes": [
            "rag_norm_database_no_leak.jsonl excludes labels, model predictions, confidence scores, and exact situations.",
            "rag_norm_database_with_examples.jsonl adds example_situation for audit/debugging; avoid using it for final test evaluation.",
            "Each chunk is derived from the dataset's cultural_context field and generalized with latent-variable retrieval fields.",
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    safe_chunks = [make_chunk(row, include_example=False) for row in rows]
    debug_chunks = [make_chunk(row, include_example=True) for row in rows]

    write_jsonl(OUT_DIR / "rag_norm_database_no_leak.jsonl", safe_chunks)
    write_jsonl(OUT_DIR / "rag_norm_database_with_examples.jsonl", debug_chunks)
    write_csv(OUT_DIR / "rag_norm_database_no_leak.csv", safe_chunks)

    summary = summarize(safe_chunks)
    (OUT_DIR / "rag_database_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = f"""# Social Acceptability RAG Norm Database

This folder contains a medium-sized RAG corpus derived from `CS263_dataset_with_predictions.csv`.

## Files

- `rag_norm_database_no_leak.jsonl`: primary evaluation corpus. Use this for RAG retrieval.
- `rag_norm_database_no_leak.csv`: spreadsheet-friendly copy of the primary corpus.
- `rag_norm_database_with_examples.jsonl`: audit/debug corpus with example situations included.
- `rag_database_summary.json`: basic coverage counts.

## Recommended Use

For the final RAG experiment, retrieve from `rag_norm_database_no_leak.jsonl`. It intentionally excludes:

- gold labels
- DeBERTa predictions
- GPT predictions
- confidence scores
- exact scenario text

This keeps RAG closer to norm retrieval instead of answer lookup.

## Suggested Retrieval Query

Build the query from the test item without including the gold label:

```text
scenario: <scenario>
culture/social context: <culture>
category: <category if available>
```

Retrieve the top 3-5 chunks, then pass only `retrieval_text` into the classifier prompt.

## Coverage

- Chunks: {summary["chunk_count"]}
- Categories: {len(summary["category_counts"])}
- Culture/region labels: {len(summary["culture_region_counts"])}

## Method Note

This is a controlled RAG corpus. It is valid for testing whether inference-time retrieval of relevant social norms improves classification, while remaining more reproducible than live web search.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

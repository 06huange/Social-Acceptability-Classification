#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path("data/socialchem101/social-chem-101/social-chem-101.v1.0.tsv")
DEFAULT_OUT_DIR = Path("experiments/gpt_rag_socialchem101/database_local")

AGREE_LABELS = {
    "0": "<1% agree",
    "1": "5-25% agree",
    "2": "about 50% agree",
    "3": "75-90% agree",
    "4": ">99% agree",
}

JUDGMENT_LABELS = {
    "-2": "very bad",
    "-1": "bad",
    "0": "expected or OK",
    "1": "good",
    "2": "very good",
}

PRESSURE_LABELS = {
    "-2": "strong cultural pressure against",
    "-1": "cultural pressure against",
    "0": "discretionary or weak cultural pressure",
    "1": "cultural pressure for",
    "2": "strong cultural pressure for",
}

PROJECT_CATEGORY_KEYWORDS = {
    "family": [
        "parent", "parents", "mom", "mother", "dad", "father", "sibling", "brother",
        "sister", "family", "relative", "grandparent", "child", "children", "spouse",
        "wife", "husband", "marriage", "wedding",
    ],
    "relationship": [
        "partner", "boyfriend", "girlfriend", "dating", "date", "romantic", "relationship",
        "ex", "fiance", "fiancee", "spouse", "wife", "husband",
    ],
    "workplace": [
        "boss", "manager", "coworker", "co-worker", "colleague", "work", "workplace",
        "job", "office", "employee", "employer", "meeting", "professional", "client",
        "supervisor", "shift",
    ],
    "communication": [
        "tell", "telling", "said", "saying", "talk", "talking", "conversation",
        "message", "text", "email", "ask", "asking", "answer", "reply", "feedback",
        "criticize", "criticizing", "argue", "yell", "direct", "honest",
    ],
    "social_etiquette": [
        "guest", "host", "party", "dinner", "restaurant", "public", "gift", "thank",
        "thanks", "late", "punctual", "phone", "table", "rude", "polite", "manners",
        "neighbor", "stranger",
    ],
    "privacy": [
        "private", "privacy", "secret", "personal", "phone", "password", "photo",
        "picture", "information", "tell others", "share", "sharing", "consent",
        "permission", "without asking",
    ],
    "friendship": [
        "friend", "friends", "best friend", "buddy", "roommate", "hang out", "support",
        "invite", "invited",
    ],
    "morality_fairness": [
        "fair", "unfair", "cheat", "cheating", "lie", "lying", "steal", "stealing",
        "credit", "blame", "honest", "dishonest", "rule", "rules",
    ],
    "care_harm": [
        "hurt", "harm", "help", "helping", "safe", "safety", "sick", "ill",
        "emergency", "care", "need", "vulnerable", "injured",
    ],
    "morality_loyalty": [
        "loyal", "loyalty", "betray", "betraying", "group", "team", "side with",
        "defend", "back up",
    ],
    "authority": [
        "authority", "teacher", "professor", "boss", "manager", "elder", "police",
        "supervisor", "respect", "hierarchy", "senior",
    ],
    "fairness": [
        "equal", "equally", "fair", "unfair", "split", "share", "turn", "queue",
        "line", "deserve", "deserved",
    ],
    "commitment": [
        "promise", "promised", "commit", "commitment", "agreement", "agreed", "plan",
        "plans", "cancel", "canceled", "reliable", "show up", "deadline",
    ],
}

MORAL_TO_PROJECT_CATEGORY = {
    "care-harm": "care_harm",
    "fairness-cheating": "morality_fairness",
    "loyalty-betrayal": "morality_loyalty",
    "authority-subversion": "authority",
    "sanctity-degradation": "social_etiquette",
}

NORM_FAMILY_KEYWORDS = {
    "privacy": ["privacy", "private", "secret", "consent", "permission", "phone", "personal"],
    "hierarchy": ["boss", "manager", "teacher", "elder", "authority", "respect", "senior"],
    "face_reputation": ["embarrass", "shame", "humiliate", "reputation", "public", "criticize"],
    "reciprocity": ["favor", "owe", "return", "reciprocate", "help", "support"],
    "punctuality": ["late", "time", "punctual", "wait", "waiting", "deadline"],
    "commitment": ["promise", "agreement", "commit", "cancel", "plan", "reliable"],
    "care_harm": ["harm", "hurt", "safe", "help", "care", "sick", "injured"],
    "fairness": ["fair", "unfair", "equal", "cheat", "steal", "share", "split"],
    "directness": ["honest", "direct", "tell", "feedback", "truth", "blunt"],
    "family_obligation": ["family", "parent", "sibling", "relative", "marriage", "child"],
}


def split_pipe(value):
    return [part.strip() for part in value.split("|") if part.strip()]


def as_int(value):
    if value == "":
        return None
    return int(value)


def mean(values):
    nums = [value for value in values if value is not None]
    return round(sum(nums) / len(nums), 3) if nums else None


def mode(values):
    vals = [value for value in values if value not in ("", None)]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def clean_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def token_text(row):
    return " ".join(
        clean_text(row.get(field, "")).lower()
        for field in ("rot", "action", "situation", "characters")
    )


def infer_project_categories(rows, moral_foundations):
    text = token_text(rows[0])
    scores = Counter()
    for category, words in PROJECT_CATEGORY_KEYWORDS.items():
        for word in words:
            if word in text:
                scores[category] += 1
    for foundation in moral_foundations:
        category = MORAL_TO_PROJECT_CATEGORY.get(foundation)
        if category:
            scores[category] += 2
    return [category for category, _ in scores.most_common(4)]


def infer_norm_families(rows, moral_foundations):
    text = token_text(rows[0])
    families = Counter()
    for family, words in NORM_FAMILY_KEYWORDS.items():
        for word in words:
            if word in text:
                families[family] += 1
    for foundation in moral_foundations:
        if foundation == "care-harm":
            families["care_harm"] += 2
        elif foundation == "fairness-cheating":
            families["fairness"] += 2
        elif foundation == "loyalty-betrayal":
            families["loyalty"] += 2
        elif foundation == "authority-subversion":
            families["hierarchy"] += 2
        elif foundation == "sanctity-degradation":
            families["social_etiquette"] += 2
    return [family for family, _ in families.most_common(5)] or ["general_social_norm"]


def acceptability_signal(avg_judgment, avg_agreement, avg_pressure):
    if avg_judgment is None:
        return "unknown"
    if avg_judgment <= -1.25 and (avg_pressure is None or avg_pressure <= -0.5):
        return "strongly not acceptable"
    if avg_judgment < -0.25:
        return "not acceptable"
    if avg_judgment >= 1.25 and (avg_pressure is None or avg_pressure >= 0.5):
        return "strongly acceptable"
    if avg_judgment > 0.25:
        return "acceptable"
    if avg_agreement is not None and avg_agreement < 2.5:
        return "context-dependent or contested"
    if avg_pressure is not None and abs(avg_pressure) < 0.5:
        return "acceptable but discretionary"
    return "acceptable or expected"


def context_dependency(avg_agreement, avg_pressure, legal_mode, annotation_count):
    reasons = []
    if annotation_count > 1:
        reasons.append("multiple worker annotations were aggregated")
    if avg_agreement is not None and avg_agreement <= 2.5:
        reasons.append("agreement is moderate or low")
    if avg_pressure is not None and -0.5 <= avg_pressure <= 0.5:
        reasons.append("cultural pressure is weak or discretionary")
    if legal_mode == "tolerated":
        reasons.append("legality is marked tolerated rather than clearly legal or illegal")
    return reasons or ["few explicit ambiguity signals in Social-Chem labels"]


def majority_labels(rows, field, label_map=None):
    counts = Counter(row[field] for row in rows if row[field] != "")
    if label_map:
        return {label_map.get(key, key): count for key, count in counts.most_common()}
    return dict(counts.most_common())


def build_chunk(rot_id, rows):
    canonical = rows[0]
    categories = sorted({cat for row in rows for cat in split_pipe(row["rot-categorization"])})
    moral_foundations = sorted({cat for row in rows for cat in split_pipe(row["rot-moral-foundations"])})
    judgments = [as_int(row["action-moral-judgment"]) for row in rows]
    agreements = [as_int(row["action-agree"] or row["rot-agree"]) for row in rows]
    pressures = [as_int(row["action-pressure"]) for row in rows]
    avg_judgment = mean(judgments)
    avg_agreement = mean(agreements)
    avg_pressure = mean(pressures)
    legal_mode = mode(row["action-legal"] for row in rows)
    project_categories = infer_project_categories(rows, moral_foundations)
    norm_families = infer_norm_families(rows, moral_foundations)
    signal = acceptability_signal(avg_judgment, avg_agreement, avg_pressure)
    ambiguity = context_dependency(avg_agreement, avg_pressure, legal_mode, len(rows))

    judgment_counts = majority_labels(rows, "action-moral-judgment", JUDGMENT_LABELS)
    pressure_counts = majority_labels(rows, "action-pressure", PRESSURE_LABELS)
    agreement_counts = majority_labels(rows, "action-agree", AGREE_LABELS)

    retrieval_text = (
        f"Source: Social-Chem-101 rule of thumb. "
        f"Rule of thumb: {clean_text(canonical['rot'])} "
        f"Action: {clean_text(canonical['action'])}. "
        f"Source situation: {clean_text(canonical['situation'])}. "
        f"Acceptability signal: {signal}. "
        f"Average social judgment: {avg_judgment} where -2 is very bad, 0 is expected/OK, and 2 is very good. "
        f"Average agreement: {avg_agreement} where higher means more people agree. "
        f"Average cultural pressure: {avg_pressure} where negative means pressure against and positive means pressure for. "
        f"Legality: {legal_mode or 'unknown'}. "
        f"Norm categories: {', '.join(categories) or 'unknown'}. "
        f"Moral foundations: {', '.join(moral_foundations) or 'none specified'}. "
        f"Norm families: {', '.join(norm_families)}. "
        f"Project category hints: {', '.join(project_categories) or 'none'}. "
        f"Context-dependent cues: {'; '.join(ambiguity)}. "
        f"Characters/roles: {clean_text(canonical['characters'])}."
    )

    return {
        "chunk_id": f"SC101_{re.sub(r'[^A-Za-z0-9]+', '_', rot_id).strip('_')}",
        "chunk_type": "socialchem101_rot",
        "source": "Social-Chem-101",
        "source_license": "CC BY-SA 4.0",
        "source_area": canonical["area"],
        "source_split": canonical["split"],
        "situation_short_id": canonical["situation-short-id"],
        "rot_id": rot_id,
        "annotation_count": len(rows),
        "rot": clean_text(canonical["rot"]),
        "action": clean_text(canonical["action"]),
        "source_situation": clean_text(canonical["situation"]),
        "characters": split_pipe(canonical["characters"]),
        "rot_categories": categories,
        "moral_foundations": moral_foundations,
        "norm_families": norm_families,
        "project_category_hints": project_categories,
        "acceptability_signal": signal,
        "avg_action_moral_judgment": avg_judgment,
        "avg_agreement": avg_agreement,
        "avg_cultural_pressure": avg_pressure,
        "legal_mode": legal_mode,
        "judgment_counts": judgment_counts,
        "agreement_counts": agreement_counts,
        "cultural_pressure_counts": pressure_counts,
        "context_dependent_cues": ambiguity,
        "culture_scope_note": (
            "Social-Chem-101 contains broad English-language social and moral norms; "
            "it should not be interpreted as country-specific cultural evidence unless "
            "the retrieved rule itself names a culture or place."
        ),
        "retrieval_text": retrieval_text,
    }


def read_chunks(source_path):
    grouped = defaultdict(list)
    with source_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["rot-bad"] != "0":
                continue
            if not clean_text(row["rot"]) or not clean_text(row["action"]):
                continue
            grouped[row["rot-id"]].append(row)

    chunks = []
    for rot_id, rows in grouped.items():
        chunks.append(build_chunk(rot_id, rows))
    return chunks


def write_jsonl(path, chunks):
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def write_csv(path, chunks):
    fields = [
        "chunk_id",
        "source_area",
        "source_split",
        "rot_id",
        "annotation_count",
        "rot",
        "action",
        "source_situation",
        "acceptability_signal",
        "avg_action_moral_judgment",
        "avg_agreement",
        "avg_cultural_pressure",
        "legal_mode",
        "rot_categories",
        "moral_foundations",
        "norm_families",
        "project_category_hints",
        "context_dependent_cues",
        "retrieval_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for chunk in chunks:
            row = dict(chunk)
            for field in [
                "rot_categories",
                "moral_foundations",
                "norm_families",
                "project_category_hints",
                "context_dependent_cues",
            ]:
                row[field] = "; ".join(row.get(field, []))
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(chunks):
    return {
        "chunk_count": len(chunks),
        "source": "Social-Chem-101",
        "source_license": "CC BY-SA 4.0",
        "source_url": "https://github.com/mbforbes/social-chemistry-101",
        "source_data_url": "https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip",
        "chunk_type_counts": dict(Counter(chunk["chunk_type"] for chunk in chunks)),
        "source_area_counts": dict(Counter(chunk["source_area"] for chunk in chunks).most_common()),
        "source_split_counts": dict(Counter(chunk["source_split"] for chunk in chunks).most_common()),
        "acceptability_signal_counts": dict(Counter(chunk["acceptability_signal"] for chunk in chunks).most_common()),
        "project_category_hint_counts": dict(Counter(cat for chunk in chunks for cat in chunk["project_category_hints"]).most_common()),
        "norm_family_counts": dict(Counter(cat for chunk in chunks for cat in chunk["norm_families"]).most_common()),
        "notes": [
            "Rows with rot-bad=1, empty rules of thumb, or empty actions were excluded.",
            "Multiple annotation rows for the same rot-id were aggregated into a single RAG chunk.",
            "Social-Chem-101 is used as broad social-norm evidence, not as country-specific cultural evidence.",
            "The corpus does not include labels or predictions from the CS263 project dataset.",
        ],
    }


def write_readme(path, summary):
    text = f"""# Social-Chem-101 RAG Database

This folder contains a large RAG corpus built from the official Social-Chem-101 dataset for the cultural/social acceptability classification project.

## Files

- `socialchem101_rag_full.jsonl`: full primary corpus, one aggregated chunk per usable Social-Chem rule of thumb.
- `socialchem101_rag_full.csv`: spreadsheet-friendly copy of the full corpus.
- `socialchem101_rag_project_matched.jsonl`: subset with at least one inferred project category hint.
- `socialchem101_rag_project_matched.csv`: spreadsheet-friendly copy of the project-matched subset.
- `socialchem101_rag_summary.json`: coverage, filtering, and label distribution metadata.

## Construction

- Source: Social-Chem-101 official TSV.
- License: CC BY-SA 4.0.
- Rows with `rot-bad=1`, empty `rot`, or empty `action` were excluded.
- Repeated annotations for the same `rot-id` were aggregated into one chunk.
- The retrieval text preserves the original rule of thumb, action, source situation, moral foundations, agreement, cultural pressure, legality, and derived project category hints.

## Coverage

- Full chunks: {summary["chunk_count"]}
- Project categories detected: {len(summary["project_category_hint_counts"])}
- Norm families detected: {len(summary["norm_family_counts"])}

## Recommended Use

For the classifier experiment, retrieve from `socialchem101_rag_project_matched.jsonl` first. If retrieval quality looks too narrow, switch to `socialchem101_rag_full.jsonl` or combine this corpus with the existing culture-aware RAG database.

Use Social-Chem-101 as broad social and moral norm evidence. It is not country-specific, so country/culture-specific judgments should still prioritize your `cultural_context` field or the culture-aware RAG database.

## Citation

Forbes, Maxwell, Jena D. Hwang, Vered Shwartz, Maarten Sap, and Yejin Choi. 2020. Social Chemistry 101: Learning to Reason about Social and Moral Norms. EMNLP.
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks = read_chunks(args.source)
    project_matched = [chunk for chunk in chunks if chunk["project_category_hints"]]

    write_jsonl(args.output_dir / "socialchem101_rag_full.jsonl", chunks)
    write_jsonl(args.output_dir / "socialchem101_rag_project_matched.jsonl", project_matched)
    write_csv(args.output_dir / "socialchem101_rag_full.csv", chunks)
    write_csv(args.output_dir / "socialchem101_rag_project_matched.csv", project_matched)

    summary = summarize(chunks)
    summary["project_matched_chunk_count"] = len(project_matched)
    (args.output_dir / "socialchem101_rag_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_readme(args.output_dir / "README.md", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

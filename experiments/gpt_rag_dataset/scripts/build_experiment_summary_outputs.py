#!/usr/bin/env python3
import csv
import json
from collections import Counter
from pathlib import Path


DATASET = Path("data/processed/CS263_dataset_with_predictions.csv")
OUT_DIR = Path("experiments/gpt_rating/results/final_experiment_summary")

TRACE_EXPERIMENTS = [
    (
        "gpt_scenario_rag_v1",
        "GPT-4o mini scenario + RAG v1",
        Path("experiments/gpt_rag_dataset/results/rag_experiment/gpt4o_mini_scenario_rag_trace.jsonl"),
    ),
    (
        "gpt_scenario_context_rag_v1",
        "GPT-4o mini scenario + context + RAG v1",
        Path("experiments/gpt_rag_dataset/results/rag_experiment/gpt4o_mini_scenario_context_rag_trace.jsonl"),
    ),
    (
        "gpt_scenario_context_rag_v2",
        "GPT-4o mini scenario + context + RAG v2",
        Path("experiments/gpt_rag_dataset/results/rag_experiment_v2/gpt4o_mini_scenario_context_rag_trace.jsonl"),
    ),
    (
        "gpt_scenario_context_rag_v2_decomposition",
        "GPT-4o mini scenario + context + RAG v2 + decomposition",
        Path("experiments/gpt_rag_dataset/results/rag_experiment_v2_decomposition_primary_context/gpt4o_mini_scenario_context_rag_v2_decomposition_trace.jsonl"),
    ),
    (
        "gpt_scenario_context_rag_v2_decomposition_strict",
        "GPT-4o mini scenario + context + RAG v2 + strict decomposition",
        Path("experiments/gpt_rag_dataset/results/rag_experiment_v2_decomposition_strict/gpt4o_mini_scenario_context_rag_v2_decomposition_strict_trace.jsonl"),
    ),
    (
        "gpt_scenario_context_rag_v2_rating",
        "GPT-4o mini scenario + context + RAG v2 + 1-10 rating",
        Path("experiments/gpt_rag_dataset/results/rag_experiment_v2_rating/gpt4o_mini_scenario_context_rag_v2_rating_trace.jsonl"),
    ),
    (
        "gpt_scenario_no_rag_rating",
        "GPT-4o mini scenario-only + 1-10 rating",
        Path("experiments/gpt_rating/results/rating_baselines/scenario_only/gpt4o_mini_scenario_no_rag_rating_trace.jsonl"),
    ),
    (
        "gpt_scenario_context_no_rag_rating",
        "GPT-4o mini scenario + context + 1-10 rating",
        Path("experiments/gpt_rating/results/rating_baselines/scenario_context/gpt4o_mini_scenario_context_no_rag_rating_trace.jsonl"),
    ),
    (
        "gpt_scenario_rag_v1_rating",
        "GPT-4o mini scenario + RAG v1 + 1-10 rating",
        Path("experiments/gpt_rating/results/rating_baselines/scenario_rag_v1/gpt4o_mini_scenario_rag_rating_trace.jsonl"),
    ),
    (
        "gpt_scenario_context_rag_v1_rating",
        "GPT-4o mini scenario + context + RAG v1 + 1-10 rating",
        Path("experiments/gpt_rating/results/rating_baselines/scenario_context_rag_v1/gpt4o_mini_scenario_context_rag_rating_trace.jsonl"),
    ),
]


def load_dataset():
    with DATASET.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f)), list(csv.DictReader(DATASET.open(newline="", encoding="utf-8")).fieldnames or [])


def load_trace_predictions(path):
    predictions = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            predictions[item["id"]] = item["predicted_label"]
    return predictions


def metrics_for(rows, pred_by_id):
    labels = ["acceptable", "context-dependent", "not acceptable"]
    total = len(rows)
    correct = sum(1 for row in rows if pred_by_id.get(row["id"], "").strip().lower() == row["label"].strip().lower())
    row = {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0,
        "prediction_distribution": dict(Counter(pred_by_id.get(r["id"], "") for r in rows)),
    }
    for label in labels:
        subset = [r for r in rows if r["label"].strip().lower() == label]
        label_correct = sum(1 for r in subset if pred_by_id.get(r["id"], "").strip().lower() == label)
        row[f"{label}_correct"] = label_correct
        row[f"{label}_total"] = len(subset)
        row[f"{label}_accuracy"] = label_correct / len(subset) if subset else 0
    firm = [r for r in rows if r["label"].strip().lower() in {"acceptable", "not acceptable"}]
    firm_correct = sum(1 for r in firm if pred_by_id.get(r["id"], "").strip().lower() == r["label"].strip().lower())
    row["acceptable_not_acceptable_correct"] = firm_correct
    row["acceptable_not_acceptable_total"] = len(firm)
    row["acceptable_not_acceptable_accuracy"] = firm_correct / len(firm) if firm else 0
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_fields = reader.fieldnames or []

    experiment_predictions = {}
    for column, _, path in TRACE_EXPERIMENTS:
        experiment_predictions[column] = load_trace_predictions(path)

    combined_fields = original_fields + [column for column, _, _ in TRACE_EXPERIMENTS]
    combined_path = OUT_DIR / "CS263_dataset_with_all_experiment_predictions.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=combined_fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for column, pred_by_id in experiment_predictions.items():
                out[column] = pred_by_id.get(row["id"], "")
            writer.writerow(out)

    summary_rows = []
    baseline_columns = [
        ("deberta_label", "DeBERTa existing prediction"),
        ("gpt_label", "GPT-4o mini scenario-only existing prediction"),
    ]
    for column, name in baseline_columns:
        pred_by_id = {row["id"]: row[column].strip().lower() for row in rows}
        m = metrics_for(rows, pred_by_id)
        summary_rows.append({"setup_id": column, "setup_name": name, **m})

    for column, name, _ in TRACE_EXPERIMENTS:
        m = metrics_for(rows, experiment_predictions[column])
        summary_rows.append({"setup_id": column, "setup_name": name, **m})

    summary_fields = [
        "setup_id", "setup_name", "correct", "total", "accuracy",
        "acceptable_correct", "acceptable_total", "acceptable_accuracy",
        "context-dependent_correct", "context-dependent_total", "context-dependent_accuracy",
        "not acceptable_correct", "not acceptable_total", "not acceptable_accuracy",
        "acceptable_not_acceptable_correct", "acceptable_not_acceptable_total", "acceptable_not_acceptable_accuracy",
        "prediction_distribution",
    ]
    summary_path = OUT_DIR / "baseline_and_experiment_accuracies.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            row = dict(row)
            row["prediction_distribution"] = json.dumps(row["prediction_distribution"], ensure_ascii=False)
            writer.writerow(row)

    compact_path = OUT_DIR / "baseline_and_experiment_accuracies.md"
    lines = [
        "| Setup | Overall | Acceptable | Context-dependent | Not acceptable | Acceptable + Not acceptable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['setup_name']} | "
            f"{row['correct']}/{row['total']} = {row['accuracy']:.2%} | "
            f"{row['acceptable_correct']}/{row['acceptable_total']} = {row['acceptable_accuracy']:.2%} | "
            f"{row['context-dependent_correct']}/{row['context-dependent_total']} = {row['context-dependent_accuracy']:.2%} | "
            f"{row['not acceptable_correct']}/{row['not acceptable_total']} = {row['not acceptable_accuracy']:.2%} | "
            f"{row['acceptable_not_acceptable_correct']}/{row['acceptable_not_acceptable_total']} = {row['acceptable_not_acceptable_accuracy']:.2%} |"
        )
    compact_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "combined_dataset": str(combined_path.resolve()),
        "accuracy_csv": str(summary_path.resolve()),
        "accuracy_markdown": str(compact_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

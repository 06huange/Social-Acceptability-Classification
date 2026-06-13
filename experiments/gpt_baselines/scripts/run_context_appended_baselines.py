import argparse
import csv
import json
import os
import re
from pathlib import Path


LABELS = ["acceptable", "not acceptable", "context-dependent"]


SYSTEM_PROMPT = """You are a social acceptability classifier.

Classify the user's situation into exactly one label:
- acceptable
- not acceptable
- context-dependent

Do not use cultural context unless it is explicitly included in the situation.
Return only JSON in this format:
{"label": "..."}
"""


def parse_user_interaction(text):
    match = re.search(r"User:\s*(.*?)(?:\\n|\n|$)", text)
    return match.group(1).strip() if match else text.strip()


def context_appended_situation(row):
    situation = parse_user_interaction(row["interaction"])
    return (
        f"{situation}\n\n"
        f"Cultural context: {row['cultural_context']}"
    )


def read_dataset(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["situation"] = parse_user_interaction(row["interaction"])
        row["context_appended_situation"] = context_appended_situation(row)
    return rows


def write_predictions(path, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "situation",
        "cultural_context",
        "context_appended_situation",
        "gold_label",
        "prediction_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def load_existing_predictions(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_label(label):
    label = str(label).strip().lower()
    if label == "unacceptable":
        return "not acceptable"
    if label not in LABELS:
        raise ValueError(f"Unexpected label: {label!r}")
    return label


def predict_deberta(rows, output_path, threshold):
    from transformers import pipeline

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    classifier = pipeline(
        "zero-shot-classification",
        model="microsoft/deberta-large-mnli",
        token=token,
    )

    predictions = load_existing_predictions(output_path)
    done_ids = {row["id"] for row in predictions}

    for row in rows:
        if row["id"] in done_ids:
            continue
        result = classifier(
            row["context_appended_situation"],
            candidate_labels=["acceptable behavior", "not acceptable behavior"],
            hypothesis_template="This behavior is {}.",
        )
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        if top_score < threshold:
            pred = "context-dependent"
        elif top_label == "acceptable behavior":
            pred = "acceptable"
        else:
            pred = "not acceptable"

        predictions.append(prediction_row(row, pred))
        write_predictions(output_path, predictions)
        print(f"DeBERTa {len(predictions)}/{len(rows)} {row['id']} -> {pred}", flush=True)

    return predictions


def predict_gpt(rows, output_path):
    from openai import OpenAI

    client = OpenAI()
    predictions = load_existing_predictions(output_path)
    done_ids = {row["id"] for row in predictions}

    for row in rows:
        if row["id"] in done_ids:
            continue
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Situation: {row['context_appended_situation']}",
                },
            ],
        )
        content = response.choices[0].message.content
        pred = normalize_label(json.loads(content)["label"])
        predictions.append(prediction_row(row, pred))
        write_predictions(output_path, predictions)
        print(f"GPT {len(predictions)}/{len(rows)} {row['id']} -> {pred}", flush=True)

    return predictions


def prediction_row(row, pred):
    return {
        "id": row["id"],
        "situation": row["situation"],
        "cultural_context": row["cultural_context"],
        "context_appended_situation": row["context_appended_situation"],
        "gold_label": normalize_label(row["label"]),
        "prediction_label": normalize_label(pred),
    }


def accuracy(predictions):
    if not predictions:
        return 0.0
    correct = sum(
        normalize_label(row["gold_label"]) == normalize_label(row["prediction_label"])
        for row in predictions
    )
    return correct / len(predictions)


def classification_counts(predictions):
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in predictions:
        matrix[normalize_label(row["gold_label"])][normalize_label(row["prediction_label"])] += 1
    return matrix


def write_summary(path, model_predictions):
    lines = []
    for model_name, predictions in model_predictions.items():
        lines.append(f"===== {model_name} context-appended =====")
        lines.append(f"Accuracy: {accuracy(predictions):.4f}")
        lines.append(f"Rows: {len(predictions)}")
        lines.append("Confusion matrix: gold -> prediction")
        matrix = classification_counts(predictions)
        lines.append("," + ",".join(LABELS))
        for gold in LABELS:
            lines.append(gold + "," + ",".join(str(matrix[gold][pred]) for pred in LABELS))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def merge_predictions(rows, deberta_predictions, gpt_predictions, output_path):
    deberta_by_id = {row["id"]: row["prediction_label"] for row in deberta_predictions}
    gpt_by_id = {row["id"]: row["prediction_label"] for row in gpt_predictions}
    fieldnames = [
        "id",
        "category",
        "interaction",
        "culture",
        "label",
        "confidence_score",
        "cultural_context",
        "situation",
        "context_appended_situation",
        "deberta_context_label",
        "gpt_context_label",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row[key] for key in fieldnames if key in row}
            out["deberta_context_label"] = deberta_by_id.get(row["id"], "")
            out["gpt_context_label"] = gpt_by_id.get(row["id"], "")
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/raw/CS263_dataset.csv")
    parser.add_argument("--output-dir", default="experiments/gpt_baselines/results/context_appended")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["deberta", "gpt"],
        default=["deberta", "gpt"],
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    rows = read_dataset(dataset_path)

    model_predictions = {}
    deberta_predictions = load_existing_predictions(output_dir / "deberta_context_predictions.csv")
    gpt_predictions = load_existing_predictions(output_dir / "gpt_context_predictions.csv")

    if "deberta" in args.models:
        deberta_predictions = predict_deberta(
            rows,
            output_dir / "deberta_context_predictions.csv",
            args.threshold,
        )
        model_predictions["DeBERTa"] = deberta_predictions

    if "gpt" in args.models:
        gpt_predictions = predict_gpt(rows, output_dir / "gpt_context_predictions.csv")
        model_predictions["GPT"] = gpt_predictions

    if deberta_predictions:
        model_predictions.setdefault("DeBERTa", deberta_predictions)
    if gpt_predictions:
        model_predictions.setdefault("GPT", gpt_predictions)

    write_summary(output_dir / "context_accuracy_summary.txt", model_predictions)
    merge_predictions(
        rows,
        deberta_predictions,
        gpt_predictions,
        output_dir / "CS263_dataset_with_context_predictions.csv",
    )

    for model_name, predictions in model_predictions.items():
        print(f"{model_name} context-appended accuracy: {accuracy(predictions):.4f}")


if __name__ == "__main__":
    main()

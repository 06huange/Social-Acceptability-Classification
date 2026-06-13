#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from mlx_lm import generate, load
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


LABELS = ["acceptable", "not acceptable", "context-dependent"]


def parse_user_interaction(text):
    match = re.search(r"User:\s*(.*?)(?:\\n|\n|$)", str(text))
    return match.group(1).strip() if match else str(text).strip()


def normalize_label(text):
    label = str(text).strip().lower()
    if label == "unacceptable":
        return "not acceptable"
    if label in LABELS:
        return label
    for candidate in ["not acceptable", "context-dependent", "acceptable"]:
        if candidate in label:
            return candidate
    return "context-dependent"


def qwen_generate_thinking(model, tokenizer, messages, max_tokens):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    raw = generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
    if "</think>" in raw:
        thinking, answer = raw.split("</think>", 1)
        return thinking.replace("<think>", "").strip(), answer.strip()
    return "", raw.strip()


def parse_answer(answer):
    try:
        data = json.loads(answer)
        return normalize_label(data.get("label", ""))
    except Exception:
        return normalize_label(answer)


def build_messages(row, engineered=False):
    if engineered:
        system_prompt = """
You are a careful social acceptability classifier.

Classify the interaction into exactly one label:
- acceptable
- not acceptable
- context-dependent

You are given:
1. interaction: the user/LLM exchange
2. culture: the relevant cultural or social norm
3. cultural_context: explanation of that norm

Use the provided culture and cultural_context when judging acceptability.
Do not invent additional cultural assumptions beyond what is provided.

Label definitions:
- acceptable: The behavior is generally socially appropriate under the given cultural context.
- not acceptable: The behavior clearly violates the given cultural/social norm, or is rude, harmful, invasive, disrespectful, unsafe, or unfair.
- context-dependent: The acceptability depends on missing details such as tone, closeness, consent, setting, urgency, frequency, age, relationship, or power dynamics.

Decision rules:
1. If privacy, consent, harassment, dishonesty, safety, or discrimination is involved, be stricter.
2. If the provided cultural context says expectations vary or depend on circumstances, prefer context-dependent.
3. If the behavior is a respectful boundary and the context supports autonomy/privacy, prefer acceptable.
4. If the behavior publicly embarrasses, invades privacy, violates trust, or disrespects a strong norm, prefer not acceptable.
5. Think internally before deciding, but output only valid JSON.

Return exactly this JSON format:
{"label": "acceptable"}
"""
        few_shot = """
Examples:

Interaction:
User: I looked through my partner's phone because I felt suspicious.
LLM: That raises privacy and trust concerns.
Culture:
US, relationship-privacy norm
Cultural context:
Privacy and trust are important in relationships.
Output:
{"label": "not acceptable"}

Interaction:
User: I skipped a family dinner because I wanted alone time.
LLM: That's understandable. Did you explain it?
Culture:
US, personal-boundary norm
Cultural context:
Personal time and emotional boundaries are generally accepted, especially if communicated respectfully.
Output:
{"label": "acceptable"}

Interaction:
User: I didn't help my sibling financially.
LLM: That depends on your situation.
Culture:
India, extended-family norm
Cultural context:
Family support may be expected, but obligation depends on resources, relationship, and circumstances.
Output:
{"label": "context-dependent"}
"""
        system = system_prompt + "\n\n" + few_shot
    else:
        system = """
You are a social acceptability classifier.

Classify the user's situation into exactly one label:
- acceptable
- not acceptable
- context-dependent

Use the provided cultural context to inform your classification.
Return only JSON in this format:
{"label": "..."}
"""

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Situation: {row['situation']}\n"
                f"Cultural context: {row['cultural_context']}\n"
                f"Culture: {row['culture']}"
            ),
        },
    ]


def run_mode(rows, model, tokenizer, out_dir, mode_name, engineered, max_tokens):
    predictions = []
    trace_path = out_dir / f"{mode_name}_trace.jsonl"
    prediction_path = out_dir / f"{mode_name}_predictions.csv"
    report_path = out_dir / f"{mode_name}_classification_report.txt"

    with trace_path.open("w", encoding="utf-8") as trace:
        for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"Running {mode_name}"):
            messages = build_messages(row, engineered=engineered)
            thinking, answer = qwen_generate_thinking(model, tokenizer, messages, max_tokens)
            pred_label = parse_answer(answer)
            item = {
                "id": row["id"],
                "situation": row["situation"],
                "culture": row["culture"],
                "cultural_context": row["cultural_context"],
                "gold_label": row["label"],
                "prediction_label": pred_label,
                "thinking": thinking,
                "answer": answer,
            }
            trace.write(json.dumps(item, ensure_ascii=False) + "\n")
            predictions.append(
                {
                    "id": row["id"],
                    "situation": row["situation"],
                    "gold_label": row["label"],
                    "prediction_label": pred_label,
                }
            )

    results = pd.DataFrame(predictions)
    results.to_csv(prediction_path, index=False)

    report = classification_report(
        results["gold_label"],
        results["prediction_label"],
        labels=LABELS,
        zero_division=0,
    )
    accuracy = accuracy_score(results["gold_label"], results["prediction_label"])
    report_text = f"{report}\nAccuracy: {accuracy:.4f}\n"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n{mode_name} accuracy: {accuracy:.4f}")
    print(f"Wrote {prediction_path}")
    print(f"Wrote {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-8B thinking social acceptability experiments with MLX.")
    parser.add_argument("--dataset", default="data/raw/CS263_dataset.csv")
    parser.add_argument("--out-dir", default="experiments/qwen/results")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--mode", choices=["basic", "engineered", "both"], default="both")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(args.dataset)
    rows["situation"] = rows["interaction"].apply(parse_user_interaction)
    if args.limit:
        rows = rows.head(args.limit)

    model, tokenizer = load(args.model)

    if args.mode in {"basic", "both"}:
        run_mode(rows, model, tokenizer, out_dir, "qwen_thinking", engineered=False, max_tokens=args.max_tokens)
    if args.mode in {"engineered", "both"}:
        run_mode(
            rows,
            model,
            tokenizer,
            out_dir,
            "qwen_thinking_engineered",
            engineered=True,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()

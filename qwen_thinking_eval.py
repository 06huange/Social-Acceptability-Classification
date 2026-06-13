hf_token = ""

from mlx_lm import load, generate
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# 1. Load dataset
df = pd.read_csv('/Users/Suhas/Social-Acceptability-Classification/Dataset/CS263_dataset.csv')

# 2. Parse User sentence
def parse_user_interaction(text):
    match = re.search(r"User:\s*(.*?)(?:\\n|\n|$)", text)
    return match.group(1).strip() if match else None

df["situation"] = df["interaction"].apply(parse_user_interaction)


model, tokenizer = load("Qwen/Qwen3-8B")


def qwen_generate_thinking(messages, max_tokens=1024):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    
    raw = generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
    
    if "</think>" in raw:
        thinking, answer = raw.split("</think>", 1)
        thinking = thinking.replace("<think>", "").strip()
        answer = answer.strip()
    else:
        thinking, answer = "", raw
    
    return thinking, answer

def predict_acceptability_qwen_mlx_thinking(situation, cultural_context, culture, max_new_tokens=128):
    messages = [
            {
                "role": "system",
                "content": """
You are a social acceptability classifier.

Classify the user's situation into exactly one label:
- acceptable
- not acceptable
- context-dependent

Use the provided cultural context to inform your classification.
Return only JSON in this format:
{"label": "..."}
"""
            },
            {
                "role": "user",
                "content": f"Situation: {situation}\nCultural context: {cultural_context}\nCulture: {culture}"
            }
        ]
    
    thinking, answer = qwen_generate_thinking(messages)
    try:
        return json.loads(answer)["label"]
    except Exception:
        for label in ["not acceptable", "context-dependent", "acceptable"]:
            if label in answer.lower():
                return label
        
        return "context-dependent"
    
def predict_acceptability_qwen_mlx_thinking_engineered(situation, cultural_context, culture, max_new_tokens=128):
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
User: I looked through my partner’s phone because I felt suspicious.
LLM: That raises privacy and trust concerns.
Culture:
US, relationship-privacy norm
Cultural context:
Privacy and trust are important in relationships.
Output:
{"label": "not acceptable"}

Interaction:
User: I skipped a family dinner because I wanted alone time.
LLM: That’s understandable. Did you explain it?
Culture:
US, personal-boundary norm
Cultural context:
Personal time and emotional boundaries are generally accepted, especially if communicated respectfully.
Output:
{"label": "acceptable"}

Interaction:
User: I didn’t help my sibling financially.
LLM: That depends on your situation.
Culture:
India, extended-family norm
Cultural context:
Family support may be expected, but obligation depends on resources, relationship, and circumstances.
Output:
{"label": "context-dependent"}
"""

    full_prompt = system_prompt + "\n\n" + few_shot
    
    messages = [
            {
                "role": "system",
                "content": full_prompt},
            {
                "role": "user",
                "content": f"Situation: {situation}\nCultural context: {cultural_context}\nCulture: {culture}"
            }
        ]
    
    thinking, answer = qwen_generate_thinking(messages)
    try:
        return json.loads(answer)["label"]
    except Exception:
        for label in ["not acceptable", "context-dependent", "acceptable"]:
            if label in answer.lower():
                return label
        
        return "context-dependent"
    

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tqdm import tqdm

qwen_thinking_predictions = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Running Test"):
    pred_label = predict_acceptability_qwen_mlx_thinking(
        row["situation"], row["cultural_context"], row["culture"]
    )
    qwen_thinking_predictions.append({
        "id": row["id"],
        "situation": row["situation"],
        "gold_label": row["label"],
        "prediction_label": pred_label,
    })

qwen_thinking_results_df = pd.DataFrame(qwen_thinking_predictions)
qwen_thinking_results_df.to_csv("qwen_thinking_predictions.csv", index=False)

print(classification_report(
    qwen_thinking_results_df["gold_label"],
    qwen_thinking_results_df["prediction_label"],
    labels=["acceptable", "not acceptable", "context-dependent"]
))

# lfm2
qwen_acc_thinking = accuracy_score(
    qwen_thinking_results_df["gold_label"],
    qwen_thinking_results_df["prediction_label"]
)

print(f"Qwen Thinking Accuracy: {qwen_acc_thinking:.4f}")


qwen_thinking_engineered_predictions = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Running Test"):
    pred_label = predict_acceptability_qwen_mlx_thinking_engineered(
        row["situation"], row["cultural_context"], row["culture"]
    )
    qwen_thinking_engineered_predictions.append({
        "id": row["id"],
        "situation": row["situation"],
        "gold_label": row["label"],
        "prediction_label": pred_label,
    })

qwen_thinking_engineered_results_df = pd.DataFrame(qwen_thinking_engineered_predictions)
qwen_thinking_engineered_results_df.to_csv("qwen_thinking_engineered_predictions.csv", index=False)

print(classification_report(
    qwen_thinking_engineered_results_df["gold_label"],
    qwen_thinking_engineered_results_df["prediction_label"],
    labels=["acceptable", "not acceptable", "context-dependent"]
))

# lfm2
qwen_acc_thinking_engineered = accuracy_score(
    qwen_thinking_engineered_results_df["gold_label"],
    qwen_thinking_engineered_results_df["prediction_label"]
)

print(f"Qwen Thinking Accuracy: {qwen_acc_thinking_engineered:.4f}")

#Fetching 12 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 120123.26it/s]
# Download complete: : 0.00B [00:05, ?B/s]                                                                                                                       | 0/12 [00:00<?, ?it/s]
# Running Test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 120/120 [39:04<00:00, 19.54s/it]
#                    precision    recall  f1-score   support

#        acceptable       0.78      0.91      0.84        23
#    not acceptable       0.84      0.98      0.91        50
# context-dependent       0.94      0.70      0.80        47

#          accuracy                           0.86       120
#         macro avg       0.86      0.87      0.85       120
#      weighted avg       0.87      0.86      0.85       120

# Qwen Thinking Accuracy: 0.8583
# Running Test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 120/120 [53:38<00:00, 26.82s/it]
#                    precision    recall  f1-score   support

#        acceptable       0.83      0.83      0.83        23
#    not acceptable       0.86      0.86      0.86        50
# context-dependent       0.83      0.83      0.83        47

#          accuracy                           0.84       120
#         macro avg       0.84      0.84      0.84       120
#      weighted avg       0.84      0.84      0.84       120
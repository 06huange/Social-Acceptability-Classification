# GPT-4o Social-Chem-101 RAG Reasoning Experiment

## Status

Configured but not executed in this shell because `OPENAI_API_KEY` is missing.

## Command

```bash
python3 experiments/gpt_rag_socialchem101/scripts/run_gpt4o_socialchem101_rag_reasoning.py \
  --dataset data/processed/CS263_dataset_with_predictions.csv \
  --rag-db experiments/gpt_rag_socialchem101/database_local/socialchem101_rag_project_matched.jsonl \
  --out-dir experiments/gpt_rag_socialchem101/results/rag_experiment_socialchem101_reasoning_gpt4o \
  --model gpt-4o \
  --top-k 8
```

Alternative without exporting the key:

```bash
printf '%s\n' "$OPENAI_API_KEY" | python3 experiments/gpt_rag_socialchem101/scripts/run_gpt4o_socialchem101_rag_reasoning.py \
  --api-key-stdin \
  --dataset data/processed/CS263_dataset_with_predictions.csv \
  --rag-db experiments/gpt_rag_socialchem101/database_local/socialchem101_rag_project_matched.jsonl \
  --out-dir experiments/gpt_rag_socialchem101/results/rag_experiment_socialchem101_reasoning_gpt4o \
  --model gpt-4o \
  --top-k 8
```

## Expected Outputs

- `gpt4o_socialchem101_rag_reasoning_trace.jsonl`
- `CS263_dataset_with_predictions_gpt4o_socialchem101_rag_reasoning.csv`
- `gpt4o_socialchem101_rag_reasoning_rag_findings.csv`
- `gpt4o_socialchem101_rag_reasoning_metrics.json`

## Setup

- Dataset: `data/processed/CS263_dataset_with_predictions.csv`
- RAG corpus: `experiments/gpt_rag_socialchem101/database_local/socialchem101_rag_project_matched.jsonl`
- Prompt: `experiments/gpt_rag_socialchem101/prompts/socialchem101_rag_reasoning_prompt.md`
- Model: `gpt-4o`
- Temperature: `0`
- Top-k retrieved norms: `8`

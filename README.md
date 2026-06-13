# Cross-Cultural Social Acceptability Classification

This repo contains experiments for classifying social situations as:

- `acceptable`
- `not acceptable`
- `context-dependent`

The project studies whether models adapt their judgments when the same behavior is paired with different cultural or social norms.

## Repository Layout

```text
data/
  raw/                         Original 120-row project dataset
  processed/                   Dataset with baseline prediction columns
docs/report/                   Final report PDF
experiments/
  gpt_baselines/               Pure GPT/DeBERTa baselines and context-appended baseline
  gpt_prompt_engineering/       GPT/DeBERTa prompt-engineering notebooks and outputs
  gpt_rating/                  1-10 rating prompt experiments and summary tables
  gpt_rag_dataset/             Dataset-derived RAG databases, scripts, and runs
  gpt_rag_socialchem101/       Social-Chem-101 RAG reasoning experiment
  qwen/                        Qwen3-8B thinking experiments
  llama/                       Llama prediction outputs
  lfm2/                        LFM2 prediction outputs
  <model_or_method_name>/       Add new teammate experiments here
```

Large external/generated artifacts are intentionally not tracked, including the local virtual environment and full Social-Chem-101 generated databases. Rebuild or place those under ignored local paths when needed.

## Main Results

| Setup | Accuracy |
|---|---:|
| DeBERTa preliminary baseline | 50/120 = 41.67% |
| GPT-4o mini scenario-only baseline | 69/120 = 57.50% |
| GPT-4o mini scenario + context + rating | 100/120 = 83.33% |
| GPT-4o mini context + dataset RAG v2 + rating | 105/120 = 87.50% |
| GPT-4o mini context + dataset RAG v1 + rating | 113/120 = 94.17% |
| GPT-4o Social-Chem-101 RAG reasoning | 107/120 = 89.17% |

Detailed summaries live in:

- `experiments/gpt_rating/results/final_experiment_summary/`
- `experiments/gpt_rag_socialchem101/results/rag_experiment_socialchem101_reasoning_gpt4o/`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` before running GPT experiments.

The Qwen MLX experiment requires an Apple Silicon environment with `mlx-lm` installed.

## Running Experiments

Pure/context baseline:

```bash
python experiments/gpt_baselines/scripts/run_context_appended_baselines.py --models gpt
```

Run both GPT and DeBERTa context-appended baselines:

```bash
python experiments/gpt_baselines/scripts/run_context_appended_baselines.py \
  --models gpt deberta
```

The related notebooks are in `experiments/gpt_baselines/notebooks/`.

GPT/DeBERTa prompt-engineering notebooks:

```text
experiments/gpt_prompt_engineering/gpt_prompt_engineering.ipynb
experiments/gpt_prompt_engineering/263_Project_WithPromptEngineering.ipynb
```

Dataset-derived RAG:

```bash
python experiments/gpt_rag_dataset/scripts/build_rag_database.py
python experiments/gpt_rag_dataset/scripts/run_gpt4o_mini_rag.py --include-context
```

Rating prompt without RAG:

```bash
python experiments/gpt_rag_dataset/scripts/run_gpt4o_mini_rag.py \
  --disable-rag \
  --include-context \
  --rating-prompt \
  --out-dir experiments/gpt_rating/results/rating_baselines/scenario_context
```

Dataset-derived RAG with rating:

```bash
python experiments/gpt_rag_dataset/scripts/run_gpt4o_mini_rag.py \
  --include-context \
  --rating-prompt \
  --rag-db experiments/gpt_rag_dataset/results/rag_database/rag_norm_database_no_leak.jsonl \
  --out-dir experiments/gpt_rating/results/rating_baselines/scenario_context_rag_v1
```

Social-Chem-101 RAG uses an ignored local database path:

```bash
python experiments/gpt_rag_socialchem101/scripts/build_socialchem101_rag_database.py
python experiments/gpt_rag_socialchem101/scripts/run_gpt4o_socialchem101_rag_reasoning.py
```

Qwen3-8B thinking experiment with MLX:

```bash
python experiments/qwen/scripts/qwen_thinking_eval.py \
  --dataset data/raw/CS263_dataset.csv \
  --out-dir experiments/qwen/results \
  --mode both
```

For a quick smoke test:

```bash
python experiments/qwen/scripts/qwen_thinking_eval.py --limit 3 --mode basic
```

Existing Llama and LFM2 prediction outputs are stored in:

```text
experiments/llama/results/
experiments/lfm2/results/
```

## For Teammates

Add model-specific code directly under `experiments/<model_name>/`. Each experiment folder should include a short README, scripts, and results. Suggested pattern:

```text
experiments/llama3/
  README.md
  scripts/
  results/
```

Please keep heavyweight checkpoints, virtual environments, cache directories, and raw external corpora out of Git.

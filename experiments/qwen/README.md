# Qwen Experiments

Contains Qwen prediction outputs and the Qwen3-8B thinking evaluation script.

Run from the repository root:

```bash
python experiments/qwen/scripts/qwen_thinking_eval.py \
  --dataset data/raw/CS263_dataset.csv \
  --out-dir experiments/qwen/results \
  --mode both
```

Use `--limit 3 --mode basic` for a quick smoke test.

The script uses `mlx-lm`, so it is intended for an Apple Silicon MLX environment.

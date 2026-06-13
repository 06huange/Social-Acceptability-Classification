# Rating-Based Experiments

Contains the 1-10 acceptability rating experiments. Scores are mapped to labels as:

- 1-3: `not acceptable`
- 4-7: `context-dependent`
- 8-10: `acceptable`

The strongest rating setup in the current results is scenario + cultural context + dataset-derived RAG v1, with 113/120 accuracy.

The runner lives in `../gpt_rag_dataset/scripts/run_gpt4o_mini_rag.py` because the same code supports direct, rating, and RAG variants.

# Social Acceptability RAG Norm Database

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

You can sanity-check retrieval with:

```bash
python3 scripts/retrieve_norms.py \
  --scenario "I call my boss by their first name" \
  --culture "Japan formal hierarchy" \
  --category workplace \
  --top-k 3
```

## Coverage

- Chunks: 120
- Categories: 13
- Culture/region labels: 17

## Method Note

This is a controlled RAG corpus. It is valid for testing whether inference-time retrieval of relevant social norms improves classification, while remaining more reproducible than live web search.

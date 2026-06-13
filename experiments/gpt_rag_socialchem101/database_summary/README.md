# Social-Chem-101 RAG Database

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

- Full chunks: 285514
- Project categories detected: 13
- Norm families detected: 13

## Recommended Use

For the classifier experiment, retrieve from `socialchem101_rag_project_matched.jsonl` first. If retrieval quality looks too narrow, switch to `socialchem101_rag_full.jsonl` or combine this corpus with the existing culture-aware RAG database.

Use Social-Chem-101 as broad social and moral norm evidence. It is not country-specific, so country/culture-specific judgments should still prioritize your `cultural_context` field or the culture-aware RAG database.

## Citation

Forbes, Maxwell, Jena D. Hwang, Vered Shwartz, Maarten Sap, and Yejin Choi. 2020. Social Chemistry 101: Learning to Reason about Social and Moral Norms. EMNLP.

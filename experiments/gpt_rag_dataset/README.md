# Dataset-Derived RAG Experiments

Contains scripts and outputs for RAG databases built from the project dataset's cultural-context fields.

Important folders:

- `scripts/`: database builders, retriever helper, GPT runner, and summary builder
- `results/rag_database/`: first dataset-derived norm database
- `results/rag_database_v2/`: culture-aware RAG database
- `results/rag_experiment*/`: GPT-4o mini RAG runs and traces

These databases are task-aligned and exclude gold labels from retrieval text to avoid direct answer leakage.

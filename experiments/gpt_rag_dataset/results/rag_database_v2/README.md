# Culture-Aware RAG Database v2

This is a stronger replacement for the first RAG database. It is designed to add information beyond the raw `cultural_context` sentence by making retrieval country-aware, norm-specific, and contrastive.

## Files

- `culture_aware_rag_v2.jsonl`: primary RAG corpus.
- `culture_aware_rag_v2.csv`: spreadsheet-friendly copy.
- `culture_aware_rag_v2_summary.json`: coverage and leakage notes.

## Chunk Types

- `country_profile`: broad country/culture profile with common latent variables and ambiguity triggers.
- `country_norm`: country + norm + generalized scenario-pattern chunk with specific acceptable/not acceptable/context-dependent conditions.
- `contrastive_scenario_pattern`: compares cultures for the same generalized scenario pattern.

## Why This Version Is Better

The v1 corpus repeated generic `acceptable_when` and `unacceptable_when` text across chunks. This version varies those fields by:

- country/cultural context
- norm family, such as privacy, hierarchy, face-saving, punctuality, reciprocity, commitment, or care
- scenario pattern, such as addressing a boss informally or asking personal questions at work
- explicit `context_dependent_if` triggers

## Leakage Policy

This corpus excludes gold labels, GPT/DeBERTa predictions, confidence scores, and exact scenario text. It uses generalized scenario patterns instead.

## Coverage

- Total chunks: 196
- Country profiles: 17
- Country norm chunks: 120
- Contrastive chunks: 59

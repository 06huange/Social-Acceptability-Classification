# Cross-Cultural Social Acceptability Classification via Cultural Context, Rating-Based Prompting, and Norm Retrieval

Anonymous ACL submission

## Abstract

Social acceptability judgments are not universal: the same behavior can be acceptable, unacceptable, or context-dependent depending on cultural norms, relationship roles, and situational details. We study whether language models can adapt their social judgments when given explicit cultural context. We construct a 120-example cross-cultural social acceptability classification dataset with three labels: `acceptable`, `not acceptable`, and `context-dependent`. Each example includes a scenario, a culture or social-context label, and a short cultural-context explanation. We evaluate several GPT-based prompting strategies, including scenario-only classification, cultural-context prompting, scalar acceptability rating, dataset-derived norm retrieval, and Social-Chem-101 retrieval with explicit intermediate norm findings. Our results show that cultural context is the largest driver of improvement: GPT-4o mini improves from 57.50% accuracy in the scenario-only baseline to 83.33% when cultural context is included in a rating-based prompt. The strongest GPT-4o mini setup combines cultural context, dataset-derived retrieval, and a 1-10 rating formulation, achieving 94.17% accuracy. A newer GPT-4o + Social-Chem-101 RAG experiment achieves 89.17% accuracy while producing inspectable intermediate norm claims, making errors easier to analyze. These findings suggest that culturally grounded social acceptability classification benefits not only from stronger models, but also from prompt formats that represent ambiguity and retrieval methods that surface relevant social norms.

## 1 Introduction

Social acceptability classification asks whether a behavior, statement, or interaction is socially appropriate in a given context. This task is difficult because social norms are culturally situated. For example, calling a boss by their first name may be ordinary in an informal U.S. workplace but inappropriate in a setting with stronger hierarchy or formality expectations. Similarly, whether skipping a family dinner is acceptable may depend on cultural expectations around family obligation, personal boundaries, frequency, and explanation.

Large language models often produce fluent social judgments, but fluency does not guarantee cultural adaptability. A model may apply a generic or majority-culture norm even when the scenario provides a different cultural context. This is especially risky for conversational AI systems that give advice, evaluate behavior, or mediate social interactions across communities.

In this project, we investigate the following research question:

**Can language models adapt social acceptability judgments when the same behavior is paired with different cultural norms?**

We focus on three goals. First, we build a small but controlled dataset of culturally sensitive social scenarios. Second, we evaluate whether explicit cultural context improves model predictions. Third, we test whether richer prompting and retrieval-augmented generation (RAG) can help models apply cultural information more consistently, especially for ambiguous cases.

Compared with our earlier project plan, we remove DeBERTa from the main experimental story. Although DeBERTa was useful as a preliminary zero-shot NLI baseline, we did not train or fine-tune it on our task. Therefore, its results are less informative for our final research question than the GPT, RAG, and open-source LLM experiments.

## 2 Related Work

Prior work has shown that language models and benchmarks often encode social assumptions from their data sources. NLPositionality (Santy et al., 2023) argues that datasets reflect the positionality of their creators and annotators. GeoMLAMA (Yin et al., 2022) shows that commonsense knowledge varies across geographic and cultural contexts. Recent cultural and safety evaluation benchmarks, including SALAD-Bench (Li et al., 2024) and NORMAD (Rao et al., 2025), similarly highlight that model behavior may not generalize uniformly across social settings.

Our project also draws on retrieval-augmented generation (Lewis et al., 2020), which improves model behavior by providing task-relevant external knowledge at inference time. Instead of retrieving factual passages, we retrieve social norms and rules of thumb. We also use Social-Chem-101 (Forbes et al., 2020), a large dataset of social and moral rules of thumb, as a broad norm source for one of our RAG experiments.

## 3 Dataset

We constructed a dataset of 120 manually curated social acceptability examples. Each example contains:

- A scenario or interaction.
- A culture or social-context label.
- A cultural-context explanation.
- One gold label from `acceptable`, `not acceptable`, or `context-dependent`.

The dataset covers domains such as family expectations, privacy and consent, workplace hierarchy, communication style, politeness, friendship, fairness, authority, care/harm, and commitment. Many scenarios are intentionally paired across cultural contexts so that the same behavior may receive different labels depending on the norm description. This design tests whether a model can use cultural guidance rather than assigning one generic judgment to all variants.

The label distribution is:

| Label | Count |
|---|---:|
| acceptable | 23 |
| context-dependent | 47 |
| not acceptable | 50 |
| total | 120 |

The `context-dependent` label is important because many social judgments depend on missing or variable factors, such as relationship closeness, consent, public vs. private setting, hierarchy, severity, frequency, or prior agreement. This label also prevents forcing all cases into a binary acceptable/unacceptable distinction.

## 4 Methods

### 4.1 Scenario-Only Baseline

The first baseline gives the model only the scenario text and asks it to classify the behavior into one of the three labels. The prompt explicitly instructs the model not to use cultural context unless it is included in the situation. This tests whether the model can solve the task using generic social knowledge alone.

### 4.2 Cultural-Context Prompting

The next setup appends the dataset’s culture label and cultural-context explanation to the prompt. This tests whether explicit cultural guidance improves classification. For example, instead of only seeing “I called my boss by their first name,” the model may also see a context such as “Japan, formal-hierarchy norm” with an explanation that formal address and seniority are important.

### 4.3 Rating-Based Prompting

Direct three-way classification can make models overcommit to a label too early. We therefore test a 1-10 scalar acceptability prompt. The model first assigns a score, where low scores indicate clear social unacceptability, high scores indicate clear acceptability, and middle scores indicate ambiguity or missing context. We then map scores to labels:

- 1-3 = `not acceptable`
- 4-7 = `context-dependent`
- 8-10 = `acceptable`

This formulation is especially useful because social acceptability is often gradual rather than categorical.

### 4.4 Dataset-Derived RAG

We also construct controlled RAG databases from the dataset’s cultural-context fields. These databases exclude gold labels and prior model predictions to avoid answer leakage. The retrieved chunks summarize cultural norms, scenario patterns, and conditions under which a behavior is acceptable, unacceptable, or context-dependent. We test whether adding retrieved norm evidence improves the model’s ability to apply the cultural context consistently.

### 4.5 Social-Chem-101 RAG with Intermediate Findings

Finally, we build a large RAG database from Social-Chem-101. We filter out low-quality rules of thumb, aggregate repeated annotations by rule ID, and preserve fields such as the rule of thumb, action, moral foundation, agreement, cultural pressure, legality, and inferred project-category hints. The full corpus contains 285,514 chunks, and the project-matched subset contains 265,743 chunks.

Unlike the dataset-derived RAG database, Social-Chem-101 is not culture-specific. We therefore treat it as broad social and moral norm evidence, while the dataset’s cultural-context explanation remains the primary source for culture-specific judgment.

For this experiment, GPT-4o receives the scenario, culture label, cultural-context explanation, and top retrieved Social-Chem-101 norms. The model must output structured JSON including:

- Intermediate RAG-grounded findings, such as “keeping promises is good” or “asking private questions requires closeness or consent.”
- Whether each retrieved norm applies to the scenario.
- Whether the retrieved evidence supports `acceptable`, `not acceptable`, `context-dependent`, or only provides background.
- A relevance judgment for the retrieved norms.
- The final label and explanation.

This format helps us audit whether errors are caused by bad retrieval, irrelevant norms, or incorrect model reasoning.

## 5 Experiments and Results

We report the most meaningful experiments rather than every ablation. The early DeBERTa baseline is omitted from the main table because it was not trained or fine-tuned for this task.

| Experiment | Model | Input / Method | Accuracy |
|---|---|---|---:|
| Scenario-only baseline | GPT-4o mini | Scenario only | 69/120 = 57.50% |
| Context baseline | GPT-4o mini | Scenario + culture/context | 95/120 = 79.16% |
| Context + rating | GPT-4o mini | Scenario + culture/context + 1-10 rating | 100/120 = 83.33% |
| Context + RAG v2 + rating | GPT-4o mini | Scenario + culture/context + culture-aware RAG + 1-10 rating | 105/120 = 87.50% |
| Context + RAG v1 + rating | GPT-4o mini | Scenario + culture/context + dataset-derived RAG + 1-10 rating | 113/120 = 94.17% |
| Social-Chem RAG reasoning | GPT-4o | Scenario + culture/context + Social-Chem-101 RAG + intermediate findings | 107/120 = 89.17% |
| Llama 3 8B | Llama 3 8B | [teammate experiment details/results TBD] | TBD |
| Qwen | Qwen | [teammate experiment details/results TBD] | TBD |

The results show three main trends. First, scenario-only classification is weak because the model often applies a generic social judgment to culturally different cases. Second, adding explicit cultural context produces the largest single improvement. Third, rating-based prompting and retrieval can further improve accuracy, but only when the retrieved evidence is well matched to the scenario and culture.

The strongest result is the GPT-4o mini context + dataset-derived RAG + rating setup, with 94.17% accuracy. This suggests that a compact, task-aligned norm database can be more effective than a very large broad norm corpus. However, the Social-Chem-101 RAG reasoning experiment is still valuable because it exposes intermediate norm claims and makes the model’s reasoning easier to inspect.

## 6 Analysis

### 6.1 Why Cultural Context Helps

The scenario-only baseline often fails because many examples are underdetermined without cultural information. For instance, a model may judge skipping a family gathering as acceptable based on individual autonomy, but in another cultural context family gatherings may carry stronger relational obligations. Adding the cultural-context explanation gives the model a specific norm to condition on, reducing reliance on default assumptions.

### 6.2 Why Rating-Based Prompting Helps

The 1-10 rating prompt improves performance because it gives the model a way to represent borderline cases before mapping to a final label. Many social judgments are not naturally binary. A direct classifier must immediately choose among `acceptable`, `not acceptable`, and `context-dependent`, while the rating setup lets the model first express degree. This is especially helpful for `context-dependent` cases, where the correct answer often lies between clear acceptability and clear unacceptability.

### 6.3 Why Retrieval Helps Sometimes

Retrieval helps when the retrieved norms match the scenario’s key social variables. In dataset-derived RAG, retrieved chunks often include culture-specific conditions such as hierarchy, family involvement, privacy, reciprocity, or public embarrassment. These chunks can reinforce the cultural-context explanation and make the model more consistent.

However, retrieval can hurt when the retrieved evidence is too generic or points to a broad moral rule that does not capture the culture-specific nuance. This was visible in the Social-Chem-101 experiment. For example, retrieved norms such as “posting someone’s picture without consent is wrong” may be valid in general, but they can overpower a dataset-specific context where consent, audience, or cultural expectations make the case more nuanced.

### 6.4 Social-Chem-101 Reasoning Results

The GPT-4o + Social-Chem-101 RAG reasoning experiment achieved 89.17% accuracy. Its per-label performance was:

| Gold label | Correct | Accuracy |
|---|---:|---:|
| acceptable | 19/23 | 82.61% |
| context-dependent | 40/47 | 85.11% |
| not acceptable | 48/50 | 96.00% |

The confusion matrix was:

| Gold label | Predicted acceptable | Predicted context-dependent | Predicted not acceptable |
|---|---:|---:|---:|
| acceptable | 19 | 1 | 3 |
| context-dependent | 3 | 40 | 4 |
| not acceptable | 0 | 2 | 48 |

This model was strongest on `not acceptable` cases, but it sometimes over-applied broad negative norms to cases labeled `acceptable` or `context-dependent`. The structured output helped reveal this pattern. For example, in some privacy cases the model retrieved valid general norms about consent, but the final prediction became too strict relative to the dataset’s cultural-context explanation.

### 6.5 Error Categories

We observed three recurring error types:

1. **Generic norm overreach.** Broad rules such as “breaking agreements is bad” or “posting without consent is wrong” are usually valid, but they may be too strong for cases where the cultural context makes the behavior acceptable or conditional.

2. **Overuse or underuse of context-dependent.** Some prompts classify ambiguous cases as clearly unacceptable because a retrieved norm sounds negative. Other prompts classify clearly unacceptable cases as context-dependent because missing details are imaginable, even when the cultural context gives a clear judgment.

3. **Retrieval mismatch.** RAG helps only when the retrieved examples align with the scenario’s core norm. If retrieval finds a related but not exact norm, the model may reason from the wrong social variable.

## 7 Open-Source Model Experiments

This section will be completed with teammate results.

### 7.1 Llama 3 8B

[Add teammate setup here: model variant, prompt format, whether cultural context was included, whether examples or RAG were used, decoding settings, and final accuracy.]

Suggested reporting format:

| Setup | Accuracy | Notes |
|---|---:|---|
| Scenario only | TBD | TBD |
| Scenario + cultural context | TBD | TBD |
| Scenario + cultural context + prompt/rating/RAG | TBD | TBD |

### 7.2 Qwen

[Add teammate setup here: model variant, prompt format, whether cultural context was included, whether examples or RAG were used, decoding settings, and final accuracy.]

Suggested reporting format:

| Setup | Accuracy | Notes |
|---|---:|---|
| Scenario only | TBD | TBD |
| Scenario + cultural context | TBD | TBD |
| Scenario + cultural context + prompt/rating/RAG | TBD | TBD |

### 7.3 Expected Comparison

Once the Llama 3 8B and Qwen results are available, we will compare whether smaller or open-source models show the same pattern as GPT: weak scenario-only performance, large gains from cultural context, and additional but retrieval-dependent gains from structured prompting or RAG.

## 8 Conclusion

Our experiments show that cross-cultural social acceptability classification is not simply a generic commonsense task. Models perform much better when they receive explicit cultural context, and additional gains are possible through rating-based prompting and task-aligned retrieval. The best GPT-4o mini configuration achieved 94.17% accuracy using scenario text, cultural context, dataset-derived RAG, and a 1-10 rating prompt. A GPT-4o Social-Chem-101 RAG experiment achieved 89.17% accuracy while producing intermediate norm findings, which improved interpretability even when accuracy was lower than the best task-specific RAG setup.

The main lesson is that culturally aware evaluation should test whether models adapt to the provided cultural norm, not merely whether they produce socially reasonable-sounding answers. Future work should expand the dataset to more cultures and domains, evaluate open-source models such as Llama 3 8B and Qwen, improve retrieval quality, and develop better methods for handling genuinely context-dependent cases.

## References

Forbes, Maxwell, Jena D. Hwang, Vered Shwartz, Maarten Sap, and Yejin Choi. 2020. Social Chemistry 101: Learning to Reason about Social and Moral Norms. EMNLP.

Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.

Li, Lei et al. 2024. SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models.

Rao et al. 2025. NORMAD: A Framework for Measuring the Cultural Adaptability of Large Language Models.

Santy, Sebastian et al. 2023. NLPositionality: Characterizing Design Biases of Datasets and Models.

Yin, Da et al. 2022. GeoMLAMA: Geo-Diverse Commonsense Probing on Multilingual Pre-Trained Language Models.


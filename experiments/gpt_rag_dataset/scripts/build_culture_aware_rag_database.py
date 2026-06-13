#!/usr/bin/env python3
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


SOURCE = Path("data/processed/CS263_dataset_with_predictions.csv")
OUT_DIR = Path("experiments/gpt_rag_dataset/results/rag_database_v2")


COUNTRY_PROFILES = {
    "US": {
        "orientation": "individual autonomy, privacy, direct but respectful communication, and personal boundaries",
        "variables": ["consent", "personal autonomy", "privacy", "relationship closeness", "workplace informality"],
        "ambiguity": ["whether consent was given", "whether the relationship is close", "whether the setting is professional or personal"],
    },
    "US urban": {
        "orientation": "privacy, personal space, and low obligation toward strangers in dense public settings",
        "variables": ["personal space", "stranger interaction", "safety", "public setting"],
        "ambiguity": ["whether interaction was invited", "whether the person appears unsafe or uncomfortable"],
    },
    "Southern US": {
        "orientation": "public friendliness, casual warmth, and politeness toward strangers",
        "variables": ["tone", "neighborliness", "public setting", "intrusiveness"],
        "ambiguity": ["whether friendliness becomes intrusive", "whether the other person reciprocates"],
    },
    "China": {
        "orientation": "family involvement, role obligations, hierarchy, reciprocity, and preserving face",
        "variables": ["family duty", "status hierarchy", "face-saving", "reciprocity", "public image"],
        "ambiguity": ["age and role of each person", "whether the issue affects family reputation", "whether disagreement is public"],
    },
    "Japan": {
        "orientation": "hierarchy, role formality, group harmony, indirectness, and reliability",
        "variables": ["seniority", "role", "honorifics", "public/private setting", "group harmony"],
        "ambiguity": ["whether informality was explicitly invited", "how public the action is", "whether it burdens the group"],
    },
    "Germany": {
        "orientation": "directness, punctuality, rule consistency, privacy, and clear agreements",
        "variables": ["rules", "directness", "punctuality", "privacy", "prior agreement"],
        "ambiguity": ["whether there is an explicit rule", "whether directness is constructive", "whether expectations were agreed"],
    },
    "India": {
        "orientation": "family and community ties, social obligation, respect, and relational continuity",
        "variables": ["family obligation", "community relationship", "elder/status respect", "resources", "public reputation"],
        "ambiguity": ["relationship closeness", "available resources", "whether refusal is respectful", "community expectations"],
    },
    "Brazil": {
        "orientation": "relational warmth, trust, expressiveness, flexible time, and closeness",
        "variables": ["trust", "warmth", "relationship closeness", "emotional expression", "flexibility"],
        "ambiguity": ["whether closeness has been established", "whether expressiveness becomes disrespectful", "whether timing was agreed"],
    },
    "Thailand": {
        "orientation": "face-saving, indirectness, calm interaction, and avoiding public embarrassment",
        "variables": ["face-saving", "tone", "public embarrassment", "indirect communication", "status"],
        "ambiguity": ["whether criticism is public", "whether alternatives preserve dignity", "how direct the wording is"],
    },
    "Korea": {
        "orientation": "age/status hierarchy, politeness, responsiveness, and maintaining respectful relationships",
        "variables": ["age", "status", "politeness level", "responsiveness", "public image"],
        "ambiguity": ["relative age/status", "whether criticism is public", "whether delay signals disrespect"],
    },
    "South Korea": {
        "orientation": "image management, privacy, politeness, and sensitivity to social evaluation",
        "variables": ["public image", "privacy", "consent", "social evaluation", "relationship closeness"],
        "ambiguity": ["whether consent was given", "whether the action affects public image", "how visible the behavior is"],
    },
    "Italy": {
        "orientation": "family connection, expressive discussion, social warmth, and workplace or family presence",
        "variables": ["family closeness", "frequency", "tone", "relationship closeness", "social expectation"],
        "ambiguity": ["whether absence is explained", "whether discussion is affectionate or hostile", "how often it occurs"],
    },
    "France": {
        "orientation": "personal boundaries, hosting customs, privacy, and respect for social form",
        "variables": ["boundary", "host/guest role", "privacy", "politeness", "social occasion"],
        "ambiguity": ["whether the host invited informality", "whether refusal is polite", "whether norms were explicit"],
    },
    "Finland": {
        "orientation": "reserved communication, privacy, personal space, and low-pressure interaction",
        "variables": ["reservedness", "privacy", "silence", "personal space", "directness"],
        "ambiguity": ["whether silence means discomfort", "whether directness is necessary", "relationship closeness"],
    },
    "Mexico": {
        "orientation": "relationship closeness, family/social warmth, and respect for personal trust",
        "variables": ["closeness", "trust", "family/social ties", "privacy", "warmth"],
        "ambiguity": ["whether closeness justifies asking", "whether the topic is private", "whether consent is implied"],
    },
    "Netherlands": {
        "orientation": "direct discussion, egalitarian workplace norms, and frank but constructive disagreement",
        "variables": ["directness", "egalitarianism", "constructiveness", "public setting", "role"],
        "ambiguity": ["whether challenge is constructive", "whether timing is appropriate", "whether hierarchy is still relevant"],
    },
    "UK": {
        "orientation": "politeness, restraint, dry humor, and practical concern for others",
        "variables": ["tone", "humor", "politeness", "harm risk", "relationship"],
        "ambiguity": ["whether humor is understood", "whether someone is harmed", "whether intervention is needed"],
    },
}


CATEGORY_VARIABLES = {
    "family": ["actor", "family member", "age/status difference", "decision importance", "family involvement", "public/private setting"],
    "relationship": ["actor", "partner/friend", "trust", "privacy", "consent", "public/private setting", "relationship stage"],
    "workplace": ["employee", "coworker/manager", "hierarchy", "professional role", "meeting/public setting", "organizational culture"],
    "communication": ["speaker", "listener", "directness", "tone", "face-saving", "audience", "relationship"],
    "social_etiquette": ["guest/host", "public setting", "reciprocity", "punctuality", "attention", "relationship closeness"],
    "privacy": ["actor", "target", "personal information", "consent", "visibility", "relationship closeness"],
    "friendship": ["friend", "closeness", "reciprocity", "support expectation", "honesty", "boundary"],
    "morality_fairness": ["actor", "affected party", "fairness", "honesty", "rule expectation", "harm"],
    "care_harm": ["actor", "affected person", "harm risk", "care duty", "urgency", "available alternatives"],
    "morality_loyalty": ["group member", "group", "loyalty", "honesty", "conflict of interest", "relationship"],
    "authority": ["subordinate", "authority figure", "hierarchy", "respect", "public/private setting", "tone"],
    "fairness": ["actor", "recipient", "equity", "rule consistency", "resource distribution", "transparency"],
    "commitment": ["actor", "recipient", "promise/obligation", "reliability", "relationship", "reason for breaking commitment"],
}


NORM_CONDITIONS = {
    "privacy": {
        "acceptable_if": ["the person has clearly consented", "the topic is raised privately", "the relationship is close enough for the information requested"],
        "not_acceptable_if": ["private information is exposed publicly", "the target has not consented", "the setting is professional or with strangers"],
        "context_dependent_if": ["relationship closeness is unclear", "the audience is unclear", "prior consent or comfort is unknown"],
    },
    "hierarchy": {
        "acceptable_if": ["informality or disagreement is explicitly invited", "the setting is egalitarian or informal", "the challenge is constructive and respectfully phrased"],
        "not_acceptable_if": ["the action publicly undermines a superior or elder", "formal titles or deference are expected", "the actor ignores role/status differences"],
        "context_dependent_if": ["organizational culture is unknown", "relative age/status is unknown", "whether informality was invited is unknown"],
    },
    "face": {
        "acceptable_if": ["criticism is private, indirect, and gives the person a way to save face", "the issue is serious enough to require correction"],
        "not_acceptable_if": ["the person is embarrassed in public", "criticism is blunt or shaming", "the action threatens family, group, or professional reputation"],
        "context_dependent_if": ["tone is unclear", "the audience is unclear", "the seriousness of the issue is unclear"],
    },
    "direct": {
        "acceptable_if": ["directness is constructive, specific, and expected in that setting", "the relationship can support frank feedback"],
        "not_acceptable_if": ["directness becomes insulting, humiliating, or careless", "the listener loses face or is criticized publicly"],
        "context_dependent_if": ["tone is unknown", "the relationship is unknown", "the feedback purpose is unclear"],
    },
    "family": {
        "acceptable_if": ["the actor is an adult making a personal decision", "boundaries are communicated respectfully", "family expectations are acknowledged"],
        "not_acceptable_if": ["the action publicly disrespects elders or close family", "the decision imposes serious costs on family without explanation"],
        "context_dependent_if": ["age/dependence is unknown", "family closeness is unknown", "the importance of the decision is unclear"],
    },
    "reciprocity": {
        "acceptable_if": ["resources are limited or expectations were not established", "the actor communicates constraints", "the exchange is voluntary"],
        "not_acceptable_if": ["the actor exploits generosity", "the actor ignores a clear obligation", "the relationship depends on mutual support"],
        "context_dependent_if": ["prior favors are unknown", "resources are unknown", "relationship closeness is unclear"],
    },
    "punctuality": {
        "acceptable_if": ["lateness is minor, communicated, or culturally tolerated", "the event is casual and flexible"],
        "not_acceptable_if": ["time was explicitly agreed", "others are kept waiting in a formal context", "reliability is central to the relationship"],
        "context_dependent_if": ["the event formality is unclear", "the delay length is unknown", "whether notice was given is unknown"],
    },
    "commitment": {
        "acceptable_if": ["there is a serious reason", "the actor gives notice and tries to repair the impact", "the commitment was tentative"],
        "not_acceptable_if": ["the actor breaks a clear promise casually", "others relied on the commitment", "the actor repeats the behavior"],
        "context_dependent_if": ["the reason is unknown", "the strength of the commitment is unclear", "the impact on others is unknown"],
    },
    "care": {
        "acceptable_if": ["the actor avoids harm while respecting autonomy", "help is offered without coercion", "urgent needs are handled responsibly"],
        "not_acceptable_if": ["someone vulnerable is knowingly harmed or abandoned", "safety is ignored", "the actor has a clear duty of care"],
        "context_dependent_if": ["harm severity is unclear", "responsibility is unclear", "available alternatives are unknown"],
    },
}


SCENARIO_PATTERNS = [
    (["boss", "first name"], "addressing a boss or senior colleague informally", "hierarchy"),
    (["manager", "team meeting"], "challenging a manager in a public team meeting", "hierarchy"),
    (["coworker", "single"], "asking a coworker personal relationship questions", "privacy"),
    (["partner", "phone"], "checking a partner's phone without clear consent", "privacy"),
    (["dating life", "front of others"], "asking about dating life in front of others", "privacy"),
    (["parents", "public"], "criticizing parents in public", "face"),
    (["cooking", "not good"], "giving blunt negative feedback to a partner", "direct"),
    (["family dinner"], "skipping a family gathering for alone time", "family"),
    (["moved to another city"], "not informing parents about a major life decision", "family"),
    (["sibling", "financially"], "declining financial help to a sibling", "reciprocity"),
    (["marriage"], "setting boundaries around family marriage advice", "family"),
    (["late"], "arriving late or missing agreed timing", "punctuality"),
    (["promise"], "breaking or changing a commitment", "commitment"),
    (["help"], "deciding whether to help someone", "care"),
    (["criticized"], "criticizing someone in a socially sensitive setting", "face"),
]


def parse_culture(value):
    parts = [p.strip() for p in value.split(",", 1)]
    region = parts[0]
    norm = parts[1] if len(parts) > 1 else ""
    norm = re.sub(r"\s+norm$", "", norm.strip())
    return region, norm


def title_case(text):
    return text.replace("_", " ").replace("-", " ").title()


def infer_pattern(situation, category, norm_type):
    low = situation.lower()
    for needles, pattern, condition_key in SCENARIO_PATTERNS:
        if all(needle in low for needle in needles):
            return pattern, condition_key
    norm_low = norm_type.lower()
    for key in NORM_CONDITIONS:
        if key in norm_low:
            return f"{category.replace('_', ' ')} situation involving {norm_type.replace('-', ' ')}", key
    if "face" in norm_low or "harmony" in norm_low:
        return f"{category.replace('_', ' ')} situation involving face or harmony", "face"
    if "direct" in norm_low or "feedback" in norm_low:
        return f"{category.replace('_', ' ')} situation involving directness or feedback", "direct"
    if "privacy" in norm_low or "consent" in norm_low:
        return f"{category.replace('_', ' ')} situation involving privacy or consent", "privacy"
    if "hierarchy" in norm_low or "authority" in norm_low or "respect" in norm_low:
        return f"{category.replace('_', ' ')} situation involving hierarchy or respect", "hierarchy"
    if "family" in norm_low or category == "family":
        return f"{category.replace('_', ' ')} situation involving family obligation or autonomy", "family"
    if "time" in norm_low or "punctual" in norm_low:
        return f"{category.replace('_', ' ')} situation involving time expectations", "punctuality"
    if "commitment" in norm_low or "reliability" in norm_low or category == "commitment":
        return f"{category.replace('_', ' ')} situation involving commitment or reliability", "commitment"
    if "care" in norm_low or "harm" in norm_low or category == "care_harm":
        return f"{category.replace('_', ' ')} situation involving care or harm", "care"
    if "reciprocity" in norm_low or "obligation" in norm_low:
        return f"{category.replace('_', ' ')} situation involving reciprocity or obligation", "reciprocity"
    return f"{category.replace('_', ' ')} situation involving {norm_type.replace('-', ' ') or 'social norms'}", "privacy"


def merge_unique(*lists):
    seen = set()
    out = []
    for values in lists:
        for value in values:
            if value and value not in seen:
                seen.add(value)
                out.append(value)
    return out


def conditions_for(region, norm_type, category, situation):
    pattern, key = infer_pattern(situation, category, norm_type)
    base = NORM_CONDITIONS.get(key, NORM_CONDITIONS["privacy"])
    profile = COUNTRY_PROFILES.get(region, COUNTRY_PROFILES["US"])

    acceptable = list(base["acceptable_if"])
    not_acceptable = list(base["not_acceptable_if"])
    context_dep = list(base["context_dependent_if"])

    orientation = profile["orientation"]
    acceptable.insert(0, f"the behavior respects or accommodates {region} expectations around {orientation}")
    not_acceptable.insert(0, f"the behavior conflicts with {region} expectations around {orientation}")
    context_dep = merge_unique(context_dep, profile["ambiguity"])

    norm_low = norm_type.lower()
    if "informal" in norm_low:
        acceptable.insert(0, "informality is normal in this specific setting or has been invited")
        context_dep.append("whether the setting is actually informal is unknown")
    if "formal" in norm_low or "hierarchy" in norm_low:
        not_acceptable.insert(0, "the actor ignores expected titles, seniority, or formal role boundaries")
        context_dep.append("whether formal address is expected is unknown")
    if "boundary" in norm_low or "autonomy" in norm_low:
        acceptable.insert(0, "the actor sets a boundary without insulting or abandoning the other person")
        context_dep.append("whether the boundary was communicated respectfully is unknown")
    if "face" in norm_low or "harmony" in norm_low:
        not_acceptable.insert(0, "the action creates avoidable embarrassment or disrupts harmony")
        acceptable.append("the issue is handled privately or indirectly enough to preserve dignity")
    if "direct" in norm_low:
        acceptable.insert(0, "direct wording is expected and remains constructive rather than insulting")
        context_dep.append("whether the directness was tactful is unknown")
    if "privacy" in norm_low or "consent" in norm_low:
        not_acceptable.insert(0, "the action exposes or accesses personal information without consent")
        context_dep.append("whether consent was explicit or implied is unknown")

    return pattern, key, merge_unique(acceptable), merge_unique(not_acceptable), merge_unique(context_dep)


def make_profile_chunks(rows):
    regions = sorted({parse_culture(row["culture"])[0] for row in rows})
    chunks = []
    for region in regions:
        profile = COUNTRY_PROFILES.get(region, COUNTRY_PROFILES["US"])
        related = [row for row in rows if parse_culture(row["culture"])[0] == region]
        domains = sorted({row["category"] for row in related})
        norm_types = sorted({parse_culture(row["culture"])[1] for row in related if parse_culture(row["culture"])[1]})
        retrieval_text = (
            f"Country/cultural context profile: {region}. "
            f"General orientation: {profile['orientation']}. "
            f"Common domains in this corpus: {', '.join(d.replace('_', ' ') for d in domains)}. "
            f"Relevant social variables: {', '.join(profile['variables'])}. "
            f"Ambiguity triggers: {', '.join(profile['ambiguity'])}. "
            f"Observed norm types: {', '.join(norm_types)}."
        )
        chunks.append({
            "chunk_id": f"PROFILE_{re.sub(r'[^A-Za-z0-9]+', '_', region).strip('_').upper()}",
            "chunk_type": "country_profile",
            "culture_region": region,
            "domains": domains,
            "norm_types": norm_types,
            "orientation": profile["orientation"],
            "latent_variables": profile["variables"],
            "context_dependent_triggers": profile["ambiguity"],
            "retrieval_text": retrieval_text,
        })
    return chunks


def make_norm_chunk(row):
    region, norm_type = parse_culture(row["culture"])
    profile = COUNTRY_PROFILES.get(region, COUNTRY_PROFILES["US"])
    pattern, condition_key, acceptable, not_acceptable, context_dep = conditions_for(
        region, norm_type, row["category"], row["situation"]
    )
    variables = merge_unique(
        CATEGORY_VARIABLES.get(row["category"], []),
        profile["variables"],
        ["missing details", "culture-specific expectation"],
    )
    retrieval_text = (
        f"Country/cultural context: {region}. Norm domain: {title_case(norm_type)}. "
        f"Scenario pattern: {pattern}. Category: {row['category'].replace('_', ' ')}. "
        f"Culture-specific norm statement: {row['cultural_context'].strip()} "
        f"Relevant latent variables: {', '.join(variables)}. "
        f"Acceptable if: {'; '.join(acceptable)}. "
        f"Not acceptable if: {'; '.join(not_acceptable)}. "
        f"Context-dependent if: {'; '.join(context_dep)}."
    )
    return {
        "chunk_id": f"NORM_{row['id']}",
        "chunk_type": "country_norm",
        "source_id": row["id"],
        "category": row["category"],
        "culture_region": region,
        "norm_type": norm_type,
        "scenario_pattern": pattern,
        "condition_family": condition_key,
        "norm_statement": row["cultural_context"].strip(),
        "latent_variables": variables,
        "acceptable_if": acceptable,
        "not_acceptable_if": not_acceptable,
        "context_dependent_if": context_dep,
        "retrieval_text": retrieval_text,
    }


def make_contrast_chunks(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["situation"]].append(row)

    chunks = []
    for idx, (situation, group) in enumerate(groups.items(), start=1):
        if len(group) < 2:
            continue
        pattern, _ = infer_pattern(group[0]["situation"], group[0]["category"], parse_culture(group[0]["culture"])[1])
        entries = []
        for row in group:
            region, norm_type = parse_culture(row["culture"])
            _, _, acceptable, not_acceptable, context_dep = conditions_for(region, norm_type, row["category"], row["situation"])
            entries.append({
                "culture_region": region,
                "norm_type": norm_type,
                "norm_statement": row["cultural_context"].strip(),
                "acceptable_if": acceptable[:4],
                "not_acceptable_if": not_acceptable[:4],
                "context_dependent_if": context_dep[:4],
            })

        contrast_lines = []
        for entry in entries:
            contrast_lines.append(
                f"{entry['culture_region']} / {title_case(entry['norm_type'])}: {entry['norm_statement']} "
                f"Acceptable if: {'; '.join(entry['acceptable_if'])}. "
                f"Not acceptable if: {'; '.join(entry['not_acceptable_if'])}. "
                f"Context-dependent if: {'; '.join(entry['context_dependent_if'])}."
            )
        retrieval_text = (
            f"Contrastive scenario pattern: {pattern}. "
            f"Use this chunk when the same behavior may be judged differently across cultures. "
            + " ".join(contrast_lines)
        )
        chunks.append({
            "chunk_id": f"CONTRAST_{idx:03d}",
            "chunk_type": "contrastive_scenario_pattern",
            "scenario_pattern": pattern,
            "source_ids": [row["id"] for row in group],
            "cultures_compared": [parse_culture(row["culture"])[0] for row in group],
            "entries": entries,
            "retrieval_text": retrieval_text,
        })
    return chunks


def write_jsonl(path, chunks):
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def flatten_for_csv(chunk):
    row = {}
    for key, value in chunk.items():
        if isinstance(value, list):
            row[key] = "; ".join(json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v) for v in value)
        elif isinstance(value, dict):
            row[key] = json.dumps(value, ensure_ascii=False)
        else:
            row[key] = value
    return row


def write_csv(path, chunks):
    fields = [
        "chunk_id", "chunk_type", "source_id", "category", "culture_region", "norm_type",
        "scenario_pattern", "condition_family", "norm_statement", "orientation",
        "latent_variables", "acceptable_if", "not_acceptable_if", "context_dependent_if",
        "cultures_compared", "source_ids", "retrieval_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for chunk in chunks:
            row = flatten_for_csv(chunk)
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(chunks):
    counts = defaultdict(int)
    regions = defaultdict(int)
    categories = defaultdict(int)
    for chunk in chunks:
        counts[chunk["chunk_type"]] += 1
        if chunk.get("culture_region"):
            regions[chunk["culture_region"]] += 1
        if chunk.get("category"):
            categories[chunk["category"]] += 1
    return {
        "chunk_count": len(chunks),
        "chunk_type_counts": dict(sorted(counts.items())),
        "culture_region_counts": dict(sorted(regions.items(), key=lambda x: (-x[1], x[0]))),
        "category_counts": dict(sorted(categories.items(), key=lambda x: (-x[1], x[0]))),
        "leakage_policy": [
            "No gold labels, model predictions, confidence scores, or exact situation text are included.",
            "Scenario patterns are generalized from recurring situation types to support retrieval without direct answer lookup.",
            "Decision fields are country/domain-specific and vary by norm family.",
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    chunks = []
    chunks.extend(make_profile_chunks(rows))
    chunks.extend(make_norm_chunk(row) for row in rows)
    chunks.extend(make_contrast_chunks(rows))

    write_jsonl(OUT_DIR / "culture_aware_rag_v2.jsonl", chunks)
    write_csv(OUT_DIR / "culture_aware_rag_v2.csv", chunks)
    summary = summarize(chunks)
    (OUT_DIR / "culture_aware_rag_v2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = f"""# Culture-Aware RAG Database v2

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

- Total chunks: {summary["chunk_count"]}
- Country profiles: {summary["chunk_type_counts"].get("country_profile", 0)}
- Country norm chunks: {summary["chunk_type_counts"].get("country_norm", 0)}
- Contrastive chunks: {summary["chunk_type_counts"].get("contrastive_scenario_pattern", 0)}
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

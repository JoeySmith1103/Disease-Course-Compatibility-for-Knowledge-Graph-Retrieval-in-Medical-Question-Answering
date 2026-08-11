"""All prompts, consolidated. Extracted verbatim from the validated scripts.
Reader prompts: VANILLA, COT_MINIMAL, RAW_KG, MEDRAG, WALKER (=original walker), NO_KG.
Walker prompt variants: WALKER_V1, WALKER_V2. Files ending in `__revised.json` were rendered
with Prompt v2 (`prompt_revised.txt` == `WALKER_V2`).
Retrieval-side prompts: TOG_REL_PRUNE, HYKGE_HYPOTHESIS.
Placeholders: {question} {options_block} {kg_block}|{evidence_block} {patient_dur_str} {top_k}
"""

VANILLA = """Solve the multiple-choice medical question. Provide your final answer as a single letter in <a></a> tags.

Question:
{question}

Options:
{options_block}

<a>?</a>"""

COT_MINIMAL = """Solve the multiple-choice medical question. Think step by step before giving your final answer. State your final answer as a single letter in <a></a> tags. Example: <a>C</a>

Question:
{question}

Options:
{options_block}"""

RAW_KG = """You are an expert medical diagnostician. Solve the multiple-choice medical question by reasoning step by step.

Question:
{question}

Options:
{options_block}

Supplementary evidence from a clinical knowledge graph (use as ONE input among many; not authoritative):
- Patient symptom duration: {patient_dur_str}
- Top candidate concepts retrieved from SNOMED via raw graph expansion from seeds (NO filtering, NO ranking — listed in graph-traversal order, may contain noise).

{kg_block}

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. Rank the candidates and select the most likely answer.

After your reasoning (under 200 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>."""

MEDRAG = """You are an expert medical diagnostician. Solve the multiple-choice medical question using the supplied medical-textbook excerpts as supporting evidence.

Question:
{question}

Options:
{options_block}

Retrieved textbook excerpts (top-{top_k} BM25 matches from the MedQA source corpus — Harrison, First Aid, Pathoma, etc.):
- Patient symptom duration: {patient_dur_str}

{evidence_block}

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. Cross-reference the retrieved textbook excerpts to support or rule out candidates.
4. Rank the candidates and select the most likely answer.

After your reasoning (under 250 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>.
"""

WALKER = """You are an expert medical diagnostician. Solve the multiple-choice medical question by reasoning step by step.

Question:
{question}

Options:
{options_block}

Supplementary evidence from a clinical knowledge graph (use as ONE input among many; not authoritative):
- Patient symptom duration: {patient_dur_str}
- Top candidate concepts retrieved from SNOMED via graph expansion from seeds (DDx hypotheses + case findings from the vignette only; option text was NOT used as seeds).

{kg_block}

How to read each entry:
  `N. [role] name (score=S: cos=C+bc=B)`
  - `score = cos + 0.3·bc − 0.08·hop`: walker's combined ranking signal.
  - `cos`: semantic similarity between the case symptoms and this concept (0..1).
  - `bc` (dur-compat): Bhattacharyya overlap between patient duration and the disease's typical clinical course (0..1; 0 if non-disease).
  - `role`: UMLS TUI-derived category — [disease] / [finding] / [organism] / [procedure] / [anatomy] etc.
  - Use as a cross-reference; do NOT let it override clinical judgment.

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. For each candidate, evaluate compatibility with the patient's symptom duration and other clinical features. High cos but low bc may indicate semantic match without duration fit.
4. Rank the candidates and select the most likely answer.

After your reasoning (under 300 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>.
"""


WALKER_ORIGINAL = WALKER

# Prompt v1: the longer structured prompt used during prompt tuning. It keeps the walker's
# numeric score explanation but asks through explicit [QUESTION]/[OPTIONS]/[ANSWER] sections.
WALKER_V1 = """You are an expert medical diagnostician. Solve the multiple-choice medical question by reasoning step by step.

Below are the [QUESTION], [OPTIONS], [SYMPTOM_DURATION], and [RETRIEVED INFORMATION] from the medical knowledge graph, UMLS. Please use this information as a guide to generate [ANSWER].

Note that each line of [RETRIEVED INFORMATION] follows this structure:
N. [category] concept name (score=S: cos=C+bc=B)
where cos is the semantic similarity between the case findings and the concept (0 to 1), bc is the overlap between the patient's symptom duration and the disease's typical clinical course (0 to 1), and score is the walker's combined ranking signal, computed as score = cos + 0.3*bc - 0.08*hop, with hop being the number of graph-expansion steps from the seed concepts. The lines are sorted by score in descending order. Note that bc is defined only for concepts of category [disease]; for [drug], [finding], [anatomy], and other categories, bc is always 0, which means "not applicable" rather than "incompatible". Also note that this list was retrieved from the case description alone, and the text of the options was not used, so the presence or absence of a concept in this list is not evidence for or against any option.

Please generate the [ANSWER], which is a single capital letter from the provided [OPTIONS] based on [QUESTION], [OPTIONS], [SYMPTOM_DURATION], and [RETRIEVED INFORMATION]
Use as a cross-reference; do NOT let it override clinical judgment.
Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. For each candidate, evaluate compatibility with the patient's symptom duration and other clinical features. High cos but low bc may indicate semantic match without duration fit.
4. Rank the candidates and select the most likely answer to generate the [ANSWER]


[QUESTION]:
{question}

[OPTIONS]:
{options}

[SYMPTOM_DURATION]:
{symptom_duration} hours

[RETRIEVED INFORMATION]:
{retrieved_information}

[ANSWER]:"""

# Prompt v2: the revised prompt used for frozen/<dataset>/<method>__revised.json.
# Keep this text identical to prompt_revised.txt unless intentionally creating a new variant.
WALKER_V2 = """You are an expert medical diagnostician answering a multiple-choice medical exam question.

Question:
{question}

Options:
{options}

Supplementary evidence from a clinical knowledge graph:
- Patient symptom duration: {patient_duration}
- The UMLS/SNOMED concepts below were retrieved from the vignette only, not from the option text.
- Use them only as cross-reference. They can be noisy or incomplete.
- score/cos reflect symptom text match; bc reflects duration fit for disease concepts. For non-disease concepts, bc=0 means not applicable, not incompatible.

{retrieved_information}

Solve by prioritizing the vignette and answer options. Use the KG only to support or challenge plausible diagnoses; ignore KG hints that conflict with core clinical facts.

Reply using exactly this format:
[Reasoning]
One concise sentence, no more than 40 words, naming the decisive clue and diagnosis.

[Answer]
A single capital letter from the provided options."""

NO_KG = """You are an expert medical diagnostician. Solve the multiple-choice medical question by reasoning step by step.

Question:
{question}

Options:
{options_block}

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. For each candidate, evaluate compatibility with the patient's symptom duration and other clinical features.
4. Rank the candidates and select the most likely diagnosis.

After your step-by-step reasoning (under 300 words), state your final answer as a single letter enclosed in <a></a> tags. Example: <a>C</a>.
"""

TOG_REL_PRUNE = """You are exploring a medical knowledge graph to answer a question.
Question: {q}
Topic entity: {ent}
Available relations from this entity:
{rels}
Retrieve up to {w} relations most useful for answering the question and rate each
0-1. Output exactly, one per line: relation (score). No other text."""

HYKGE_HYPOTHESIS = """You are a medical expert. For the question below, write a concise hypothetical
answer (3-4 sentences) describing the most likely diagnosis and the reasoning, THEN on a final
line starting with "ENTITIES:" list the key medical entities you mentioned (comma-separated).
Question:
{q}"""

# ── ReinRAG-style output section for small/medium open models ──────────────────
# Small models (Qwen etc.) are unreliable with the <a>X</a> convention, so answers become
# unparseable. run_reader_block.py STRIPS the frozen prompt's <a></a> instruction (so there is
# no contradictory second answer spec) and appends this ReinRAG-style bracketed section.
# Everything ELSE stays the frozen, method-correct ReinRAG prompt — byte-identical to the
# big-model run — so the two differ only in how the answer is requested.
REINRAG_OUTPUT = """[Output Format]
Reply using exactly these two sections, in this order, and nothing else:

[Reasoning]
Your concise reasoning, under 200 words.

[Answer]
A single capital letter from the provided options."""


# Shown as {patient_dur_str} when the vignette carries no usable symptom duration. Those
# questions still get retrieval, but with bc disabled (score degrades to cos − μ·hop), so the
# bc column is all-zero — say so explicitly or the reader misreads it as "all non-disease".
NO_DURATION_STR = ("not stated in the vignette — temporal compatibility (bc) is therefore "
                   "not applicable and is 0 for every candidate below; rank on symptom match alone")


# ── Prompt post-processing (single home for ALL prompt surgery) ────────────────
# Readers apply these to the FROZEN per-question prompts. Kept here, not in the readers,
# so every change to what a model actually sees lives in this one file.
import re as _re


def strip_path_legend(text: str) -> str:
    """Remove the `path:` legend from an already-frozen prompt.

    WALKER_WITH_PATH is off, so no kg_block ever contained a path line (0/655) — the legend
    described evidence that was never supplied. It is now gone from the WALKER template above,
    but prompts frozen BEFORE that change still carry it, so both readers strip it at read time
    and get the identical, path-free prompt without needing to re-freeze.
    """
    kept = []
    for line in (text or "").split("\n"):
        s = line.strip()
        if s.startswith("path: origin_seed") or s.startswith("- `path`:"):
            continue
        kept.append(line)
    return "\n".join(kept).replace(
        "Use the KG paths to verify which candidates have well-supported "
        "retrieved-evidence chains; high cos", "High cos")


def strip_answer_instruction(text: str) -> str:
    """Drop the <a></a> answer-format sentences so they cannot contradict the [Answer] spec.
    Sentence-level, so only those sentences go — evidence and reasoning text is untouched."""
    kept = []
    for line in (text or "").split("\n"):
        if "<a>" in line:
            line = " ".join(s for s in _re.split(r"(?<=\.)\s+", line) if "<a>" not in s).strip()
            if not line:
                continue
        kept.append(line)
    return "\n".join(kept).rstrip()


def to_big_model_prompt(frozen_prompt: str) -> str:
    """Big models (gpt/gemini): frozen prompt minus the dead path legend. Keeps <a></a>."""
    return strip_path_legend(frozen_prompt or "")


def to_small_model_prompt(frozen_prompt: str) -> str:
    """Small/medium open models (Qwen etc.): additionally drop the <a></a> spec and append the
    ReinRAG [Answer] section. This is the ONLY small-model-specific divergence — it exists to
    cut parse errors, not to change the evidence the model reasons over."""
    return strip_answer_instruction(strip_path_legend(frozen_prompt or "")) + "\n\n" + REINRAG_OUTPUT


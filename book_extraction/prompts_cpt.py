"""Prompts and Templates for CPT Corpus Generation (v2 — Fact-Anchored Suite).

Designed to cure "Parametric Arrogance": every representation forces verbatim
anchoring to numbers, chemical formulas, and state-wise statistics from the
source text, with explicit negative constraints against generic expert filler.
"""

GROUND_TRUTH_CONTRACT = """\
ANTI-HALLUCINATION CONTRACT (NON-NEGOTIABLE):
1. GROUNDING: Every factual claim you write MUST be traceable to the SOURCE CONTENT. \
If a fact is not in the source, you must NOT write it.
2. VERBATIM NUMBERS: Reproduce every number EXACTLY as written in the source \
(e.g. "6.5", "34%", "million hectares", "kg/ha", "pH < 5.5"). NEVER round, \
approximate, invent precision, or substitute your own estimates.
3. NO GENERIC FILLER: BANNED phrases include: "plays a crucial/vital role", \
"it is important to note that", "various factors", "significant impact", \
"several studies suggest", "proper management is essential", "and many more". \
Every sentence must carry at least one concrete entity, number, place, crop, \
chemical, or mechanism from the source.
4. UNITS AND FORMULAS: Preserve units, chemical formulas with their charges and \
subscripts (e.g. Al3+, HCO3-, CaCO3, NH4+), equations, and named references \
(e.g. "Maji et al. 2012") exactly as given.
5. COVERAGE: Do not silently drop data points. If the source lists N items \
(states, values, crops), your output must reflect all N unless explicitly told otherwise.
6. EMPTY SOURCE PROTOCOL: If the source content genuinely lacks the kind of information \
this representation requires, output EXACTLY this single line and nothing else: NO_USABLE_DATA
"""

FEW_SHOT_GROUNDING = """\
FEW-SHOT GROUNDING EXAMPLE (illustrates the required fidelity):

SOURCE EXCERPT:
"Acid soils are generally defined as those having a pH of less than 6.5 for most of
the year. Approximately one-third of all soils of the world are acidic, and nearly
50% of the world's potentially arable lands fall under this category (von Uexkull
and Mutert 1995). In India, nearly 34% of cropped land are estimated to have acid
soils (Maji et al. 2012)."

BAD OUTPUT (generic filler — REJECTED):
"Acid soils are a major problem in India and around the world. They affect crop
growth significantly and proper management is essential for sustainable agriculture."

GOOD OUTPUT (fact-anchored — ACCEPTED):
"Acid soils are defined as soils with a pH below 6.5 for most of the year.
Approximately one-third of all soils worldwide are acidic, covering nearly 50%
of the world's potentially arable land (von Uexkull and Mutert 1995). In India,
an estimated 34% of cropped land is acid soil (Maji et al. 2012)."
"""

CPT_REPRESENTATION_TYPES = {
    "quant_fact_ledger": {
        "system": (
            "You are a quantitative data auditor for ICAR soil science publications. "
            "Convert the source content into a dense FACT LEDGER: an exhaustive, itemized "
            "prose inventory of EVERY quantitative datum and hard fact present. For each entry "
            "state: the exact value with units, what it measures, its geographic/entity scope "
            "(state, district, country, crop), and the cited source if named.\n\n"
            + GROUND_TRUTH_CONTRACT
            + "\n"
            + FEW_SHOT_GROUNDING
        ),
        "temperature": 0.15,
    },
    "technical_factsheet": {
        "system": (
            "You are an ICAR technical documentation specialist. Convert the source content into "
            "a formal structured factsheet with plain-text section markers such as DEFINITION, "
            "CLASSIFICATION, QUANTITATIVE THRESHOLDS, REGIONAL DISTRIBUTION, DIAGNOSIS, MANAGEMENT. "
            "Under each marker, list facts as compact declarative statements. Every statement must "
            "contain at least one verbatim number, chemical term, taxonomic name, or place name "
            "from the source.\n\n"
            + GROUND_TRUTH_CONTRACT
        ),
        "temperature": 0.25,
    },
    "causal_chemical_chain": {
        "system": (
            "You are an agricultural systems analyst specializing in soil chemistry. Rewrite the "
            "source content as explicit causal chains in the form: CAUSE -> MECHANISM -> EFFECT -> "
            "MANAGEMENT. Write out every chemical reaction, hydrolysis step, equilibrium expression, "
            "and nutrient-availability threshold mentioned, preserving ion charges and formulas "
            "exactly (e.g. Al3+, Al(OH)3, HCO3-, exchangeable acidity). Link each chain to the exact "
            "numeric thresholds given in the source.\n\n"
            + GROUND_TRUTH_CONTRACT
        ),
        "temperature": 0.35,
    },
    "regional_mapping_narrative": {
        "system": (
            "You are a geographer compiling the National Bureau of Soil Survey regional atlas "
            "narratives. Rewrite the source content as a region-by-region geographic narrative of "
            "India. Enumerate EVERY state, district, or agro-climatic zone mentioned together with "
            "its exact figures (area in million hectares or lakh hectares, percentage share, severity "
            "class). Never merge two regions into one vague statement; each region gets its own "
            "concrete, number-bearing sentences.\n\n"
            + GROUND_TRUTH_CONTRACT
        ),
        "temperature": 0.35,
    },
    "kvk_advisory_sop": {
        "system": (
            "You are a senior Krishi Vigyan Kendra (KVK) extension officer writing Standard Operating "
            "Procedure sheets for field staff. Convert the source content into numbered SOP steps. Each "
            "step must specify exact inputs: product/chemical name, dose per unit area (e.g. kg/ha, "
            "g/plant), timing relative to crop stage or season, application method, and the numeric "
            "trigger condition (e.g. apply lime when soil pH falls below X). Never replace a dose with "
            "'as recommended' or 'appropriate amount'.\n\n"
            + GROUND_TRUTH_CONTRACT
        ),
        "temperature": 0.45,
    },
    "farmer_expert_diagnostic": {
        "system": (
            "You are writing transcripts of authentic Kisan Call Centre diagnostic sessions between a "
            "progressive Indian farmer and a KVK soil scientist. Build a multi-turn dialogue where the "
            "farmer describes symptoms/context and the scientist responds with SPECIFIC, number-bearing "
            "answers drawn strictly from the source content (exact pH cutoffs, exact doses, named "
            "amendments, state-specific figures). The farmer should probe for precise quantities; the "
            "scientist must always answer with the verbatim figures, never vague guidance.\n\n"
            + GROUND_TRUTH_CONTRACT
        ),
        "temperature": 0.55,
    },
    "pedagogical_tutorial": {
        "system": (
            "You are a professor of soil science recording a deep tutorial transcript with a bright MSc "
            "student. Walk through the source content concept by concept. The student must ask pointed "
            "verification questions ('What is the exact extent in Assam?', 'Which equation describes this?') "
            "and the professor answers using ONLY verbatim source data, explaining the underlying mechanism. "
            "Include at least one recall-check where the student repeats exact numbers back and the professor "
            "confirms or corrects them from the source.\n\n"
            + GROUND_TRUTH_CONTRACT
        ),
        "temperature": 0.5,
    },
    "exam_bank_grounded": {
        "system": (
            "You are setting the ICAR-NET question bank for soil science. From the source content, generate "
            "10 to 20 examination items in plain text using the format:\n"
            "Q: <question targeting a specific fact, number, formula, or regional statistic>\n"
            "A: <precise answer quoting the verbatim value(s) plus one sentence of context>\n"
            "Mix factual-recall, numeric-threshold, comparison, and applied-diagnosis questions. Answers must "
            "be self-contained and quote exact figures.\n\n"
            + GROUND_TRUTH_CONTRACT
        ),
        "temperature": 0.3,
    },
    "encyclopedic_macro_synthesis": {
        "system": (
            "You are writing the definitive encyclopedia article for an Indian agricultural reference work. "
            "Synthesize ALL of the source content into one dense, self-contained passage that preserves every "
            "key concept, entity, relationship, and data point. This representation trains whole-topic recall: "
            "structure it as definition -> classification -> quantitative extent (all figures preserved) -> "
            "processes and mechanisms -> regional distribution -> management practices. Nothing substantive "
            "from the source may be omitted.\n\n"
            + GROUND_TRUTH_CONTRACT
        ),
        "temperature": 0.3,
    },
    "numeric_comparison_matrix": {
        "system": (
            "You are a data analyst converting agricultural tables into comparative prose matrices. Extract "
            "every tabular or comparative relationship in the source and render it as parallel contrastive "
            "sentences of the form: 'Compared to <entity A> at <value A>, <entity B> records <value B>, a "
            "difference of <derived delta>'. Cover crops vs crops, state vs state, treatment vs control, depth "
            "vs depth, exactly as the source tabulates them. Keep every unit identical to the source.\n\n"
            + GROUND_TRUTH_CONTRACT
        ),
        "temperature": 0.25,
    },
}

LEGACY_ALIASES = {
    "article": "encyclopedic_macro_synthesis",
    "technical_doc": "technical_factsheet",
    "causal_chain": "causal_chemical_chain",
    "factsheet": "quant_fact_ledger",
    "kvk_narrative": "kvk_advisory_sop",
}


def resolve_representation_type(rep_type: str) -> str:
    return LEGACY_ALIASES.get(rep_type, rep_type)


CPT_USER_TEMPLATE = (
    "Book: {book_title}\n"
    "Chapter: {chapter_title}\n"
    "Source numeric anchors detected: {numeric_anchor_count} "
    "(your output must retain at least {numeric_anchor_target} of them verbatim)\n\n"
    "FIGURE ANNOTATIONS: Diagrams, charts, maps and tables from the original book have been "
    "replaced inline by dense technical descriptions inside [FIGURE n: ...] markers at their "
    "exact original position. Treat these annotations as first-class source facts and integrate "
    "their data (axes, legends, mapped regions, tabulated values) into your output.\n\n"
    "SOURCE CONTENT:\n"
    "\"\"\"\n"
    "{text}\n"
    "\"\"\"\n\n"
    "Generate the representation now. Output plain text only — no markdown headers, no JSON, "
    "no code blocks, no meta-commentary about the task. Just the finished passage."
)


def count_numeric_anchors(text: str) -> int:
    import re
    return len(re.findall(r"\d+(?:[.,]\d+)?", text))

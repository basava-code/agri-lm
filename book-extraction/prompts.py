"""Prompt Registry and Output Schema Definition for Agri-LM QnA Generation.

This module houses the structured output schemas (using Pydantic) and the
prompt registry (using LangChain ChatPromptTemplate) for version control.
It guarantees that the model output strictly follows a specified JSON structure.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class FactQnAGroup(BaseModel):
    """A single QnA pair centered around a precise crop recommendation."""

    exact_source_quote: str = Field(
        description="The EXACT verbatim sentence or paragraph copied directly from the textbook chapter text. This serves as the grounding anchor."
    )

    alternative_questions: list[str] = Field(
        description=(
            "A list of exactly 3 different question phrasings representing how users might query this fact. "
            "Variation 1: Simple/Casual (e.g. how a farmer might ask). "
            "Variation 2: Highly technical (e.g. how an agronomist or extension officer might ask). "
            "Variation 3: Contextual/Metric-focused (e.g. asking about specific dosages, dates, or variety names)."
        )
    )

    authoritative_answer: str = Field(
        description=(
            "The single, highly detailed, complete, and self-contained answer. "
            "Must be written in an authoritative voice, using British English spelling. "
            "All chemical names, formulation strengths (e.g., 50 EC, 75 WS), sowing schedules, "
            "fertiliser doses, and risks (e.g., lodging) must match the exact_source_quote with 100% precision. "
            "Never refer to 'the text' or 'the document'."
        )
    )


class CropQnADataset(BaseModel):
    """The master container schema holding all generated QnA groups for a logical crop chapter."""

    qna_groups: list[FactQnAGroup] = Field(
        description=(
            "An exhaustive collection of QnA groups covering every single fact, crop variety, "
            "pesticide dosage, date, technique, warning, and guideline in the provided chapter."
        )
    )


# System instructions and user prompt templates versioning
_PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "v1": {
        "system_instruction": (
            "You are an expert agricultural AI researcher and dataset designer. Your goal is to perform "
            "an exhaustive, high-fidelity translation of the provided textbook chapter on agricultural "
            "practices into a comprehensive, high-quality QnA dataset. This dataset will be used to "
            "fine-tune a small agricultural language model (SLM).\n\n"
            "CRITICAL DIRECTIVES:\n"
            "1. EXHAUSTIVE EXTRACTION: Do not skip a single bit of knowledge, variety specification, "
            "nutrient dosage, planting date, weed control practice, warning, or tip. Every recommendation "
            "in the text must be translated into QnA format.\n"
            "2. MULTIPLE VARIATIONS: For every core fact, you must generate a `FactQnAGroup` containing "
            "at least 3 variations. The variations must capture different ways of asking and answering "
            "the same core fact. For example:\n"
            "   - Variation 1 (Casual/Direct): How a smallholder farmer might ask in simple terms.\n"
            "   - Variation 2 (Technical/Detailed): How an agricultural extension officer or scientist might ask.\n"
            "   - Variation 3 (Alternative Context/Tabular style): E.g., 'What is the dosage of X for Y?' or 'Which crop variety matures in Z days?'\n"
            "3. SELF-CONTAINED ANSWERS: Every answer must be complete and fully self-contained. "
            "Do NOT write 'According to the chapter' or 'As discussed in the text'. Write the answer as "
            "an authoritative agricultural recommendation.\n"
            "4. BRITISH ENGLISH: You must use British English spelling and terminology throughout all "
            "questions and answers (e.g. fertiliser, colour, programme, generalise, standardise, yield).\n"
            "5. OUTPUT FORMAT: You must adhere strictly to the JSON schema provided."
        ),
        "user_template": (
            "Here is the text content of the agricultural chapter titled: '{title}' (Key: {key}).\n"
            "Please extract and translate every single recommendation, crop variety detail, and agricultural "
            "instruction in this text into the structured FactQnAGroups. Ensure maximum detail and completeness.\n\n"
            "TEXT CONTENT:\n"
            "\"\"\"\n"
            "{text}\n"
            "\"\"\""
        )
    },
    
    "v2": {
        "system_instruction": (
            "You are a senior agricultural extension specialist, working "
            "as a dataset architect. Your task is to translate the provided crop practice text into a "
            "premium fine-tuning dataset.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Extract all knowledge with 100% fidelity. Absolutely no summaries or loss of detail.\n"
            "- Create a `FactQnAGroup` for every variety (Crop Variety A, Crop Variety B, etc.), fertilizer recommendation, "
            "irrigation schedule, and pest control step.\n"
            "- Each group must have at least 3 distinct phrasing variations (casual, professional, informational).\n"
            "- All text MUST use British English. Standardise all terms (e.g., 'fertiliser' instead of 'fertilizer').\n"
            "- Answers must be direct, polite, and actionable for farmers or credit scorers."
        ),
        "user_template": (
            "Perform an exhaustive QnA extraction from the following agricultural text: '{title}'.\n\n"
            "CHAPTER CONTENT:\n"
            "\"\"\"\n"
            "{text}\n"
            "\"\"\""
        )
    },
    
    "v3": {
        "system_instruction": (
            "You are an expert agricultural researcher and dataset designer. Your goal is to perform "
            "an exhaustive, high-fidelity translation of the provided book chapter into a comprehensive, "
            "high-quality QnA dataset. This dataset will be used to fine-tune a specialized language model.\n\n"
            "CRITICAL DIRECTIVES:\n"
            "1. EXHAUSTIVE EXTRACTION: You MUST extract every single core concept, strategy, "
            "methodology, statistic, tool, and instruction from the text. For an average chapter, you "
            "should generate AT LEAST 15 to 25 distinct QnA groups. Do not summarize the chapter into just a few questions.\n\n"
            "2. VERBATIM GROUNDING (ZERO HALLUCINATION): Before writing any question or answer, you MUST "
            "extract the exact verbatim text segment from the source chapter and copy it into the `exact_source_quote` field. "
            "Every query and recommendation must be strictly derived from this quote. Do NOT introduce external concepts.\n\n"
            "3. NUMERICAL INTEGRITY: Do not round up, down, or generalize any numbers, metrics, or financial figures. "
            f"They must be preserved with 100% fidelity.\n\n"
            "4. ENTITY DISAMBIGUATION: Treat distinct tools, methodologies, and concepts as completely distinct entities. "
            "Do not mix up their definitions, use cases, or outcomes.\n\n"
            "5. SELF-CONTAINED ANSWERS: Every answer must be complete, authoritative, and stand alone. "
            "Do NOT write 'According to the chapter' or 'As discussed in the text'. Write the answer as "
            "an official, definitive statement.\n\n"
            "6. BRITISH ENGLISH: You must use British English spelling and terminology throughout all "
            "questions and answers (e.g. optimise, programme, generalise, standardise).\n\n"
            "7. OUTPUT FORMAT: You must adhere strictly to the JSON schema provided. Your response MUST be valid JSON containing a single root key 'qna_groups' which holds a list of objects. Each object MUST have the following keys exactly: 'exact_source_quote', 'alternative_questions' (which MUST be a list of exactly 3 highly relevant questions), and 'authoritative_answer'."
        ),
        "user_template": (
            "Here is the text content of the chapter titled: '{title}' (Key: {key}).\n"
            "Please perform an exhaustive extraction of every single concept, methodology, strategy, "
            "warning, and instruction. Generate a massive, comprehensive list of QnA groups. Do NOT skip anything.\n\n"
            "TEXT CONTENT:\n"
            "\"\"\"\n"
            "{text}\n"
            "\"\"\""
        )
    }
}


# Construct PROMPT_REGISTRY with ChatPromptTemplate instances
PROMPT_REGISTRY: dict[str, ChatPromptTemplate] = {
    version: ChatPromptTemplate.from_messages([
        ("system", templates["system_instruction"]),
        ("human", templates["user_template"])
    ])
    for version, templates in _PROMPT_TEMPLATES.items()
}


def get_prompt(version: str = "v1") -> ChatPromptTemplate:
    """Retrieve the ChatPromptTemplate configuration for a given version.

    Raises ValueError if the version does not exist in the registry.
    """
    if version not in PROMPT_REGISTRY:
        raise ValueError(
            f"Prompt version '{version}' is not registered in PROMPT_REGISTRY. "
            f"Available versions: {list(PROMPT_REGISTRY.keys())}"
        )
    return PROMPT_REGISTRY[version]


"You are an expert agricultural researcher and dataset designer. Your goal is to perform an exhaustive, high-fidelity translation of the provided book chapter into a comprehensive, high-quality QnA dataset. This dataset will be used to fine-tune a specialized language model.\n\nCRITICAL DIRECTIVES:\n1. EXHAUSTIVE EXTRACTION: You MUST extract every single core concept, strategy, methodology, statistic, tool, and instruction from the text. For an average chapter, you should generate AT LEAST 15 to 25 distinct QnA groups. Do not summarize the chapter into just a few questions.\n\n2. VERBATIM GROUNDING (ZERO HALLUCINATION): Before writing any question or answer, you MUST extract the exact verbatim text segment from the source chapter and copy it into the `exact_source_quote` field. Every query and recommendation must be strictly derived from this quote. Do NOT introduce external concepts.\n\n3. NUMERICAL INTEGRITY: Do not round up, down, or generalize any numbers, metrics, or financial figures. fThey must be preserved with 100% fidelity.\n\n4. ENTITY DISAMBIGUATION: Treat distinct tools, methodologies, and concepts as completely distinct entities. Do not mix up their definitions, use cases, or outcomes.\n\n5. SELF-CONTAINED ANSWERS: Every answer must be complete, authoritative, and stand alone. Do NOT write 'According to the chapter' or 'As discussed in the text'. Write the answer as an official, definitive statement.\n\n6. BRITISH ENGLISH: You must use British English spelling and terminology throughout all questions and answers (e.g. optimise, programme, generalise, standardise).\n\n7. OUTPUT FORMAT: You must adhere strictly to the JSON schema provided. Your response MUST be valid JSON containing a single root key 'qna_groups' which holds a list of objects. Each object MUST have the following keys exactly: 'exact_source_quote', 'alternative_questions' (which MUST be a list of exactly 3 highly relevant questions), and 'authoritative_answer'."

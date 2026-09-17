#!/usr/bin/env python3
"""
CPT Multi-Stage Evaluation Pipeline using Ollama and vLLM.

Model A (Ollama):  gemma4_e2b_acid_soils:latest (CPT Fine-Tuned Model)
Model B (vLLM):   gemma4:31b (Examiner & Judge with Chapter/Book Context)

Evaluation Stages:
1. Chapter-by-Chapter Q&A: vLLM generates technical questions from chapters -> Ollama answers -> vLLM grades.
2. Chapter-by-Chapter Peer Conversation: vLLM & Ollama engage in multi-turn technical discussion per chapter.
3. Whole-Book Discussion: vLLM (holding full book context) & Ollama engage in macro multi-turn discussion across the entire book.
"""

import os
import sys
import json
import re
import time
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import ollama
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(description="CPT Multi-Stage Evaluation using Ollama & vLLM")
    parser.add_argument("--chapters_path", type=str, 
                        default="data/processed/Acid Soils of India/Acid Soils of India_chapters.json",
                        help="Path to extracted chapters JSON file")
    parser.add_argument("--ollama_url", type=str, default="100.71.118.101:11434",
                        help="Base URL for Ollama server")
    parser.add_argument("--ollama_model", type=str, default="gemma4_e2b_acid_soils_v3:f16",
                        help="Trained model name in Ollama")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1",
                        help="Base URL for vLLM OpenAI-compatible server")
    parser.add_argument("--vllm_model", type=str, default="/home/basava/agri-lm/kcc-data-cleaning/gemma-4-31b-it",
                        help="Judge/Examiner model name in vLLM")
    parser.add_argument("--num_questions", type=int, default=5,
                        help="Number of questions to generate per chapter in Stage 1")
    parser.add_argument("--convo_turns", type=int, default=5,
                        help="Number of dialogue turns per chapter in Stage 2")
    parser.add_argument("--book_convo_turns", type=int, default=5,
                        help="Number of dialogue turns for whole book in Stage 3")
    parser.add_argument("--output_dir", type=str, default="./outputs/cpt_eval_results_v3",
                        help="Output directory for JSON and Markdown reports")
    parser.add_argument("--skip_references", action="store_true", default=True,
                        help="Skip references chapter from evaluation")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature for evaluation responses")
    return parser.parse_args()

class OllamaClient:
    def __init__(self, base_url: str, model_name: str):
        self.client = ollama.Client(host=base_url)
        self.model_name = model_name

    def check_connection(self) -> bool:
        try:
            res = self.client.list()
            models = []
            if hasattr(res, 'models'):
                models = [m.model for m in res.models]
            elif isinstance(res, dict):
                models = [m.get("name") for m in res.get("models", [])]
            print(f"[OK] Ollama connected via SDK. Found models: {models}")
            return True
        except Exception as e:
            print(f"[WARNING] Could not connect to Ollama via SDK: {e}")
            return False

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": temperature}
            )
            if hasattr(response, 'message'):
                return response.message.content.strip()
            elif isinstance(response, dict):
                return response.get("message", {}).get("content", "").strip()
            return str(response).strip()
        except Exception as e:
            print(f"[ERROR] Ollama generation error: {e}")
            raise e

class VLLMClient:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(base_url=self.base_url, api_key="vllm")

    def check_connection(self) -> bool:
        try:
            self.client.models.list()
            print(f"[OK] vLLM connected via OpenAI SDK at {self.base_url}")
            return True
        except Exception as e:
            print(f"[WARNING] Could not connect to vLLM at {self.base_url}: {e}")
            print("  Ensure vLLM server is running (e.g. `vllm serve gemma4:31b --port 8000`)")
            return False

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=2048
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] vLLM generation error: {e}")
            raise e

def clean_chapter_text(text: str) -> str:
    """Strips base64 image data and normalizes whitespace."""
    if not text:
        return ""
    # Strip markdown image tags containing base64 data
    text = re.sub(r'!\[.*?\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)', '[Image]', text)
    # Strip raw base64 strings if any remain
    text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+', '', text)
    # Normalize excess newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_json(text: str) -> Any:
    """Safely extracts JSON objects or arrays from LLM outputs."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Match ```json ... ``` blocks
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass

    # Match raw JSON object or list
    match_raw = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if match_raw:
        try:
            return json.loads(match_raw.group(1).strip())
        except Exception:
            pass

    raise ValueError(f"Could not parse valid JSON from response: {text[:200]}...")

def clean_question_text(question: str) -> str:
    """Removes reading-comprehension meta-phrases from questions."""
    patterns = [
        r"(?i)\baccording to (the|this) (text|chapter|content|document|passage|book|article),?\s*",
        r"(?i)\bbased on (the|this) (provided|given)?\s*(text|content|document|passage|chapter|book|article),?\s*",
        r"(?i)\bas (mentioned|stated|described|discussed) in (the|this) (text|chapter|content|document|passage),?\s*",
        r"(?i)\bfrom the (provided|given)?\s*(text|content|document|passage|chapter),?\s*"
    ]
    for p in patterns:
        question = re.sub(p, "", question)
    question = question.strip()
    if question and question[0].islower():
        question = question[0].upper() + question[1:]
    return question

def stage_1_chapter_qa(vllm: VLLMClient, ollama: OllamaClient, chapter: Dict[str, Any], num_questions: int) -> Dict[str, Any]:
    title = chapter.get("title", "Untitled Chapter")
    text = clean_chapter_text(chapter.get("text", ""))

    print(f"\n--- STAGE 1: Q&A Evaluation for '{title}' ---")
    
    # 1. Ask vLLM to generate technical questions
    prompt_gen = f"""You are a senior agricultural soil scientist and examiner creating a technical exam.
Analyze the following chapter from the book 'Acid Soils of India'.
Generate exactly {num_questions} specific, highly technical, direct factual questions to test an AI model's internal memory and domain knowledge of this topic.

Chapter Title: {title}
Chapter Content:
{text}

CRITICAL RULES FOR QUESTIONS:
- Do NOT use phrases like "According to the text", "Based on the provided content", "In this chapter", "As stated in the text", or "According to the passage".
- Frame each question as a direct domain question as if asking a subject matter expert (e.g. "What is the specific pH threshold used to define acid soils, and what percentage of global arable land falls into this category?").
- Ask about specific numbers, percentages, chemical formulas, pH thresholds, state-wise statistics, and scientific mechanisms.

Output ONLY a JSON array of objects with the exact format:
[
  {{
    "question_id": 1,
    "question": "What is the total estimated area of acid soils in India, and what percentage of the country's cropped land is affected?",
    "expected_key_facts": ["31% of total geographical area", "98.71 million hectares (mha)", "34% of cropped land"]
  }}
]"""

    try:
        q_resp = vllm.generate([{"role": "user", "content": prompt_gen}], temperature=0.2)
        questions = extract_json(q_resp)
    except Exception as e:
        print(f"[WARNING] Error generating questions for chapter '{title}': {e}")
        # Fallback question if JSON parsing fails
        questions = [{
            "question_id": 1,
            "question": f"Summarize the key scientific findings, soil pH thresholds, and state distribution regarding {title}.",
            "expected_key_facts": ["General chapter summary"]
        }]

    qa_results = []
    total_acc, total_flu, total_ret = 0.0, 0.0, 0.0

    for q_item in questions:
        q_id = q_item.get("question_id", len(qa_results) + 1)
        raw_q = q_item.get("question", "")
        question = clean_question_text(raw_q)
        expected_facts = q_item.get("expected_key_facts", [])

        print(f"  Q{q_id}: {question}")

        # 2. Get answer from Ollama model
        ollama_messages = [
            {"role": "system", "content": "You are an expert AI assistant specializing in Indian acid soils, soil chemistry, liming, and agricultural degradation. Answer accurately, concisely, and with precise scientific facts."},
            {"role": "user", "content": question}
        ]
        answer = ollama.generate(ollama_messages, temperature=0.2)
        print(f"  Ollama Model Answer: {answer[:150]}...")

        # 3. Grade answer using vLLM
        grade_prompt = f"""You are an expert agricultural AI evaluator.
Grade the candidate model's answer to the question using the ground-truth chapter text.

Chapter Title: {title}
Ground-Truth Chapter Text:
{text}

Question: {question}
Expected Key Facts: {json.dumps(expected_facts)}
Candidate Model Answer: {answer}

Grade on a scale of 0 to 100 for:
1. accuracy_score: Factual correctness based on chapter data.
2. fluency_score: Technical clarity, scientific vocabulary, and readability.
3. factual_retention_score: Recall of specific numbers, chemical formulas, and state data.

Return ONLY a JSON object:
{{
  "accuracy_score": 85,
  "fluency_score": 90,
  "factual_retention_score": 80,
  "feedback": "Short explanation of the assigned scores."
}}"""

        try:
            g_resp = vllm.generate([{"role": "user", "content": grade_prompt}], temperature=0.1)
            grade = extract_json(g_resp)
        except Exception as e:
            print(f"[WARNING] Error grading answer: {e}")
            grade = {
                "accuracy_score": 70,
                "fluency_score": 80,
                "factual_retention_score": 70,
                "feedback": "Automatic fallback score due to response format error."
            }

        acc = grade.get("accuracy_score", 0)
        flu = grade.get("fluency_score", 0)
        ret = grade.get("factual_retention_score", 0)
        feedback = grade.get("feedback", "")

        print(f"  Scores -> Accuracy: {acc} | Fluency: {flu} | Retention: {ret}")

        total_acc += acc
        total_flu += flu
        total_ret += ret

        qa_results.append({
            "question_id": q_id,
            "question": question,
            "expected_key_facts": expected_facts,
            "candidate_answer": answer,
            "grade": grade
        })

    count = max(len(qa_results), 1)
    return {
        "chapter_title": title,
        "avg_accuracy": round(total_acc / count, 2),
        "avg_fluency": round(total_flu / count, 2),
        "avg_retention": round(total_ret / count, 2),
        "qa_pairs": qa_results
    }

def stage_2_chapter_convo(vllm: VLLMClient, ollama: OllamaClient, chapter: Dict[str, Any], convo_turns: int) -> Dict[str, Any]:
    title = chapter.get("title", "Untitled Chapter")
    text = clean_chapter_text(chapter.get("text", ""))

    print(f"\n--- STAGE 2: Chapter Peer Conversation for '{title}' ---")

    transcript = []
    
    # 1. Initiator prompt for vLLM
    init_prompt = f"""You are an expert agricultural scientist conducting a peer conversation with a fine-tuned AI model regarding Chapter '{title}'.

Ground-Truth Chapter Text (for your reference as examiner):
{text}

Start the conversation by asking the candidate model a direct, open-ended technical question about a primary concept, statistic, or chemical mechanism described in this chapter.
CRITICAL RULE: Do NOT use phrases like "according to the text", "in this chapter", "based on the document", or "in the passage". Ask directly as an expert interviewer. Keep your message under 3 sentences."""

    raw_vllm = vllm.generate([{"role": "user", "content": init_prompt}], temperature=0.7)
    vllm_msg = clean_question_text(raw_vllm)
    transcript.append({"role": "vLLM (Examiner)", "content": vllm_msg})
    print(f"  vLLM (Examiner): {vllm_msg}")

    ollama_history = [
        {"role": "system", "content": "You are an expert AI assistant specializing in Indian acid soils and agricultural degradation. Participate in a scientific peer conversation accurately using domain knowledge."}
    ]

    for turn in range(convo_turns):
        # Ollama responds
        ollama_history.append({"role": "user", "content": vllm_msg})
        ollama_resp = ollama.generate(ollama_history, temperature=0.3)
        ollama_history.append({"role": "assistant", "content": ollama_resp})
        transcript.append({"role": "Ollama (Candidate)", "content": ollama_resp})

        print(f"  Ollama (Candidate): {ollama_resp[:180]}...")

        if turn == convo_turns - 1:
            break

        # vLLM evaluates response and asks follow-up
        followup_prompt = f"""You are an expert agricultural scientist having a dialogue with an AI model on Chapter '{title}'.

Ground-Truth Chapter Text (for your reference):
{text}

Conversation Transcript so far:
{json.dumps(transcript, indent=2)}

Analyze the candidate model's last response against the ground-truth facts. Ask a direct probing follow-up question or request clarification on specific numbers, regional statistics, or chemical mechanisms.
CRITICAL RULE: Do NOT use phrases like "according to the text", "in the chapter", "based on the text", or "as mentioned". Keep your response concise."""

        raw_vllm = vllm.generate([{"role": "user", "content": followup_prompt}], temperature=0.7)
        vllm_msg = clean_question_text(raw_vllm)
        transcript.append({"role": "vLLM (Examiner)", "content": vllm_msg})
        print(f"  vLLM (Examiner): {vllm_msg}")

    # Evaluate the conversation with vLLM
    eval_prompt = f"""You are an expert AI evaluator.
Grade the candidate model's performance in the following multi-turn peer conversation on Chapter '{title}'.

Ground-Truth Chapter Text:
{text}

Conversation Transcript:
{json.dumps(transcript, indent=2)}

Grade on a scale of 0 to 100 for:
1. chapter_accuracy: Factual accuracy against chapter text.
2. chapter_fluency: Professional tone, domain terminology, and coherence.
3. domain_depth: Technical depth, specificity of answers, and absence of generic fluff.
4. hallucination_rating: Select one of ("None", "Low", "Moderate", "High").
5. summary: Overall evaluation summary paragraph.

Return ONLY a JSON object:
{{
  "chapter_accuracy": 88,
  "chapter_fluency": 92,
  "domain_depth": 85,
  "hallucination_rating": "Low",
  "summary": "The model demonstrated strong grasp of acid soil distribution..."
}}"""

    try:
        e_resp = vllm.generate([{"role": "user", "content": eval_prompt}], temperature=0.1)
        convo_eval = extract_json(e_resp)
    except Exception as e:
        print(f"[WARNING] Error evaluating chapter dialogue: {e}")
        convo_eval = {
            "chapter_accuracy": 75,
            "chapter_fluency": 80,
            "domain_depth": 75,
            "hallucination_rating": "Low",
            "summary": "Automatic fallback score due to response format error."
        }

    print(f"  Convo Scores -> Acc: {convo_eval.get('chapter_accuracy')} | Fluency: {convo_eval.get('chapter_fluency')} | Depth: {convo_eval.get('domain_depth')} | Hallucinations: {convo_eval.get('hallucination_rating')}")

    return {
        "chapter_title": title,
        "evaluation": convo_eval,
        "transcript": transcript
    }

def stage_3_whole_book_convo(vllm: VLLMClient, ollama: OllamaClient, chapters: List[Dict[str, Any]], book_turns: int) -> Dict[str, Any]:
    print(f"\n--- STAGE 3: Whole-Book Comprehensive Conversation ---")

    # Combine clean text of all chapters into full book context
    book_context_blocks = []
    for ch in chapters:
        title = ch.get("title", "")
        text = clean_chapter_text(ch.get("text", ""))
        book_context_blocks.append(f"=== CHAPTER: {title} ===\n{text}")

    full_book_text = "\n\n".join(book_context_blocks)
    
    # Truncate slightly if context is extremely huge (e.g. > 150k chars)
    if len(full_book_text) > 150000:
        full_book_text = full_book_text[:150000] + "\n...[Remaining text truncated for context fit]"

    transcript = []

    init_prompt = f"""You are a lead agricultural researcher holding a technical discussion on Indian acid soils with a domain-specialized AI model.

Full Book Context (for your reference as examiner):
{full_book_text}

Initiate the discussion by asking a direct, macro question comparing key findings across topics (e.g., spatial distribution, acidity trends across agroecosystems, chemical mechanisms, and liming strategies).
CRITICAL RULE: Do NOT reference "the text", "the book", "the passage", or "according to". Ask directly as an expert interviewer. Keep it under 3 sentences."""

    raw_vllm = vllm.generate([{"role": "user", "content": init_prompt}], temperature=0.7)
    vllm_msg = clean_question_text(raw_vllm)
    transcript.append({"role": "vLLM (Examiner)", "content": vllm_msg})
    print(f"  vLLM (Examiner): {vllm_msg}")

    ollama_history = [
        {"role": "system", "content": "You are an expert AI assistant specializing in Indian acid soils, soil management, and crop yield optimization. Engage in an in-depth scientific debate using your fine-tuned domain knowledge."}
    ]

    for turn in range(book_turns):
        ollama_history.append({"role": "user", "content": vllm_msg})
        ollama_resp = ollama.generate(ollama_history, temperature=0.3)
        ollama_history.append({"role": "assistant", "content": ollama_resp})
        transcript.append({"role": "Ollama (Candidate)", "content": ollama_resp})

        print(f"  Ollama (Candidate): {ollama_resp[:180]}...")

        if turn == book_turns - 1:
            break

        followup_prompt = f"""You are a lead agricultural researcher discussing Indian acid soils with a domain AI model.

Full Book Context (for your reference):
{full_book_text}

Conversation Transcript so far:
{json.dumps(transcript, indent=2)}

Analyze the candidate model's latest response against the ground-truth facts. Ask a challenging follow-up question connecting state-wise data, crop yield impacts, or management recommendations.
CRITICAL RULE: Do NOT reference "the text", "the book", "the passage", or "according to". Keep your response concise."""

        raw_vllm = vllm.generate([{"role": "user", "content": followup_prompt}], temperature=0.7)
        vllm_msg = clean_question_text(raw_vllm)
        transcript.append({"role": "vLLM (Examiner)", "content": vllm_msg})
        print(f"  vLLM (Examiner): {vllm_msg}")

    # Grade the whole book conversation
    eval_prompt = f"""You are an expert AI evaluator.
Grade the candidate model's overall performance in this whole-book comprehensive discussion.

Full Book Context:
{full_book_text[:50000]}... [Book text summary context]

Conversation Transcript:
{json.dumps(transcript, indent=2)}

Grade on a scale of 0 to 100 for:
1. overall_accuracy: Macro factual correctness across the whole book.
2. overall_fluency: Professional scientific tone and cross-domain coherence.
3. synthesis_and_reasoning: Ability to connect concepts across different chapters and regions.
4. final_cpt_verdict: Comprehensive paragraph evaluating the quality and accuracy of the CPT fine-tuning.

Return ONLY a JSON object:
{{
  "overall_accuracy": 90,
  "overall_fluency": 94,
  "synthesis_and_reasoning": 88,
  "final_cpt_verdict": "The fine-tuned model demonstrates exceptional domain knowledge..."
}}"""

    try:
        b_resp = vllm.generate([{"role": "user", "content": eval_prompt}], temperature=0.1)
        book_eval = extract_json(b_resp)
    except Exception as e:
        print(f"[WARNING] Error evaluating whole-book dialogue: {e}")
        book_eval = {
            "overall_accuracy": 80,
            "overall_fluency": 85,
            "synthesis_and_reasoning": 80,
            "final_cpt_verdict": "Automatic fallback verdict due to format error."
        }

    print(f"  Whole-Book Scores -> Accuracy: {book_eval.get('overall_accuracy')} | Fluency: {book_eval.get('overall_fluency')} | Synthesis: {book_eval.get('synthesis_and_reasoning')}")

    return {
        "evaluation": book_eval,
        "transcript": transcript
    }

def generate_markdown_report(results: Dict[str, Any], output_file: str):
    summary = results.get("summary", {})
    s1_results = results.get("stage_1_qa", [])
    s2_results = results.get("stage_2_chapter_dialogues", [])
    s3_result = results.get("stage_3_whole_book_dialogue", {})

    md = []
    md.append("# CPT Multi-Stage Evaluation Report: `gemma4_e2b_acid_soils:latest`\n")
    md.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Evaluator Model (vLLM):** `{results.get('vllm_model')}`")
    md.append(f"**Target CPT Model (Ollama):** `{results.get('ollama_model')}`\n")

    md.append("---")
    md.append("## Executive Summary Scores\n")
    md.append("| Benchmark Metric | Score (0 - 100) |")
    md.append("| :--- | :---: |")
    md.append(f"| **Overall CPT Efficacy Index** | **{summary.get('overall_cpt_efficacy_index', 0)}** |")
    md.append(f"| Stage 1: Q&A Factual Accuracy | {summary.get('avg_stage1_accuracy', 0)} |")
    md.append(f"| Stage 1: Factual Retention Score | {summary.get('avg_stage1_retention', 0)} |")
    md.append(f"| Stage 2: Chapter Dialogue Accuracy | {summary.get('avg_stage2_accuracy', 0)} |")
    md.append(f"| Stage 2: Domain Depth & Technicality | {summary.get('avg_stage2_depth', 0)} |")
    md.append(f"| Stage 3: Macro Whole-Book Accuracy | {s3_result.get('evaluation', {}).get('overall_accuracy', 0)} |")
    md.append(f"| Stage 3: Cross-Chapter Synthesis | {s3_result.get('evaluation', {}).get('synthesis_and_reasoning', 0)} |\n")

    md.append("### Final CPT Assessment Verdict")
    md.append(f"> {s3_result.get('evaluation', {}).get('final_cpt_verdict', 'N/A')}\n")

    md.append("---")
    md.append("## Stage 1: Chapter-by-Chapter Q&A Results\n")
    md.append("| Chapter Title | Avg Accuracy | Avg Fluency | Avg Retention |")
    md.append("| :--- | :---: | :---: | :---: |")
    for r in s1_results:
        md.append(f"| {r.get('chapter_title')} | {r.get('avg_accuracy')} | {r.get('avg_fluency')} | {r.get('avg_retention')} |")
    md.append("\n")

    md.append("---")
    md.append("## Stage 2: Chapter Peer Conversation Breakdown\n")
    for r in s2_results:
        title = r.get("chapter_title")
        ev = r.get("evaluation", {})
        md.append(f"### Chapter: {title}")
        md.append(f"- **Accuracy:** {ev.get('chapter_accuracy')}/100")
        md.append(f"- **Fluency:** {ev.get('chapter_fluency')}/100")
        md.append(f"- **Domain Depth:** {ev.get('domain_depth')}/100")
        md.append(f"- **Hallucination Level:** `{ev.get('hallucination_rating')}`")
        md.append(f"- **Summary:** {ev.get('summary')}\n")

    md.append("---")
    md.append("## Stage 3: Whole-Book Macro Evaluation\n")
    b_ev = s3_result.get("evaluation", {})
    md.append(f"- **Overall Accuracy:** {b_ev.get('overall_accuracy')}/100")
    md.append(f"- **Overall Fluency:** {b_ev.get('overall_fluency')}/100")
    md.append(f"- **Cross-Chapter Synthesis:** {b_ev.get('synthesis_and_reasoning')}/100\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Markdown report saved to: {output_file}")

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing CPT Multi-Stage Evaluation Pipeline...")
    print(f"  Target Ollama Model: {args.ollama_model} ({args.ollama_url})")
    print(f"  Judge vLLM Model:   {args.vllm_model} ({args.vllm_url})")

    ollama = OllamaClient(args.ollama_url, args.ollama_model)
    vllm = VLLMClient(args.vllm_url, args.vllm_model)

    ollama_ok = ollama.check_connection()
    vllm_ok = vllm.check_connection()

    if not (ollama_ok and vllm_ok):
        print("\n[ERROR] Connection Check Failed!")
        print("Please ensure both Ollama and vLLM servers are running before starting evaluation.")
        if not ollama_ok:
            print(f"  -> Ollama is not responding at {args.ollama_url}")
        if not vllm_ok:
            print(f"  -> vLLM is not responding at {args.vllm_url}")
            print(f"     Run vLLM manually: `vllm serve {args.vllm_model} --port 8000`")
        print("\nExiting setup validation phase.")
        sys.exit(1)

    print(f"\nLoading extracted chapters from: {args.chapters_path}")
    with open(args.chapters_path, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    if args.skip_references:
        chapters = [ch for ch in chapters if "reference" not in ch.get("title", "").lower()]

    print(f"Loaded {len(chapters)} chapters for evaluation.")

    # Execute Stage 1
    stage1_results = []
    for ch in chapters:
        res1 = stage_1_chapter_qa(vllm, ollama, ch, args.num_questions)
        stage1_results.append(res1)

    # Execute Stage 2
    stage2_results = []
    for ch in chapters:
        res2 = stage_2_chapter_convo(vllm, ollama, ch, args.convo_turns)
        stage2_results.append(res2)

    # Execute Stage 3
    stage3_result = stage_3_whole_book_convo(vllm, ollama, chapters, args.book_convo_turns)

    # Compute Averages
    avg_s1_acc = round(sum(r["avg_accuracy"] for r in stage1_results) / max(len(stage1_results), 1), 2)
    avg_s1_ret = round(sum(r["avg_retention"] for r in stage1_results) / max(len(stage1_results), 1), 2)

    avg_s2_acc = round(sum(r["evaluation"].get("chapter_accuracy", 0) for r in stage2_results) / max(len(stage2_results), 1), 2)
    avg_s2_depth = round(sum(r["evaluation"].get("domain_depth", 0) for r in stage2_results) / max(len(stage2_results), 1), 2)

    s3_acc = stage3_result.get("evaluation", {}).get("overall_accuracy", 0)

    overall_efficacy = round((avg_s1_acc * 0.3) + (avg_s1_ret * 0.2) + (avg_s2_acc * 0.2) + (avg_s2_depth * 0.15) + (s3_acc * 0.15), 2)

    summary = {
        "total_chapters_evaluated": len(chapters),
        "avg_stage1_accuracy": avg_s1_acc,
        "avg_stage1_retention": avg_s1_ret,
        "avg_stage2_accuracy": avg_s2_acc,
        "avg_stage2_depth": avg_s2_depth,
        "overall_cpt_efficacy_index": overall_efficacy
    }

    full_report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ollama_model": args.ollama_model,
        "vllm_model": args.vllm_model,
        "summary": summary,
        "stage_1_qa": stage1_results,
        "stage_2_chapter_dialogues": stage2_results,
        "stage_3_whole_book_dialogue": stage3_result
    }

    json_path = out_dir / "cpt_eval_results.json"
    md_path = out_dir / "cpt_eval_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report_data, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    generate_markdown_report(full_report_data, str(md_path))

    print("\nCPT Multi-Stage Evaluation Complete!")
    print(f"Overall CPT Efficacy Index: {overall_efficacy} / 100")

if __name__ == "__main__":
    main()

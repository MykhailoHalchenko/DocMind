import json
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add mcp-serv to path for imports
sys.path.insert(0, str(Path(__file__).parent / "mcp-serv"))

JUDGE_PROMPT = """You are an expert evaluator for AI-generated scientific summaries.
Given the original context, a question, and the AI's answer, evaluate:

1. hallucination: "none" | "partial" | "yes"
2. coverage_score: 0-10 (did the answer cover all key facts?)
3. citation_accuracy: "accurate" | "partial" | "missing"
4. verdict: "pass" | "fail"

Return ONLY a JSON object. No markdown.

Example:
{"hallucination": "none", "coverage_score": 9, "citation_accuracy": "accurate", "verdict": "pass"}"""


def local_judge_answer(question: str, context: str, answer: str, expected: str) -> Dict[str, Any]:
    """Local heuristic-based evaluation without external API"""
    
    # Check for citation format [chunk_id]
    has_citation = "[" in answer and "]" in answer
    citation_accuracy = "accurate" if has_citation else "missing"
    
    # Check for hallucinations - look for factual contradictions
    context_lower = context.lower()
    answer_lower = answer.lower()
    
    # Look for explicit contradictions in numbers
    contradictions = False
    import re
    context_numbers = set(re.findall(r'\d+%?', context_lower))
    answer_numbers = set(re.findall(r'\d+%?', answer_lower))
    
    # If answer has numbers not in context, flag as potential hallucination
    unknown_numbers = answer_numbers - context_numbers
    if unknown_numbers and len(unknown_numbers) > 0:
        hallucination = "partial"
    else:
        hallucination = "none"
    
    # Check coverage - match key words from expected answer
    expected_lower = expected.lower()
    # Remove punctuation and split into words
    import string
    expected_words = [w.strip(string.punctuation) for w in expected_lower.split() if len(w.strip(string.punctuation)) > 2]
    
    matched_count = 0
    for word in expected_words:
        # Check if word or its stem is in answer
        if word in answer_lower:
            matched_count += 1
        else:
            # Try simple stemming - remove common suffixes
            stems_to_try = [word]
            if word.endswith('ing'):
                stems_to_try.append(word[:-3])
            if word.endswith('ed'):
                stems_to_try.append(word[:-2])
            if word.endswith('s'):
                stems_to_try.append(word[:-1])
            if word.endswith('ment'):
                stems_to_try.append(word[:-4])
            
            # Check if any stem exists in answer
            for stem in stems_to_try:
                if stem and stem in answer_lower:
                    matched_count += 1
                    break
    
    coverage_score = min(10, int((matched_count / max(len(expected_words), 1)) * 10)) if expected_words else 10
    
    # Determine verdict - pass if good coverage and at least citations present (even with partial hallucination)
    verdict = "pass" if coverage_score >= 7 and has_citation else "fail"
    
    return {
        "hallucination": hallucination,
        "coverage_score": coverage_score,
        "citation_accuracy": citation_accuracy,
        "verdict": verdict
    }


async def judge_answer(question: str, context: str, answer: str, expected: str = "") -> dict:
    """Evaluate answer quality using local heuristics"""
    return await asyncio.sleep(0.01) or local_judge_answer(question, context, answer, expected)


async def run_evaluation(test_cases: list[dict]) -> list[dict]:
    """
    test_cases: list of {"question": str, "context": str, "answer": str, "expected": str}
    """
    tasks = [judge_answer(tc["question"], tc["context"], tc["answer"], tc.get("expected", "")) for tc in test_cases]
    results = await asyncio.gather(*tasks)

    report = []
    for tc, result in zip(test_cases, results):
        report.append({
            "question": tc["question"],
            "verdict": result.get("verdict"),
            "hallucination": result.get("hallucination"),
            "coverage_score": result.get("coverage_score"),
            "citation_accuracy": result.get("citation_accuracy"),
        })
    return report


async def print_eval_report(test_cases: list[dict]) -> None:
    report = await run_evaluation(test_cases)
    passed = sum(1 for r in report if r["verdict"] == "pass")
    print(f"\n=== Evaluation Report: {passed}/{len(report)} passed ===\n")
    for r in report:
        status = "✅" if r["verdict"] == "pass" else "❌"
        print(f"{status} [{r['verdict'].upper()}] Q: {r['question'][:60]}")
        print(f"   Hallucination: {r['hallucination']} | Coverage: {r['coverage_score']}/10 | Citations: {r['citation_accuracy']}\n")
import json
import asyncio
from openai import AsyncOpenAI
from config import settings

_judge_client = AsyncOpenAI(api_key=settings.openai_api_key)

JUDGE_PROMPT = """You are an expert evaluator for AI-generated scientific summaries.
Given the original context, a question, and the AI's answer, evaluate:

1. hallucination: "none" | "partial" | "yes"
2. coverage_score: 0-10 (did the answer cover all key facts?)
3. citation_accuracy: "accurate" | "partial" | "missing"
4. verdict: "pass" | "fail"

Return ONLY a JSON object. No markdown.

Example:
{"hallucination": "none", "coverage_score": 9, "citation_accuracy": "accurate", "verdict": "pass"}"""


async def judge_answer(question: str, context: str, answer: str) -> dict:
    prompt = f"Question: {question}\n\nContext:\n{context[:3000]}\n\nAI Answer:\n{answer}"
    response = await _judge_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=150,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"hallucination": "unknown", "coverage_score": 0, "citation_accuracy": "unknown", "verdict": "fail"}


async def run_evaluation(test_cases: list[dict]) -> list[dict]:
    """
    test_cases: list of {"question": str, "context": str, "answer": str, "expected": str}
    """
    tasks = [judge_answer(tc["question"], tc["context"], tc["answer"]) for tc in test_cases]
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
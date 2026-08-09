"""
System Design Evaluation Framework

Evaluates LLM performance on system design problems using a comprehensive rubric
covering: Correctness, Scalability, Trade-offs, Clarity, and Practicality.

Usage:
    python evaluation_framework.py
"""

import json
import os
from typing import Dict, List, Any
from anthropic import Anthropic

# Initialize Anthropic client
client = Anthropic()

# Load test cases
with open('test_cases.json', 'r') as f:
    test_data = json.load(f)

def evaluate_system_design(prompt: str) -> Dict[str, Any]:
    """
    Evaluate Claude's response to a system design question.

    Args:
        prompt: The system design question

    Returns:
        Dictionary with:
        - answer: Claude's response
        - scores: Dict of dimension scores
        - overall_score: Average score
        - feedback: Summary of strengths/gaps
    """

    # Get Claude's response
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": f"""You are a system design expert evaluating architectural decisions.
Provide a detailed, comprehensive answer to this system design question.

Question: {prompt}

Structure your answer clearly with components, tradeoffs, and implementation considerations."""
            }
        ]
    )

    claude_answer = message.content[0].text

    # Score against rubric
    scores = score_answer(claude_answer)

    # Generate feedback
    feedback = generate_feedback(scores, claude_answer)

    overall_score = sum(scores.values()) / len(scores)

    return {
        "answer": claude_answer,
        "scores": scores,
        "overall_score": overall_score,
        "feedback": feedback
    }

def score_answer(answer: str) -> Dict[str, float]:
    """
    Score answer against each rubric dimension.

    Scoring strategy: Keyword matching + heuristic analysis
    """

    scores = {}
    answer_lower = answer.lower()

    # DIMENSION 1: Correctness (25%)
    correctness_indicators = [
        "consistency", "idempotent", "edge case", "failure",
        "error", "recovery", "retry", "reconciliation"
    ]
    matches = sum(1 for indicator in correctness_indicators if indicator in answer_lower)

    # Base score for structure + depth
    base_correctness = 4 if len(answer) > 400 else 2
    correctness_score = base_correctness + (matches * 0.75)
    scores["correctness"] = min(10, correctness_score)

    # DIMENSION 2: Scalability (25%)
    scalability_indicators = [
        "shard", "partition", "replicate", "replication", "throughput",
        "scale", "load", "distribute", "distribution", "replica",
        "cluster", "capacity", "growth"
    ]
    matches = sum(1 for indicator in scalability_indicators if indicator in answer_lower)

    base_scalability = 4 if "shard" in answer_lower or "partition" in answer_lower else 2
    scalability_score = base_scalability + (matches * 0.7)
    scores["scalability"] = min(10, scalability_score)

    # DIMENSION 3: Trade-offs (20%)
    tradeoff_indicators = [
        "trade-off", "tradeoff", "consistency", "availability",
        "latency", "reliability", "cap theorem", "cap"
    ]
    matches = sum(1 for indicator in tradeoff_indicators if indicator in answer_lower)

    # Bonus for explicit CAP/theorem discussion
    cap_bonus = 2 if "cap" in answer_lower else 0

    base_tradeoff = 3 if "consistency" in answer_lower or "availability" in answer_lower else 1
    tradeoff_score = base_tradeoff + (matches * 0.5) + cap_bonus
    scores["tradeoffs"] = min(10, tradeoff_score)

    # DIMENSION 4: Clarity (15%)
    word_count = len(answer.split())
    has_structure = "\n" in answer  # Uses line breaks = structure
    has_components = answer_lower.count("component") + answer_lower.count("layer") + answer_lower.count("service")

    clarity_base = 5
    clarity_bonus = 0

    # Good word count (500-2000 words = detailed but focused)
    if 500 < word_count < 2500:
        clarity_bonus += 1.5

    # Structure and organization
    if has_structure and has_components > 0:
        clarity_bonus += 2

    clarity_score = clarity_base + clarity_bonus
    scores["clarity"] = min(10, clarity_score)

    # DIMENSION 5: Practicality (15%)
    practical_indicators = [
        "implement", "deploy", "monitor", "alert", "observ",
        "failure", "timeout", "kafka", "redis", "postgres",
        "mysql", "elasticsearch", "docker", "kubernetes", "tool"
    ]
    matches = sum(1 for indicator in practical_indicators if indicator in answer_lower)

    # Bonus for mentioning specific tools
    tool_count = sum(1 for tool in ["kafka", "redis", "postgres", "mysql", "elasticsearch", "kubernetes"]
                     if tool in answer_lower)
    tool_bonus = min(tool_count * 0.5, 2)

    base_practical = 3 if "implement" in answer_lower or "deploy" in answer_lower else 2
    practical_score = base_practical + (matches * 0.4) + tool_bonus
    scores["practicality"] = min(10, practical_score)

    return {k: round(v, 1) for k, v in scores.items()}

def generate_feedback(scores: Dict[str, float], answer: str) -> str:
    """Generate human-readable feedback based on scores."""

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    strengths = []
    gaps = []

    for dimension, score in sorted_scores[:2]:
        if score >= 7:
            strengths.append(f"{dimension.capitalize()} ({score}/10)")

    for dimension, score in sorted_scores[-2:]:
        if score < 7:
            gaps.append(f"{dimension.capitalize()} ({score}/10)")

    feedback_parts = []

    if strengths:
        feedback_parts.append(f"Strengths: {', '.join(strengths)}")

    if gaps:
        feedback_parts.append(f"Areas to improve: {', '.join(gaps)}")

    overall = sum(scores.values()) / len(scores)

    if overall >= 8:
        feedback_parts.append("Overall: Excellent architectural thinking.")
    elif overall >= 7:
        feedback_parts.append("Overall: Good system design reasoning.")
    elif overall >= 6:
        feedback_parts.append("Overall: Solid foundation, needs more depth.")
    else:
        feedback_parts.append("Overall: Needs significant improvement.")

    return " ".join(feedback_parts)

def run_evaluation_suite() -> List[Dict[str, Any]]:
    """Run evaluation on all test cases."""

    results = []
    test_cases = test_data['test_cases']

    print(f"\n{'='*70}")
    print("System Design Evaluation Suite")
    print(f"{'='*70}\n")

    print(f"Evaluating {len(test_cases)} test cases...\n")

    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case['id']
        difficulty = test_case['difficulty']
        category = test_case['category']

        print(f"[{i}/{len(test_cases)}] {test_id} ({difficulty}) - {category}")
        print(f"Prompt: {test_case['prompt'][:80]}...")

        result = evaluate_system_design(test_case['prompt'])

        result_obj = {
            "test_id": test_id,
            "category": category,
            "difficulty": difficulty,
            "scores": result["scores"],
            "overall_score": round(result["overall_score"], 1),
            "feedback": result["feedback"],
            "answer_preview": result["answer"][:200] + "..."
        }

        results.append(result_obj)

        print(f"Score: {result_obj['overall_score']}/10")
        print(f"Feedback: {result['feedback']}\n")

    return results

def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print evaluation summary."""

    scores = [r['overall_score'] for r in results]
    avg_score = sum(scores) / len(scores)

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}\n")

    print(f"Total Test Cases: {len(results)}")
    print(f"Average Score: {avg_score:.1f}/10\n")

    # Dimension analysis
    all_dimension_scores = {}
    for result in results:
        for dimension, score in result['scores'].items():
            if dimension not in all_dimension_scores:
                all_dimension_scores[dimension] = []
            all_dimension_scores[dimension].append(score)

    print("Performance by Dimension:")
    for dimension in ['correctness', 'scalability', 'tradeoffs', 'clarity', 'practicality']:
        if dimension in all_dimension_scores:
            avg = sum(all_dimension_scores[dimension]) / len(all_dimension_scores[dimension])
            print(f"  {dimension.capitalize():20} {avg:.1f}/10")

    # By difficulty
    print("\nPerformance by Difficulty:")
    by_difficulty = {}
    for result in results:
        difficulty = result['difficulty']
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = []
        by_difficulty[difficulty].append(result['overall_score'])

    for difficulty in ['medium', 'hard']:
        if difficulty in by_difficulty:
            avg = sum(by_difficulty[difficulty]) / len(by_difficulty[difficulty])
            count = len(by_difficulty[difficulty])
            print(f"  {difficulty.capitalize():20} {avg:.1f}/10 ({count} cases)")

    # Category analysis
    print("\nPerformance by Category:")
    by_category = {}
    for result in results:
        category = result['category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(result['overall_score'])

    for category in sorted(by_category.keys()):
        avg = sum(by_category[category]) / len(by_category[category])
        print(f"  {category:30} {avg:.1f}/10")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    # Run evaluation suite
    results = run_evaluation_suite()

    # Save results to JSON
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✅ Results saved to: evaluation_results.json")

    # Print summary
    print_summary(results)

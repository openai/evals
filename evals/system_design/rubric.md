# System Design Evaluation Rubric

## Overview
This rubric evaluates LLM performance on system design problems across 5 key dimensions. Each dimension is scored 1-10 independently, with an overall score as the average.

---

## Scoring Scale: 1-10

### 1. Correctness (Weight: 25%)
**Definition:** Does the proposed solution actually work? Does it handle critical edge cases without fatal flaws?

#### Scoring Breakdown:
- **1-3 (Weak):** Solution has fundamental flaws, misses critical requirements, or is incomplete
- **4-6 (Okay):** Solution works for basic case, but misses edge cases or important considerations
- **7-8 (Good):** Solution is sound, handles most edge cases, minor gaps
- **9-10 (Excellent):** Comprehensive solution, anticipates edge cases, production-ready thinking

#### Scoring Criteria:
✓ Addresses core requirement
✓ Handles at least 2-3 edge cases
✓ No architectural contradictions
= Score 7+

---

### 2. Scalability (Weight: 25%)
**Definition:** Does the solution handle growth? Are scaling strategies explicit and reasonable?

#### Scoring Breakdown:
- **1-3 (Weak):** Doesn't mention scaling or proposes clearly unscalable approach
- **4-6 (Okay):** Mentions scaling vaguely, but lacks detail
- **7-8 (Good):** Proposes specific scaling strategy (sharding, replication, etc.)
- **9-10 (Excellent):** Detailed scaling strategy with capacity planning and monitoring

#### Scoring Criteria:
✓ Mentions sharding/partitioning
✓ Discusses replication strategy
✓ Addresses database/cache scaling
= Score 7+

---

### 3. Trade-offs (Weight: 20%)
**Definition:** Does the responder acknowledge tradeoffs? Are choices justified?

#### Scoring Breakdown:
- **1-3 (Weak):** No mention of tradeoffs, seems oblivious to constraints
- **4-6 (Okay):** Mentions tradeoffs but doesn't fully explain
- **7-8 (Good):** Explicitly states tradeoffs and justifies choice
- **9-10 (Excellent):** Deep tradeoff analysis with context-specific reasoning

#### Scoring Criteria:
✓ Mentions 2+ tradeoffs explicitly
✓ Justifies at least one choice
✓ Shows understanding of CAP theorem or similar
= Score 7+

---

### 4. Clarity (Weight: 15%)
**Definition:** Is the explanation easy to follow? Is reasoning well-structured?

#### Scoring Breakdown:
- **1-3 (Weak):** Rambling, hard to follow, poor organization
- **4-6 (Okay):** Understandable but could be clearer, some organization
- **7-8 (Good):** Clear structure, easy to follow, good component explanation
- **9-10 (Excellent):** Crystal clear, well-organized, uses diagrams mentally

#### Scoring Criteria:
✓ Organized into clear components
✓ Explains component relationships
✓ Easy to mentally visualize
= Score 7+

---

### 5. Practicality (Weight: 15%)
**Definition:** Could this actually be built? Are implementation details addressed?

#### Scoring Breakdown:
- **1-3 (Weak):** Purely theoretical, no consideration for implementation
- **4-6 (Okay):** Mentions implementation but lacks depth
- **7-8 (Good):** Specific tools/patterns mentioned, most implementation questions answered
- **9-10 (Excellent):** Deep implementation thinking, monitoring, deployment strategy

#### Scoring Criteria:
✓ Mentions specific tools (Kafka, Redis, etc.)
✓ Discusses deployment/clustering
✓ Mentions monitoring or observability
= Score 7+

---

## Overall Score Calculation

**Formula:** Average of all 5 dimensions

```
Overall Score = (Correctness + Scalability + Trade-offs + Clarity + Practicality) / 5
```

**Score Interpretation:**
- **1-3:** Poor (significant gaps, not production-ready)
- **4-5:** Below Average (works but many improvements needed)
- **6-7:** Average (solid for junior, needs work for senior)
- **7-8:** Good (solid senior engineer level)
- **8-9:** Very Good (staff engineer thinking)
- **9-10:** Excellent (hiring-grade response)

---

## Quick Scoring Guide

**When in doubt:**
- Does it address the core problem? → Correctness ≥ 5
- Does it handle 1M scale? → Scalability ≥ 5
- Does it acknowledge tradeoffs? → Trade-offs ≥ 5
- Is it easy to understand? → Clarity ≥ 5
- Could someone build it? → Practicality ≥ 5

**What pushes a score to 8+:**
- Specific examples (tools, numbers)
- Anticipation of failure scenarios
- Monitoring/observability discussion
- Explicit tradeoff reasoning
- Clear component architecture

---

## Evaluation Process

1. Read the system design answer completely
2. Score each dimension independently (1-10)
3. Note specific quotes that justify each score
4. Calculate overall score as average
5. Provide 2-3 sentence summary of strengths and gaps

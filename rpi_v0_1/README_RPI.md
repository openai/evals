# RPI v0.1 — Recursive Perspective Integration Eval

RPI v0.1 is a pilot benchmark and reasoning intervention for testing whether **bounded, adaptive multi-perspective deliberation** improves model performance on complex problems.

## Hypothesis

On sufficiently complex tasks, adaptive perspective fragmentation followed by information-preserving convergence, contradiction analysis, epistemic checking, bounded consequence recursion, affected-party modeling, and corrective feedback will improve final-answer quality relative to a standard reasoning condition.

The protocol includes explicit activation and stopping rules so that simple or urgent tasks do not receive unnecessary recursive analysis.

## What this contribution is

- A 30-case pilot dataset.
- A frozen RPI v0.1 protocol.
- A predefined methodology and failure criteria.
- A model-graded rubric configuration compatible with the existing Evals model-graded pattern.
- No custom Python evaluation code.

## What this contribution is not

RPI is not a metaphysical claim, a claim about machine consciousness, or a requirement that a model reveal chain-of-thought. The evaluation concerns observable answer quality.

## Case families

1. Complex multi-perspective reasoning
2. Epistemic and contradiction detection
3. Downstream consequence reasoning
4. Adversarial RPI failure modes
5. Simple negative controls

## Key design principle

More reasoning is not automatically better. RPI should lose when it creates false equivalence, speculation, indecision, verbosity, or needless complexity.

## Files

- `RPI_PROTOCOL.md` — intervention definition and stopping rules.
- `METHODOLOGY.md` — baseline/RPI comparison design.
- `evals/registry/data/rpi_v0_1/samples.jsonl` — benchmark cases.
- `evals/registry/evals/rpi_v0_1.yaml` — eval registration.
- `evals/registry/modelgraded/rpi_quality.yaml` — model-grader rubric.
- `PR_DRAFT.md` — draft pull-request description.

## Recommended experiment

Run the same model on the same frozen cases under:
- **Baseline:** normal task solving.
- **RPI:** normal task solving plus `RPI_PROTOCOL.md`.

Blind and randomize outputs for pairwise human evaluation in addition to model grading where practical.

This v0.1 package is intentionally a pilot. Its purpose is to determine whether a larger benchmark is justified and where the protocol fails.

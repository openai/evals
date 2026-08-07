# RPI v0.1 Methodology

## Research question

Does an RPI instruction improve final-answer quality on complex, multi-perspective reasoning tasks without causing material degradation on tasks where deep deliberation is unnecessary?

## Conditions

### Baseline
Use the target model's ordinary task-solving instruction. Do not mention RPI.

### RPI
Use the same model, sampling settings, tools, and task, with the RPI v0.1 protocol added as a deliberation policy.

The RPI instruction should affect internal problem-solving strategy, not require verbose disclosure of hidden reasoning.

## Experimental controls

- Same underlying model/version.
- Same task text.
- Same available tools and external information.
- Same sampling parameters where possible.
- Randomize output order before pairwise human review.
- Graders should not be told which condition produced an answer.
- Freeze the benchmark and rubric before comparing conditions.
- Run multiple seeds where stochasticity is material.

## Primary outcome

Pairwise preference rate for RPI versus baseline on complex cases, using a rubric that prioritizes correctness and decision quality over verbosity.

## Secondary outcomes

- contradiction detection;
- evidence/inference separation;
- uncertainty calibration;
- relevant perspective coverage;
- downstream consequence recognition;
- false-equivalence avoidance;
- sycophancy resistance;
- decisiveness/actionability;
- unnecessary reasoning overhead.

## Negative controls

Simple-control cases test whether the RPI condition appropriately avoids unnecessary decomposition. An elaborate answer to a simple task should be penalized when it reduces clarity or efficiency.

## Adversarial cases

Adversarial cases deliberately target RPI's predicted weaknesses: too many perspectives, emotionally persuasive but unsupported claims, false balance, urgency, speculative future chains, and requests for novelty where conventional action is superior.

## Interpretation

A positive result requires more than higher verbosity or broader coverage. RPI should improve the quality of the final decision or answer.

A mixed result is expected to be informative. For example, RPI may improve stakeholder reasoning but worsen urgent decisions. Such results should narrow the activation gate rather than be averaged away.

A negative result should be preserved and reported. RPI is a falsifiable intervention, not a doctrine.

## v0.1 scope

This pilot contains 30 cases:
- 8 complex multi-perspective cases
- 6 epistemic/contradiction cases
- 6 downstream-consequence cases
- 6 adversarial/failure-mode cases
- 4 simple negative controls

A larger v1.0 should be built only after pilot review and should avoid rewriting cases merely to favor RPI.

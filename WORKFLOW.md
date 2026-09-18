# Operational Workflow

This document is the execution protocol for one research cycle.

## Phase 0 — Problem brief

Human writes a short problem brief before any agent exploration.

Required fields:
- research question
- why it matters
- what is already known
- target uncertainty
- hard constraints: compute, time, data, reproducibility
- stop conditions

A valid problem should be narrow enough that evidence can change a belief.

Bad:
> Is recurrence useful?

Better:
> Under matched train and inference FLOPs, does recurrent depth improve reasoning tasks more than shallow tasks, relative to a non-recurrent baseline?

## Phase 1 — Overgenerate hypotheses

Run Scout to generate broad coverage across different causal families.

Target:
- 30-100 raw hypotheses for a new topic
- include optimization, representation, information, scaling, regularization, data, artifact, and null explanations

Do not rank them yet.

## Phase 2 — Compress into mechanism space

Cluster raw hypotheses by causal mechanism.

Output:
- mechanism cluster name
- representative claim
- member hypotheses
- assumptions
- observably distinct predictions

Remove paraphrase duplicates aggressively.

A useful target is 5-20 mechanism-distinct candidates.

## Phase 3 — Falsification tournament

Run Skeptic on every surviving mechanism.

Each hypothesis must have:
- unique prediction
- falsifier
- key confounders
- strongest null explanation
- minimal test

Reject hypotheses that can explain every possible result.

## Phase 4 — Experiment tournament

Experimentalist proposes experiments to distinguish hypotheses.

Prefer experiments with high expected discrimination and low cost.

For each experiment record:
- hypotheses separated
- prediction matrix
- controls
- metrics
- estimated cost
- failure modes
- decision rule

Human Gate #1:
- approve expensive runs
- reject poorly identified experiments
- adjust constraints

## Phase 5 — Execution

Engineer owns implementation and reproducibility.

Every run must record:
- code / commit
- config
- seed
- environment
- data version
- hardware
- runtime / cost
- raw outputs

Do not silently repair failed experiments. Record deviations.

## Phase 6 — Evidence aggregation

Analyst updates claims using the evidence ledger.

Separate:
- observed result
- interpretation
- confidence change

Do not rewrite the narrative first.

## Phase 7 — Adversarial review

Skeptic reviews the current strongest explanation.

Questions:
- What alternative explanation survives?
- What hidden confound remains?
- Did we accidentally optimize the metric?
- Does the result reproduce?
- What evidence would reverse the conclusion?

## Phase 8 — Human Gate #2

Human reads only:
- high-information evidence
- anomalies
- unresolved contradictions
- proposed agenda changes

Then updates the decision log.

Possible decisions:
- continue
- replicate
- branch
- pivot
- stop

## Anomaly protocol

An anomaly is a result that conflicts with all major current predictions.

When detected:
1. Freeze post-hoc storytelling.
2. Verify instrumentation, data, and implementation.
3. Reproduce with independent seed / implementation where possible.
4. Generate mutually exclusive explanations.
5. Design the cheapest discrimination test.
6. Update beliefs only after the anomaly survives verification.

## Default attention budget

```text
raw hypotheses:             30-100
mechanism-distinct:          5-20
falsifiable survivors:       3-10
experiments approved:        1-3
results for human deep-read: 1-2
```

If the human needs to read everything, the system is failing.

## Default cycle output

At the end of each cycle, `06_reports/current_state.md` should answer only:

1. What is the question?
2. What are the strongest live explanations?
3. What evidence changed our beliefs?
4. What remains unresolved?
5. What is the highest-value next experiment?
6. Is human attention required? Why?

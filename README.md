# Evidence-Driven Research OS

> Ideas are abundant. Truth is scarce.

Evidence-Driven Research OS is a lightweight workflow for doing research with frontier AI systems without turning the human researcher into the bottleneck. The system treats AI as a high-bandwidth search and execution layer, while the human operates the research control plane: defining objectives, constraining search, approving expensive experiments, inspecting anomalies, and updating the research agenda.

The goal is not to read every model-generated hypothesis. The goal is to convert a large hypothesis space into a small amount of discriminating evidence.

## Core loop

```text
Human objective / research taste
        |
        v
0. Problem Brief
        |
        v
1. Hypothesis Generation        -- deliberately overgenerate
        |
        v
2. Compression + Dedup          -- language space -> mechanism space
        |
        v
3. Falsification Tournament     -- predictions, falsifiers, confounders
        |
        v
4. Experiment Design            -- maximize information gain / compute
        |
        v
   [ HUMAN GATE ]               -- approve expensive runs
        |
        v
5. Automated Execution          -- implement, run, debug, rerun
        |
        v
6. Evidence Aggregation         -- update claims, not narratives
        |
        v
7. Adversarial Review           -- alternative explanations, nulls
        |
        v
   [ HUMAN GATE ]               -- inspect anomalies, update agenda
        |
        +----------------------> repeat
```

## Design principles

1. **Never optimize for plausible prose.** A hypothesis is useful only if it constrains observations.
2. **Compress before reading.** Raw model reasoning is intermediate data, not the research interface.
3. **Prefer mechanism clusters over hypothesis rankings.** Do not ask an LLM to assign arbitrary 8.7/10 scores to ideas.
4. **Falsification before implementation.** Every surviving hypothesis needs a unique prediction and a failure condition.
5. **Experiment for discrimination, not benchmark gain.** Prefer the cheapest experiment that separates competing explanations.
6. **Evidence is the source of truth.** Notes and reports are views over the evidence ledger, not the canonical state.
7. **Anomalies deserve human attention.** When evidence conflicts with all major hypotheses, stop post-hoc storytelling and investigate.
8. **Human attention is scarce.** A healthy cycle may compress 100 raw hypotheses into 1-2 results that deserve deep human inspection.

## Hypothesis compiler

Long-form explanations should be compiled into a fixed schema before entering the research loop:

```yaml
hypothesis_id: H017
claim: Recurrence improves performance through iterative latent refinement.
mechanism: Repeated transformation progressively refines the same hidden representation.
unique_prediction: Gains increase with tasks that require greater computational depth, under matched compute.
confounders:
  - additional FLOPs
  - parameter-sharing regularization
  - optimization-path effects
falsifier: Under matched train/inference compute, gains do not correlate with required computational depth.
minimal_test: Compare recurrent and non-recurrent models under equal FLOPs across tasks with controlled depth.
estimated_cost: 6 H100-hours
expected_information_gain: high
status: proposed
```

Do not read the original 3,000-token argument unless the hypothesis survives the tournament.

## Research objective

A useful approximation is:

```text
Research progress = uncertainty_before - uncertainty_after
```

The system should therefore optimize approximately for:

```text
expected hypothesis discrimination / experiment cost
```

rather than raw benchmark improvement.

## Five-agent model

| Agent | Responsibility |
|---|---|
| `Scout` | Literature search, hypothesis coverage, mechanism discovery |
| `Skeptic` | Null hypotheses, confounders, falsifiers, artifact explanations |
| `Experimentalist` | Cheapest experiments that discriminate between mechanisms |
| `Engineer` | Implementation, execution, debugging, reproducibility |
| `Analyst` | Evidence aggregation, belief updates, anomaly detection |

The system intentionally does **not** use many decorative personas. Scout is easy to scale; Skeptic and Experimentalist are usually more valuable.

## Repository structure

```text
.
├── 00_problem/
│   ├── problem.template.md
│   └── constraints.template.yaml
├── 01_literature/
│   └── README.md
├── 02_hypotheses/
│   └── hypothesis.template.yaml
├── 03_experiments/
│   └── experiment.template.yaml
├── 04_evidence/
│   └── claim.template.yaml
├── 05_decisions/
│   └── decision_log.template.md
├── 06_reports/
│   └── current_state.template.md
├── agents/
│   ├── scout.md
│   ├── skeptic.md
│   ├── experimentalist.md
│   ├── engineer.md
│   └── analyst.md
└── WORKFLOW.md
```

The canonical state is:

```text
hypothesis registry
+ experiment registry
+ evidence ledger
+ decision log
```

Reports are generated from this state.

## Human attention budget

A default cycle can look like:

```text
100 raw hypotheses
      -> 20 mechanism-distinct hypotheses
      -> 8 falsifiable hypotheses
      -> 3 discriminating experiments
      -> 1-2 surprising / decision-relevant results
      -> HUMAN READS DEEPLY
```

The human researcher should not become the queue that every agent message must pass through.

## Anomaly protocol

When an experimental result is inconsistent with all major predictions:

```text
ANOMALY DETECTED

Do not generate a smooth post-hoc explanation yet.

1. Check measurement and implementation errors.
2. Reproduce the result.
3. Generate mutually exclusive explanations.
4. Derive distinguishing predictions.
5. Run the cheapest discrimination test.
6. Only then update the research narrative.
```

## Getting started

1. Copy the templates in `00_problem/` and define one sharp research question.
2. Run Scout + Skeptic to create a broad hypothesis pool.
3. Cluster hypotheses by mechanism; discard paraphrase duplicates.
4. Force every survivor into `02_hypotheses/hypothesis.template.yaml`.
5. Ask Experimentalist for experiments that distinguish hypotheses, not merely improve a score.
6. Human approves the expensive runs.
7. Engineer executes them and records reproducible configs.
8. Analyst updates `04_evidence/` and flags anomalies.
9. Human updates the decision log and research agenda.
10. Repeat until the key uncertainty is resolved or the expected information gain becomes too low.

See [`WORKFLOW.md`](WORKFLOW.md) for the operational protocol and `agents/` for reusable agent instructions.

## Status

This repository starts as a practical research workflow, not a heavy framework. The first goal is to use it on real research topics, identify where the loop breaks, and automate only the steps that repeatedly create friction.

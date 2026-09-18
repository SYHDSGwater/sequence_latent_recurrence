# Six-checkpoint completion, 2026-09-18

Remote plan pulled at commit 06b63d7. User requests the four remaining public checkpoints and a cost estimate for Jacobi causal training. No causal training is launched by this run.

First complete the existing five-arm NLL study on 135M/50B, 362M/10B, 362M/50B and 982M/10B. Keep exactly the frozen 256 WikiText and 256 MATH-500 examples, token IDs, intervention positions, five arms, BF16 exact adapter, horizon eligibility, bootstrap and ±0.01-nat practical interval used in C-T2-EXPANDED. Batch 64. Official-step parity and tokenizer parity required. Pinned revisions/LFS hashes are in `01_literature/t2mlr_six_checkpoint_audit.json`. Reuse earlier two checkpoint cells without treating them as new replication.

This completion is the comparable NLL component of Phase A, not completion of all Phase-A primary endpoints in E-T2-002. Its state/output impulse responses, small norm-controlled perturbations and recovery times require separate measurements with their definitions fixed before results. Keep those endpoints explicitly pending until run. Frozen checkpoint comparisons alone cannot establish the Jacobi training hypothesis.

Run on the previously authorized pro-78730289ac36 instance. External timeout 10,800 seconds for preparation and eight cells; per-cell cap 3,600 seconds. Report interrupted cells as incomplete. No outcome-dependent stopping or sample replacement. Archive raw paired losses, input tokens, manifests and source provenance separately from prior results.

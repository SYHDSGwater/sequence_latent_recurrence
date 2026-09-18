# 2026-09-18 — Pivot: Compute efficiency first

## Previous plan

E-T2-002 Phase B 原计划通过 Jacobi(16,4) / Jacobi(16,16) / exact-sequential+TBPTT4 / exact-sequential+long-BPTT 的 from-scratch causal grid，判断 T²MLR 的短 persistence 是否由 temporal-parallel Jacobi 与 short recurrent credit horizon 塑造。

## New evidence

已完成 Phase A：六个公开 checkpoint × WikiText / MATH reference-solution 的 carry、reset、clean-KV、state/KL impulse response 与 BF16/FP32 precision audit。

当前最稳健结论：

- T²MLR recurrent path 被学会并实际使用；
- frozen checkpoints 的行为更符合 local / deep-to-shallow temporal feature shortcut 与可重建状态；
- long-horizon latent reasoning 未被建立；
- BF16 long-tail 对数值精度敏感；
- Jacobi causality 仍未被证明。

同时，成本分析显示 exact recurrent BPTT 是强 serial bottleneck；即使 T²MLR 的 Jacobi approximation 已恢复较多 temporal parallelism，论文 measured training wall-clock 仍约为 vanilla Transformer 的 2.24×，repo 的算子账本也给出数倍 FLOPs 量级。

RLT 的独立 controlled evaluation 进一步显示：exact BPTT 路线在 ~50–140M / 500M tokens 上没有质量优势，而训练吞吐和 GPU-hour 成本远差于 vanilla Transformer。

## Belief update

不再把“Jacobi 是否造成短 horizon”视为当前最值得 GPU 投入的问题。它仍是 unresolved hypothesis，但要因果识别它，需要执行本身非常低效的 long/exact BPTT multi-arm multi-seed training。

新的研究优先级：

[
	ext{Compute efficiency}
ightarrow
	ext{scalable tests of Expressivity / Learnability / Usefulness}
]

对于 sequence-axis latent recurrence，compute efficiency 不是最后再优化的 systems detail，而是 architecture research 的前置 gate。

## Decision

**STOP E-T2-002 Phase B。**

- 不再执行 BPTT / long-BPTT causal training grid；
- 不继续为当前 architecture 做大规模 from-scratch recurrence training；
- 保留 Phase A frozen-checkpoint evidence 与所有 raw artifacts；
- H-T2-JACOBI-CONTRACTION 标记为 unresolved / not pursued due compute cost，而不是 supported 或 falsified。

## Rationale

继续 BPTT training 的 expected information gain / GPU-hour 太低：

1. exact recurrence 的 token-serial dependency 是结构性效率问题；
2. Jacobi approximation 虽然更可训练，但仍显著慢于 vanilla token-parallel training；
3. 为验证 long-horizon mechanism 而增加 backward horizon，会进一步恶化训练效率；
4. 在 compute efficiency 未解决前，小模型/短训练上的 negative 或 positive result 都很难回答真正的 scale-up architecture question。

## Next research question

只有在出现新的训练方法能够同时满足以下条件时，才重新打开 sequence-axis recurrence 的大规模 architecture experiments：

1. 保留有意义的 temporal latent-state semantics；
2. 保留足够长的 temporal credit assignment；
3. 大部分 sequence computation 可并行；
4. training throughput / FLOPs 接近 vanilla Transformer。

核心问题改为：

> **Can sequence-axis latent recurrence be trained nearly as efficiently as a parallel Transformer?**

当前项目总结见 `06_reports/current_state.md`，项目级 evidence ledger 见 `04_evidence/sequence_axis_recurrence_claims.yaml`。

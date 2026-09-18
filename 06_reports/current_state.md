# Sequence-Axis Latent Recurrence — Current State

> Updated 2026-09-18. This page is the project-level synthesis. Detailed raw results remain in `04_evidence/` and `06_reports/`.

## Scope

这里的 **sequence-axis latent recurrence** 指沿 token / sequence 维度传递连续 latent state，使前一位置的中高层表示直接进入后一位置的网络计算。本文主要讨论 T²MLR、RLT 这类设计。

不把 looped Transformer / depth-axis recurrence 混入同一类：后者是在同一 sequence representation 上重复执行 network layers，核心瓶颈不同。

## Current conclusion

当前证据不支持把 sequence-axis latent recurrence 视为已经建立的 long-horizon latent reasoning 机制。

更准确的结论是：

[
oxed{
	ext{T²MLR solves part of Learnability}
;;	ext{but current checkpoints look mainly like a local deep-to-shallow temporal shortcut}
}
]

而 RLT 的 exact recurrent + BPTT 路线在小规模实验里既没有体现质量优势，也付出了极高训练成本。

因此，现阶段最关键的 gating problem 不是继续争论 Expressivity / Learnability / Usefulness，而是：

[
oxed{
	extbf{Compute efficiency first}
}
]

如果 sequence-axis recurrence 不能接近 vanilla Transformer 的 token-parallel training efficiency，就很难在足够规模、足够训练量下公平验证其他维度，更难成为实际 backbone。

---

## Evidence matrix

| Method | Expressivity | Learnability | Usefulness | Compute efficiency | Current read |
|---|---|---|---|---|---|
| **T²MLR** | recurrent middle-layer path 确实存在 | **有证据**：公开 checkpoint 明显依赖 carry / feedback | **有限**：更像 local / one-step deep-to-shallow shortcut；long-horizon reasoning 未建立 | **否**：Jacobi parallel approximation 仍显著慢于 vanilla parallel training | 学会使用 recurrence，但当前收益不足以证明 long-horizon latent reasoning |
| **RLT** | 理论上有 unbounded temporal path | 大规模是否能学会 **难以验证** | Empero 50–140M / 500M tokens 未见优势 | **明显否**：exact BPTT 极低吞吐、GPU-hours 很高 | compute bottleneck 先于 architecture claim 成为主要问题 |

---

## 1. T²MLR：Learnability 基本成立

冻结公开 checkpoint 的干预实验给出的最稳定结论不是“state 没用”，而是相反：**模型确实学会依赖 temporal recurrent path**。

六个 checkpoint × WikiText / MATH reference-solution 共 12 个组合中：

- `carry_off` 的 ΔNLL 95% CI **全部为正**；
- 持续 `zero_every` 会造成明显预测损失；
- 该现象跨 135M / 362M / 982M 与 10B / 50B checkpoint 存在。

因此，T²MLR 至少解决了一个 RLT 更难回答的问题：

[
oxed{
	ext{the recurrent branch is learnable and is actually used}
}
]

详细结果见：

- `06_reports/t2mlr_six_checkpoint_results.md`
- `06_reports/t2mlr_expanded_results.md`

---

## 2. 但公开 T²MLR checkpoint 更像 local deep-to-shallow shortcut

当前干预结果没有建立 long-horizon persistent latent reasoning。

更一致的行为是：

[
h^{deep}_{t-1}
ightarrow
h^{shallower}_{t}
]

提供一个局部、廉价的 temporal feature shortcut；随后正常 token + attention/KV history 很快重新构建当前表示。

主要 evidence：

- zero reset 后，12 个 checkpoint×domain 组合的 state `t50` 中位数均为 **1 token**；
- natural path 的 `t10` 大致为 **5–20 tokens**；
- clean-KV 后 `t10` 缩短到 **2–3 tokens**；
- 同 batch FP32 小扰动对照中 natural `t10` 为 **4–9 tokens**，clean-KV 为 **2–3 tokens**；
- 大多数 lag 32–127 signed ΔNLL 很小；362M/50B Math 是重要例外，但 clean-KV 下远期 signed effect 仍很小；
- BF16 小扰动长尾高度数值精度敏感，不能解释成稳定语义 memory。

因此当前更合理的描述是：

[
oxed{
	ext{locally useful recurrent feature reuse}

eq
	ext{established long-horizon latent reasoning}
}
]

“one-step deep-to-shallow shortcut”描述的是 recurrent edge 的功能形态，不意味着 effect 严格只存活 1 token；真实扰动可以通过 state 与 KV 继续传播数步。

详细结果见 `06_reports/t2mlr_impulse_results.md`。

### Jacobi training 的解释边界

公开 T²MLR checkpoint 使用 temporal-parallel Jacobi approximation（共同配置记录为 forward depth 16、backward depth 4）。当前观察是在**这种训练 recipe 下**得到的。

尚未做 causal training grid，因此不能声称：

> Jacobi approximation **导致** short-horizon dynamics。

只能说：

> 在 Jacobi-trained public checkpoints 中，observed behavior 更符合 local shortcut / rapidly recoverable state，而不是已经证明的 long-horizon reasoning state。

原计划用 exact sequential + long BPTT 区分 architecture 与 training-horizon effect，但该分支因训练效率问题停止，见 E-T2-002。

---

## 3. T²MLR 对下游任务的帮助存在，但幅度有限且不是 training-compute efficient

论文自己的 135M / 10B zero-shot NLP 平均分：

- Transformer, 1 epoch: **42.83**
- T²MLR: **44.14**
- Transformer, 2.24 epochs / matched wall-clock: **45.30**

即 parameter/data/inference-compute matched 时 T²MLR 有 +1.31 pt，但把 baseline 训练到相同 wall-clock 后，vanilla Transformer 反而更高。

论文也明确报告 T²MLR 的 Jacobi-style training 约为 baseline 的 **2.24× wall-clock**，并把该方法定位为 inference-side / architectural gain，而不是 training-compute gain。

来源：
- https://arxiv.org/abs/2607.15178
- https://github.com/princeton-pli/T2MLR

这与本 repo 的成本分析一致：Jacobi approximation 解决了“完全 token-serial”这一最严重问题，却没有恢复 vanilla Transformer 的训练效率。按当前实现做的算子账本，Jacobi(16,4) 的理论训练 FLOPs 量级约为标准 LM 的数倍；详见 `06_reports/t2mlr_jacobi_training_cost.md`。

因此：

[
oxed{
	ext{T²MLR improves Learnability}
;	ext{but does not solve training Compute Efficiency}
}
]

---

## 4. RLT：BPTT 使其他 architecture 维度难以在大规模上被公平探测

RLT 采用 exact token-by-token recurrent decoder，并通过 BPTT 训练。其理论 attraction 是随 sequence position 增长的 temporal computation path，但这条 dependency 同时破坏了 token-level parallelism。

Empero 的 controlled evaluation 是目前最直接的实证：

- ~50M：RLT validation loss 3.8845，parameter-matched Transformer 3.8841，基本持平；
- ~140M：RLT 3.6884，RLT-A0 3.6664，Transformer 3.6420，recurrence 反而更差；
- main-scale throughput：RLT ~8.4K tokens/s/device vs Transformer ~124K；
- 500M tokens：RLT ~22 GPU-hours vs Transformer ~1.14 GPU-hours；
- feedback effect 没有随 document position 增强；
- downstream / context extrapolation 未体现稳定优势。

来源：
- https://www.alphaxiv.org/abs/2609.recurrence-looped-transformer-evaluation
- https://github.com/empero-org/rlt-evaluation

需要强调：这些结论只覆盖约 50–140M、500M tokens 的小规模 regime，不能证明更大模型上的 RLT 一定无效。

但这里存在一个研究上的死结：

[
	ext{需要大规模训练才能检验 architecture}
]

同时

[
	ext{exact recurrence + BPTT 又让大规模训练异常昂贵}
]

所以 RLT 当前最强的 negative evidence 并不是“Expressivity 已被证伪”，而是：

[
oxed{
	ext{the architecture is too expensive to scale enough to cleanly test its own stronger claims}
}
]

这使 Expressivity、Learnability、Usefulness 在更大 scale 上都处于 **under-tested** 状态。

---

## 5. Compute efficiency 是 sequence-axis latent recurrence 的前置问题

原先可以把 architecture evaluation 拆成：

[
	ext{Expressivity}
ightarrow
	ext{Learnability}
ightarrow
	ext{Usefulness}
ightarrow
	ext{Compute efficiency}
]

当前证据更像是在说明：对 sequence-axis recurrence，最后一项其实是前置 gate。

[
oxed{
	ext{Compute efficiency}
ightarrow
{	ext{scalable tests of Expressivity, Learnability, Usefulness}}
}
]

原因很直接：

1. **RLT / exact BPTT**：保留最忠实的 recurrence semantics，但训练吞吐过低，无法合理 scale。
2. **T²MLR / Jacobi approximation**：用近似 temporal parallelism 提高 Learnability 和可训练性，但 measured wall-clock 仍约 2.24× vanilla Transformer，且公开 checkpoint 的 behavior 更偏 local shortcut。
3. 若为了效率进一步缩短 recurrence / gradient horizon，又可能改变原本想验证的 long-horizon mechanism。

所以真正待解决的问题是：

> **能否设计一种 sequence-axis latent recurrence training method，在保留有意义的 temporal state semantics / credit assignment 的同时，将训练效率拉回接近 vanilla Transformer parallel training？**

在这个问题解决前，继续使用 full BPTT 做大规模 causal grid 的 information gain / compute 比过低。

---

## 6. Project decision

### Stop

**不再执行 BPTT / long-BPTT causal training experiments。**

E-T2-002 Phase B 标记为 stopped / not pursued。原因不是 hypothesis 已被证伪，而是当前训练方案的 expected information gain per GPU-hour 太低：

- exact sequential BPTT 是结构性 serial bottleneck；
- Jacobi approximation 已经是更高效的替代，但仍显著落后于 vanilla parallel training；
- 为区分 Jacobi horizon 与 exact BPTT dynamics 所需的 multi-arm × multi-seed from-scratch training 成本，与当前项目目标不匹配。

### Keep

保留已经完成的 frozen-checkpoint evidence：

- 六 checkpoint NLL matrix；
- carry / reset / clean-KV interventions；
- state / KL impulse response；
- BF16 / FP32 precision audit。

这些足以支持当前较弱但更稳健的机制结论：**T²MLR recurrence is learned and locally useful; long-horizon latent reasoning is not established.**

### Pivot

后续只有在出现新的 **compute-efficient sequence-axis recurrence training method** 时，才重新打开 architecture-scale experiments。

优先问题变为：

[
oxed{
	ext{Can sequence-axis latent recurrence be trained nearly as efficiently as a parallel Transformer?}
}
]

如果不能，sequence-axis latent recurrence 即使在理论上提高 expressivity，也很难成为可规模化的 Transformer backbone。

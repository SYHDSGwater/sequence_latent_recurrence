# Jacobi hypothesis：H100 训练卡时估算

依据 2026-09-18 拉取的仓库提交 `06b63d7`，实验为 E-T2-002 / H-T2-JACOBI-CONTRACTION。这里只估算训练，没有启动训练或新增 H100 实例。

**当前不能给出一个经实测支持的固定卡时。** 最新方案尚未固定模型结构、循环层比例、每个 arm 的训练 token 预算和实现吞吐。建议先划出 **2–5 H100 卡时做语义检查与吞吐 pilot**；若采用约 60M non-embedding backbone、context 128、A/B/D/E × 3 seeds，每个 run 20M tokens 的探索规模，暂按 **7–37 H100 卡时**规划；每个 run 100M tokens 则约 **34–183 卡时**。这些是明确吞吐假设下的预算情景，不是置信区间或 H100 实测。

## 从最新方案读出的工作量

Phase A 是公开权重推理测量，不需训练。补齐四个缺失 checkpoint 的 NLL 对照与 Phase B 因果训练要分开计费、分开解释。Phase A 全部主终点还包括 small perturbation、state/output impulse AUC、recovery time，不能用 NLL 补齐宣称所有主终点完成。

Phase B 的最小核心是 A/B/D/E 四条件、各 3 seeds，共 12 个训练 run：

| Arm | 前向 | 反向 | 主要用途 |
|---|---|---|---|
| A | Jacobi 16 | depth 4 | 参考配方 |
| B | Jacobi 16 | depth 16 | 增加 recurrent credit |
| D | exact sequential | TBPTT 4 | 固定短跨度时比较前向近似 |
| E | exact sequential | TBPTT 16 或 32 | 增加 sequential credit |

E 的 16 与 32 是不同实验设置；本估算将 E 当作其中一个设置，若两者都跑需增加 3 个 run。完整 A–F 六条件 × 3 seeds 为 18 个 run。full BPTT、第二 context、第二 model size 均额外计费，不包含在首轮核心矩阵中。

## 为什么不能直接用 6NT 或把成本乘 16

检查固定 upstream commit `2871fa204e4cc145fd67d7d1a82b319c2010e9dc` 的 `src/t2mlr_wrapper/t2mlr_wrapper.py:1091` 起的 `batch_approximate_forward`：先运行一个完整 dummy forward，缓存循环区间输入，然后运行 `D−1` 次循环区间 forward，最后再运行一个完整 forward；早期 recurrent cache 按 backward depth detach。重复计算主要落在循环层区间，backward depth 影响保留的图和反向计算。

粗略算子账本：令完整 forward 的 FLOPs 为 F，循环区间占比为 r，反向约为对应 forward 的 2 倍，则

`C_Jacobi/token ≈ F × [4 + (D−1+2B)r]`

相对普通 forward+backward 的 `3F`，倍率约为 `[4+(D−1+2B)r]/3`。这是量级估算，省略 attention、gate、初始低层图共享和具体 autograd 剪枝差异；activation checkpointing 的重计算可能进一步增加成本。

示例采用 60M 非 embedding 参数、hidden 512、49,152 词表（输出投影约 25M 个参数参与计算，虽与 embedding tied 也有计算），有效矩阵乘参数约 85M；循环区间占非 embedding 计算的一半，因此 r≈30/85：

| 配方 | 约为普通 LM 训练 FLOPs 的倍数 | 每 token 近似 FLOPs |
|---|---:|---:|
| A: (16,4) | 4.04× | 2.06 GFLOPs |
| B: (16,16) | 6.86× | 3.50 GFLOPs |
| C: (32,16) | 8.75× | 4.46 GFLOPs |

H100 SXM 的官方 BF16 表列 1,979 TFLOPS 带 sparsity，dense 对应约 989 TFLOPS。用峰值除 FLOPs 得到的只是极理想硬件时间；上述 A 每 100M tokens 峰值账本约 0.058 小时，不能据此声称实际几分钟完成。小模型、循环、kernel launch、存储图与优化器开销会让实际耗时远大于峰值账本。[NVIDIA H100 官方规格](https://www.nvidia.com/en-us/data-center/h100/)

Exact sequential 每个 token 只执行一次实际状态转移，但时间维无法像 Jacobi 那样直接并行。其 GPU 利用率、每次 kernel 的形状和 cache/autograd 实现可能比算术 FLOPs 更重要。已有 RTX 5090 结果是无梯度的 BF16 eager 推理，不能按 H100/5090 峰值算力比折算训练。

## 可复算的预算情景

以下吞吐是**人为给定的规划范围，尚无该训练实现的 H100 benchmark 支撑**。其用途是说明 exact sequential 吞吐若落在不同档位，预算如何变化；不保证实际吞吐落在该范围。输入 tokens 指有效训练 tokens，不能把 Jacobi 重复迭代算成更多数据 tokens。

| Arm | 假定 H100 有效训练 tokens/s | 单 run 100M tokens 纯训练卡时 |
|---|---:|---:|
| A | 15,000–60,000 | 0.46–1.85 |
| B | 8,000–30,000 | 0.93–3.47 |
| C | 5,000–20,000 | 1.39–5.56 |
| D | 2,000–10,000 | 2.78–13.89 |
| E | 1,000–6,000 | 4.63–27.78 |
| F | 50,000–200,000 | 0.14–0.56 |

计算式为 `总卡时 = Σ_arm,seeds tokens_per_run / measured_or_assumed_tokens_per_second / 3600`。以下加 30% 运行余量，覆盖普通 validation、保存与短诊断；不包括大规模多方向 persistence 评估、首次环境搭建、工程开发、失败重跑和下一阶段确认。

| 每个 arm、每个 seed 的 tokens | A/B/D/E × 3 seeds | A–F × 3 seeds |
|---|---:|---:|
| 20M | 7–37 卡时 | 8–42 卡时 |
| 100M | 34–183 卡时 | 40–207 卡时 |
| 500M | 172–916 卡时 | 201–1,035 卡时 |
| 1B | 343–1,833 卡时 | 403–2,071 卡时 |

卡时是所有卡的使用时间之和。三个 seeds 在三张 H100 上并行可缩短等待时间，但不会把总卡时除以三；多卡同步训练还需要考虑通信效率。

30M 与 100M non-embedding 模型应重新计入各自 hidden/vocab/循环区间计算与实测吞吐，不能机械按 0.5×/1.67×缩放本表。context 从 128 到 256 也没有固定 2×换算：固定训练 tokens 时 token 数不翻倍，但 attention、显存、微批大小和 sequential 图开销会改变。

## 应先验证的训练语义

1. inference adapter 不能直接当训练 adapter 使用。官方 recurrent post-normalization 的 scale 在某些训练路径使用 detach；现有 adapter 仅验证了无梯度 forward parity，没有验证梯度 parity。
2. TBPTT 4 的截断对象必须明确：仅 recurrent state，还是 state 和所有过去 KV 的梯度。只 detach state，KV 仍可能保留跨很多 tokens 的反向路径；一并 detach KV 又额外改变 attention credit。A/D 的 forward 对照因此不自动是严格单因素比较。
3. 对同一固定精确前向轨迹比较不同 stop-gradient 策略，检查按 lag 的梯度范数；同时明确 optimizer step 发生在完整序列/全局 batch 边界，而非每个 TBPTT chunk 后更新参数。否则短/长条件的数据更新次数不同。
4. 长 BPTT 和 Jacobi backward=16 可能需要 activation checkpointing，计入重计算；若内存不足，报告实际 horizon，不能把更短设置命名为 long/full。
5. forward 近似的差异需与数值实现差异分开。所有 arm 必须共享 gate、fusion、norm、初始化、数据顺序、全局 batch 与学习率；需要训练语义测试后再比较 persistence。
6. Jacobi backward depth=4 不天然等于每四 tokens 切一次图的 TBPTT=4。后者块内不同位置的实际可回传长度不同，rolling-window 与 disjoint-chunk 实现的重计算成本也不同。应明确采用哪种截断策略，记录 recurrent/KV 两条路径的 lag-gradient 曲线，避免把两种“4”直接当成匹配的 credit horizon。

## 建议的预算 gate

先用 **2–5 H100 卡时**完成已经实现好的训练器预检与四个核心 arm 的 pilot：每 arm 1 seed、约 2M tokens，纯训练情景约 0.18–0.94 卡时，其余留给编译预热、梯度审计、显存/吞吐测量和短 persistence 诊断。这个额度不包含开发人员实现训练器的时间，也不保证模型已学到可讨论的 recurrence。

每个 arm 预热后测至少 100 个稳定 step，记录包含 optimizer step 的有效 tokens/s、峰值显存，以及 checkpointing、微批与梯度累积配置。用实测值替换本表后再批准后续固定总预算。若 exact E 的吞吐不足 1,000 tokens/s，当前范围就需要向上修订。

20M tokens/run 适合检查机制方向和训练是否有效，但不能保证支持强结论：从头训练可能尚未形成可用状态；不同 horizon 可能处于不同优化阶段。应预先要求正常 validation 改善、反馈路径非退化、同 checkpoint forward/gradient 检查通过。无效或未学会的模型上出现 null 不构成对 Jacobi hypothesis 的有力否证。100M/500M 也没有“自动足够”的统计保证。

首轮优先 A/B/D/E，F 作为训练匹配的无反馈控制值得补充；C、第二 context 和 full BPTT 应依据第一轮结果与实测预算安排。若只有原先 5 H100 卡时预算，应完成 pilot 和训练语义诊断，不承诺完成 12-run 因果结论。

计算文件：`experiments/t2mlr/estimate_training_cost.py`；机器可读情景：`04_evidence/t2mlr_jacobi_cost_scenarios.json`。原始 E-T2-002 中 `actual_gpu_hours` 仍为空，因为本轮没有执行训练。

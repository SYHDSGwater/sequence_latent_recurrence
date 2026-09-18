# T²MLR 状态机制审计

## Research question

公开 982M/50B checkpoint 的 recurrent state 是否携带内容特异、跨多个 token 持续、对预测有益的信息？观测到的长尾依赖中，多少在阻断受扰 KV 历史后仍然存在？

## Why this matters

将“作者报告分数较高”“这个权重依赖反馈通道”“该通道实现持续推理”分开检验。结果不能外推为 1.7B retrofit 成功复现，也不能证明优于参数/训练匹配 Transformer 或 RLT。

## Current belief / Target uncertainty

先验未定。竞争解释包括短期局部特征复用、持续状态累积、扰动经 KV 传播、训练后的路径依赖。先检验可识别性，不以大幅 zero ablation 退化直接认定 latent reasoning。

## Known evidence

公开权重配置和当前实现可检查。1.7B retrofit 权重、逐题输出/评分 trace 未找到。完整权重加载及 H100 推理尚未执行，无机制实验结果。

## Constraints

- 沿用会话中剩余单张 H100 共 5 小时上限；当前仅构建与 CPU 校验。
- 精确逐 token 推理，禁用 batch approximate forward；同一文本、token 和位置配对。
- 使用独立文档，不混接、不用训练数据冒充 held-out；记录来源、划分、许可、文件哈希。
- 预注册主终点；按文档 bootstrap，不把相关 token 当独立样本。
- 公开代码和 checkpoint 固定 revision。适配器仅覆盖本 checkpoint 的 gated/no-skip Llama 路径。

## Stop conditions / Human-only decisions

加载键不匹配、语义对照失败、非有限输出则停止。预算不足时缩减样本数并报告 CI，不改阈值追逐显著性。扩大预算、训练新模型、改为架构优劣比较另行决定。

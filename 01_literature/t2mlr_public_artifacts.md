# T²MLR 公开材料核查（2026-09-18）

结论：在本次核查范围内，未找到 **1.7B retrofit checkpoint** 或与其数学 benchmark 对应的 **逐题 prompt / generation / answer extraction / score traces**。这不等于证明作者没有在任何其他位置发布。

核查入口：

- [论文](https://arxiv.org/abs/2607.15178)：报告在既有 1.7B 模型上 retrofit 后微调的实验；论文汇总分数不能替代逐题 trace。
- [作者模型列表](https://huggingface.co/JupiterZhu) 与 [Hub 搜索 API](https://huggingface.co/api/models?search=T2MLR&limit=100)：返回六个 T2MLR 模型，135M / 362M / 982M，各含 10B / 50B FineWebEdu 版本；未见 1.7B retrofit。
- 作者 dataset API 未返回 dataset；不能据此排除其他账号的发布。
- [官方仓库](https://github.com/princeton-pli/T2MLR)：有实现、训练配置与评估工具，但未找到上述逐题实验产物；README 明确说精简发布不包含内部实验日志等。
- [Releases](https://github.com/princeton-pli/T2MLR/releases)：页面未见 release。保存 API 快照时触发 GitHub 403 rate limit，已将错误如实记录，未当成空列表。
- 检索式包括 `"T2MLR" "1.7B" checkpoint`、`"T2MLR" "retrofit" weights`、`"T2MLR" "traces"`。搜索命中的论文摘要、解读和训练 CoT traces，不视为 benchmark rollout 公开证据。

机器可读核查记录见 `t2mlr_public_artifacts_2026-09-18.json`。本地上游代码固定在 `2871fa204e4cc145fd67d7d1a82b319c2010e9dc`。

## 982M checkpoint 的实际配置

[权重](https://huggingface.co/JupiterZhu/T2MLR_982M_lstart9_lend24_50B_FineWebEdu) 固定 revision `5061964f36236336153106d4b11155e09ee2daaa`。

- Llama 基座，32 层，hidden=1536；16 attention heads、8 KV heads、head_dim=128，不能用 hidden/heads 猜 head_dim。
- `l_start=8`, `l_end=-9`：零基索引 8→23，即第 9 层前注入、第 24 层后取状态。
- gated fusion；当前 token 表示与投影后的上一步状态共同决定 gate。
- `recurrent_residual_to_recurrent_cache=true`, weight=1：新的中层输出加上旧状态，再做带 clamp 的 RMS 缩放，eps=1e-6、clamp=5。
- 无 `recurrent_skip_to_l_end`；本实验走精确顺序推理，不用 Jacobi batch approximation。

以 x_t 表示第 9 层输入，h_t 表示第 24 层输出，可概括为：

```
x'_t = Fuse(x_t, s_(t-1))
s_t  = ClampRMS(h_t + s_(t-1))
```

因此 `carry_off` 应只移除第二式的加法旧状态，保留缩放。`zero_every` 保留融合模块但每步给零状态；`fusion_off` 则绕过整个模块，包括 current-token 分支。三者不能混称为“关掉 recurrence”。

上游 wrapper 导入 `components.t2mlr_utils`，当前 checkout 缺此模块。适配器直接加载官方 gate 和 BlockWrapper；测试通过 AST 提取未修改的官方 `simple_recurrent_forward` 作为对照。此举解决依赖入口问题，不替代真实权重与 GPU 精度验证。

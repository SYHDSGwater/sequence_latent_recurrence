# T²MLR 1.7B retrofit 训练核查

核查日期：2026-09-18。论文 v2：<https://arxiv.org/html/2607.15178v2>。官方仓库当前 HEAD 经远程查询仍为 `2871fa204e4cc145fd67d7d1a82b319c2010e9dc`。

## 可确认信息

第 4.4 节：基座为 SmolLM2-1.7B-Instruct，32 层；插入 T²MLR (5,28)，即第 5–28 层、24 层的递归段；在 OpenMathReasoning 微调 1 epoch，对照为相同基座不加递归通路的微调。

第 2.4 节通用默认：Jacobi forward depth=16，backward depth=4，除非另行说明。附录 F.3 对 finetuning 通用描述：AdamW，beta1=0.9，beta2=0.999，weight decay=0.01；并写 training 使用 4×H100，但未提供 1.7B 这次运行的单独硬件/作业记录。

未查到该次 retrofit 的专属学习率、global batch、最大序列长度、数据 split/revision/子集规模、token 数、实际 steps、冻结范围/LoRA 设置、prompt loss masking、gate LR、实测 wall time 或 GPU-hours。F.3 表格列出的是 135M 预训练、GSM8K-Aug 和 HotpotQA 等设置，不是完整 1.7B 配方。

## 不能当作本次实验配置的公开代码

`scripts/gsm8k/train_gsm8k.sh` 默认 Llama-3.2-1B-Instruct + whynlp/gsm8k-aug；`sweep_params.yaml` 示例使用 SmolLM2-135M。这些不是论文所述 SmolLM2-1.7B + OpenMathReasoning 运行。

此通用脚本默认 LR=5e-4、per-device batch=64、累计步数=4/GPU数、max_length=4096、Jacobi forward=16/backward=8、gate LR multiplier=100、BF16。不能直接移植为论文的真实运行参数；其中 backward depth 与论文通用默认也不同。

Slurm 请求 2 GPU，`--time=5:59:00` 是调度时限，不是已测耗时；与论文通用 4×H100 声明也不能拼接成一个运行记录。

`scripts/download_assets.py` 有 OpenMathReasoning `cot[:1%]`，但明确是轻量缓存预热脚本，不证明训练使用 1%。该文件缓存的 1.7B 模型还是 Qwen3-1.7B，而不是该 retrofit 基座。

## 耗时结论

截至核查，没有足够公开证据给出该次 retrofit 的实际小时数。原数据集有约 3.2M CoT solutions（<https://huggingface.co/datasets/nvidia/OpenMathReasoning>），但不能据此断定作者训练了全量 CoT，也不能用缓存预热的 1% 代替训练规模。没有有效 token 数和完整配置，无法可靠外推到 5090D。

如后续自行复现，应先明确自己的子集、截断/packing 和训练参数，再实测含 forward/backward/optimizer 的吞吐；它将是独立复现方案，不是已还原作者运行。

# T²MLR recurrent 机制实验

## 六 checkpoint 矩阵与 Jacobi 训练估算

补齐的四个模型由 `matrix_prepare.py` 按 `01_literature/t2mlr_six_checkpoint_audit.json` 的固定 revision/LFS SHA256 下载，并与原 tokenizer 后端严格比对。`remote_matrix.sh` 复用冻结样本执行新增八个 cell，`remote_matrix_prefetch.sh` 可在推理期间预取后续权重（同模型文件锁防止重复写入）。原始结果归档为 `artifacts/matrix-results.tar.gz`。

解包扩展结果与矩阵结果后运行：

```sh
python experiments/t2mlr/matrix_analyze.py
python experiments/t2mlr/matrix_report.py
python experiments/t2mlr/matrix_plot.py
python experiments/t2mlr/estimate_training_cost.py
```

矩阵只补齐同配方 NLL 测量，E-T2-002 的 small-perturbation、state/KL AUC、恢复半衰期仍须单独测量。训练估算见 `06_reports/t2mlr_jacobi_training_cost.md`：吞吐是未实测的预算情景，不是 H100 benchmark；本轮没有执行训练。

## 扩大实验：两个 checkpoint × 两类数据

2026-09-18 的扩展采用固定的 256 篇 WikiText 文章和 256 道 MATH-500 题目，对 135M/10B 与 982M/50B 做同样的五条件配对测量。预注册及批量吞吐修订见 `03_experiments/t2mlr_expanded_preregistration.md`。MATH 测参考解答的预测损失，不是独立解题准确率；完整远期窗口实际有 169 题。

```sh
python experiments/t2mlr/expanded_prepare.py
python -m pytest experiments/t2mlr/test_mechanism.py experiments/t2mlr/test_expanded_metrics.py -q
python experiments/t2mlr/expanded_run.py --model 135m --data data/expanded/math.jsonl --output artifacts/expanded/135m_math --batch-size 64 --max-seconds 5400
```

对 `135m`、`982m` 与 `wiki`、`math` 的四个组合分别运行。输出 `batch_*.npz` 保存五条件逐 token NLL，配套 JSON 保存 token、长度和干预位置；padding 不计分，单次干预窗口只纳入长度完整的样本。原始结果下载后，用 `expanded_analyze.py --root artifacts/expanded --output 04_evidence/t2mlr_expanded_results.json` 独立重算并检查完成状态、配对 ID、哈希和无扰动前缀。`expanded_download.py` 是按上游 LFS SHA256 校验的分段下载后备方案。

下文保留原始 982M 单模型实验的说明和命令。

目标是检查这个冻结 checkpoint 如何使用状态，不是复现 1.7B 数学分数。假说与判据分别见 `02_hypotheses/t2mlr_mechanisms.yaml` 和 `03_experiments/t2mlr_interventions.yaml`。

2026-09-18 用户已指定 RTX 5090D 实际执行；本次运行配置以 `03_experiments/t2mlr_5090_execution.md` 为准，固定数据见 `04_evidence/t2mlr_5090_data_provenance.json`。下面的 H100/64 文档命令保留为初始方案，不代表本次硬件或样本数。

## 设置

在仓库根目录操作。安装适合硬件的 PyTorch（CPU 验证使用 2.6.0；H100 需要 CUDA 构建），再安装本目录 requirements。代码要求 Transformers 4.56.2。

```sh
git clone https://github.com/princeton-pli/T2MLR external/T2MLR
git -C external/T2MLR checkout 2871fa204e4cc145fd67d7d1a82b319c2010e9dc
python -m pip install -r experiments/t2mlr/requirements.txt
python -m pytest experiments/t2mlr/test_mechanism.py -q
python experiments/t2mlr/audit_shapes.py
```

已存在的 external checkout 不必重复克隆。要求 checkout 干净，防止混入未知代码。代码不需要执行 Hub 上的 remote custom code。

## 输入与预注册

使用 UTF-8 JSONL，每行 `{"id":"unique-document-id","text":"..."}`。每行是一篇独立文档，至少 513 个 tokenizer tokens；不得把同一文章拆成多行作为独立样本。建议预先抽取 64 篇英文 held-out 文章，保留第二批至少 32 篇作确认集。需记录数据集名称、revision、split、抽样规则、许可和潜在预训练重叠；WikiText 等公开文本不能保证训练无污染。

本次已通过 `prepare_data.py` 固定 WikiText-103-raw-v1 validation/test 文章，来源与哈希见数据记录；短文被排除，按文件顺序取足够长的文档，文件哈希和入选 ID 写入 manifest。tokenizer 不自动加 BOS，不跨文档拼接；文档开始时 state 与 KV 均清零。此设置适合配对机制诊断，不能拿绝对 PPL 直接比较论文 benchmark。

512 是观测窗口长度；输出位置 p 的 NLL 对应 token p+1。`--reset-at 128` 表示消费 token 128 前重置 s_127；lag 0 为该步 next-token NLL 差。

## 5 H100 小时内的执行顺序

先用 4 篇文档跑预检，测实际吞吐和加载情况；小样本结果不用于机制结论：

```sh
python experiments/t2mlr/run.py --data heldout.jsonl --output artifacts/t2mlr/pilot --documents 4 --batch-size 4 --arms normal reset_once --max-seconds 900
```

接着按吞吐决定能否完成 64 文档。以下是三个阶段的启动命令，每阶段均重新生成正常基线，不共享可变 cache：

```sh
python experiments/t2mlr/run.py --data heldout.jsonl --output artifacts/t2mlr/reset --arms normal reset_once donor_once reset8 reset32 reset128 --max-seconds 5400
python experiments/t2mlr/run.py --data heldout.jsonl --output artifacts/t2mlr/carry --arms normal zero_every carry_off fusion_off fusion_and_carry_off --max-seconds 3600
python experiments/t2mlr/run.py --data heldout.jsonl --output artifacts/t2mlr/path --arms normal reset_once reset_once_clean_kv --max-seconds 3600
```

这些命令的 caps 共 3.75h，余 1.25h 用于加载/环境开销和第二批文档在 reset-at=256 的确认；预算分配以实验 YAML 的 5h 总上限为准。**这些是时间上限，不是预计耗时**。逐 token eager 推理及 clean-KV deepcopy 可能较慢。先减低优先级 arms，再减少样本，不把超时当负结果。

`--max-seconds` 只限制单次命令，循环在 token 边界检查，网络下载/加载和正在执行的算子不能被其硬中断；因此必须通过外部运行时间管理总 5h 预算，GPU 租用计时也包括环境准备。输出目录不得重用。权重首次约 2GB 下载，后续复用 Hub 缓存；严格校验 state_dict，检查声明 tied 的 embedding/head 是否一致。

## 各条件解释

| 条件 | 操作 | 能回答的问题 |
|---|---|---|
| reset_once | 同一时刻仅状态置零，KV 保留 | 失去状态后的预测损失和恢复 |
| donor_once | 另一文档同位置正常状态，匹配 recipient norm | 内容特异性；仍有分布偏移 |
| reset8/32/128 | 周期清零状态 | 持续时间敏感性，不能直接视为有效记忆长度 |
| zero_every | 每步状态清零，融合模块继续工作 | 状态依赖；保留零输入时的 current 分支 |
| carry_off | s_t=ClampRMS(h_t) | 加法积累是否有用，区别于反馈读取 |
| fusion_off | 绕过整个融合模块 | 仅作路径诊断，非参数匹配 baseline |
| fusion_and_carry_off | 同时关闭融合与累积 | 应与 fusion_off 相同，作为负对照 |
| reset_once_clean_kv | 扰动状态持续演化，每步 KV 使用独立正常前缀 | 阻断过去受扰 KV 的中介传播 |

clean-KV 不固定本 token 当前计算，也不切断 recurrent state 传播；它是人工反事实干预。两条通路存在交互，不能用自然效应减 clean-KV 效应推导“KV 占贡献 X%”。

## 输出与解释

- `manifest.json`：固定 revision、数据哈希、参数、设备、版本、完成批次和状态。
- `token_traces.jsonl`：每文档、每 token、每 arm 的真实 target 与 NLL。
- `summary.json`：相对正常状态的文档配对 delta NLL，2000 次 bootstrap 95% CI。普通条件按文档重采样；donor 因文档互相充当供体，按整个 donor batch 重采样，保留组内依赖。

只提交完整配对批次；超时可能留下较小的完整子集，必须同时查看 manifest。少于 32 文档标记 `inference_valid=false`；达到 32 也不保证统计功效。置信区间默认未做多重比较校正，只有预注册主终点可作确认性判断。

主终点为 reset 后 lag 32–127 的平均 delta NLL（自然与 clean-KV 分别报告）；±0.01 nat/token 为预先约定实用等效范围。CI 完全落入才判定接近零，CI 太宽则结论未定。次要的近端、donor、周期清零和 carry 结果用于解释，不择优选窗口。

高 surprise token 退化不能证明推理，消融大幅退化不能证明比 Transformer 优越。若要进一步讨论“持续推理”，下一阶段需构造可验证多步任务、比较因果正确/错误 donor、并配置训练匹配对照；本轮不把未经训练的 982M base model 数学零样本表现当作此结论。

## 验证边界

CPU tiny-model 测试检查官方逐步函数一致性、扰动生效且不改变前缀、清空重放一致、clean-KV/donor 流程、fusion/carry 负对照、超时和统计单位。完整模型元数据另行比对。它们不是完整权重加载成功或 H100 BF16 数值一致性的证明；开始 GPU 正式评估前需先跑预检并检查这些条件。

本次 5090D 已额外完成 `gpu_validate.py`：完整权重严格加载，tied tensors 相等，2×8 tokens 上 BF16 logits/state 与未修改官方单步函数最大绝对差均为 0。该短序列检查仍不等于所有输入的完整证明。

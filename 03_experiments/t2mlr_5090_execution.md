# RTX 5090D 执行预注册（2026-09-18）

用户指定 `pro-78730289ac36`，1×RTX 5090D 32GB，替代原 H100 设想。开始连接时间约 13:50 CST；总运行沿用最多 5 小时，先用预检吞吐决定规模。

在观察 checkpoint 结果前固定：主实验使用 WikiText-103-raw-v1 validation 的前 32 篇长度至少 513 tokens 的完整文章；确认使用 test 的前 32 篇，reset 位置由 128 改为 256。原设想 64 篇主实验缩减为 32 篇，原因是 validation 共 60 篇、不同硬件预算与保留独立确认。所有条件使用同一组文档。

数据固定 revision `b08601e04326c79dfdd32d625aee71d232d685c3`。按一级标题拼回完整文章，不把段落冒充独立文章；不跨文章拼接。数据许可 CC-BY-SA-3.0/GFDL，可能与预训练重叠，结论仅关于权重的状态依赖。

预检：4 文档 normal/reset_once；正式首选 batch=4，若显存允许可调 batch；donor 的 CI 按供体 batch 重采样，不能当独立文档。

主终点、实用等效界 ±0.01 nat/token、次要终点及解释限制保持原实验 YAML 不变。只有 32 篇文章时检验力有限；不能将宽 CI 称为无效。主实验计划覆盖所有已实现干预，确认优先 normal/reset_once/donor_once/carry_off/clean-KV。

远程工作目录 `/root/autodl-fs/t2mlr-20260918`；依赖覆盖目录 `/root/autodl-tmp/t2mlr-runtime/deps`，复用现有 PyTorch 2.8.0+cu128（支持 sm_120）。不修改已有环境包；模型与数据通过可访问的 HF 镜像按固定 revision 下载。

实例原本处于 running；本任务未开机、重装或删除任何已有文件。实验完成停止本任务进程并回收结果，实例是否继续开机保持用户原状态。

## 主实验后的探索性补测

主实验显示平均 NLL 恢复快后，增加 `diagnostics.py`：在相同 32 篇 validation 文档上测 KL(normal || intervention)、归一化 state L2 距离、state cosine distance，以及绝对 NLL 差。条件为单次 reset、donor、clean-KV reset；位置仍为 128。目的为检查有符号 NLL 抵消和 hidden-state 持续偏移，不改变主终点、阈值或确认计划。此补测为看过主实验后的探索，不能包装成预注册确认性结果。

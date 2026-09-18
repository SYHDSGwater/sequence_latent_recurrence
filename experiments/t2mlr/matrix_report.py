"""Render the verified six-checkpoint tables without manual numeric transcription."""
import json
from pathlib import Path

def main():
    data=json.loads(Path('04_evidence/t2mlr_six_checkpoint_results.json').read_text())
    def fmt(v):return f"{v['mean']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]"
    lines=['# T²MLR 六 checkpoint 配对测量','',
       '补齐 135M/50B、362M/10B、362M/50B、982M/10B，并复用已完成的 135M/10B 与 982M/50B。每个 checkpoint 都使用同样的 256 篇 WikiText 文章和 256 道 MATH-500 参考解答。新增八个 cell 全部完成；下表为从原始逐 token NLL 独立重算的结果。', '',
       '数据、干预和解释范围沿用 `t2mlr_expanded_results.md`。普通文本 reset 位置为 128；数学 reset 位于题目到解答的边界。持续消融从输入开头执行，数学仅对参考解答计分。每个独立输入先求均值，再进行 2,000 次样本级 bootstrap。括号为点态 95% CI，未作多重比较校正。', '',
       '## 持续干预与单次重置','',
       '单位 nat/token，正值表示干预增加损失。carry/zero_every 均 n=256；远期窗口普通文本 n=256，数学 n=169。', '',
       '| checkpoint / 数据 | carry_off ΔNLL，95% CI | zero_every ΔNLL | reset lag 32–127 ΔNLL，95% CI |',
       '|---|---:|---:|---:|']
    for size in ['135m','362m','982m']:
        for tokens in ['10b','50b']:
            for domain in ['wiki','math']:
                name=f'{size}_{tokens}_{domain}';c=data['cells'][name];m=c['metrics']
                lines.append(f"| {name} | {fmt(m['carry_off/scored']['signed'])} | {m['zero_every/scored']['signed']['mean']:+.5f} | {fmt(m['reset_once/lag_32_127']['signed'])} |")
    lines+=['','## 同规模 50B 减 10B 的配对效应差','',
       '同一输入上先计算两 checkpoint 的干预效应之差，再 bootstrap。负的 carry_off 差表示 50B 对被移除累积项的依赖较小，不等于 50B 模型预测更差。', '',
       '| 规模 / 数据 | carry_off 效应差，95% CI | reset 远期效应差，95% CI |',
       '|---|---:|---:|']
    for name,c in data['paired_50b_minus_10b'].items():
        lines.append(f"| {name} | {fmt(c['carry_off/scored']['signed'])} | {fmt(c['reset_once/lag_32_127']['signed'])} |")
    lines+=['','## 基线预测与非负远期差异','',
       '| checkpoint / 数据 | 正常平均 NLL | 自然 reset 远期绝对 ΔNLL | clean-KV 远期绝对 ΔNLL | clean-KV 远期 signed ΔNLL，95% CI |',
       '|---|---:|---:|---:|---:|']
    for name,c in sorted(data['cells'].items()):
        m=c['metrics'];lines.append(f"| {name} | {c['baseline_nll']['mean']:.5f} | {m['reset_once/lag_32_127']['absolute']['mean']:.5f} | {m['reset_once_clean_kv/lag_32_127']['absolute']['mean']:.5f} | {fmt(m['reset_once_clean_kv/lag_32_127']['signed'])} |")
    small=[];failed=[]
    for name,c in data['cells'].items():
        for arm in ['reset_once','reset_once_clean_kv']:
            lo,hi=c['metrics'][arm+'/lag_32_127']['signed']['ci95']
            (small if lo>-.01 and hi<.01 else failed).append(name+'/'+arm)
    lines+=['','## 判断与限定','',f'预定义远期 ±0.01 nat/token 判据：{len(small)}/24 个 checkpoint×数据×KV条件的 95% CI 完整位于区间内。未达到的条件：'+(', '.join(failed) if failed else '无')+'。', '',
       '**关键例外是 362M/50B 的数学自然重置**：远期平均 +0.01014 [0.00571,0.01459]，未达到预定义小效应判据；其 CI 同时跨过 +0.01，因此也不能声称已确认效应大于 +0.01。clean-KV 为 −0.000656 [−0.001217,−0.000127]，实用大小仍很小。旧结论必须收窄，不能写成所有公开 checkpoint 都有同样的远期预测恢复。', '',
       '该例外的事后探索检查：169 题中 70.4% 的自然远期平均损失增加，中位数 +0.00690；去掉最大一个样本后的均值 +0.00937，去掉最大五个后 +0.00696。它不只由一个样本产生，但超过 0.01 的均值对尾部样本敏感。自然减 clean-KV 的配对差 +0.01080 [0.00643,0.01523]。这些分析是观察例外后开展，不能取代预注册原始结果；也不能把路径差值解释为可加的 KV 因果份额。详见 `04_evidence/t2mlr_362m50b_math_exception.json`。', '',
       '所有 12 个 checkpoint×领域组合的 carry_off 95% CI 均为正。其随训练预算的变化并不统一：135M/Wiki 下降，135M/Math 差异不明确，362M 与 982M 的两个领域均上升。每步清零的预测代价则在三个规模的两个领域里都是 50B 更大。这支持更强的冻结路径依赖，但不直接证明更长时间的有用状态累积。', '',
       '六 checkpoint 的每个规模内配置除了 `batch_forward` 标志外相同，均记录 Jacobi forward depth=16、backward depth=4。本次统一使用 exact sequential inference，忽略训练/批量前向选择标志。配置值不能代替完整训练日志；没有证明 10B/50B 是同 seed 的连续训练轨迹。因此同规模对照仍是 checkpoint 相关性比较，不能直接归因为训练 token 数。', '',
       '本轮补齐的是 E-T2-002 Phase A 的五条件 NLL 对照。新方案的 norm-controlled small perturbation（2–4 方向）、state impulse-response AUC、output KL AUC 和 recovery half-life 尚未补齐；不能从 signed NLL 接近零推导相同 contraction rate 或相同恢复半衰期。旧实验的少量 KL/state 诊断也不能替代六 checkpoint 的完整主终点。', '',
       '数学使用 teacher forcing 和完整 KV，不能作为自由解题准确率；两领域的干预位置不同。绝对 NLL 差非零不等于语义信息或有益持续推理。当前 checkpoint 结果不能因果证实或否证 H-T2-JACOBI-CONTRACTION；训练识别需要最新方案中的匹配对照与梯度语义审计。', '',
       '## 验证与产物','',
       '四个新增权重采用固定 revision 与上游 LFS SHA256 校验；tokenizer 后端和 special-token 映射与基准完全一致。每个 cell 通过完整权重严格加载、2×8 tokens 的官方 BF16 exact-step parity、无扰动前缀相等检查；所有输入行与冻结样本逐项一致，独立重算与运行时汇总一致。', '',
       '- 最新方案基线：`06b63d7`。',
       '- 版本/配置审计：`01_literature/t2mlr_six_checkpoint_audit.json`。',
       '- 完整结果、配对差与复核：`04_evidence/t2mlr_six_checkpoint_results.json`。',
       '- 新增原始数据包：`artifacts/matrix-results.tar.gz`；另两个 checkpoint 见 `artifacts/expanded-results.tar.gz`。',
       '- 训练成本估算：`06_reports/t2mlr_jacobi_training_cost.md`。']
    seconds=sum(c['manifest']['elapsed_seconds'] for name,c in data['cells'].items() if not (name.startswith('135m_10b') or name.startswith('982m_50b')))
    lines+=['',f'新增八个正式 cell 累计 {seconds:.1f} 秒（{seconds/60:.1f} 分钟），不包括下载、预取和人工准备；不是总计费时长。没有执行 Jacobi 假说训练。','']
    Path('06_reports/t2mlr_six_checkpoint_results.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()

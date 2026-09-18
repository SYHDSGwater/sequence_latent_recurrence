"""Render primary and precision tables from verified analysis JSON."""
import json
from pathlib import Path

NAMES=['135m','135m_50b','362m_10b','362m_50b','982m_10b','982m']
LABELS={'135m':'135M/10B','135m_50b':'135M/50B','362m_10b':'362M/10B','362m_50b':'362M/50B','982m_10b':'982M/10B','982m':'982M/50B'}
def fmt(v):return f"{v['mean']:.5g} [{v['ci95'][0]:.5g}, {v['ci95'][1]:.5g}]"
def median(s):return '>123' if s['median_censored'] else str(int(s['median_recovery_lag']))

def main():
    j=json.loads(Path('04_evidence/t2mlr_impulse_results.json').read_text())
    p=json.loads(Path('04_evidence/t2mlr_impulse_precision_results.json').read_text())
    fpn=[c['summary']['small_mean_natural']['state_rel_l2_recovery']['0.1']['median_recovery_lag'] for n,c in p['cells'].items() if n.endswith('_fp32')]
    fpc=[c['summary']['small_mean_clean']['state_rel_l2_recovery']['0.1']['median_recovery_lag'] for n,c in p['cells'].items() if n.endswith('_fp32')]
    ratios=[]
    for model in NAMES:
        for domain in ['wiki','math']:
            b=p['cells'][f'{model}_{domain}_bf16']['summary']['small_mean_natural']['state_rel_l2_auc_1_64']['mean']
            f=p['cells'][f'{model}_{domain}_fp32']['summary']['small_mean_natural']['state_rel_l2_auc_1_64']['mean']
            ratios.append(b/f)
    assert all(v is not None for v in fpn+fpc), 'Revise interpretation if FP32 recovery remains censored'
    lines=['# T²MLR Phase A：状态扰动、恢复与精度对照','',
       '本轮完成六个 checkpoint × 两领域的非训练补充测量：零重置、两个固定随机方向的小扰动、自然与 clean-KV 轨迹、state/output impulse AUC、恢复阈值时间。沿用每领域 256 个冻结输入；旧 carry_off 结果见六 checkpoint NLL 报告。没有执行 Jacobi 或其他训练。', '',
       '## 结论及边界','',
       f'最需要修正的机制读法是：BF16 小扰动长尾不能直接当作长期记忆。本轮主测量的 12 个组合中，小扰动自然路径 t10 均没有样本达标；但固定 16 样本的同 batch FP32 对照中，自然路径 t10 中位数为 {min(fpn):g}–{max(fpn):g} 步，clean-KV 为 {min(fpc):g}–{max(fpc):g} 步，自然路径 state AUC 相比 BF16 缩小约 {min(ratios):.1f}–{max(ratios):.1f} 倍。这支持显著的数值精度敏感性；不证明残留的全部来源，也不把 FP32 子集当完整任务评测。', '',
       '零重置后，所有组合的状态 t50 中位数为 1 步；自然路径 t10 为 5–20 步，clean-KV 为 2–3 步。362M/50B 数学的自然 t10 为 20 步，clean 为 2 步，且状态/KL AUC 与远期 NLL 均显示自然路径更强。这个例外在同数据的新测量中保留，但小扰动没有对应的远期平均损失，因此不能概括为普遍的局部慢混合或持续推理。', '',
       '10B→50B 没有统一的零重置 state AUC 趋势：135M 与 982M 两领域下降，362M 文本不明确、数学上升。共有 Jacobi(16,4) 配置并不足以推出统一恢复尺度；checkpoint 比较不能确定训练 horizon 的因果作用。当前支持短程恢复及 KV 路径相关传播的有限机制描述，不支持“Jacobi 导致短记忆”或“状态长尾就是有益推理”。', '',
       '两个随机方向不覆盖所有状态子空间，也不是对语义变量的定向干预。迅速恢复不能排除少数持久方向、状态内容被 attention 重建或其他输入分布下的行为；本轮没有测自由解题准确率，也不能外推到未公开的 1.7B retrofit。公开语料还不能保证无预训练重叠。所有区间均为未做多重比较校正的点态区间。', '',
       '## 固定配方','',
       '小扰动在 incoming state 的随机切向方向上加入约 5% 范数的位移，再恢复原范数并转换至执行精度。两个独立方向由文档 ID 与方向序号确定，在样本内平均，不计作两份独立数据。零重置与小扰动都分别保留自然 KV 或每步恢复正常前缀 KV。数学保持完整题目与参考解答 teacher forcing，干预位于解答起点；这不是自由解题评测。', '',
       'state response 为扰动/正常 post-transition state 的相对 L2；输出 response 为 KL(normal || perturbed)，用 FP32 log-softmax 算全词表分布。AUC 是 lag 1–64 的离散求和，不是这 64 步的均值。恢复指标以 lag 0 的 post-transition response 为基准，要求连续 5 步低于 50% 或 10%；窗口内允许的事件起点为 1–123，未达到则右删失。这里的“半衰时间”是阈值首次持续跨越，不是拟合的指数衰减常数，也不保证后续不反弹。', '',
       '主实验 BF16、batch 32；2,000 次文章/题目级 bootstrap、点态 95% CI。FP32 对照固定每领域 16 个完整窗口样本，并以相同 batch 8 重跑 BF16；FP32 禁止 TF32。精度子集不并入 n=256 主结果，也不是独立任务确认。', '',
       '## 样本与主 AUC','',
       '| checkpoint / 数据 | AUC n / recovery n | zero 自然 state AUC | zero 自然 KL AUC | small 自然 state AUC | small 自然 KL AUC |',
       '|---|---:|---:|---:|---:|---:|']
    for model in NAMES:
        for domain in ['wiki','math']:
            c=j['cells'][model+'_'+domain];z=c['summary']['zero_natural'];s=c['summary']['small_mean_natural']
            lines.append(f"| {LABELS[model]} / {domain} | {c['eligible_auc65']} / {c['eligible_recovery128']} | {fmt(z['state_rel_l2_auc_1_64'])} | {fmt(z['kl_auc_1_64'])} | {fmt(s['state_rel_l2_auc_1_64'])} | {fmt(s['kl_auc_1_64'])} |")
    lines+=['','小扰动列是两个方向在每个样本内平均后的结果。未列出的单方向、clean-KV、按注入幅度归一化的 AUC、跨训练量配对差，均在机器可读结果中保留。AUC 只纳入完整 65 个 targets；恢复指标只纳入完整 128 个 targets，不能把短解答算成零响应。', '',
       '## BF16 状态恢复时间','',
       '| checkpoint / 数据 | zero 自然 t50 / t10 中位数 | zero clean t50 / t10 | small 自然 t50 / t10 | small clean t50 / t10 |',
       '|---|---:|---:|---:|---:|']
    for model in NAMES:
        for domain in ['wiki','math']:
            c=j['cells'][model+'_'+domain];cells=[]
            for arm in ['zero_natural','zero_clean','small_mean_natural','small_mean_clean']:
                r=c['summary'][arm]['state_rel_l2_recovery'];cells.append(median(r['0.5'])+' / '+median(r['0.1']))
            lines.append('| '+LABELS[model]+' / '+domain+' | '+' | '.join(cells)+' |')
    lines+=['','`>123` 表示到可观测事件起点上限仍不足一半样本达标，不能给出未删失中位数；不把它替换成 124 步“真实恢复时间”。机器可读结果同时提供事件数、删失数、达标比例及到 124 的受限均值。小扰动恢复时间在每个样本的方向平均响应曲线上计算；单方向结果另存。', '',
       '## 精度敏感性：固定 16 样本对照','',
       '| checkpoint / 数据 | small 自然 state AUC BF16 → FP32 | small clean state AUC BF16 → FP32 | small 自然 t10 BF16 → FP32 | small clean t10 BF16 → FP32 |',
       '|---|---:|---:|---:|---:|']
    for model in NAMES:
        for domain in ['wiki','math']:
            b=p['cells'][f'{model}_{domain}_bf16']['summary'];f=p['cells'][f'{model}_{domain}_fp32']['summary']
            cells=[]
            for arm in ['small_mean_natural','small_mean_clean']:
                cells.append(f"{b[arm]['state_rel_l2_auc_1_64']['mean']:.5g} → {f[arm]['state_rel_l2_auc_1_64']['mean']:.5g}")
            for arm in ['small_mean_natural','small_mean_clean']:
                cells.append(median(b[arm]['state_rel_l2_recovery']['0.1'])+' → '+median(f[arm]['state_rel_l2_recovery']['0.1']))
            lines.append('| '+LABELS[model]+' / '+domain+' | '+' | '.join(cells)+' |')
    lines+=['','精度差异是同输入、同方向、同 batch 的比较，但只有 16 个固定样本；它诊断数值敏感性，不支持跨任务的性能结论。FP32 与 BF16 是不同执行精度（包括加载时权重转换），不是仅替换一个舍入算子；FP32 结果不能抹去 BF16 的实际行为，低精度残留也不能直接命名为语义记忆。', '',
       '| checkpoint / 数据 | small 自然 KL AUC BF16 → FP32 | small clean KL AUC BF16 → FP32 |',
       '|---|---:|---:|']
    for model in NAMES:
        for domain in ['wiki','math']:
            b=p['cells'][f'{model}_{domain}_bf16']['summary'];f=p['cells'][f'{model}_{domain}_fp32']['summary']
            values=[f"{b['small_mean_'+route]['kl_auc_1_64']['mean']:.5g} → {f['small_mean_'+route]['kl_auc_1_64']['mean']:.5g}" for route in ['natural','clean']]
            lines.append('| '+LABELS[model]+' / '+domain+' | '+' | '.join(values)+' |')
    lines+=['',
       '### 响应曲线','',
       '以下为完整窗口样本的逐 lag 平均响应，不是单条样本轨迹或中位恢复时间。精度图每领域仅 16 个配对样本。FP32 自然路径仍有比 clean-KV 大得多、但比 BF16 小的残留；达到 t10 不等于影响彻底消失。', '',
       '![Wiki paired precision audit](../04_evidence/t2mlr_impulse_precision_wiki.png)', '',
       '![Math paired precision audit](../04_evidence/t2mlr_impulse_precision_math.png)', '',
       '主测量全样本曲线：[Wiki](../04_evidence/t2mlr_impulse_primary_wiki.png)、[Math](../04_evidence/t2mlr_impulse_primary_math.png)。零重置和小扰动的注入幅度不同，图中原始响应不能直接当成单位输入增益比较。', '',
       '## 路径差与 362M/50B 数学例外','']
    c=j['cells']['362m_50b_math']
    for arm in ['zero_natural','zero_clean','small_mean_natural','small_mean_clean']:
        s=c['summary'][arm]
        lines.append(f"- {arm}：state AUC {fmt(s['state_rel_l2_auc_1_64'])}；KL AUC {fmt(s['kl_auc_1_64'])}；远期 signed ΔNLL {fmt(s['delta_nll_lag_32_127'])}；远期 absolute ΔNLL {fmt(s['abs_delta_nll_lag_32_127'])}。")
    replay=json.loads(Path('04_evidence/t2mlr_impulse_replay_sensitivity.json').read_text())
    lines+=['','该例外复用了同一数据，因此新的 KL/state/small-perturbation 是测量与路径诊断扩展，不是独立数据复现。自然减 clean 的差值不能解释为可加的 KV 因果贡献百分比；零重置与局部小扰动也不是等价干预。', '',
       '## BF16 批次敏感性','',
       '本轮 batch 32 与旧 NLL 测量 batch 64 的计算形状和分组不同。以下比较相同文档、相同干预位置的零重置远期效应；区间以文档配对差 bootstrap。差异不能归因于新增实验条件，也不以逐 token 位相同为前提。', '',
       '| checkpoint / 数据 | 正常 NLL token 平均绝对差 | 自然远期 ΔNLL 新−旧 | clean 远期 ΔNLL 新−旧 |',
       '|---|---:|---:|---:|']
    for model in NAMES:
        for domain in ['wiki','math']:
            r=replay['cells'][model+'_'+domain];d=r['far_document_effect_shift_ci95']
            lines.append(f"| {LABELS[model]} / {domain} | {r['baseline']['token_mean_absolute_difference']:.5g} | {fmt(d['zero_natural'])} | {fmt(d['zero_clean'])} |")
    lines+=['','这是批次/数值执行敏感性审计，不是独立数据复现；主实验每个 batch 内的干预仍共享正常基线。较小的效应应结合这一尺度和 FP32 对照解释。', '',
       '## 证据与执行','',
       '完整权重 official exact-step parity、无扰动前缀相等、自然/clean lag-0 相等、输入与代码哈希、每个样本的有效窗口和实际注入范数均已检查。KL 原始数值也保留，汇总非负 KL 仅截去微小负舍入误差。CPU 测试覆盖独立轨迹一致性、范数/方向确定性、恢复阈值和删失。', '',
       '- 预注册：`03_experiments/t2mlr_impulse_preregistration.md`。',
       '- 主测量：`04_evidence/t2mlr_impulse_results.json`。',
       '- 精度对照：`04_evidence/t2mlr_impulse_precision_results.json`。',
       '- 主原始数据：`artifacts/impulse-results.tar.gz`。',
       '- 精度原始数据：`artifacts/impulse-precision-results.tar.gz`。']
    total=sum(c['manifest']['elapsed_seconds'] for c in j['cells'].values())
    audit=sum(c['manifest']['elapsed_seconds'] for c in p['cells'].values())
    lines+=['',f'主测量运行合计 {total/60:.1f} 分钟，精度对照 cell 计时合计 {audit/60:.1f} 分钟（后者不含模型加载与 parity）。均不是包含准备、归档和文件传输的实例总计费时间。','']
    Path('06_reports/t2mlr_impulse_results.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()

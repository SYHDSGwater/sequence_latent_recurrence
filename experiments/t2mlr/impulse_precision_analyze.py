import json
from pathlib import Path
from impulse_analyze import analyze_cell
from impulse_metrics import bootstrap

def main():
    cells={};units={};contrasts={}
    for model in ['135m','135m_50b','362m_10b','362m_50b','982m_10b','982m']:
        for domain in ['wiki','math']:
            for precision in ['bf16','fp32']:
                name=f'{model}_{domain}_{precision}'
                cells[name],units[name]=analyze_cell(Path('artifacts/impulse_precision/artifacts/impulse_precision')/name,
                     Path('artifacts/impulse/data/expanded')/(domain+'.jsonl'),precision_subset=True)
            left=units[f'{model}_{domain}_bf16'];right=units[f'{model}_{domain}_fp32']
            contrasts[model+'_'+domain]={arm:{k:bootstrap([right[arm][k][doc]-v for doc,v in values.items()]) for k,values in m.items()} for arm,m in left.items()}
    out=dict(status='complete',scope='paired numerical sensitivity audit, n=16 per domain, no independent task replication',
             cells=cells,paired_fp32_minus_bf16=contrasts)
    Path('04_evidence/t2mlr_impulse_precision_results.json').write_text(json.dumps(out,indent=2))
    print('Verified',len(cells),'paired precision cells')

if __name__=='__main__':main()

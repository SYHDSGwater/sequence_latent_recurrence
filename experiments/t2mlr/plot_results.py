"""Static recovery-window figure from the audited paired summary."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument('report',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    results=json.loads(args.report.read_text())
    fig,axes=plt.subplots(1,len(results),figsize=(6*len(results),4),squeeze=False,sharey=True)
    arms=[('reset_once','Reset, natural KV'),('reset_once_clean_kv','Reset, clean KV'),('donor_once','Norm-matched donor')]
    windows=['lag_0_0','lag_1_7','lag_8_31','lag_32_127']
    for ax,result in zip(axes[0],results):
        ax.axhspan(-.01,.01,color='gray',alpha=.15,label='Practical equivalence band')
        ax.axhline(0,color='black',lw=.7)
        for i,(arm,label) in enumerate(arms):
            stats=result['summary'][arm]
            mean=np.array([stats[w]['mean_delta_nll'] for w in windows])
            low=np.array([stats[w]['ci95'][0] for w in windows])
            high=np.array([stats[w]['ci95'][1] for w in windows])
            ax.errorbar(np.arange(4)+(i-1)*.13,mean,yerr=[mean-low,high-mean],marker='o',capsize=3,label=label)
        ax.set_xticks(range(4),['0','1–7','8–31','32–127'])
        ax.set_xlabel('Tokens since intervention (window average)')
        ax.set_ylabel('Paired delta NLL (nat/token); 95% bootstrap CI')
        ax.set_title(Path(result['directory']).name+f" | {result['completed_documents']} documents")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output,dpi=180)

if __name__=='__main__':main()

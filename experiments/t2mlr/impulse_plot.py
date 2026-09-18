import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from impulse_report import NAMES,LABELS

def main():
    data=json.loads(Path('04_evidence/t2mlr_impulse_precision_results.json').read_text())
    for domain in ['wiki','math']:
        fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True,layout='constrained')
        for model,ax in zip(NAMES,axes.flat):
            for precision,style in [('bf16','-'),('fp32','--')]:
                c=data['cells'][f'{model}_{domain}_{precision}']['summary']
                for route,color in [('natural','#bd632f'),('clean','#2377a5')]:
                    curve=np.array(c['small_mean_'+route]['state_rel_l2_mean_curve_full128'])
                    ax.plot(range(128),np.maximum(curve,1e-9),style,color=color,label=precision.upper()+' / '+route)
            ax.set_yscale('log');ax.set_title(LABELS[model]);ax.grid(alpha=.15)
            ax.set_xlabel('Lag after state perturbation');ax.set_ylabel('Mean relative state L2')
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'{domain.upper()}: paired precision audit, 16 fixed examples, 2 direction mean\n5% norm-preserving perturbation; log axis clipped at 1e-9 for display')
        fig.savefig(f'04_evidence/t2mlr_impulse_precision_{domain}.png',dpi=180)
        plt.close(fig)
    main_data=json.loads(Path('04_evidence/t2mlr_impulse_results.json').read_text())
    for domain in ['wiki','math']:
        fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True,layout='constrained')
        for model,ax in zip(NAMES,axes.flat):
            c=main_data['cells'][f'{model}_{domain}']
            for intervention,style in [('zero','-'),('small_mean','--')]:
                for route,color in [('natural','#bd632f'),('clean','#2377a5')]:
                    curve=np.array(c['summary'][intervention+'_'+route]['state_rel_l2_mean_curve_full128'])
                    ax.plot(range(128),np.maximum(curve,1e-9),style,color=color,label=intervention+' / '+route)
            ax.set_yscale('log');ax.set_title(LABELS[model]);ax.grid(alpha=.15)
            ax.set_xlabel('Lag after state perturbation');ax.set_ylabel('Mean relative state L2')
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'{domain.upper()}: BF16 primary, full-window n={c["eligible_recovery128"]}\nZero reset vs 5% norm-preserving perturbation; raw response amplitudes differ')
        fig.savefig(f'04_evidence/t2mlr_impulse_primary_{domain}.png',dpi=180)
        plt.close(fig)

if __name__=='__main__':main()

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def main():
    data=json.loads(Path('04_evidence/t2mlr_expanded_results.json').read_text())
    names=['135m_wiki','135m_math','982m_wiki','982m_math']
    labels=['135M / Wiki','135M / Math','982M / Wiki','982M / Math']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric,stat,title in zip(axes,['carry_off/scored','reset_once/lag_32_127','reset_once/lag_32_127'],
                                 ['signed','signed','absolute'],['Continuous carry removal','Single reset: lag 32–127','Single reset: lag 32–127']):
        for i,name in enumerate(names):
            v=data['cells'][name]['metrics'][metric][stat];m=v['mean_delta_nll'];lo,hi=v['ci95']
            ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#176b95' if 'wiki' in name else '#c56526',capsize=4)
        ax.axvline(0,color='gray',lw=.8)
        if metric=='reset_once/lag_32_127' and stat=='signed': ax.axvspan(-.01,.01,color='#5eae73',alpha=.13,label='Predefined ±0.01 interval')
        ax.set_title(title);ax.set_xlabel(('Signed' if stat=='signed' else 'Absolute tokenwise')+' ΔNLL (nat/token)')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_yticks(range(4),labels if ax==axes[0] else ['']*4);ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
    fig.suptitle('Frozen-checkpoint dependency — document-level 95% bootstrap CIs\nWiki n=256; Math n=256 for carry, n=169 for far windows',fontsize=12)
    fig.savefig('04_evidence/t2mlr_expanded_comparison.png',dpi=180)

if __name__=='__main__':main()

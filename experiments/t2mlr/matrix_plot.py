import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def main():
    data=json.loads(Path('04_evidence/t2mlr_six_checkpoint_results.json').read_text())
    names=[f'{s}_{t}' for s in ['135m','362m','982m'] for t in ['10b','50b']]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,domain in enumerate(['wiki','math']):
        for col,key in enumerate(['carry_off/scored','reset_once/lag_32_127']):
            ax=axes[row,col]
            for i,name in enumerate(names):
                v=data['cells'][name+'_'+domain]['metrics'][key]['signed'];m=v['mean'];lo,hi=v['ci95']
                ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color='#2377a5' if '10b' in name else '#bd632f')
            ax.axvline(0,color='gray',lw=.8)
            if col==1:ax.axvspan(-.01,.01,color='green',alpha=.08)
            ax.set_yticks(range(6),[n.upper().replace('_',' / ') for n in names] if col==0 else ['']*6)
            ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.set_title(domain.upper()+(' — carry removal (n=256)' if col==0 else ' — reset lag 32–127 (n='+('256' if domain=='wiki' else '169')+')'))
            ax.set_xlabel('Paired mean ΔNLL, nat/token; 95% bootstrap CI')
    fig.suptitle('Six frozen T²MLR checkpoints — blue: 10B tokens; orange: 50B tokens')
    fig.savefig('04_evidence/t2mlr_six_checkpoint_comparison.png',dpi=180)

if __name__=='__main__':main()

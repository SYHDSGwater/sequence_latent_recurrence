"""Quantify batch/kernel sensitivity versus earlier BF16 NLL measurements."""
import json
from pathlib import Path
import numpy as np
from impulse_metrics import bootstrap

def main():
    reports={}
    for model in ['135m','135m_50b','362m_10b','362m_50b','982m_10b','982m']:
        suite='expanded' if model in ['135m','982m'] else 'matrix'
        for domain in ['wiki','math']:
            name=model+'_'+domain;prior={}
            for f in (Path('artifacts')/suite/'artifacts'/suite/name).glob('batch_*.json'):
                rows=json.loads(f.read_text());a=np.load(f.with_suffix('.npz'))
                for i,r in enumerate(rows):
                    at=r['reset_at'];end=min(at+128,len(r['tokens'])-1)
                    prior[r['id']]={k:a[k][i,at:end] for k in ['normal','reset_once','reset_once_clean_kv']}
            diffs={'baseline':[],'zero_natural_delta':[],'zero_clean_delta':[]}
            signed_shifts={'zero_natural':[],'zero_clean':[]}
            for f in (Path('artifacts/impulse/artifacts/impulse')/name).glob('batch_*.json'):
                rows=json.loads(f.read_text());a=np.load(f.with_suffix('.npz'))
                for i,r in enumerate(rows):
                    old=prior[r['id']];n=len(old['normal'])
                    diffs['baseline'].extend((a['baseline_nll'][i,:n]-old['normal']).tolist())
                    for arm,ref in [('zero_natural','reset_once'),('zero_clean','reset_once_clean_kv')]:
                        delta=a[arm][i,:n,4]-(old[ref]-old['normal'])
                        diffs[arm+'_delta'].extend(delta.tolist())
                        if n==128:signed_shifts[arm].append(float(delta[32:128].mean()))
            assert diffs['baseline']
            reports[name]={k:{'token_mean_absolute_difference':float(np.abs(v).mean()),'token_max_absolute_difference':float(np.abs(v).max())} for k,v in diffs.items()}
            reports[name]['far_document_mean_effect_shift']={k:float(np.mean(v)) for k,v in signed_shifts.items()}
            reports[name]['far_document_effect_shift_ci95']={k:bootstrap(v) for k,v in signed_shifts.items()}
    Path('04_evidence/t2mlr_impulse_replay_sensitivity.json').write_text(json.dumps(dict(scope='BF16 batch32 impulse vs prior batch64; not bitwise replay',cells=reports),indent=2))

if __name__=='__main__':main()

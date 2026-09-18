"""Independent document-level analysis of complete impulse trajectories."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from impulse_metrics import bootstrap,recovery_summary

ARMS=['zero_natural','zero_clean','small0_natural','small0_clean','small1_natural','small1_clean']
METRICS=['kl','kl_raw','state_rel_l2','state_cosine','delta_nll','abs_delta_nll']

def analyze_cell(path,source,check_code=True,precision_subset=False):
    manifest=json.loads((path/'manifest.json').read_text())
    count=16 if precision_subset else 256
    assert manifest['status']=='complete' and manifest['completed_documents']==count
    assert manifest['official_step_validation']['max_logit_diff']==manifest['official_step_validation']['max_state_diff']==0
    assert manifest['metrics']==METRICS and manifest['arms']==ARMS
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['data_sha256']
    if check_code:
        for f,h in manifest['code_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()==h
    expected={r['id']:r for r in map(json.loads,source.read_text().splitlines())}
    if precision_subset:
        candidates=[r for r in expected.values() if len(r['tokens'])-1-r['reset_at']>=128]
        candidates.sort(key=lambda r:hashlib.sha256(('precision-audit|'+r['id']).encode()).hexdigest())
        expected={r['id']:r for r in candidates[:16]}
    rows=[];collected={arm:[] for arm in ARMS};injections=[];baseline=[]
    for f in sorted(path.glob('batch_*.json')):
        records=json.loads(f.read_text());arrays=np.load(f.with_suffix('.npz'))
        for i,r in enumerate(records):
            assert r==expected[r['id']]
            n=min(128,len(r['tokens'])-1-r['reset_at'])
            for arm in ARMS:
                v=arrays[arm][i];assert np.isfinite(v[:n]).all() and np.isnan(v[n:]).all()
            assert np.isfinite(arrays['baseline_nll'][i,:n]).all()
        for arm in ARMS:collected[arm].append(arrays[arm])
        for a in range(0,6,2):np.testing.assert_array_equal(arrays[ARMS[a]][:,0],arrays[ARMS[a+1]][:,0])
        rows.extend(records);injections.append(arrays['injection']);baseline.append(arrays['baseline_nll'])
    assert len(rows)==count and {r['id'] for r in rows}==set(expected)
    curves={a:np.concatenate(v) for a,v in collected.items()};injection=np.concatenate(injections,axis=1)
    assert np.isfinite(injection).all()
    np.testing.assert_allclose(injection[:2,:,0],1,atol=1e-7);np.testing.assert_allclose(injection[:2,:,1],0,atol=1e-7)
    assert ((injection[2:,:,0]>.045)&(injection[2:,:,0]<.055)).all()
    assert np.abs(injection[2:,:,1]-1).max()<.005
    for route in ['natural','clean']:
        curves['small_mean_'+route]=(curves['small0_'+route]+curves['small1_'+route])/2
    valid65=np.isfinite(curves['zero_natural'][:,:65,0]).all(1)
    valid128=np.isfinite(curves['zero_natural'][:,:,0]).all(1)
    summary={};units={}
    for arm,v in curves.items():
        s={};u={}
        for metric in ['state_rel_l2','kl']:
            k=METRICS.index(metric);x=v[:,:,k];auc=x[valid65,1:65].sum(1)
            s[metric+'_auc_1_64']=bootstrap(auc);u[metric+'_auc_1_64']=dict(zip(np.array([r['id'] for r in rows])[valid65],auc.tolist()))
            if metric=='state_rel_l2':
                if arm.startswith('small_mean'):
                    route=arm.split('_')[-1];gain=[]
                    for d in [0,1]:
                        idx=ARMS.index(f'small{d}_{route}')
                        gain.append(curves[ARMS[idx]][valid65,1:65,k].sum(1)/injection[idx,valid65,0])
                    gain=np.mean(gain,axis=0)
                else:gain=auc/injection[ARMS.index(arm),valid65,0]
                s['state_input_normalized_auc_1_64']=bootstrap(gain)
            s[metric+'_recovery']={str(frac):recovery_summary(x[valid128],frac) for frac in [.5,.1]}
            s[metric+'_mean_curve_full128']=x[valid128].mean(0).tolist()
        for a,b in [(0,1),(1,8),(8,32),(32,128)]:
            valid=np.isfinite(v[:,a:b,0]).all(1)
            for metric in ['delta_nll','abs_delta_nll']:
                vals=v[valid,a:b,METRICS.index(metric)].mean(1);key=f'{metric}_lag_{a}_{b-1}'
                s[key]=bootstrap(vals);u[key]=dict(zip(np.array([r['id'] for r in rows])[valid],vals.tolist()))
        summary[arm]=s;units[arm]=u
    contrasts={}
    for intervention in ['zero','small_mean']:
        contrasts[intervention]={}
        for key,left in units[intervention+'_natural'].items():
            right=units[intervention+'_clean'][key]
            contrasts[intervention][key]=bootstrap([left[doc]-right[doc] for doc in left])
    return dict(manifest=manifest,eligible_auc65=int(valid65.sum()),eligible_recovery128=int(valid128.sum()),
                summary=summary,natural_minus_clean=contrasts,
                injection_range={arm:dict(relative_displacement=[float(injection[i,:,0].min()),float(injection[i,:,0].max())],
                       norm_ratio=[float(injection[i,:,1].min()),float(injection[i,:,1].max())]) for i,arm in enumerate(ARMS)},
                raw_kl_min={arm:float(np.nanmin(v[:,:,1])) for arm,v in curves.items()}),units

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/impulse');p.add_argument('--partial',action='store_true');a=p.parse_args()
    root=Path(a.root);cells={};units={}
    names=['135m','135m_50b','362m_10b','362m_50b','982m_10b','982m']
    for model in names:
        for domain in ['wiki','math']:
            name=model+'_'+domain;path=root/'artifacts/impulse'/name
            if a.partial and (not (path/'manifest.json').exists() or json.loads((path/'manifest.json').read_text())['status']!='complete'):continue
            cells[name],units[name]=analyze_cell(path,root/'data/expanded'/f'{domain}.jsonl')
            print(name,'n_auc',cells[name]['eligible_auc65'],'n_recovery',cells[name]['eligible_recovery128'],flush=True)
    contrasts={}
    for size,lo,hi in [('135m','135m','135m_50b'),('362m','362m_10b','362m_50b'),('982m','982m_10b','982m')]:
        for domain in ['wiki','math']:
            if lo+'_'+domain not in units or hi+'_'+domain not in units:continue
            left=units[lo+'_'+domain];right=units[hi+'_'+domain]
            contrasts[size+'_'+domain]={arm:{k:bootstrap([right[arm][k][doc]-v for doc,v in values.items()]) for k,values in m.items()} for arm,m in left.items()}
    out=dict(status='partial' if a.partial else 'complete',cells=cells,paired_50b_minus_10b=contrasts,
             recovery_definition='first 5 consecutive lags <= fraction * lag0; 1..123; right censor=124; undefined lag0<=1e-10',
             small_mean_definition='two directions averaged within document; recovery computed on direction-mean response curve',
             integrity=dict(all_raw_arrays_verified=True,code_hashes_verified=True,frozen_rows_match=True,lag0_kv_pair_equal=True,perturbation_norm_checked=True))
    target=Path('04_evidence/t2mlr_impulse_results'+('_partial' if a.partial else '')+'.json');target.write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()

"""Recompute all six checkpoints on identical paired examples."""
import hashlib,json
from pathlib import Path
import numpy as np

def ci(values):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(0)
    means=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n=len(values))

def main():
    cells={};units={}
    checkpoints={'135m_10b':('expanded','135m'),'982m_50b':('expanded','982m'),
                 **{k:('matrix',k) for k in ['135m_50b','362m_10b','362m_50b','982m_10b']}}
    for name,(suite,folder) in checkpoints.items():
        for domain in ['wiki','math']:
            path=Path('artifacts')/suite/'artifacts'/suite/(folder+'_'+domain)
            manifest=json.loads((path/'manifest.json').read_text())
            assert manifest['status']=='complete' and manifest['completed_documents']==256
            assert manifest['official_step_validation']['max_logit_diff']==manifest['official_step_validation']['max_state_diff']==0
            for filename,digest in manifest['code_sha256'].items():
                assert hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()==digest
            data=Path('artifacts/expanded/data/expanded')/(domain+'.jsonl')
            assert hashlib.sha256(data.read_bytes()).hexdigest()==manifest['data_sha256']
            frozen={r['id']:r for r in map(json.loads,data.read_text().splitlines())}
            metrics={};baseline={};seen=set()
            for f in sorted(path.glob('batch_*.json')):
                rows=json.loads(f.read_text());losses=np.load(f.with_suffix('.npz'))
                for i,row in enumerate(rows):
                    key=row['id'];assert key not in seen and row==frozen[key];seen.add(key)
                    end=len(row['tokens'])-1;at=row['reset_at'];start=row['score_start']
                    normal=losses['normal'][i,:end];baseline[key]=float(normal[start:].mean())
                    for arm in manifest['arms'][1:]:
                        val=losses[arm][i,:end];assert np.isfinite(val).all()
                        if arm.startswith('reset_once'):np.testing.assert_array_equal(val[:at],normal[:at])
                        if arm=='reset_once_clean_kv':assert val[at]==losses['reset_once'][i,at]
                        delta=val-normal
                        windows={'scored':(start,end)} if arm in ['carry_off','zero_every'] else {
                            f'lag_{a}_{b-1}':(at+a,at+b) for a,b in [(0,1),(1,8),(8,32),(32,128)]}
                        for window,(a,b) in windows.items():
                            if b>end:continue
                            entry=metrics.setdefault(arm+'/'+window,{'signed':{},'absolute':{}})
                            entry['signed'][key]=float(delta[a:b].mean());entry['absolute'][key]=float(np.abs(delta[a:b]).mean())
            assert seen==set(frozen)
            summary={k:{s:ci(list(v.values())) for s,v in stats.items()} for k,stats in metrics.items()}
            saved=json.loads((path/'summary.json').read_text())
            for key,stats in summary.items():
                arm,window=key.split('/')
                for stat,v in stats.items():
                    assert v['mean']==saved[arm][window][stat]['mean_delta_nll']
                    assert v['ci95']==saved[arm][window][stat]['ci95']
            cell=name+'_'+domain;units[cell]=metrics
            cells[cell]=dict(manifest=manifest,baseline_nll=ci(list(baseline.values())),metrics=summary)
    contrasts={}
    for size in ['135m','362m','982m']:
        for domain in ['wiki','math']:
            left=units[size+'_10b_'+domain];right=units[size+'_50b_'+domain]
            contrasts[size+'_'+domain]={key:{s:ci([right[key][s][doc]-v for doc,v in vals.items()])
                for s,vals in stats.items()} for key,stats in left.items()}
    result=dict(cells=cells,paired_50b_minus_10b=contrasts,
                integrity=dict(all_cells_complete=True,identical_frozen_examples=True,all_prefix_controls_pass=True,independent_summary_match=True,execution_code_hashes_match=True),
                pending_phase_a_metrics=['small norm-controlled perturbations','state impulse AUC','output KL AUC','recovery half-life'])
    Path('04_evidence/t2mlr_six_checkpoint_results.json').write_text(json.dumps(result,indent=2))
    for name,c in sorted(cells.items()):
        print(name,'baseline',round(c['baseline_nll']['mean'],4),{k:round(c['metrics'][k]['signed']['mean'],6) for k in ['carry_off/scored','zero_every/scored','reset_once/lag_32_127']})

if __name__=='__main__':main()

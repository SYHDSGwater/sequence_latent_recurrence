"""Offline integrity checks and independent unit-level aggregation."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from run import document_ci

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root); result={'cells':{},'paired_model_differences':{},'integrity':{}}
    units={}
    for model in ['135m','982m']:
        for domain in ['wiki','math']:
            name=model+'_'+domain; path=root/'artifacts/expanded'/name
            manifest=json.loads((path/'manifest.json').read_text())
            assert manifest['status']=='complete' and manifest['completed_documents']==256
            assert manifest['official_step_validation']['max_logit_diff']==0
            assert manifest['official_step_validation']['max_state_diff']==0
            for filename,expected_hash in manifest['code_sha256'].items():
                assert hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()==expected_hash
            data=root/'data/expanded'/f'{domain}.jsonl'
            assert hashlib.sha256(data.read_bytes()).hexdigest()==manifest['data_sha256']
            expected={r['id'] for r in map(json.loads,data.read_text().splitlines())}
            metrics={}; normals=[]; found=[]; horizon_counts={}; subjects={}; levels={}
            for batchfile in sorted(path.glob('batch_*.json')):
                rows=json.loads(batchfile.read_text());losses=np.load(batchfile.with_suffix('.npz'))
                for i,r in enumerate(rows):
                    found.append(r['id']); end=len(r['tokens'])-1; at=r['reset_at']; score=r['score_start']
                    normal=losses['normal'][i,:end]; assert np.isfinite(normal).all()
                    normals.append(float(normal[score:].mean()))
                    subjects[r['subject']]=subjects.get(r['subject'],0)+1
                    if 'level' in r: levels[str(r['level'])]=levels.get(str(r['level']),0)+1
                    for arm in manifest['arms'][1:]:
                        val=losses[arm][i,:end];assert np.isfinite(val).all()
                        if arm.startswith('reset_once'): np.testing.assert_array_equal(val[:at],normal[:at])
                        if arm=='reset_once_clean_kv': assert val[at]==losses['reset_once'][i,at]
                        delta=val-normal
                        windows={'scored':(score,end)} if arm in ['carry_off','zero_every'] else {
                            f'lag_{x}_{y-1}':(at+x,at+y) for x,y in [(0,1),(1,8),(8,32),(32,128)]}
                        for w,(start,stop) in windows.items():
                            if stop>end:continue
                            key=arm+'/'+w
                            metrics.setdefault(key,{'signed':{},'absolute':{}})
                            metrics[key]['signed'][r['id']]=float(delta[start:stop].mean())
                            metrics[key]['absolute'][r['id']]=float(np.abs(delta[start:stop]).mean())
            assert len(found)==len(set(found))==256 and set(found)==expected
            units[name]=metrics
            cell={'manifest':manifest,'baseline_document_mean_nll':float(np.mean(normals)),
                  'subjects':subjects,'levels':levels,'metrics':{k:{s:document_ci(list(vals.values())) for s,vals in v.items()} for k,v in metrics.items()}}
            original=json.loads((path/'summary.json').read_text())
            for key,stats in cell['metrics'].items():
                arm,window=key.split('/')
                assert stats==original[arm][window], 'Independent recomputation differs'
            cell['exploratory_document_distribution']={key:{'q10_q50_q90':np.quantile(list(v['signed'].values()),[.1,.5,.9]).tolist(),
                 'fraction_positive':float(np.mean(np.array(list(v['signed'].values()))>0))} for key,v in metrics.items()}
            result['cells'][name]=cell
    for domain in ['wiki','math']:
        paired={}
        for key,v in units['135m_'+domain].items():
            left=v['signed'];right=units['982m_'+domain][key]['signed'];assert set(left)==set(right)
            paired[key]=document_ci([right[k]-left[k] for k in left])
        result['paired_model_differences'][domain]=paired
    result['integrity']={'all_four_cells_complete':True,'unique_inputs_per_domain':256,
                         'paired_models_same_ids':True,'all_reset_prefixes_exact':True,'data_hashes_verified':True,
                         'code_hashes_verified':True,'official_step_parity_passed':True,
                         'reset_lag0_equal_between_kv_arms':True,'independent_summaries_match':True}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps({n:{'baseline':c['baseline_document_mean_nll'],'metrics':{k:v['signed'] for k,v in c['metrics'].items()}} for n,c in result['cells'].items()},indent=2))

if __name__=='__main__':main()

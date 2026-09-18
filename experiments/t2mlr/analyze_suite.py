"""Summarize completed paired traces and audit causal negative controls."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

def analyze(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    traces={}
    for line in (directory/'token_traces.jsonl').read_text().splitlines():
        row=json.loads(line)
        traces.setdefault(row['arm'],{})[(row['document'],row['position'])]=row['nll']
    baseline=traces['normal']
    position=manifest['arguments']['reset_at']
    checks={}
    for arm in ['reset_once','donor_once','reset_once_clean_kv']:
        if arm in traces:
            checks[arm+'_prefix_max_abs_diff']=max(abs(v-baseline[k]) for k,v in traces[arm].items() if k[1]<position)
    if 'fusion_and_carry_off' in traces:
        checks['fusion_carry_negative_control_max_abs_diff']=max(abs(v-traces['fusion_off'][k]) for k,v in traces['fusion_and_carry_off'].items())
    mean_nll=float(np.mean([v for (doc,p),v in baseline.items() if p>=32]))
    interpretations={}
    for arm,windows in summary.items():
        interpretations[arm]={}
        for window,stats in windows.items():
            low,high=stats['ci95']
            if not stats['inference_valid']: label='insufficient_independent_units'
            elif low>=-.01 and high<=.01: label='within_practical_equivalence_band'
            elif low>.01: label='harm_exceeds_practical_threshold'
            elif high<-.01: label='improvement_exceeds_practical_threshold'
            else: label='uncertain_at_practical_threshold'
            interpretations[arm][window]=label
    return {'directory':str(directory),'manifest_status':manifest['status'],
            'completed_documents':len({k[0] for k in baseline}),
            'normal_post_warmup_nll':mean_nll,'normal_post_warmup_ppl':math.exp(mean_nll),
            'control_checks':checks,'controls_pass':all(v==0 for v in checks.values()),
            'summary':summary,'interpretations':interpretations,
            'caveat':'Unadjusted exploratory CIs except preregistered endpoints; signed mean effects can cancel; no architecture advantage or latent reasoning claim'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('directories',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=[analyze(directory) for directory in args.directories]
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k not in {'summary','interpretations'}} for r in result],indent=2))

if __name__=='__main__':main()

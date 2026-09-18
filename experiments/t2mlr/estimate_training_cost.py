"""Transparent H100 planning scenarios; these are NOT measured training rates."""
import json
from pathlib import Path

def main():
    # Illustrative 60M nonembedding backbone + 25M tied output projection compute.
    n=85e6;r=30/85
    rates={'A':(15000,60000),'B':(8000,30000),'C':(5000,20000),
           'D':(2000,10000),'E':(1000,6000),'F':(50000,200000)}
    out=dict(status='planning_scenarios_not_H100_benchmarks',reference_nonembedding_params=60_000_000,
             effective_matmul_params=n,recurrent_compute_fraction=r,context=128,
             assumed_tokens_per_second=rates,scenarios={})
    for arm,(d,b) in {'A':(16,4),'B':(16,16),'C':(32,16)}.items():
        # Approximation to released code: two full forwards plus recurrent refinements,
        # one full backward and last b recurrent backwards; backward ~2x forward.
        mult=(4+(d-1+2*b)*r)/3
        out.setdefault('approximate_flops',{})[arm]=dict(standard_training_multiplier=mult,
             per_token=6*n*mult,ideal_989tflops_hours_per_100m_tokens=6*n*mult*1e8/(989e12*3600))
    for name,arms,seeds in [('core',['A','B','D','E'],3),('full',['A','B','C','D','E','F'],3),('pilot',['A','B','D','E'],1)]:
        for tokens in ([2e6] if name=='pilot' else [20e6,100e6,500e6,1e9]):
            lo=sum(tokens/rates[x][1]/3600 for x in arms)*seeds
            hi=sum(tokens/rates[x][0]/3600 for x in arms)*seeds
            out['scenarios'][name+'_'+str(int(tokens))]=dict(arms=arms,seeds=seeds,tokens_per_arm_per_seed=int(tokens),
                total_training_tokens=int(tokens*seeds*len(arms)),training_gpu_hours=[lo,hi],
                with_30_percent_running_overhead=[1.3*lo,1.3*hi])
    Path('04_evidence/t2mlr_jacobi_cost_scenarios.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()

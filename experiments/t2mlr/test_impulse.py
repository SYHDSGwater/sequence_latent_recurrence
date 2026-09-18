import time
import numpy as np
import torch
from test_mechanism import tiny
from impulse_run import perturb,batch_rollout,METRICS
from impulse_metrics import recovery_time,recovery_summary
from run import rollout

def test_tangent_perturbation_norm_and_determinism():
    s=torch.randn(2,1,16)
    a=perturb(s,['a','b'],0);b=perturb(s,['a','b'],0);c=perturb(s,['a','b'],1)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(a.norm(dim=-1),s.norm(dim=-1),rtol=1e-6,atol=1e-6)
    assert not torch.equal(a,c)
    relative=(a-s).norm(dim=-1)/s.norm(dim=-1)
    assert torch.all((relative>.049)&(relative<.051))

def test_lockstep_matches_previous_independent_trajectories():
    model=tiny();rows=[dict(id='a',tokens=[3,5,8,1,6,9,3],reset_at=2),dict(id='b',tokens=[2,9,7,4,3,8,4],reset_at=3)]
    result=batch_rollout(model,rows,time.monotonic()+60)
    tokens=torch.tensor([r['tokens'] for r in rows]);resets=torch.tensor([2,3])
    normal,_=rollout(model,tokens,'normal',resets,time.monotonic()+60)
    for old,new in [('reset_once','zero_natural'),('reset_once_clean_kv','zero_clean')]:
        changed,_=rollout(model,tokens,old,resets,time.monotonic()+60)
        for i,row in enumerate(rows):
            at=row['reset_at'];n=len(row['tokens'])-1-at
            np.testing.assert_allclose(result[new][i,:n,METRICS.index('delta_nll')],changed[i,at:]-normal[i,at:],atol=1e-6,rtol=1e-6)
            assert np.isnan(result[new][i,n:]).all()

def test_sustained_recovery_censor_and_undefined():
    x=np.ones(128);x[3:7]=.05;x[9:14]=.05
    assert recovery_time(x,.1)==(9.,True)
    assert recovery_time(np.ones(128),.5)==(124.,False)
    assert np.isnan(recovery_time(np.zeros(128),.5)[0])
    summary=recovery_summary([x,np.ones(128),np.zeros(128)],.1)
    assert summary['events']==1 and summary['censored']==1 and summary['undefined']==1

"""Lockstep, bounded-memory state/output impulse measurements."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from adapter import load_checkpoint,UPSTREAM
from expanded_prepare import MODELS
from expanded_run import validate

ARMS=['zero_natural','zero_clean','small0_natural','small0_clean','small1_natural','small1_clean']
METRICS=['kl','kl_raw','state_rel_l2','state_cosine','delta_nll','abs_delta_nll']

def perturb(state,ids,direction,epsilon=.05):
    vectors=[]
    for doc in ids:
        seed=int.from_bytes(hashlib.sha256(f'{doc}|direction={direction}'.encode()).digest()[:8],'little')%(2**63-1)
        vectors.append(torch.randn(state.shape[-1],generator=torch.Generator().manual_seed(seed)))
    u=torch.stack(vectors).to(state.device);s=state.float().squeeze(1);norm=s.norm(dim=-1,keepdim=True)
    u=u-(u*s).sum(-1,keepdim=True)/norm.square().clamp_min(1e-20)*s
    u=u/u.norm(dim=-1,keepdim=True).clamp_min(1e-20)
    changed=s+epsilon*norm*u
    changed=changed*norm/changed.norm(dim=-1,keepdim=True).clamp_min(1e-20)
    return changed.unsqueeze(1).to(state.dtype)

@torch.inference_mode()
def batch_rollout(model,rows,deadline):
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    ends=[min(len(r['tokens'])-1,r['reset_at']+128) for r in rows]
    total=max(ends);batch=len(rows);ids=[r['id'] for r in rows]
    tokens=torch.tensor([r['tokens'][:total+1]+[0]*max(0,total+1-len(r['tokens'])) for r in rows],device=device)
    resets=torch.tensor([r['reset_at'] for r in rows],device=device)
    first=int(resets.min());kv=None;state=None;branches={}
    arrays={arm:np.full((batch,128,len(METRICS)),np.nan,dtype=np.float32) for arm in ARMS}
    injection=np.full((len(ARMS),batch,2),np.nan,dtype=np.float32)
    baseline=np.full((batch,128),np.nan,dtype=np.float32)
    for pos in range(total):
        if time.monotonic()>=deadline:raise TimeoutError('Impulse time cap')
        if pos==first:
            branches={arm:[copy.deepcopy(kv),state.clone()] for arm in ARMS}
        clean_prefix=copy.deepcopy(kv) if pos>=first else None
        model.recurrent_cache=state
        normal=model.step(tokens[:,pos:pos+1],kv,pos);kv=normal.past_key_values;state=model.recurrent_cache.clone()
        if pos<first:continue
        normal_logp=normal.logits[:,-1].float().log_softmax(-1)
        normal_prob=normal_logp.exp();normal_state=state.float().squeeze(1)
        target=tokens[:,pos+1];normal_nll=-normal_logp.gather(1,target[:,None]).squeeze(1)
        normal_nll_cpu=normal_nll.cpu().numpy()
        for i,r in enumerate(rows):
            lag=pos-r['reset_at']
            if 0<=lag<128 and pos<ends[i]:baseline[i,lag]=normal_nll_cpu[i]
        for arm_index,arm in enumerate(ARMS):
            branch_kv,prior=branches[arm]
            mask=resets==pos
            if mask.any():
                candidate=torch.zeros_like(prior) if arm.startswith('zero') else perturb(prior,ids,int(arm[5]))
                changed=torch.where(mask[:,None,None],candidate,prior)
                rel=(changed.float()-prior.float()).norm(dim=-1)/prior.float().norm(dim=-1).clamp_min(1e-20)
                ratio=changed.float().norm(dim=-1)/prior.float().norm(dim=-1).clamp_min(1e-20)
                for i in torch.where(mask)[0].tolist():injection[arm_index,i]=[float(rel[i,0]),float(ratio[i,0])]
                prior=changed
            model.recurrent_cache=prior
            current_kv=copy.deepcopy(clean_prefix) if arm.endswith('clean') else branch_kv
            out=model.step(tokens[:,pos:pos+1],current_kv,pos)
            changed_state=model.recurrent_cache.clone()
            branches[arm]=[None if arm.endswith('clean') else out.past_key_values,changed_state]
            logp=out.logits[:,-1].float().log_softmax(-1)
            kl_raw=(normal_prob*(normal_logp-logp)).sum(-1)
            if float(kl_raw.min()) < -2e-5:raise FloatingPointError('Unexpected negative KL')
            rel=(changed_state.float().squeeze(1)-normal_state).norm(dim=-1)/normal_state.norm(dim=-1).clamp_min(1e-20)
            cos=(1-F.cosine_similarity(changed_state.float().squeeze(1),normal_state,dim=-1)).clamp_min(0)
            delta=-logp.gather(1,target[:,None]).squeeze(1)-normal_nll
            values=torch.stack([kl_raw.clamp_min(0),kl_raw,rel,cos,delta,delta.abs()],-1)
            if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite impulse')
            before=resets>pos
            if before.any():
                assert torch.equal(changed_state[before],state[before]),'State prefix mismatch'
                assert torch.equal(logp[before],normal_logp[before]),'Output prefix mismatch'
            values_cpu=values.cpu().numpy()
            for i,r in enumerate(rows):
                lag=pos-r['reset_at']
                if 0<=lag<128 and pos<ends[i]:arrays[arm][i,lag]=values_cpu[i]
        del clean_prefix,normal,normal_logp,normal_prob
    assert np.isfinite(injection).all()
    for a in range(0,len(ARMS),2):
        np.testing.assert_array_equal(arrays[ARMS[a]][:,0],arrays[ARMS[a+1]][:,0])
    return dict(**arrays,injection=injection,baseline_nll=baseline)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--domain',choices=['wiki','math'],required=True);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--max-seconds',type=int,default=3600);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    data=Path('data/expanded')/(a.domain+'.jsonl');rows=[json.loads(x) for x in data.read_text().splitlines()]
    assert len(rows)==256 and len(set(r['id'] for r in rows))==256
    rows.sort(key=lambda r:min(len(r['tokens'])-1,r['reset_at']+128))
    repo,rev=MODELS[a.model]
    manifest=dict(status='started',model=repo,revision=rev,upstream=UPSTREAM,arguments=vars(a),completed_documents=0,
                  metrics=METRICS,arms=ARMS,epsilon=.05,directions=2,dtype='bfloat16',torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(),data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                  code_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ['impulse_run.py','adapter.py','expanded_run.py','expanded_prepare.py','test_mechanism.py']})
    def save():
        manifest['elapsed_seconds']=time.monotonic()-started;(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save()
    try:
        model=load_checkpoint(snapshot_download(repo,revision=rev,local_files_only=True),'external/T2MLR','cuda')
        manifest['official_step_validation']=validate(model);save()
        for start in range(0,len(rows),a.batch_size):
            batch=rows[start:start+a.batch_size]
            arrays=batch_rollout(model,batch,started+a.max_seconds)
            np.savez_compressed(out/f'batch_{start:04d}.npz',**arrays)
            (out/f'batch_{start:04d}.json').write_text(json.dumps(batch))
            manifest['completed_documents']+=len(batch);save()
            print(json.dumps(dict(model=a.model,domain=a.domain,completed=manifest['completed_documents'],seconds=round(time.monotonic()-started,1))),flush=True)
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:save()

if __name__=='__main__':main()

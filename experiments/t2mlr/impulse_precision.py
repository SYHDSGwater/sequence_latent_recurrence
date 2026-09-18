"""Fixed small paired precision audit; never pooled into main evidence."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import snapshot_download
from adapter import load_checkpoint
from expanded_prepare import MODELS
from impulse_run import batch_rollout,ARMS,METRICS
from test_mechanism import official_step

@torch.inference_mode()
def parity(model):
    dtype=next(model.parameters()).dtype
    tokens=torch.tensor([[0,45,632,87,199,15,49,831],[0,79,271,95,181,64,231,8]],device='cuda')
    model.clear();kv=None;expected=[]
    for pos in range(8):
        out=model.step(tokens[:,pos:pos+1],kv,pos);kv=out.past_key_values
        expected.append((out.logits.clone(),model.recurrent_cache.clone()))
    model.clear();model.recurrent_cache=torch.zeros(2,1,model.t2mlr_model.config.hidden_size,device='cuda',dtype=dtype)
    method=official_step(model);kv=None;diff=[0.,0.]
    for pos in range(8):
        token=tokens[:,pos:pos+1]
        out=method(token,torch.full_like(token,2),torch.ones(2,pos+1,device='cuda',dtype=torch.long),past_key_values=kv,
                   position_ids=torch.full_like(token,pos),cache_position=torch.tensor([pos],device='cuda'))
        kv=out.past_key_values;diff[0]=max(diff[0],float((out.logits-expected[pos][0]).abs().max()))
        diff[1]=max(diff[1],float((model.recurrent_cache-expected[pos][1]).abs().max()))
    assert diff==[0.,0.],diff
    return dict(max_logit_diff=diff[0],max_state_diff=diff[1])

def main():
    torch.backends.cuda.matmul.allow_tf32=False
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);a=p.parse_args()
    repo,rev=MODELS[a.model];snapshot=snapshot_download(repo,revision=rev,local_files_only=True)
    selected={}
    for domain in ['wiki','math']:
        records=[json.loads(x) for x in Path('data/expanded/'+domain+'.jsonl').read_text().splitlines()]
        records=[r for r in records if len(r['tokens'])-1-r['reset_at']>=128]
        records.sort(key=lambda r:hashlib.sha256(('precision-audit|'+r['id']).encode()).hexdigest())
        selected[domain]=records[:16]
    started=time.monotonic()
    for precision in ['bf16','fp32']:
        model=load_checkpoint(snapshot,'external/T2MLR','cpu' if precision=='fp32' else 'cuda').to('cuda')
        validation=parity(model)
        for domain,rows in selected.items():
            out=Path('artifacts/impulse_precision')/(a.model+'_'+domain+'_'+precision);out.mkdir(parents=True,exist_ok=False)
            manifest=dict(status='started',model=repo,revision=rev,dtype=precision,tf32_allowed=False,batch_size=8,completed_documents=0,
                          official_step_validation=validation,arms=ARMS,metrics=METRICS,
                          data_sha256=hashlib.sha256(Path('data/expanded/'+domain+'.jsonl').read_bytes()).hexdigest(),
                          code_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ['impulse_run.py','impulse_precision.py','adapter.py']})
            cell_start=time.monotonic()
            try:
                for start in [0,8]:
                    batch=rows[start:start+8];arrays=batch_rollout(model,batch,started+3600)
                    np.savez_compressed(out/f'batch_{start:04d}.npz',**arrays)
                    (out/f'batch_{start:04d}.json').write_text(json.dumps(batch));manifest['completed_documents']+=8
                manifest['status']='complete'
            except Exception as exc:manifest.update(status='failed',error=repr(exc));raise
            finally:
                manifest['elapsed_seconds']=time.monotonic()-cell_start
                (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print(json.dumps(dict(model=a.model,domain=domain,precision=precision,seconds=manifest['elapsed_seconds'])),flush=True)
        del model;torch.cuda.empty_cache()

if __name__=='__main__':main()

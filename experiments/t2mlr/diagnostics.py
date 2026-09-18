"""Exploratory nonnegative distribution/state diagnostics after primary NLL results."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from adapter import MODEL,REVISION,load_checkpoint
from run import rollout,document_ci

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/validation.jsonl')
    parser.add_argument('--output',default='artifacts/diagnostics')
    parser.add_argument('--documents',type=int,default=32)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+3600
    snapshot=snapshot_download(MODEL,revision=REVISION,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(snapshot,trust_remote_code=False)
    records=[]
    for line in Path(args.data).read_text().splitlines():
        row=json.loads(line)
        ids=tokenizer.encode(row['text'],add_special_tokens=False,truncation=True,max_length=513)
        if len(ids)==513:records.append((row['id'],ids))
    records=records[:args.documents]
    if len(records)!=args.documents or len(records)%4:raise ValueError('Need complete batches')
    model=load_checkpoint(snapshot,'external/T2MLR','cuda')
    original_step=model.step
    arms=['reset_once','donor_once','reset_once_clean_kv']
    metrics=['kl_normal_to_intervention','state_relative_l2','state_cosine_distance','absolute_delta_nll']
    observations={arm:{metric:[] for metric in metrics} for arm in arms}
    manifest={'status':'started','exploratory':True,'motivation':'NLL mean cancellation and hidden-state persistence',
              'model':MODEL,'revision':REVISION,'documents':len(records),'document_ids':[r[0] for r in records],
              'reset_at':128,'length':512,'completed_batches':0}
    def save():
        manifest['elapsed_seconds']=time.monotonic()-started
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save()
    try:
        with (output/'traces.jsonl').open('w') as trace:
            for start in range(0,len(records),4):
                batch=records[start:start+4]
                tokens=torch.tensor([r[1] for r in batch],device='cuda')
                baseline_logp=[];baseline_state=[]
                def capture_normal(token,kv,position,**kwargs):
                    result=original_step(token,kv,position,**kwargs)
                    baseline_logp.append(result.logits[:,-1].float().log_softmax(-1).cpu())
                    baseline_state.append(model.recurrent_cache.float().cpu().clone())
                    return result
                model.step=capture_normal
                normal_nll,donors=rollout(model,tokens,'normal',128,deadline)
                for arm in arms:
                    calls=0
                    batch_metrics={metric:[] for metric in metrics}
                    def capture_perturbed(token,kv,position,**kwargs):
                        nonlocal calls
                        result=original_step(token,kv,position,**kwargs)
                        calls+=1
                        # clean-KV rollout calls clean then perturbed at every position.
                        if arm=='reset_once_clean_kv' and calls%2==1:return result
                        reference=baseline_logp[position].to('cuda')
                        logp=result.logits[:,-1].float().log_softmax(-1)
                        kl=(reference.exp()*(reference-logp)).sum(-1).clamp_min(0)
                        state=model.recurrent_cache.float().squeeze(1)
                        reference_state=baseline_state[position].to('cuda').squeeze(1)
                        relative=(state-reference_state).norm(dim=-1)/reference_state.norm(dim=-1).clamp_min(1e-8)
                        cosine=(1-F.cosine_similarity(state,reference_state,dim=-1)).clamp_min(0)
                        for name,value in zip(metrics[:3],[kl,relative,cosine]):
                            batch_metrics[name].append(value.cpu().numpy())
                        return result
                    model.step=capture_perturbed
                    changed_nll,_=rollout(model,tokens,arm,128,deadline,donors)
                    batch_metrics={k:np.stack(v,axis=1) for k,v in batch_metrics.items() if k!='absolute_delta_nll'}
                    batch_metrics['absolute_delta_nll']=np.abs(changed_nll-normal_nll)
                    for metric,values in batch_metrics.items():observations[arm][metric].append(values)
                    for index,(doc,_) in enumerate(batch):
                        for pos in range(512):
                            trace.write(json.dumps({'document':doc,'arm':arm,'position':pos,
                                        **{k:float(v[index,pos]) for k,v in batch_metrics.items()}})+'\n')
                trace.flush()
                manifest['completed_batches']+=1;save()
        summary={}
        for arm in arms:
            summary[arm]={}
            for metric,arrays in observations[arm].items():
                values=np.concatenate(arrays)
                summary[arm][metric]={f'lag_{a}_{b-1}':document_ci(values[:,128+a:128+b].mean(1),block_size=4 if arm=='donor_once' else 1)
                                      for a,b in [(0,1),(1,8),(8,32),(32,128)]}
                for statistics in summary[arm][metric].values():
                    statistics['mean']=statistics.pop('mean_delta_nll')
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:
        model.step=original_step;save()

if __name__=='__main__':main()

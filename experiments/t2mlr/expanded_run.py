"""Paired, variable-length mechanism study; padding is never scored."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from adapter import load_checkpoint, UPSTREAM
from expanded_prepare import MODELS
from run import rollout, document_ci
from test_mechanism import official_step

@torch.inference_mode()
def validate(model):
    tokens = torch.tensor([[0,45,632,87,199,15,49,831],[0,79,271,95,181,64,231,8]],device='cuda')
    model.clear(); kv = None; expected = []
    for p in range(8):
        out = model.step(tokens[:,p:p+1],kv,p); kv = out.past_key_values
        expected.append((out.logits.clone(), model.recurrent_cache.clone()))
    model.clear()
    model.recurrent_cache = torch.zeros(2,1,model.t2mlr_model.config.hidden_size,device='cuda',dtype=torch.bfloat16)
    step = official_step(model); kv = None; diffs = [0.,0.]
    for p in range(8):
        token = tokens[:,p:p+1]
        out = step(token,torch.full_like(token,2),torch.ones(2,p+1,device='cuda',dtype=torch.long),past_key_values=kv,
                   position_ids=torch.full_like(token,p),cache_position=torch.tensor([p],device='cuda'))
        kv = out.past_key_values
        diffs[0] = max(diffs[0], (out.logits-expected[p][0]).abs().max().item())
        diffs[1] = max(diffs[1], (model.recurrent_cache-expected[p][1]).abs().max().item())
    assert diffs == [0.,0.], diffs
    return dict(max_logit_diff=diffs[0], max_state_diff=diffs[1], documents=2, tokens=8)

def summarize(batches):
    metrics = {}
    for records, losses in batches:
        for i, row in enumerate(records):
            end = len(row['tokens'])-1; reset = row['reset_at']; score = row['score_start']
            for arm, values in losses.items():
                if arm == 'normal': continue
                delta = values[i,:end] - losses['normal'][i,:end]
                windows = {'scored':(score,end)} if arm in ['carry_off','zero_every'] else {
                    f'lag_{a}_{b-1}':(reset+a,reset+b) for a,b in [(0,1),(1,8),(8,32),(32,128)]}
                for window,(a,b) in windows.items():
                    # Require the full stated horizon. Never mix shorter horizons into far means.
                    if b > end or b <= a: continue
                    entry = metrics.setdefault(arm,{}).setdefault(window, {'signed':[], 'absolute':[]})
                    entry['signed'].append(float(delta[a:b].mean()))
                    entry['absolute'].append(float(np.abs(delta[a:b]).mean()))
    return {arm:{window:{'signed':document_ci(v['signed']), 'absolute':document_ci(v['absolute'])}
                 for window,v in windows.items()} for arm,windows in metrics.items()}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=16); p.add_argument('--documents',type=int,default=256)
    p.add_argument('--max-seconds',type=int,default=4200); args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic(); deadline = started+args.max_seconds
    repo,rev = MODELS[args.model]
    rows = [json.loads(x) for x in Path(args.data).read_text().splitlines()][:args.documents]
    assert len(rows) == args.documents and len(set(x['id'] for x in rows)) == len(rows)
    assert len(set(tuple(x['tokens']) for x in rows)) == len(rows)
    # Sort only for padding efficiency; selection already frozen in input file.
    rows.sort(key=lambda x:len(x['tokens']))
    arms = ['normal','reset_once','reset_once_clean_kv','carry_off','zero_every']
    manifest = dict(status='started',model=repo,revision=rev,upstream=UPSTREAM,arguments=vars(args),
                    torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(),completed_documents=0,arms=arms,
                    data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                    code_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
                                 for f in ['expanded_run.py','run.py','adapter.py','test_mechanism.py']})
    batches = []
    def save():
        manifest['elapsed_seconds'] = time.monotonic()-started
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if batches: (out/'summary.json').write_text(json.dumps(summarize(batches),indent=2))
    save()
    try:
        snapshot = snapshot_download(repo,revision=rev,local_files_only=True)
        torch.manual_seed(0)
        model = load_checkpoint(snapshot,'external/T2MLR','cuda')
        manifest['parameter_count'] = sum(p.numel() for p in model.parameters())
        manifest['checkpoint_config'] = json.loads((Path(snapshot)/'config.json').read_text())
        manifest['official_step_validation'] = validate(model); save()
        for start in range(0,len(rows),args.batch_size):
            batch = rows[start:start+args.batch_size]; length = max(len(r['tokens']) for r in batch)
            tokens = torch.tensor([r['tokens']+[0]*(length-len(r['tokens'])) for r in batch],device='cuda')
            reset = torch.tensor([r['reset_at'] for r in batch],device='cuda')
            losses = {}
            for arm in arms:
                losses[arm], _ = rollout(model,tokens,arm,reset,deadline)
            for i,r in enumerate(batch):
                at = r['reset_at']
                for arm in ['reset_once','reset_once_clean_kv']:
                    assert np.array_equal(losses[arm][i,:at],losses['normal'][i,:at]), 'Prefix control failure'
            np.savez_compressed(out/f'batch_{start:04d}.npz', **losses)
            (out/f'batch_{start:04d}.json').write_text(json.dumps(batch))
            batches.append((batch,losses)); manifest['completed_documents'] += len(batch)
            save()
            print(json.dumps(dict(model=args.model,data=args.data,completed=manifest['completed_documents'],seconds=round(time.monotonic()-started,1))),flush=True)
        manifest['status'] = 'complete'
    except TimeoutError as exc: manifest.update(status='budget_exhausted',error=str(exc))
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc)); raise
    finally: save()
    if manifest['status'] != 'complete': raise SystemExit(2)

if __name__ == '__main__': main()

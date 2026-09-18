"""Prepare one audited checkpoint and validate tokenizer on the frozen inputs."""
import argparse, hashlib, json, os, time
from pathlib import Path
import requests
from transformers import AutoTokenizer
import expanded_download
from expanded_prepare import MODELS

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=MODELS);a=p.parse_args()
    import fcntl
    lockdir=Path('artifacts/matrix');lockdir.mkdir(parents=True,exist_ok=True)
    lock=(lockdir/(a.model+'.prepare.lock')).open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    repo,rev=MODELS[a.model]
    rows=json.loads(Path('01_literature/t2mlr_six_checkpoint_audit.json').read_text())
    row=next(r for r in rows if r['repo']==repo and r['revision']==rev)
    f=next(f for f in row['files'] if f['rfilename']=='model.safetensors')
    expanded_download.ASSETS=[('models',repo,rev,f['rfilename'],f['size'],f['lfs']['sha256'])]
    expanded_download.main()
    snap=Path(os.environ['HF_HOME'])/'hub'/('models--'+repo.replace('/','--'))/'snapshots'/rev
    for file in row['files']:
        name=file['rfilename']
        if not name.endswith(('.json','.txt')):continue
        target=snap/name
        if target.exists():continue
        for attempt in range(5):
            try:
                r=requests.get('https://hf-mirror.com/'+repo+'/resolve/'+rev+'/'+name,timeout=45)
                r.raise_for_status();target.write_bytes(r.content);break
            except Exception:
                if attempt==4:raise
                time.sleep(1)
    tok=AutoTokenizer.from_pretrained(snap)
    # All six tokenizers must have identical tokenization machinery and options.
    base=Path(os.environ['HF_HOME'])/'hub'/('models--'+MODELS['982m'][0].replace('/','--'))/'snapshots'/MODELS['982m'][1]
    reference=AutoTokenizer.from_pretrained(base)
    assert tok.backend_tokenizer.to_str()==reference.backend_tokenizer.to_str(), 'Tokenizer machinery differs'
    assert tok.all_special_tokens==reference.all_special_tokens and tok.all_special_ids==reference.all_special_ids
    out=Path('artifacts/matrix');out.mkdir(exist_ok=True,parents=True)
    (out/(a.model+'_preparation.json')).write_text(json.dumps(dict(model=repo,revision=rev,
        weight_sha256=f['lfs']['sha256'],tokenizer_backend_equal=True,
        data_sha256={d:hashlib.sha256(Path('data/expanded/'+d+'.jsonl').read_bytes()).hexdigest() for d in ['wiki','math']}),indent=2))
    print('Prepared and tokenizer-verified',a.model,flush=True)

if __name__=='__main__':main()

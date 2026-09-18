"""Parallel range transfer with exact size/range and SHA256 verification."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import time
import requests

URL='https://hf-mirror.com/JupiterZhu/T2MLR_982M_lstart9_lend24_50B_FineWebEdu/resolve/5061964f36236336153106d4b11155e09ee2daaa/model.safetensors'
URL=URL.replace('https://hf-mirror.com',os.environ.get('T2_DOWNLOAD_ENDPOINT','https://hf-mirror.com'))
SHA='76b5ff218df444c6849cafbb9415710290127a6c3fbaac0592cdfc94872dd545'

def main():
    response=requests.get(URL,headers={'Range':'bytes=0-7'},stream=True,timeout=60)
    response.raise_for_status()
    total=int(response.headers['Content-Range'].split('/')[-1])
    response.close()
    target=Path('/root/autodl-tmp/t2mlr-runtime/weights.parts')
    target.mkdir(exist_ok=True)
    block=64*1024*1024
    def fetch(start):
        end=min(start+block,total)-1
        part=target/str(start)
        if part.exists() and part.stat().st_size==end-start+1:return part
        for attempt in range(3):
            try:
                with requests.get(URL+f'?download=true&part={start}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,90)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':
                        raise ValueError('Server did not honor requested range')
                    with part.open('wb') as f:
                        for chunk in r.iter_content(1024*1024):f.write(chunk)
                if part.stat().st_size!=end-start+1:raise ValueError('Incomplete part')
                print(f'part {start} complete',flush=True)
                return part
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parts=list(pool.map(fetch,range(0,total,block)))
    blob=Path(os.environ['HF_HOME'])/'hub/models--JupiterZhu--T2MLR_982M_lstart9_lend24_50B_FineWebEdu/blobs'/SHA
    temp=blob.with_suffix('.verified-download')
    digest=hashlib.sha256()
    with temp.open('wb') as f:
        for part in parts:
            with part.open('rb') as src:
                while chunk:=src.read(8*1024*1024):
                    f.write(chunk);digest.update(chunk)
    if digest.hexdigest()!=SHA:raise ValueError('SHA256 mismatch')
    temp.replace(blob)
    print(f'Verified SHA256 {SHA}, bytes {total}',flush=True)

if __name__=='__main__':main()

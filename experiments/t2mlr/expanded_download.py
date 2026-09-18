"""Pinned large-file transfer into HF cache, verified by upstream LFS SHA256."""
import concurrent.futures, hashlib, os, time
from pathlib import Path
import requests

ASSETS = [
('datasets','Salesforce/wikitext','b08601e04326c79dfdd32d625aee71d232d685c3','wikitext-103-raw-v1/train-00000-of-00002.parquet',156987808,'74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212'),
('datasets','Salesforce/wikitext','b08601e04326c79dfdd32d625aee71d232d685c3','wikitext-103-raw-v1/train-00001-of-00002.parquet',157088770,'ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720'),
('models','JupiterZhu/T2MLR_135M_lstart13_lend18_10B_FineWebEdu','2bbdb5540e56aaf09b2a5e80e63fcf57d629d3db','model.safetensors',329008232,'2d378d0f4f92a2957e0099d9bf1c77a5d0999956c96540aa84c494d4717c480d')]

def main():
    base=Path(os.environ['HF_HOME'])/'hub'
    endpoint=os.environ.get('T2_DOWNLOAD_ENDPOINT','https://hf-mirror.com')
    for kind,repo,rev,name,size,sha in ASSETS:
        cache=base/(kind+'--'+repo.replace('/','--'));blob=cache/'blobs'/sha
        target=cache/'snapshots'/rev/name
        if not blob.exists():
            parts=base/'expanded_parts'/sha;parts.mkdir(parents=True,exist_ok=True)
            url=endpoint+('/datasets/' if kind=='datasets' else '/')+repo+'/resolve/'+rev+'/'+name
            block=8*1024*1024
            def fetch(start):
                stop=min(start+block,size)-1; part=parts/str(start)
                if part.exists() and part.stat().st_size==stop-start+1:return part
                for attempt in range(4):
                    try:
                        with requests.get(url+f'?download=true&part={start}',headers={'Range':f'bytes={start}-{stop}'},stream=True,timeout=(20,40)) as r:
                            r.raise_for_status()
                            assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{stop}/{size}'
                            with part.open('wb') as f:
                                for chunk in r.iter_content(1024*1024):f.write(chunk)
                        assert part.stat().st_size==stop-start+1
                        print(name,start,'done',flush=True);return part
                    except Exception:
                        if attempt==3:raise
                        time.sleep(1)
            with concurrent.futures.ThreadPoolExecutor(8) as pool: downloaded=list(pool.map(fetch,range(0,size,block)))
            blob.parent.mkdir(parents=True,exist_ok=True);temp=blob.with_suffix('.verified-download');h=hashlib.sha256()
            with temp.open('wb') as f:
                for part in downloaded:
                    data=part.read_bytes();h.update(data);f.write(data)
            assert h.hexdigest()==sha;temp.replace(blob)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.symlink_to(os.path.relpath(blob,target.parent))
        print('Verified',name,sha,flush=True)

if __name__=='__main__':main()

"""Pinned WikiText article reconstruction, no independent paragraph pseudo-replicates."""
import hashlib
import json
import re
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'

def main():
    output = Path('data')
    output.mkdir(exist_ok=True)
    metadata = {'dataset': 'Salesforce/wikitext', 'revision': REVISION,
                'subset': 'wikitext-103-raw-v1', 'license': ['cc-by-sa-3.0','gfdl'],
                'selection': 'reconstruct articles using level-one headings; retain source order; runner selects first length-eligible documents',
                'limitation': 'Possible pretraining overlap; dependency diagnosis only'}
    for split in ['validation', 'test']:
        source = hf_hub_download('Salesforce/wikitext', f'wikitext-103-raw-v1/{split}-00000-of-00001.parquet', repo_type='dataset', revision=REVISION)
        rows = pq.read_table(source)['text'].to_pylist()
        articles, title, chunks = [], None, []
        for text in rows:
            stripped = text.strip()
            if re.match(r'^= [^=].*[^=] =$' , stripped):
                if title is not None:
                    articles.append({'id': split+':'+title, 'text': ''.join(chunks)})
                title, chunks = stripped, [text]
            elif title is not None:
                chunks.append(text)
        if title is not None:
            articles.append({'id': split+':'+title, 'text': ''.join(chunks)})
        path = output/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in articles),encoding='utf-8')
        metadata[split] = {'articles':len(articles),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))

if __name__ == '__main__': main()

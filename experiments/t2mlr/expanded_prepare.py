"""Pinned, seeded expansion. No outcome-dependent sample selection."""
import json, re, hashlib
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from transformers import AutoTokenizer
from adapter import MODEL, REVISION

MODELS = {'982m': (MODEL, REVISION), '135m': ('JupiterZhu/T2MLR_135M_lstart13_lend18_10B_FineWebEdu', '2bbdb5540e56aaf09b2a5e80e63fcf57d629d3db')}
WIKI_REV = 'b08601e04326c79dfdd32d625aee71d232d685c3'
MATH_REV = '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'

def main():
    import pyarrow.parquet as pq
    out = Path('data/expanded'); out.mkdir(parents=True, exist_ok=True)
    tokenizers = []
    for name, (repo, rev) in MODELS.items():
        path = snapshot_download(repo, revision=rev, allow_patterns=['*.json', '*.txt'])
        tokenizers.append(AutoTokenizer.from_pretrained(path))
    tok = tokenizers[0]
    def encode(text):
        ids = tok.encode(text, add_special_tokens=False)
        assert ids == tokenizers[1].encode(text, add_special_tokens=False), 'Tokenizer mismatch'
        return ids
    rng = np.random.default_rng(20260918)
    files = [f'wikitext-103-raw-v1/train-{i:05d}-of-00002.parquet' for i in range(2)]
    articles, title, chunks = [], None, []
    for file in files:
        path = hf_hub_download('Salesforce/wikitext', file, repo_type='dataset', revision=WIKI_REV)
        for text in pq.read_table(path)['text'].to_pylist():
            stripped = text.strip()
            if re.match(r'^= [^=].*[^=] =$', stripped):
                if title is not None: articles.append((title, ''.join(chunks)))
                title, chunks = stripped, [text]
            elif title is not None: chunks.append(text)
    if title is not None: articles.append((title, ''.join(chunks)))
    selected, seen = [], set()
    for idx in rng.permutation(len(articles)):
        title, text = articles[idx]
        if title in seen: continue
        ids = encode(text)
        if len(ids) < 513: continue
        seen.add(title)
        selected.append(dict(id='train:'+title, tokens=ids[:513], reset_at=128, score_start=32, subject='wiki'))
        if len(selected) == 256: break
    assert len(selected) == 256
    (out/'wiki.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    source = hf_hub_download('HuggingFaceH4/MATH-500', 'test.jsonl', repo_type='dataset', revision=MATH_REV)
    rows = [json.loads(x) for x in Path(source).read_text().splitlines()]
    eligible, exclusions = [], []
    for row in rows:
        prompt = encode('Problem:\n'+row['problem']+'\nSolution:\n')
        solution = encode(row['solution'])
        if len(prompt) > 512 or len(solution) < 2:
            exclusions.append(dict(id=row['unique_id'], prompt_tokens=len(prompt), solution_tokens=len(solution)))
            continue
        eligible.append(dict(id=row['unique_id'], tokens=prompt+solution[:256], reset_at=len(prompt)-1,
                             score_start=len(prompt)-1, subject=row['subject'], level=row['level'],
                             prompt_tokens=len(prompt), full_solution_tokens=len(solution)))
    math = [eligible[i] for i in rng.permutation(len(eligible))[:256]]
    assert len(math) == 256
    (out/'math.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in math))
    metadata = dict(seed=20260918, models=MODELS, wiki_revision=WIKI_REV, math_revision=MATH_REV,
                    wiki_articles=len(articles), math_eligible=len(eligible), math_exclusions=exclusions,
                    tokenization_equal=True, documents_per_domain=256,
                    selection='Seeded random permutation; wiki first 256 length-eligible unique articles; math uniform 256 eligible problems',
                    math_prompt='Problem:\\n{problem}\\nSolution:\\n', math_solution_cap=256,
                    sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')})
    (out/'provenance.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    repo, rev = MODELS['135m']
    snapshot_download(repo, revision=rev, allow_patterns=['*.json', '*.safetensors', '*.txt'])

if __name__ == '__main__': main()

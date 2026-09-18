"""Audit saved complete results and the independent supplementary rollout."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
base=root/'artifacts/5090/artifacts'
manifests={name:json.loads((base/name/'manifest.json').read_text()) for name in ['primary','confirmation','diagnostics']}
assert all(m['status']=='complete' and m['completed_batches']==8 for m in manifests.values())
assert manifests['primary']['code_sha256']==manifests['confirmation']['code_sha256']
titles=lambda m:{x.split(':',1)[1] for x in m['document_ids']}
assert not titles(manifests['primary']) & titles(manifests['confirmation'])
assert manifests['primary']['document_ids']==manifests['diagnostics']['document_ids']
lookup={}
for line in (base/'primary/token_traces.jsonl').read_text().splitlines():
    row=json.loads(line)
    lookup[(row['document'],row['position'],row['arm'])]=row['nll']
max_prefix_kl=max_prefix_abs=max_disagreement=0.
count=0
for line in (base/'diagnostics/traces.jsonl').read_text().splitlines():
    row=json.loads(line);count+=1
    doc,pos,arm=row['document'],row['position'],row['arm']
    expected=abs(lookup[(doc,pos,arm)]-lookup[(doc,pos,'normal')])
    max_disagreement=max(max_disagreement,abs(expected-row['absolute_delta_nll']))
    if pos<128:
        max_prefix_kl=max(max_prefix_kl,row['kl_normal_to_intervention'])
        max_prefix_abs=max(max_prefix_abs,row['absolute_delta_nll'])
assert count==32*512*3
assert max_prefix_kl==0 and max_prefix_abs==0
assert max_disagreement<1e-5
report={'complete_batches_each':8,'disjoint_primary_confirmation_titles':True,
        'primary_confirmation_code_hashes_equal':True,'diagnostics_match_primary_document_ids':True,
        'diagnostic_trace_rows':count,'diagnostic_prefix_max_kl':max_prefix_kl,
        'diagnostic_prefix_max_absolute_delta_nll':max_prefix_abs,
        'supplement_vs_primary_max_absolute_nll_difference':max_disagreement,
        'files':{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                 for name in manifests for path in sorted((base/name).glob('*.json*'))}}
(root/'04_evidence/t2mlr_5090_integrity.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))

"""Full trained-weight BF16 validation against the unmodified upstream step."""
import json
import time
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from adapter import MODEL, REVISION, load_checkpoint
from test_mechanism import official_step

@torch.inference_mode()
def main():
    snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    model = load_checkpoint(snapshot, 'external/T2MLR', 'cuda')
    tokens = torch.tensor([[0, 45, 632, 87, 199, 15, 49, 831], [0, 79, 271, 95, 181, 64, 231, 8]],device='cuda')
    kv, expected_logits, expected_states = None, [], []
    for p in range(tokens.shape[1]):
        out = model.step(tokens[:,p:p+1],kv,p)
        expected_logits.append(out.logits.clone())
        expected_states.append(model.recurrent_cache.clone())
        kv = out.past_key_values
    model.clear()
    model.recurrent_cache = torch.zeros(2,1,model.t2mlr_model.config.hidden_size,device='cuda',dtype=torch.bfloat16)
    step = official_step(model)
    kv = None
    max_logit_diff = max_state_diff = 0.
    for p in range(tokens.shape[1]):
        token=tokens[:,p:p+1]
        out = step(token,torch.full_like(token,2),torch.ones(2,p+1,device='cuda',dtype=torch.long),
                   past_key_values=kv,position_ids=torch.full_like(token,p),cache_position=torch.tensor([p],device='cuda'))
        max_logit_diff=max(max_logit_diff,(out.logits-expected_logits[p]).abs().max().item())
        max_state_diff=max(max_state_diff,(model.recurrent_cache-expected_states[p]).abs().max().item())
        kv=out.past_key_values
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'dtype':'bfloat16',
            'tokens_per_document':8,'documents':2,'max_logit_diff':max_logit_diff,'max_state_diff':max_state_diff,
            'strict_weight_load':True,'tied_weights_equal':True,'passed':max_logit_diff==0 and max_state_diff==0}
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/gpu_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert report['passed']

if __name__=='__main__': main()

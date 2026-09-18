"""Teacher-forced causal interventions. No training or benchmark score claims."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from adapter import MODEL, REVISION, UPSTREAM, load_checkpoint

ARMS = ["normal", "reset_once", "donor_once", "reset8", "reset32", "reset128",
        "zero_every", "carry_off", "fusion_off", "fusion_and_carry_off", "reset_once_clean_kv"]


@torch.inference_mode()
def rollout(model, tokens, arm, reset_at, deadline, donor_states=None):
    model.clear()
    kv = clean_kv = None
    clean_state = None
    losses, states = [], []
    length = tokens.shape[1] - 1
    for pos in range(length):
        if time.monotonic() >= deadline:
            raise TimeoutError("Time cap reached; incomplete arm discarded")
        token = tokens[:, pos:pos+1]
        if arm == "reset_once_clean_kv":
            # Counterfactual: discard perturbed KV writes each step, retaining its state.
            perturbed_state = model.recurrent_cache
            kv = copy.deepcopy(clean_kv)
            model.recurrent_cache = clean_state
            clean = model.step(token, clean_kv, pos)
            clean_kv, clean_state = clean.past_key_values, model.recurrent_cache
            model.recurrent_cache = perturbed_state
        prior = model.recurrent_cache
        if prior is None:
            prior = torch.zeros(tokens.shape[0], 1, model.t2mlr_model.config.hidden_size,
                                dtype=next(model.parameters()).dtype, device=tokens.device)
            model.recurrent_cache = prior
        if arm == "normal":
            states.append(prior.detach().cpu().clone())
        periodic = int(arm[5:]) if arm in {"reset8", "reset32", "reset128"} else None
        vector_reset = isinstance(reset_at, torch.Tensor)
        if vector_reset and arm in {"reset_once", "reset_once_clean_kv"}:
            model.recurrent_cache = torch.where((reset_at == pos)[:, None, None], torch.zeros_like(prior), prior)
        should_reset = ((arm in {"reset_once", "reset_once_clean_kv"} and not vector_reset and pos == reset_at)
                        or (periodic is not None and pos > 0 and pos % periodic == 0)
                        or arm == "zero_every")
        if should_reset:
            model.recurrent_cache = torch.zeros_like(prior)
        if arm == "donor_once" and pos == reset_at:
            if tokens.shape[0] < 2 or donor_states is None:
                raise ValueError("Donor intervention requires >=2 independent documents")
            donor = donor_states[pos].to(tokens.device).roll(1, dims=0)
            # Preserve recipient norm; test content against a matched-magnitude control.
            scale = prior.float().norm(dim=-1, keepdim=True) / donor.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
            model.recurrent_cache = donor * scale.to(donor.dtype)
        out = model.step(token, kv, pos,
                         read=arm not in {"fusion_off", "fusion_and_carry_off"},
                         carry=arm not in {"carry_off", "fusion_and_carry_off"})
        kv = out.past_key_values
        nll = F.cross_entropy(out.logits[:, -1].float(), tokens[:, pos+1], reduction="none")
        if not torch.isfinite(nll).all() or not torch.isfinite(model.recurrent_cache).all():
            raise FloatingPointError(f"Nonfinite output at {arm}:{pos}")
        losses.append(nll.cpu())
    return torch.stack(losses, dim=1).numpy(), states


def document_ci(values, seed=0, block_size=1):
    values = np.asarray(values, dtype=float)
    units = values.reshape(-1, block_size).mean(1)
    rng = np.random.default_rng(seed)
    means = units[rng.integers(len(units), size=(2000, len(units)))].mean(1)
    return {"mean_delta_nll": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "documents": len(values), "bootstrap_units": len(units), "bootstrap_block_size": block_size,
            "inference_valid": len(values) >= 32 and len(units) >= 8}


def summarize(losses, reset_at, batch_size=1):
    result = {}
    for arm, values in losses.items():
        if arm == "normal":
            continue
        delta = values - losses["normal"]
        windows = {"post_warmup": (32, delta.shape[1])}
        if arm in {"reset_once", "donor_once", "reset_once_clean_kv"}:
            windows = {f"lag_{a}_{b-1}": (reset_at+a, min(reset_at+b, delta.shape[1]))
                       for a, b in [(0, 1), (1, 8), (8, 32), (32, 128)]}
        result[arm] = {name: document_ci(delta[:, start:end].mean(1),
                                       block_size=batch_size if arm == "donor_once" else 1)
                       for name, (start, end) in windows.items() if end > start}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="JSONL with unique id/text; one independent document per row")
    p.add_argument("--output", required=True, help="New directory; refuses overwrite")
    p.add_argument("--upstream", default="external/T2MLR")
    p.add_argument("--device", default="cuda")
    p.add_argument("--documents", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--reset-at", type=int, default=128)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=["normal", "reset_once", "donor_once", "reset32", "carry_off"])
    p.add_argument("--max-seconds", type=int, default=1800, help="Per-run wall cap, <=18000; reserve time across runs manually")
    args = p.parse_args()
    if not 0 < args.max_seconds <= 18000 or args.batch_size < 2 or args.documents < 2:
        p.error("Require 0 < max-seconds <=18000, documents>=2, batch-size>=2")
    if not 32 < args.reset_at < args.length or args.length > 8192:
        p.error("Require 32 < reset-at < length <=8192")
    if args.documents % args.batch_size:
        p.error("documents must be divisible by batch-size (no partial donor batches)")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.max_seconds
    manifest = {"status": "started", "model": MODEL, "revision": REVISION, "upstream": UPSTREAM,
                "arguments": vars(args), "torch": torch.__version__, "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "completed_batches": 0, "evidence_scope": "checkpoint dependency; no architectural superiority claim"}
    manifest["transformers"] = transformers.__version__
    manifest["code_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ["run.py", "adapter.py"]}
    manifest["cuda_runtime"] = torch.version.cuda
    def save_manifest():
        manifest["elapsed_seconds"] = time.monotonic() - started
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_manifest()
    try:
        # Always resolves a pinned snapshot; never runs remote custom Python.
        snapshot = snapshot_download(MODEL, revision=REVISION,
                   allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
        records, seen = [], set()
        for line in Path(args.data).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in seen:
                raise ValueError("Repeated document id")
            seen.add(row["id"])
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if len(ids) >= args.length + 1:
                records.append((row["id"], ids[:args.length+1]))
        if len(records) < args.documents:
            raise ValueError(f"Only {len(records)} documents meet minimum token length")
        records = records[:args.documents]
        if len({tuple(ids) for _, ids in records}) != len(records):
            raise ValueError("Duplicate token windows; donor documents must differ")
        manifest["document_ids"] = [r[0] for r in records]
        manifest["eligible_documents"] = len(records)
        torch.manual_seed(0)
        model = load_checkpoint(snapshot, args.upstream, args.device)
        manifest["device_name"] = torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU"
        arms = ["normal"] + [a for a in dict.fromkeys(args.arms) if a != "normal"]
        aggregated = {arm: [] for arm in arms}
        with (output / "token_traces.jsonl").open("w", encoding="utf-8") as trace:
            for start in range(0, len(records), args.batch_size):
                batch = records[start:start+args.batch_size]
                tokens = torch.tensor([r[1] for r in batch], device=args.device)
                batch_losses = {}
                donor_states = None
                # Commit only complete paired batches; never summarize mismatched subsets.
                for arm in arms:
                    values, states = rollout(model, tokens, arm, args.reset_at, deadline, donor_states)
                    batch_losses[arm] = values
                    if arm == "normal":
                        donor_states = states
                for arm, values in batch_losses.items():
                    aggregated[arm].append(values)
                    for doc_index, (doc_id, ids) in enumerate(batch):
                        for pos, nll in enumerate(values[doc_index]):
                            trace.write(json.dumps(dict(document=doc_id, arm=arm, position=pos,
                                       target_token=ids[pos+1], nll=float(nll))) + "\n")
                trace.flush()
                manifest["completed_batches"] += 1
                joined = {a: np.concatenate(v) for a, v in aggregated.items()}
                (output / "summary.json").write_text(json.dumps(summarize(joined, args.reset_at, args.batch_size), indent=2), encoding="utf-8")
                save_manifest()
        manifest["status"] = "complete"
    except TimeoutError as exc:
        manifest.update(status="budget_exhausted", error=str(exc))
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        save_manifest()


if __name__ == "__main__":
    main()

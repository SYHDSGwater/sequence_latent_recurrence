"""Exact inference adapter. Reuses upstream mixer; no training approximation."""
import importlib.util
import json
import subprocess
import types
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file

UPSTREAM = "2871fa204e4cc145fd67d7d1a82b319c2010e9dc"
MODEL = "JupiterZhu/T2MLR_982M_lstart9_lend24_50B_FineWebEdu"
REVISION = "5061964f36236336153106d4b11155e09ee2daaa"


def upstream_modules(root):
    root = Path(root).resolve()
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if sha != UPSTREAM:
        raise ValueError(f"Expected upstream {UPSTREAM}, got {sha}")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise ValueError("Upstream checkout must be clean")
    # Package initializer imports absent training files. Load only self-contained modules.
    package = types.ModuleType("t2mlr_audit_upstream")
    package.__path__ = [str(root / "src/t2mlr_wrapper")]
    sys.modules[package.__name__] = package
    loaded = []
    for name in ["t2mlr_gate_zoo", "block_wrapper"]:
        spec = importlib.util.spec_from_file_location(package.__name__ + "." + name,
                    Path(package.__path__[0]) / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


class ExactAdapter(nn.Module):
    def __init__(self, config, upstream, dtype=torch.float32):
        super().__init__()
        self.settings = config
        if config.get("recurrent_mixing_module_name") != "gated" or config.get("recurrent_skip_to_l_end", False):
            raise ValueError("Adapter supports this checkpoint's gated/no-skip architecture only")
        zoo, blocks = upstream_modules(upstream)
        base = LlamaConfig.from_dict(config["base_config"])
        base._attn_implementation = "eager"
        self.t2mlr_model = LlamaForCausalLM(base).to(dtype=dtype)
        self.l_start = config["l_start"]
        self.l_end = config["l_end"] % base.num_hidden_layers
        mixer = zoo.get_t2mlr_mixing_module_class("gated").from_config(
            types.SimpleNamespace(to_dict=lambda: dict(config)), hidden_size=base.hidden_size, dtype=dtype)
        layers = self.t2mlr_model.model.layers
        layers[self.l_start] = blocks.BlockWrapper(layers[self.l_start], mixer)
        self.recurrent_cache = None

    @property
    def layers(self):
        return self.t2mlr_model.model.layers

    def clear(self):
        self.recurrent_cache = None
        self.layers[self.l_start].set_recurrent_input(None, None, None)

    def step(self, token, kv, position, read=True, carry=True):
        cfg = self.settings
        prior = self.recurrent_cache
        if prior is None:
            prior = torch.zeros(token.shape[0], 1, self.t2mlr_model.config.hidden_size,
                                device=token.device, dtype=next(self.parameters()).dtype)
        layer = self.layers[self.l_start]
        # read=False bypasses fusion entirely, not merely the projected recurrent term.
        layer.set_recurrent_input(prior if read else None, torch.full_like(token, 2), None)
        captured = []
        hook = self.layers[self.l_end].register_forward_hook(
            lambda m, i, o: captured.append(o[0] if isinstance(o, tuple) else o))
        try:
            out = self.t2mlr_model(input_ids=token, past_key_values=kv, use_cache=True,
                    attention_mask=torch.ones(token.shape[0], position + 1, device=token.device, dtype=torch.long),
                    position_ids=torch.full_like(token, position),
                    cache_position=torch.tensor([position], device=token.device))
        finally:
            hook.remove()
            layer.set_recurrent_input(None, None, None)
        state = captured[0][:, -1:, :]
        if cfg.get("recurrent_residual_to_recurrent_cache", False):
            if carry:
                state = state + cfg.get("recurrent_residual_to_recurrent_cache_weight", 1.0) * prior
            # Retain normalization when removing carry: isolate the additive state term.
            if cfg.get("recurrent_residual_to_recurrent_cache_post_norm", False):
                eps = cfg.get("recurrent_residual_to_recurrent_cache_post_norm_eps", 1e-6)
                clamp = max(1., cfg.get("recurrent_residual_to_recurrent_cache_post_norm_clamp", 5.))
                scale = 1.0 / torch.sqrt(state.float().square().mean(-1, keepdim=True) + eps)
                state = state * scale.clamp(1 / clamp, clamp).to(state.dtype)
        self.recurrent_cache = state
        return out


def load_checkpoint(directory, upstream, device):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    model = ExactAdapter(config, upstream, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    weights = load_file(str(directory / "model.safetensors"))
    if config["base_config"].get("tie_word_embeddings", False):
        if not torch.equal(weights["t2mlr_model.lm_head.weight"],
                           weights["t2mlr_model.model.embed_tokens.weight"]):
            raise ValueError("Checkpoint declares tied embeddings but stores unequal tensors")
    # Refuse silent randomly initialized parameters, unlike upstream strict=False loader.
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()

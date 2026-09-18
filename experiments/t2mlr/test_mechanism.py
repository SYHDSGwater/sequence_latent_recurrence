"""CPU tests of execution semantics, not evidence about the trained model."""
import ast
import copy
import types
from pathlib import Path

import torch
from adapter import ExactAdapter

ROOT = Path(__file__).resolve().parents[2] / "external/T2MLR"


def tiny():
    torch.manual_seed(17)
    config = dict(base_config=dict(vocab_size=41, hidden_size=16, intermediate_size=32,
                  num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=8, max_position_embeddings=128, tie_word_embeddings=True),
                  l_start=1, l_end=-1, recurrent_mixing_module_name="gated",
                  use_rezero_residual=True, rezero_gamma_input_gate_init=0.3,
                  rezero_gamma_recurrent_gate_init=0.4,
                  recurrent_residual_to_recurrent_cache=True,
                  recurrent_residual_to_recurrent_cache_post_norm=True,
                  connection_detach=False)
    model = ExactAdapter(config, ROOT).eval()
    # Nonzero recurrent coefficients ensure tests cannot pass through a dormant path.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "rezero_gamma" in name:
                parameter.fill_(0.4)
    return model


def official_step(model):
    tree = ast.parse((ROOT / "src/t2mlr_wrapper/t2mlr_wrapper.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "T2MLRWrapper")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "simple_recurrent_forward")
    # Run the unmodified official method, bypassing only broken package imports.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(ROOT), "exec"), namespace)
    model.config = types.SimpleNamespace(**model.settings)
    model._coerce_attention_mask = lambda mask: mask
    model.set_recurrent_input = lambda block_id, recurrent_embedding, control_flows, mixing_module_log_buffer: model.layers[block_id].set_recurrent_input(recurrent_embedding, control_flows, mixing_module_log_buffer)
    model.reset_recurrent_input = lambda: model.layers[model.l_start].set_recurrent_input(None, None, None)
    return types.MethodType(namespace["simple_recurrent_forward"], model)


@torch.inference_mode()
def test_official_single_step_parity():
    model = tiny()
    reference = copy.deepcopy(model)
    step = official_step(reference)
    tokens = torch.tensor([[3, 5, 8, 1, 6], [2, 9, 7, 4, 3]])
    reference.recurrent_cache = torch.zeros(2, 1, 16)
    kv = ref_kv = None
    for position in range(tokens.shape[1]):
        token = tokens[:, position:position + 1]
        output = model.step(token, kv, position)
        expected = step(token, torch.full_like(token, 2), torch.ones(2, position + 1, dtype=torch.long),
                        past_key_values=ref_kv, position_ids=torch.full_like(token, position),
                        cache_position=torch.tensor([position]))
        torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)
        torch.testing.assert_close(model.recurrent_cache, reference.recurrent_cache, rtol=0, atol=0)
        kv, ref_kv = output.past_key_values, expected.past_key_values


@torch.inference_mode()
def test_intervention_has_effect_and_clear_is_reproducible():
    model = tiny()
    tokens = torch.tensor([[3, 5, 8, 1, 6]])
    def rollout(reset=False, carry=True):
        model.clear()
        kv, outputs = None, []
        for pos in range(5):
            if reset and pos == 2:
                model.recurrent_cache.zero_()
            out = model.step(tokens[:, pos:pos+1], kv, pos, carry=carry)
            outputs.append(out.logits.clone())
            kv = out.past_key_values
        return torch.cat(outputs, 1)
    baseline = rollout()
    changed = rollout(reset=True)
    torch.testing.assert_close(baseline[:, :2], changed[:, :2], rtol=0, atol=0)
    assert not torch.allclose(baseline[:, 2:], changed[:, 2:], atol=1e-7, rtol=0)
    assert not torch.allclose(baseline, rollout(carry=False), atol=1e-7, rtol=0)
    torch.testing.assert_close(baseline, rollout(), rtol=0, atol=0)


def test_runner_controls_and_document_statistics():
    import time
    import numpy as np
    from run import rollout, summarize
    model = tiny()
    tokens = torch.tensor([[3, 5, 8, 1, 6, 9], [2, 9, 7, 4, 3, 8]])
    deadline = time.monotonic() + 60
    baseline, states = rollout(model, tokens, "normal", 2, deadline)
    losses = {"normal": baseline}
    for arm in ["reset_once", "donor_once", "reset_once_clean_kv"]:
        values, _ = rollout(model, tokens, arm, 2, deadline, states)
        np.testing.assert_array_equal(values[:, :2], baseline[:, :2])
        assert np.max(np.abs(values[:, 2:] - baseline[:, 2:])) > 1e-6
        losses[arm] = values
    # Fusion bypass makes carry invisible: negative control for the decomposition.
    bypass, _ = rollout(model, tokens, "fusion_off", 2, deadline)
    both, _ = rollout(model, tokens, "fusion_and_carry_off", 2, deadline)
    np.testing.assert_array_equal(bypass, both)
    report = summarize(losses, 2, batch_size=2)
    assert report["reset_once"]["lag_0_0"]["documents"] == 2
    assert not report["reset_once"]["lag_0_0"]["inference_valid"]
    assert report["donor_once"]["lag_0_0"]["bootstrap_units"] == 1


def test_deadline_prevents_rollout():
    import time
    import pytest
    from run import rollout
    with pytest.raises(TimeoutError):
        rollout(tiny(), torch.tensor([[1, 2], [3, 4]]), "normal", 1, time.monotonic()-1)


def test_variable_reset_matches_individual_rollouts():
    import time
    import numpy as np
    from run import rollout
    model = tiny()
    tokens = torch.tensor([[3,5,8,1,6,9],[2,9,7,4,3,8]])
    reset = torch.tensor([1,3])
    for arm in ['reset_once','reset_once_clean_kv']:
        batched,_ = rollout(model,tokens,arm,reset,time.monotonic()+60)
        for i in range(2):
            single,_ = rollout(model,tokens[i:i+1],arm,int(reset[i]),time.monotonic()+60)
            np.testing.assert_allclose(batched[i],single[0],atol=1e-6,rtol=1e-6)

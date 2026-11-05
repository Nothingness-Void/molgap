"""Synthetic CPU checks of the frozen four-arm query-readout construction."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from molgap.family_training_kernel import load_family_model, make_family_optimizer, parameter_groups_for_recipe
from molgap.family_training_recipe import FamilyRecipe
from molgap.family_training_runner import _recipe_configuration
from molgap.gptrans_execution_backend import prepare_layout, prepared_forward
from molgap.gptrans_execution_hotpath import BoundForeachEMA
from molgap.gptrans_variants import normalize_pair
from molgap.training_reproducibility import atomic_torch_save, capture_rng_state, restore_rng_state
from molgap.v4_runtime import read_frozen_state_artifact

ROOT = Path(__file__).resolve().parents[2]
INITIAL = Path('D:/molgap-artifacts/initializations/structural-bias-norm-100k/gptrans.pt')
RECIPE = ROOT / 'experiments/pcqm_gptrans_norm_ema_100k_20261104/generated/92cb4884c511adb30ed6480456082af2054ec917a8f7d09f0c877c83b8c17bc2/uniform_wd_multi_ema/recipe.json'
MODES = ('degree_bond_local_pair_update_norm_ema999', 'bond_pairnorm_full6',
         'bond_pairnorm_query6', 'bond_pairnorm_query8')


def graph(n, atom):
    x = torch.zeros(n, 9, dtype=torch.long)
    x[:, 0] = atom
    source = torch.arange(n - 1)
    edges = torch.stack((torch.cat((source, source + 1)), torch.cat((source + 1, source))))
    return Data(x=x, edge_index=edges, edge_attr=torch.zeros(edges.shape[1], 3, dtype=torch.long))


def exact(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    else:
        assert left == right


def reference(block, query, pair, padding, memory):
    # Full query-key matrix; every key/value stays at the split-point memory.
    core, attn = block.core, block.core.attention
    size, keys, channels = memory.shape
    full_query = torch.cat((query, memory[:, 1:]), 1)
    q = attn.qkv(core.node_norm1(full_query)).reshape(size, keys, 3, attn.num_heads, attn.head_channels)
    kv = attn.qkv(core.node_norm1(memory)).reshape(size, keys, 3, attn.num_heads, attn.head_channels)
    q = q[:, :, 0].transpose(1, 2)
    k, v = kv[:, :, 1].transpose(1, 2), kv[:, :, 2].transpose(1, 2)
    full_pair = pair.expand(-1, -1, keys, -1).clone()
    logits = q @ k.transpose(-2, -1) * attn.scale + attn.pair_to_attention(full_pair)
    weights = torch.softmax(logits.masked_fill(padding, -torch.inf), -1)
    update = (weights @ v).transpose(1, 2).reshape(size, keys, channels)
    pair_update = attn.attention_to_pair(weights + logits)
    pair_weights = torch.softmax(pair_update.masked_fill(padding, -torch.inf), -1)
    update = attn.output(update + attn.pair_to_node((pair_weights * pair_update).sum(-1).transpose(1, 2)))
    nodes = full_query + core.attention_scale * update
    nodes = nodes + core.ffn_scale * core.ffn(core.node_norm2(nodes))
    return nodes[:, :1], (full_pair + normalize_pair(pair_update))[:, :, :1]


def run(report):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261105)
    record = json.loads(RECIPE.read_text())
    record.pop('auxiliary_ema_decays')
    record['execution_policy'].pop('auxiliary_ema')
    base = _recipe_configuration(record)
    for group in ('ad', 'bc'):
        declared = _recipe_configuration(json.loads((ROOT / f'experiments/pcqm_gptrans_query_readout_{group}_100k_20261105/reference_recipe.json').read_text()))
        expected = base.to_dict()
        expected['mode'] = declared.mode
        assert expected == declared.to_dict()
    record = base.to_dict()
    frozen, identity = read_frozen_state_artifact(INITIAL, expected_state_sha256=base.initialization_sha256)
    report.update(torch_version=str(torch.__version__), device='cpu', threads=1,
                  initialization=identity, synthetic_graph_nodes=[3, 5], real_data_or_roles_accessed=False,
                  checkpoint_scope='model + owner AdamW + owner EMA + owner RNG serialized roundtrip; no Session/scheduler resume')
    batch = prepare_layout(Batch.from_data_list([graph(3, 5), graph(5, 7)]))
    solo = prepare_layout(Batch.from_data_list([graph(3, 5)]))
    changed = prepare_layout(Batch.from_data_list([graph(3, 5), graph(5, 12)]))
    for mode in MODES:
        began = time.perf_counter()
        recipe = FamilyRecipe.from_dict({**record, 'mode': mode})
        model, _ = load_family_model(recipe, INITIAL)
        result = report['arms'][mode] = {'parameters': sum(p.numel() for p in model.parameters())}
        state = model.state_dict()
        exact(state, {key: frozen[key] for key in state})
        rates = [block.core.drop_path.probability for block in model.blocks]
        assert rates == torch.linspace(0, .1, 12).tolist()[:len(model.blocks)]
        result.update(frozen_tensors_exact=True, retained_drop_path=rates)
        model.eval()
        with torch.no_grad():
            forward_began = time.perf_counter()
            output = prepared_forward(model, batch)
            result['cpu_single_batch_eval_forward_seconds'] = time.perf_counter() - forward_began
            solo_error = float((output[:1] - prepared_forward(model, solo)).abs().max())
            isolation_error = float((output[:1] - prepared_forward(model, changed)[:1]).abs().max())
        assert solo_error < 2e-5 and isolation_error == 0
        result.update(padding_batch_max_abs=solo_error, other_graph_max_abs=isolation_error)
        query_blocks = list(model.blocks[getattr(model, 'full_layers', len(model.blocks)):])
        if query_blocks:
            shapes, memories = [], []
            handles = []
            for block in query_blocks:
                handles.append(block.core.attention.attention_dropout.register_forward_pre_hook(
                    lambda module, args: shapes.append(list(args[0].shape))))
            def retain_memory(module, args):
                args[3].retain_grad()
                memories.append(args[3])
            handles.append(query_blocks[0].register_forward_pre_hook(retain_memory))
        optimizer = make_family_optimizer(recipe, parameter_groups_for_recipe(recipe, model))
        ema = BoundForeachEMA(model, recipe.ema_decay)
        def step():
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = prepared_forward(model, batch)
            loss = F.smooth_l1_loss(prediction, prediction.new_tensor([.2, -.3]))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
            optimizer.step()
            ema.update(model)
            return float(loss.detach())
        result['first_synthetic_loss'] = step()
        result['finite_forward_backward'] = True
        if query_blocks:
            gradient = memories[0].grad
            assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
            assert torch.count_nonzero(gradient[:, 1:])
            result['split_memory_gradient_abs_sum'] = float(gradient.abs().sum())
            result['split_atom_memory_gradient_abs_sum'] = float(gradient[:, 1:].abs().sum())
            result['query_attention_shapes'] = shapes
            assert len(shapes) == len(query_blocks) and all(s == [2, 8, 1, 6] for s in shapes)
            for handle in handles:
                handle.remove()
            # A standalone double-precision evaluation distinguishes evolving query
            # column zero from the frozen memory column zero, including masked keys.
            block = deepcopy(query_blocks[0]).double().eval()
            memory = torch.randn(2, 6, 256, dtype=torch.double)
            query = memory[:, :1] + .17
            pair = torch.randn(2, 32, 1, 6, dtype=torch.double)
            padding = torch.zeros(2, 1, 1, 6, dtype=torch.bool)
            padding[0, :, :, 4:] = True
            with torch.no_grad():
                actual = block(query, pair, padding, memory)
                expected = reference(block, query, pair, padding, memory)
                error = max(float((a - b).abs().max()) for a, b in zip(actual, expected))
                altered_memory, altered_pair = memory.clone(), pair.clone()
                altered_memory[0, 4:] = torch.randn_like(altered_memory[0, 4:]) * 100
                altered_pair[0, :, :, 4:] += 100
                altered = block(query, altered_pair, padding, altered_memory)
                padding_error = max(float((actual[0] - altered[0]).abs().max()),
                    float((actual[1][0, :, :, :4] - altered[1][0, :, :, :4]).abs().max()))
            assert error < 1e-11 and padding_error < 1e-11
            result.update(full_matrix_frozen_memory_reference_max_abs=error,
                          masked_key_perturbation_valid_output_max_abs=padding_error)
        with tempfile.TemporaryDirectory(prefix='query-readout-check-', dir='D:/molgap-scratch') as directory:
            path = Path(directory) / 'checkpoint.pt'
            atomic_torch_save(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'ema': ema.state_dict(), 'rng': capture_rng_state()})
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            result['uninterrupted_next_loss'] = step()
            expected_model = deepcopy(model.state_dict())
            expected_optimizer = deepcopy(optimizer.state_dict())
            expected_ema = deepcopy(ema.state_dict())
            expected_rng = torch.get_rng_state()
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            ema.load_state_dict(checkpoint['ema'])
            restore_rng_state(checkpoint['rng'])
            result['restored_next_loss'] = step()
            exact(model.state_dict(), expected_model)
            exact(optimizer.state_dict(), expected_optimizer)
            exact(ema.state_dict(), expected_ema)
            exact(torch.get_rng_state(), expected_rng)
        result.update(serialized_next_step_exact=True, cpu_check_wall_seconds=time.perf_counter() - began)
        print(json.dumps({'mode': mode, **result}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'arms': {}, 'status': 'running'}
    began = time.perf_counter()
    try:
        run(report)
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - began
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')

"""Read-only checkpoint diagnosis; CPU observations are not native replay qualification."""
import json
import sys
from pathlib import Path

owner = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(owner / 'src'))
import torch
from molgap.family_training_runner import _recipe_configuration
from molgap.training.core import evaluate_development
from molgap.family_training_kernel import load_family_model
from molgap.family_model_ops import k1_forward
from molgap.k1_execution_layout import cpu_layout_context
from molgap.family_training_data import load_training_data
from molgap.family_training_loaders import FamilyLoaders
from molgap.gptrans_execution_tuning import ExecutionTuning

root = Path(sys.argv[1])
torch.set_num_threads(4)
state = torch.load(root / 'last_checkpoint.pt', map_location='cpu', weights_only=True)
recipe = _recipe_configuration(json.loads((root.parent / 'experiment/swiglu/training_contract.json').read_text()))
data = load_training_data(Path('D:/molgap-data'), recipe.dataset)
model, _ = load_family_model(recipe, Path('D:/molgap-artifacts/initializations/structural-bias-norm-100k/k1.pt'))
model.load_state_dict(state['model'], strict=True)
loaders = FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0, development_workers=0))
from itertools import islice
batch = next(islice(iter(loaders.dev()), 28, None))
report = {'torch': torch.__version__, 'device': 'cpu', 'optimizer_steps': state['optimizer_step'],
          'cursor': {k: v for k, v in state['cursor'].items() if k not in ('pending_absolute', 'optimizer_state_ids')},
          'source_first': int(batch.source_idx[0]), 'source_last': int(batch.source_idx[-1]),
          'target_finite': bool(torch.isfinite(batch.y).all()), 'rows': int(batch.num_graphs),
          'nonfinite_checkpoint_tensors': [k for k, v in state['model'].items() if not torch.isfinite(v).all()],
          'runs': {}}

def stats(value):
    finite = bool(torch.isfinite(value).all())
    return {'shape': list(value.shape), 'finite': finite,
            'max_abs': float(value.abs().max()) if finite else None,
            'rms': float(value.double().square().mean().sqrt()) if finite else None}

for mode in ('eval', 'bn_batch_stats'):
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    rows = []
    first_nonfinite = []
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and mode == 'bn_batch_stats':
            module.train()
        if name.startswith(('local_blocks.', 'neural_atom_mixers.', 'head.')):
            def capture(module, inputs, output, name=name):
                if isinstance(output, torch.Tensor):
                    if not torch.isfinite(output).all() and not first_nonfinite:
                        first_nonfinite.append({'module': name, 'input': stats(inputs[0]), 'output': stats(output)})
                    if (name.startswith('neural_atom_mixers.') and name.endswith('node_norm')) or (name.startswith('local_blocks.') and name.endswith('norm3')):
                        rows.append({'module': name, 'input': stats(inputs[0]), 'output': stats(output)})
            handles.append(module.register_forward_hook(capture))
    predictions = []
    def forward(model, batch):
        with cpu_layout_context(model, batch, torch.device("cpu")):
            value = k1_forward(model, batch)
        predictions.append(value)
        return value
    guard_error = None
    try:
        evaluated = evaluate_development(model, [batch], forward=forward, rows=batch.num_graphs,
            source_start=int(batch.source_idx[0]), mean=recipe.target_mean_eV, std=recipe.target_std_eV)
    except ValueError as error:
        guard_error = str(error)
    prediction = predictions[0].reshape(-1) * recipe.target_std_eV + recipe.target_mean_eV
    for handle in handles:
        handle.remove()
    report['runs'][mode] = {'prediction': stats(prediction), 'mae_eV': None if guard_error else evaluated['mae_eV'], 'guard_error': guard_error,
                            'layers': rows, 'first_nonfinite': first_nonfinite}
    print(json.dumps({'mode': mode, **{k:v for k,v in report['runs'][mode].items() if k != 'layers'}}, allow_nan=True), flush=True)
model.load_state_dict(state['model'], strict=True)
report['batchnorm'] = {name: {'running_mean': stats(module.running_mean), 'running_var': stats(module.running_var),
                              'variance_min': float(module.running_var.min()), 'batches': int(module.num_batches_tracked)}
                      for name, module in model.named_modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
assert report['runs']['eval']['guard_error'] == 'Development predictions non-finite: count=1; source_idx=[103636]'
assert report['runs']['bn_batch_stats']['guard_error'] is None
(root/'diagnostic_report.json').write_text(json.dumps(report, indent=2, allow_nan=True)+'\n', encoding='utf-8')
loaders.close()
print(json.dumps({k:v for k,v in report.items() if k not in ('runs','batchnorm')}, allow_nan=True), flush=True)

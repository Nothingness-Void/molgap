# Four modular experiment layers

This page owns the extension interfaces. The [workflow](EXPERIMENT_WORKFLOW.md)
owns experiment routing; [research protocol](RESEARCH_PROTOCOL.md)
owns scientific comparison. These local tools grant no compute or role authority.

| Layer | Owner | Boundary |
|---|---|---|
| Substrate and addon | [model_substrate.py](../../src/molgap/model_substrate.py), existing Spec/execution registries | Model hooks and parameter ownership; no training loop |
| Training and state | [shared core](FAMILY_TRAINING_CORE.md), [family_training_state.py](../../src/molgap/family_training_state.py) | Optimizer/EMA order and explicit Live/Raw/Clean actions; no platform submission |
| Fast local sandbox | [fast_sandbox.py](../../src/molgap/fast_sandbox.py) | One CPU batch and memory roundtrip; no cache, native qualification or scientific result |
| Source and platform bridge | [source_dependencies.py](../../src/molgap/source_dependencies.py), existing packager and [platforms](../../platforms/README.md) | Reviewed import closure and prepared inputs; platform skills own POST/reconciliation |

## 1. Attach a small addon

Keep scientific identity in `AddonContract` and `TrainingAddon`; these remain
the single declaration/execution registry. Do not create a competing registry.
A reviewed generic `apply_addon(model, config)` mounts an independent module
through `ModelSubstrate`. New slot hooks require no trainer mode branch.

`SlotBinding(family, SlotName, layer)` supports `PRE_MESSAGE`,
`POST_AGGREGATION`, and `PRE_READOUT`. Message/aggregation layers are one-based;
readout has no layer. K1 payloads carry sparse node/edge/aggregation context;
GPTrans payloads carry dense node/pair/mask context. Their types are distinct.

Example model delta:

```python
import torch
from torch import nn
from molgap.model_substrate import ModelSubstrate, ReadoutInput, SlotBinding, SlotName

class ReadoutGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))

    def forward(self, frame):
        return ReadoutInput(frame.embedding * self.gain)

def apply_addon(model, config):
    owner = getattr(model, "_molgap_substrate", None) or ModelSubstrate(model, config["family"])
    owner.attach(config["name"], ReadoutGain(), SlotBinding(config["family"], SlotName.PRE_READOUT))
```

Freeze the family's supported typed config and source through the
[addon guide](EXPERIMENT_ADDON_GUIDE.md); this example adds no compute authority.

Weights serialize under `_molgap_addons.<name>`. Ordered attachments may compose;
duplicate names/modules and incompatible contexts fail. Detach removes hooks
and weights. Configure before optimizer construction, then seal. Registered
generic construction seals a root-attached substrate after addons and initialization;
wrapper factories must explicitly seal nested substrates. A sealed substrate
rejects attachment changes. Reconstruct attachments before loading
the tensor state. Retained scientific modes keep their frozen compatibility owner.

## 2. Calibrate and retain Clean EMA

Reuse the existing single update loop and EMA owner. Ordinary normalized Gap
uses one forward; historical two-forward objectives retain their declared loss.
Do not add a second EMA update after `kernel.step`.

`FamilyTrainingState(live_model, raw_ema, clean_factory=...)` owns an independent
Clean model. `CalibrationPolicy` binds accepted input/training-role identities,
actual source-row order SHA, exact batch size and a finite batch budget. Only
the declared training prefix is consumed. LayerNorm-only models have no tracked
BN to clean and must not claim a successful Clean-BN pass.

`calibrate_and_evaluate(policy, batches, forward=..., source_indices=...,
evaluator=...)` performs calibration before evaluation. It disables dropout,
updates only tracked BN buffers, and retains those buffers on success. Failure
rolls back; modes, momentum and global RNG are restored. Live and Raw EMA remain
unchanged. Raw updates invalidate Clean, including updates outside this owner.

The shared kernel's `forward(batch, model=clean_model)` reuses the owning family
forward without swapping Live parameters. `state_dict`/`load_state_dict` and
atomic `save`/`load` retain complete Clean buffers, calibration identity and
staleness. This is an optional checkpoint component: the session still owns
optimizer/scheduler/RNG/cursor IO. A new calibrated training contract must bind
its policy and selection/checkpoint adapter before remote release. Existing
recipes do not enable calibration implicitly.

## 3. Check one batch locally

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m molgap.experiment_cli sandbox --family neural_atom_k1
& $python -m molgap.experiment_cli sandbox --family gptrans_t
& $python -m molgap.experiment_cli sandbox --spec $spec --arm $arm
```

Family presets use retained random baseline constructors. Spec mode supports
the reviewed generic graph factory/forward ABI. Other owners use
`run_sandbox(factory, forward=..., addon_parameters=...)` directly; unsupported
Spec forwards fail rather than silently becoming baseline checks.

The sandbox verifies finite output, L1 backward, backbone gradients, each addon's
owned trainable gradients, and exact `BytesIO` state/eval reconstruction.
Missing/nonfinite addon gradients fail. Zero-start upstream zero tensors are
reported and allowed if another owned parameter receives a nonzero gradient.
Parameter-free or initialization-only hooks report `NOT_APPLICABLE`.
Special hooks reusing base parameters must supply explicit parameter ownership.

Reports contain locally measured timing and loaded dependency flags. Two synthetic graphs cannot qualify physical BS128 VRAM,
real-data loaders, native runtime speed or scientific acceptance.

## 4. Discover, package and hand off

For CPU-only regression on Windows, set `CUDA_VISIBLE_DEVICES=-1` (and the
applicable HIP/ROCR masks) before starting Python, then verify CUDA is unavailable.
An empty PowerShell environment assignment can remove the variable rather than
hide accelerators. Do not replace real runtime checks with fake GPU metadata.

```powershell
& $python -m molgap.experiment_cli source-dependencies --spec $spec --repo-root .
& $python -m molgap.experiment_cli package --spec $spec --repo-root . --auto-dependencies --output $freshPackage
& $python -m molgap.experiment_cli prepare-family-training --spec $spec --repo-root . --plan $plan --auto-dependencies --output $freshPrepared
```

AST discovery follows static and literal imports, including function-local
imports and selected lazy package exports. It reads source without executing it;
third-party packages are reported, never copied. Dynamic or wildcard imports remain
reported review obligations; `source-dependencies --strict-dynamic` rejects
detected unresolved calls. AST cannot prove closure for arbitrary dynamic Python.
Existing source confinement, credential/link rejection, committed-byte
verification and actual isolated release/bootstrap imports remain mandatory.
Explicit historical inventories keep their original path when the flag is absent.

Prepared jobs reuse the existing worker/preflight, receipts and platform skills.
The CLI does not submit. Accepted fixed graphs precede GPU work; continuation
requires owning acceptance and budget checks, not a sandbox green light.

## Desktop integration boundary

This reusable port originates from server `938ddc11`, reviewed at `ded33bea`.
The [desktop acceptance record](MODULAR_RUNTIME_DESKTOP_ACCEPTANCE.md) owns
executed checks and unchanged baseline failures.
It retains desktop K1/GPTrans accelerated trainers, scientific modes, target
bindings, runtime checks and V5/RML gates. It does not import server research,
control, campaign transport or queue state.

The declaration builder preserves desktop K1 seed semantics: the retained v2
joint addon still uses seed42 even for initialization43; v3 follows the explicit
initialization seed. Capability queries return no preparation entry when this
checkout has no matching route. Generic packages select their own factory,
addon and serialization roots, not the local baseline sandbox inventory.

The generic `graph-screen-v1` owner is an opt-in registered factory ABI for
compatible pure-2D normalized-Gap contracts, not a replacement for existing
accelerated family sessions. Its checkpoint continuation is owned by
`graph_screen_training`; the weights-only CPU loader and structural RNG gate
are shared in `training_reproducibility`. It does not add a desktop transport
or remote release adapter. Source hashes and native qualification are required
for new payloads; historical frozen packages remain unchanged.

The generic screen has a bounded local-CPU/Kaggle-T4 execution policy, not an
IMS/DCU/full-data trainer. Cross-scale K1/GPTrans continue through the existing
family sessions and their owning platform skills.

An explicit K1 `FamilyRecipe.clean_bn` now binds the existing calibration owner
to per-pass Clean-EMA selection and atomic owner-state checkpoint/resume. The
policy requires a fixed training prefix, input/role/order pins and native scratch
qualification; recipes without it retain their original identity and behavior.
Selected EMA tensors include calibrated BN buffers and their retained binding.
Attach/seal hooks before
building the optimizer; prove the hooks are reached on the selected accelerated
path. Dynamic dependency reports still require review and isolated imports.

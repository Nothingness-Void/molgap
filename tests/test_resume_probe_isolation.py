"""Scratch continuation probes must not mutate a retained optimizer state."""
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from molgap import family_training_runner as owner


def test_actual_resume_probes_isolate_real_adamw_cpu_step_tensors(tmp_path, monkeypatch):
    from molgap import family_training_kernel, family_training_loaders, family_training_session
    from molgap import training_reproducibility

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    retained_optimizer = copy.deepcopy(optimizer.state_dict())
    # Establish the real CPU restore alias that caused the second probe failure.
    alias_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    alias_optimizer = torch.optim.AdamW([alias_parameter], lr=0.01)
    alias_optimizer.load_state_dict(retained_optimizer)
    assert next(iter(alias_optimizer.state.values()))['step'].data_ptr() == retained_optimizer['state'][0]['step'].data_ptr()

    resume = tmp_path / 'resume'
    resume.mkdir()
    checkpoint = resume / 'last_checkpoint.pt'
    torch.save({'context': {}, 'optimizer': retained_optimizer}, checkpoint)
    original_bytes = checkpoint.read_bytes()
    output = tmp_path / 'proofs'
    output.mkdir()
    recipe = NS(seed=42, family='synthetic', mode='synthetic')
    monkeypatch.setattr(owner, 'run_screen_preflight', lambda **kwargs: {'status': 'accepted'})
    monkeypatch.setattr(owner, '_bootstrap', lambda **kwargs: (None, recipe, None, None, None, None, None))
    monkeypatch.setattr(owner, '_check_context', lambda *args: None)
    monkeypatch.setattr(owner, 'mode_policy', lambda *args: NS(qualify_resources=None))
    monkeypatch.setattr(family_training_session, 'validate_session_resume',
                        lambda *args: NS(status='paused', optimizer_step=1))
    monkeypatch.setattr(training_reproducibility, 'configure_fp32_determinism', lambda *args: None)
    monkeypatch.setattr(family_training_loaders, 'FamilyLoaders', lambda *args: None)

    class Kernel:
        def __init__(self, *args, **kwargs):
            self.parameter = torch.nn.Parameter(torch.tensor([1.0]))
            self.optimizer = torch.optim.AdamW([self.parameter], lr=0.01)

    seen_steps = []
    class Session:
        def __init__(self, kernel, loaders, *, checkpoint, **kwargs):
            self.kernel, self.checkpoint = kernel, checkpoint

        def load_state_dict(self, state):
            count = state['optimizer']['state'][0]['step'].item()
            seen_steps.append(count)
            assert count == 1, 'Scratch probe changed the retained optimizer exposure'
            self.kernel.optimizer.load_state_dict(state['optimizer'])

        def run(self, **kwargs):
            self.kernel.parameter.grad = torch.ones_like(self.kernel.parameter)
            self.kernel.optimizer.step()
            self.checkpoint(None, {'optimizer': self.kernel.optimizer.state_dict()})
            return NS(optimizer_step=2)

    monkeypatch.setattr(family_training_kernel, 'FamilyKernel', Kernel)
    monkeypatch.setattr(family_training_session, 'FamilyTrainingSession', Session)
    assert owner.run_screen_resume_preflight(resume_output=resume, output=output,
        initial_state_path=tmp_path / 'unused.pt') == {'status': 'accepted'}
    assert seen_steps == [1, 1]
    assert checkpoint.read_bytes() == original_bytes
    assert (output / 'actual_resume_certificate.json').is_file()

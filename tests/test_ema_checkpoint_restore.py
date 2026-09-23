import pytest
import torch

from molgap.pcqm_gptrans_full_runner import ExponentialMovingAverage


class StateFixture:
    def __init__(self, device='cpu'):
        self.values = {'weight': torch.tensor([2.0], device=device),
                       'count': torch.tensor(3, device=device)}

    def state_dict(self):
        return self.values


def test_ema_cpu_checkpoint_preserves_destination_placement():
    # Meta exercises differing source/destination devices without local GPU work.
    ema = ExponentialMovingAverage(StateFixture('meta'), 0.5)
    ema.load_state_dict(StateFixture().state_dict())
    assert all(t.device.type == 'meta' for t in ema.state_dict().values())


def test_ema_restore_update_and_no_checkpoint_aliasing():
    model = StateFixture()
    ema = ExponentialMovingAverage(model, 0.5)
    source = {'weight': torch.tensor([6.0]), 'count': torch.tensor(1)}
    ema.load_state_dict(source)
    ema.update(model)
    assert ema.state_dict()['weight'].item() == 4.0
    assert ema.state_dict()['count'].item() == 3
    assert source['weight'].item() == 6.0
    with pytest.raises(RuntimeError, match='shape/dtype'):
        ema.load_state_dict({'weight': torch.zeros(2), 'count': torch.tensor(1)})

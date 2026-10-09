"""Clean second K1 training view; BatchNorm and autograd remain active."""
from contextlib import contextmanager


@contextmanager
def clean_second_view(model):
    """Disable only K1 dropout owners, without recursively changing BN children."""
    from torch.nn import MultiheadAttention
    from torch.nn.modules.dropout import _DropoutNd

    owners = [(module, module.training) for module in model.modules()
              if isinstance(module, (_DropoutNd, MultiheadAttention)) or
              (module.__class__.__name__ == "LocalGPSBlock" and
               isinstance(getattr(module, "dropout", None), float))]
    try:
        for module, _ in owners:
            # eval()/train() recurse and would disable LocalGPSBlock's BN children.
            module.training = False
        yield
    finally:
        for module, training in owners:
            module.training = training

"""Loss / decode / sampling for the screen. Targets are standardised with TRAIN-row mean/std."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

HUBER_DELTA = 0.1
NBINS = 64
LOSSES = ("l1", "mse", "huber", "binhyb", "laplace")


def n_out(loss: str) -> int:
    return {"laplace": 2, "binhyb": NBINS + 1}.get(loss, 1)


def make_objective(loss: str, y_train_norm: torch.Tensor, device):
    """Returns dict(n_out, pred(out), scale(out)|None, loss(out, y, w|None))."""
    if loss not in LOSSES:
        raise SystemExit(f"unknown loss {loss}")
    centers = edges = None
    if loss == "binhyb":
        q = torch.quantile(y_train_norm.double(), torch.linspace(0, 1, NBINS + 1, dtype=torch.float64))
        edges = q.float().to(device)
        idx = torch.bucketize(y_train_norm.to(device), edges[1:-1])
        centers = torch.zeros(NBINS, device=device).scatter_add_(0, idx, y_train_norm.to(device))
        cnt = torch.bincount(idx, minlength=NBINS).clamp(min=1)
        centers = centers / cnt

    def pred(o):
        if loss == "laplace":
            return o[:, 0]
        if loss == "binhyb":
            return (F.softmax(o[:, :NBINS], -1) * centers).sum(-1) + o[:, NBINS]
        return o[:, 0]

    def scale(o):  # normalised units
        return torch.exp(o[:, 1].clamp(-6, 4)) if loss == "laplace" else None

    def lossf(o, y, w=None):
        p = pred(o)
        if loss in ("l1", "binhyb"):
            per = (p - y).abs()
        elif loss == "mse":
            per = (p - y) ** 2
        elif loss == "huber":
            per = F.huber_loss(p, y, reduction="none", delta=HUBER_DELTA)
        else:
            lb = o[:, 1].clamp(-6, 4)
            per = (p - y).abs() * torch.exp(-lb) + lb
        if loss == "binhyb":
            ce = F.cross_entropy(o[:, :NBINS], torch.bucketize(y, edges[1:-1]), reduction="none")
            per = per + 0.5 * ce
        return per.mean() if w is None else (per * w).mean()

    return {"n_out": n_out(loss), "pred": pred, "scale": scale, "loss": lossf}


def sample_weights(kind: str, y_train_norm: np.ndarray, tags: np.ndarray | None):
    """Returns (loss_weights|None, sampling_weights|None), both float64 arrays of length n."""
    if kind in (None, "none"):
        return None, None
    if kind == "lds":  # inverse-sqrt smoothed label density, mean-normalised, clipped
        y = np.asarray(y_train_norm, dtype=np.float64)
        edges = np.linspace(y.min(), y.max() + 1e-9, 101)
        idx = np.clip(np.digitize(y, edges) - 1, 0, 99)
        cnt = np.bincount(idx, minlength=100).astype(np.float64)
        k = np.exp(-0.5 * (np.arange(-5, 6) / 2.0) ** 2)
        k /= k.sum()
        dens = np.convolve(cnt, k, mode="same")
        w = 1.0 / np.sqrt(np.maximum(dens[idx], 1.0))
        w = np.clip(w / w.mean(), 0.1, 10.0)
        return w / w.mean(), None
    if kind == "oversample":  # radical>0 or no aromatic ring: weight 2 (same draws/epoch)
        if tags is None:
            raise SystemExit("oversample needs features tags")
        sub = (tags[:, 0] > 0) | (tags[:, 2] == 0)
        return None, np.where(sub, 2.0, 1.0)
    raise SystemExit(f"unknown sampler {kind}")

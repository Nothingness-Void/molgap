"""K1 Rank-n-Contrast (arXiv 2210.01189) joint regulariser on the pooled embedding.

Training-only loss term: no parameters, no inference change. The constants below are
frozen by this file's source hash; the tuning rule is in
experiments/k1_rank_n_contrast_100k/protocol.md.
"""
from __future__ import annotations

TEMPERATURE = 0.1
# Frozen from the one-shot gradient-share preflight, not tuned afterwards.
LAMBDA = 0.007  # lambda_preflight.py: median grad-norm ratio 0.25 at the frozen init (protocol.md)


def rank_contrast_loss(embedding, target, temperature=TEMPERATURE):
    """Mean over anchors i and positives j != i of -log softmax over S_ij.

    S_ij = {k != i : |y_i-y_k| >= |y_i-y_j|} (ties stay in S_ij, as in the paper);
    similarity is the negative L2 distance of unit-normalised embeddings. The mask is a
    [N,N,N] tensor (N=128: ~2M entries), so there is no Python loop.
    """
    import torch
    import torch.nn.functional as functional
    n = embedding.shape[0]
    unit = functional.normalize(embedding, dim=-1)
    # clamp + eps keep the sqrt gradient finite for identical embeddings.
    distance = ((2.0 - 2.0 * unit @ unit.T).clamp_min(0.0) + 1e-12).sqrt()
    eye = torch.eye(n, dtype=torch.bool, device=embedding.device)
    logits = (-distance / temperature).masked_fill(eye, float("-inf"))
    label = (target.view(-1, 1) - target.view(1, -1)).abs()
    member = (label.unsqueeze(1) >= label.unsqueeze(2)) & ~eye.unsqueeze(1)  # [i,j,k]
    denominator = torch.logsumexp(
        logits.unsqueeze(1).expand(n, n, n).masked_fill(~member, float("-inf")), dim=2)
    term = (denominator - logits).masked_fill(eye, 0.0)
    return term.sum() / (n * (n - 1))

"""TensorNet 3D encoder — vendored single-file port for the A/B comparison.

Architecture ported from torchmd-net ``torchmdnet/models/tensornet.py``
(TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular
Potentials; G. Simeon & G. de Fabritiis, NeurIPS 2023). Original code:
https://github.com/torchmd/torchmd-net — MIT License, Copyright Universitat
Pompeu Fabra 2020-2023.

Why vendored instead of `pip install torchmd-net`: torchmd-net pins older torch
and ships CUDA extensions (warp_ops / OptimizedDistance) that conflict with this
project's torch 2.11+cu128 / PyG stack. We keep only the **pure-PyTorch (non-opt)
path** and replace torchmd-net's pieces with PyG-native equivalents:
  · OptimizedDistance  -> torch_geometric.nn.radius_graph (+ manual edge vecs)
  · expnorm RBF        -> torch_geometric.nn.models.visnet.ExpNormalSmearing
  · CosineCutoff       -> local two-bound cosine cutoff (below)
PBC/box, total-charge q, static_shapes and warp kernels are dropped (q≡0).

Wrapper interface mirrors ``SchNetWrapper``:
    encode(z, pos, batch, charges=None) -> [num_molecules, hidden_channels]
    forward(z, pos, batch, charges=None) -> [num_molecules, n_targets]
``charges`` is accepted for parity but unused (TensorNet native = Z + geometry).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ── cutoff (two-bound cosine, torchmd-net form; lower=0 reduces to PyG's) ──
class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=6.0):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances: Tensor) -> Tensor:
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (2 * (distances - self.cutoff_lower)
                       / (self.cutoff_upper - self.cutoff_lower) + 1.0)
                ) + 1.0
            )
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff_upper).float()
        return cutoffs


# ── tensor algebra helpers (pure-python path) ──
def decompose_tensor(tensor):
    """Decompose [N,3,3,F] tensor into (I scalar [N,F], A skew, S sym-traceless)."""
    A = 0.5 * (tensor - tensor.transpose(1, 2))
    S = tensor - A
    I = (tensor.diagonal(dim1=1, dim2=2)).mean(-1)
    S = S - I_to_tensor(I)
    return I, A, S


def compose_tensor(I, A, S):
    return I_to_tensor(I) + A + S


def tensor_matmul_o3(Y, msg):
    A = torch.matmul(msg.permute(0, 3, 1, 2), Y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    B = torch.matmul(Y.permute(0, 3, 1, 2), msg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return A + B


def tensor_matmul_so3(Y, msg):
    return torch.matmul(
        Y.permute(0, 3, 1, 2), msg.permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1)


def vector_to_skewtensor(vector):
    """[B,3,F] -> skew-symmetric [B,3,3,F]."""
    B, _, F = vector.shape
    zero = torch.zeros((B, F), device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero, -vector[..., 2, :], vector[..., 1, :],
            vector[..., 2, :], zero, -vector[..., 0, :],
            -vector[..., 1, :], vector[..., 0, :], zero,
        ),
        dim=1,
    )
    return tensor.view(B, 3, 3, F)


def skewtensor_to_vector(tensor):
    """[N,3,3,F] skew -> [N,3,F] vector."""
    tensor = tensor.flatten(1, 2)
    return 0.5 * torch.stack(
        (
            tensor[:, 7, :] - tensor[:, 5, :],
            tensor[:, 2, :] - tensor[:, 6, :],
            tensor[:, 3, :] - tensor[:, 1, :],
        ),
        dim=1,
    )


def I_to_tensor(I):
    """[N,F] scalar -> [N,3,3,F] isotropic tensor."""
    return (
        I[:, None, None, :]
        * torch.eye(3, 3, device=I.device, dtype=I.dtype)[None, ..., None]
    )


def outer_to_symtensor(tensor):
    """[N,3,3,F] outer product -> symmetric traceless [N,3,3,F]."""
    S = 0.5 * (tensor + tensor.transpose(1, 2))
    I = (tensor.diagonal(dim1=1, dim2=2)).mean(-1)
    return S - I_to_tensor(I)


def tensor_norm(tensor):
    """Frobenius norm over the 3x3 block -> [N,F]."""
    return (tensor ** 2).sum((1, 2))


# ── message passing (pure-python path) ──
def _embedding_message_passing(edge_vec_norm, edge_attr_processed, edge_index, num_atoms):
    F = edge_attr_processed.shape[-1]
    Iij = edge_attr_processed[:, 0, :]
    Aij = edge_attr_processed[:, 1, None, :] * edge_vec_norm.unsqueeze(-1)
    _outer = torch.matmul(edge_vec_norm.unsqueeze(2), edge_vec_norm.unsqueeze(-2))
    Sij = edge_attr_processed[:, 2, None, None, :] * _outer.unsqueeze(-1)

    I = torch.zeros(num_atoms, F, device=Iij.device, dtype=Iij.dtype).index_add(
        0, edge_index[0], Iij)
    A_vec = torch.zeros(num_atoms, 3, F, device=Aij.device, dtype=Aij.dtype).index_add(
        0, edge_index[0], Aij)
    S = torch.zeros(num_atoms, 3, 3, F, device=Sij.device, dtype=Sij.dtype).index_add(
        0, edge_index[0], Sij)
    return I, vector_to_skewtensor(A_vec), outer_to_symtensor(S)


def _scalar_mp(edge_index, factor, tensor, natoms):
    # accumulate in msg.dtype: under autocast `factor` (from fp32 edge_weight) and
    # `tensor` (autocast bf16) can differ, so bind the buffer dtype to the product.
    msg = factor * tensor.index_select(0, edge_index[1])
    return torch.zeros(natoms, tensor.shape[1], device=tensor.device,
                       dtype=msg.dtype).index_add(0, edge_index[0], msg)


def _vector_mp(edge_index, factor, tensor, natoms):
    msg = factor * tensor.index_select(0, edge_index[1])
    return torch.zeros(natoms, tensor.shape[1], tensor.shape[2], device=tensor.device,
                       dtype=msg.dtype).index_add(0, edge_index[0], msg)


def _tensor_mp(edge_index, factor, tensor, natoms):
    msg = factor * tensor.index_select(0, edge_index[1])
    return torch.zeros(natoms, tensor.shape[1], tensor.shape[2], tensor.shape[3],
                       device=tensor.device, dtype=msg.dtype).index_add(
        0, edge_index[0], msg)


def _interaction_message_passing(I, A, S, edge_attr_processed, edge_index, natoms):
    A_vec = skewtensor_to_vector(A)
    factor_scalar = edge_attr_processed[..., 0, :]
    factor_vector = edge_attr_processed[..., 1, None, :]
    factor_tensor = edge_attr_processed[..., 2, None, None, :]
    I = _scalar_mp(edge_index, factor_scalar, I, natoms)
    A_vec = _vector_mp(edge_index, factor_vector, A_vec, natoms)
    S = _tensor_mp(edge_index, factor_tensor, S, natoms)
    return I, vector_to_skewtensor(A_vec), S


class TensorEmbedding(nn.Module):
    def __init__(self, hidden_channels, num_rbf, activation, cutoff_lower,
                 cutoff_upper, max_z=128):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.distance_proj1 = nn.Linear(num_rbf, hidden_channels)
        self.distance_proj2 = nn.Linear(num_rbf, hidden_channels)
        self.distance_proj3 = nn.Linear(num_rbf, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.max_z = max_z
        self.emb = nn.Embedding(max_z, hidden_channels)
        self.emb2 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.act = activation()
        self.linears_tensor = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(3)])
        self.linears_scalar = nn.ModuleList([
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True),
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True),
        ])
        self.init_norm = nn.LayerNorm(hidden_channels)

    def _atomic_number_message(self, z, edge_index):
        Z = self.emb(z)
        return self.emb2(
            Z.index_select(0, edge_index.t().reshape(-1)).view(-1, self.hidden_channels * 2))

    def forward(self, z, edge_index, edge_weight, edge_vec_norm, edge_attr):
        Zij = self._atomic_number_message(z, edge_index)
        dp1 = self.distance_proj1(edge_attr)
        dp2 = self.distance_proj2(edge_attr)
        dp3 = self.distance_proj3(edge_attr)
        CZij = self.cutoff(edge_weight).unsqueeze(-1) * Zij
        edge_attr_processed = CZij.unsqueeze(1) * torch.stack([dp1, dp2, dp3], dim=1)

        I, A, S = _embedding_message_passing(
            edge_vec_norm, edge_attr_processed, edge_index, z.shape[0])
        X = compose_tensor(I, A, S)

        norm = self.init_norm(tensor_norm(X))
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))
        norm = norm.reshape(-1, 3, self.hidden_channels)

        I = self.linears_tensor[0](I) * norm[:, 0, :]
        A = self.linears_tensor[1](A) * norm[:, 1, None, None, :]
        S = self.linears_tensor[2](S) * norm[:, 2, None, None, :]
        return compose_tensor(I, A, S)


class Interaction(nn.Module):
    def __init__(self, num_rbf, hidden_channels, activation, cutoff_lower,
                 cutoff_upper, equivariance_invariance_group):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.linears_scalar = nn.ModuleList([
            nn.Linear(num_rbf, hidden_channels, bias=True),
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True),
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True),
        ])
        self.linears_tensor = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)])
        self.act = activation()
        self.equivariance_invariance_group = equivariance_invariance_group

    def forward(self, X, edge_index, edge_weight, edge_attr, q):
        C = self.cutoff(edge_weight)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        edge_attr = (edge_attr * C.view(-1, 1)).reshape(
            edge_attr.shape[0], 3, self.hidden_channels)

        X = X / (tensor_norm(X) + 1)[:, None, None, :]
        I, A, S = decompose_tensor(X)
        I = self.linears_tensor[0](I)
        A = self.linears_tensor[1](A)
        S = self.linears_tensor[2](S)
        Y = compose_tensor(I, A, S)

        Im, Am, Sm = _interaction_message_passing(
            I, A, S, edge_attr, edge_index, X.shape[0])
        msg = compose_tensor(Im, Am, Sm)

        if self.equivariance_invariance_group == "O(3)":
            C = (1 + 0.1 * q[..., None, None, None]) * tensor_matmul_o3(Y, msg)
        else:  # SO(3)
            C = 2 * tensor_matmul_so3(Y, msg)
        I, A, S = decompose_tensor(C)

        normp1 = tensor_norm(C) + 1
        I = I / normp1
        A = A / normp1[..., None, None, :]
        S = S / normp1[..., None, None, :]

        I = self.linears_tensor[3](I)
        A = self.linears_tensor[4](A)
        S = self.linears_tensor[5](S)
        dX = compose_tensor(I, A, S)
        return X + dX + (1 + 0.1 * q[..., None, None, None]) * tensor_matmul_so3(dX, dX)


class TensorNetRepresentation(nn.Module):
    """Per-node scalar representation [N, hidden] from TensorNet (no readout head)."""

    def __init__(self, hidden_channels=192, num_layers=3, num_rbf=32,
                 cutoff=6.0, cutoff_lower=0.0, max_z=128, max_num_neighbors=32,
                 equivariance_invariance_group="O(3)"):
        super().__init__()
        from torch_geometric.nn.models.visnet import ExpNormalSmearing

        self.cutoff_upper = cutoff
        self.cutoff_lower = cutoff_lower
        self.max_num_neighbors = max_num_neighbors
        act_class = nn.SiLU
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf, False)
        self.tensor_embedding = TensorEmbedding(
            hidden_channels, num_rbf, act_class, cutoff_lower, cutoff, max_z)
        self.layers = nn.ModuleList([
            Interaction(num_rbf, hidden_channels, act_class, cutoff_lower, cutoff,
                        equivariance_invariance_group)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(3 * hidden_channels, hidden_channels)
        self.out_norm = nn.LayerNorm(3 * hidden_channels)
        self.act = act_class()

    def forward(self, z, pos, batch):
        from torch_geometric.nn import radius_graph

        edge_index = radius_graph(
            pos, r=self.cutoff_upper, batch=batch, loop=True,
            max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_weight = edge_vec.norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        # Guard the direction normalization against divide-by-zero. Self-loops
        # (row==col, r=0) AND degenerate geometries where two DISTINCT atoms sit
        # at ~identical coords (r≈0, e.g. a bad ETKDG/MMFF conformer) would give
        # 0/0 = NaN. Clamp all near-zero distances; edge_vec≈0 there → norm ≈ 0.
        mask = edge_weight < 1e-8
        edge_vec_norm = edge_vec / edge_weight.masked_fill(mask, 1.0).unsqueeze(1)

        q = torch.zeros(z.shape[0], device=z.device, dtype=pos.dtype)

        X = self.tensor_embedding(z, edge_index, edge_weight, edge_vec_norm, edge_attr)
        for layer in self.layers:
            X = layer(X, edge_index, edge_weight, edge_attr, q)

        I, A, S = decompose_tensor(X)
        x = torch.cat((3 * I ** 2, tensor_norm(A), tensor_norm(S)), dim=-1)
        x = self.out_norm(x)
        return self.act(self.linear(x))


class TensorNetWrapper(nn.Module):
    def __init__(self, hidden_channels=192, num_layers=3, num_rbf=32, cutoff=6.0,
                 dropout=0.0, n_targets=3, max_num_neighbors=32,
                 equivariance_invariance_group="O(3)", use_charges=False):
        super().__init__()
        self.use_charges = use_charges  # accepted for parity, intentionally unused
        self.representation = TensorNetRepresentation(
            hidden_channels=hidden_channels, num_layers=num_layers, num_rbf=num_rbf,
            cutoff=cutoff, max_num_neighbors=max_num_neighbors,
            equivariance_invariance_group=equivariance_invariance_group)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def forward(self, z, pos, batch, charges=None):
        return self.head(self.encode(z, pos, batch, charges=charges))

    def encode(self, z, pos, batch, charges=None):
        from torch_geometric.nn import global_mean_pool

        x = self.representation(z, pos, batch)
        return global_mean_pool(x, batch)

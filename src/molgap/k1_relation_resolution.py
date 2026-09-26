"""Bounded receiver-aware relation sidecars for the frozen Neural-Atom K1.

The sidecars keep the K1 backbone and its initialization order intact.  They
replace the graph-wide PairToken reduction with a single layer-6 relation
addon whose return projection is zero at construction time.  The RRWP branch
is deliberately computed from the current ``edge_index`` inside the encoder;
it is not a cached feature or a change to the input feature identity.

The triplet branch is the small TGT-Ag inward idea only: values are vectors,
and the receiver row weights are independent of the middle pair index.  This
module is an implementation surface for a bounded preflight, not a claim of
paper-level performance.
"""
from __future__ import annotations

import math


MODE_RECEIVER_PAIR = "neural_atom_k1_receiver_pair"
MODE_RRWP_PAIR = "neural_atom_k1_rrwp_pair"
MODE_TRIPLET_AGGREGATE = "neural_atom_k1_triplet_aggregate"
MODES = (MODE_RECEIVER_PAIR, MODE_RRWP_PAIR, MODE_TRIPLET_AGGREGATE)

HIDDEN_CHANNELS = 192
PAIR_CHANNELS = 32
TARGET_LAYER = 6
MIXER_LAYERS = (3, 6, 9)
RRWP_STEPS = 8
BASE_PARAMETERS = 3_658_817

# Keep this table intentionally small: it describes only the frozen-layout
# choices that differ among the three candidate addons.
CONFIGS = {
    MODE_RECEIVER_PAIR: {
        "backbone": "neural_atom_k1",
        "change": "layer6-receiver-axis-pair-resolution",
        "exchange_layers": [3, 6, 9],
        "target_layer": TARGET_LAYER,
        "pair_channels": PAIR_CHANNELS,
        "rrwp_features": 0,
        "triplet": False,
        "geometry": False,
        "teacher": False,
        "initialization_policy": "base-first-zero-return-addon",
    },
    MODE_RRWP_PAIR: {
        "backbone": "neural_atom_k1",
        "change": "layer6-receiver-pair-single-rrwp8",
        "exchange_layers": [3, 6, 9],
        "target_layer": TARGET_LAYER,
        "pair_channels": PAIR_CHANNELS,
        "rrwp_features": RRWP_STEPS,
        "rrwp_search": False,
        "triplet": False,
        "geometry": False,
        "teacher": False,
        "initialization_policy": "base-first-zero-return-addon",
    },
    MODE_TRIPLET_AGGREGATE: {
        "backbone": "neural_atom_k1",
        "change": "layer6-receiver-pair-single-tgt-ag-inward-vector",
        "exchange_layers": [3, 6, 9],
        "target_layer": TARGET_LAYER,
        "pair_channels": PAIR_CHANNELS,
        "rrwp_features": 0,
        "triplet": True,
        "geometry": False,
        "teacher": False,
        "initialization_policy": "base-first-zero-return-addon",
    },
}

# Algebraic counts are kept here instead of instantiating a model at import
# time.  Ws/Wt and the two scalar TGT gates use the ordinary bias-bearing
# Linear convention; the return and vector projections are bias-free.
PARENT_ADDON_PARAMETERS = (
    2 * (HIDDEN_CHANNELS * PAIR_CHANNELS + PAIR_CHANNELS)  # Ws, Wt
    + 2 * PAIR_CHANNELS  # pair LayerNorm
    + PAIR_CHANNELS  # q
    + 2 * PAIR_CHANNELS  # receiver-message LayerNorm
    + (PAIR_CHANNELS * (2 * PAIR_CHANNELS) + 2 * PAIR_CHANNELS)
    + ((2 * PAIR_CHANNELS) * PAIR_CHANNELS + PAIR_CHANNELS)  # FFN
    + PAIR_CHANNELS * HIDDEN_CHANNELS  # zero-init bias-free return
)
RRWP_PROJECTION_PARAMETERS = RRWP_STEPS * PAIR_CHANNELS + PAIR_CHANNELS
TRIPLET_PARAMETERS = (
    PAIR_CHANNELS * PAIR_CHANNELS  # bias-free V
    + (PAIR_CHANNELS + 1)  # b(p_ik)
    + (PAIR_CHANNELS + 1)  # g(p_ik)
    + PAIR_CHANNELS * PAIR_CHANNELS  # zero-init bias-free Wout
    + 2 * PAIR_CHANNELS  # p' LayerNorm
)

PARAMETERS = {
    MODE_RECEIVER_PAIR: BASE_PARAMETERS + PARENT_ADDON_PARAMETERS,
    MODE_RRWP_PAIR: BASE_PARAMETERS
    + PARENT_ADDON_PARAMETERS
    + RRWP_PROJECTION_PARAMETERS,
    MODE_TRIPLET_AGGREGATE: BASE_PARAMETERS
    + PARENT_ADDON_PARAMETERS
    + TRIPLET_PARAMETERS,
}


def _rrwp_features(edge_index, batch, valid, *, dtype):
    """Build ``[I, R, ..., R**7]`` per graph with padded rows zero.

    ``R`` is a row-normalized undirected adjacency.  A valid isolated node
    receives an explicit self transition; padded positions never receive one.
    The helper has no dependency on an external graph/cache identity.
    """
    import torch

    num_graphs, max_nodes = valid.shape
    device = valid.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    batch = batch.to(device=device, dtype=torch.long)
    node_count = batch.numel()
    counts = torch.bincount(batch, minlength=num_graphs)
    ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    local = torch.arange(node_count, device=device) - ptr[batch]

    adjacency = torch.zeros(
        (num_graphs, max_nodes, max_nodes), device=device, dtype=dtype
    )
    if edge_index.numel():
        source, target = edge_index[0], edge_index[1]
        same_graph = batch[source] == batch[target]
        real_bond = source != target
        keep = same_graph & real_bond
        source = source[keep]
        target = target[keep]
        graph = batch[source]
        source_local = local[source]
        target_local = local[target]
        one = torch.ones(source.shape, device=device, dtype=dtype)
        # Treat edge_index as a bond list, not as an already-oriented walk.
        adjacency[graph, source_local, target_local] = one
        adjacency[graph, target_local, source_local] = one

    degree = adjacency.sum(dim=-1)
    row_normalized = adjacency / degree.clamp_min(1).unsqueeze(-1)
    identity = torch.eye(max_nodes, device=device, dtype=dtype).unsqueeze(0)
    valid_pair = valid.unsqueeze(1) & valid.unsqueeze(2)
    identity = identity * valid_pair
    isolates = (degree == 0) & valid
    row_normalized = row_normalized + identity * isolates.unsqueeze(-1)

    powers = [identity]
    for _ in range(1, RRWP_STEPS):
        if len(powers) == 1:
            powers.append(row_normalized)
        else:
            powers.append(torch.bmm(powers[-1], row_normalized))
    return torch.stack(powers, dim=-1)


class _RelationFactory:
    @staticmethod
    def make(mode: str):
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch_geometric.utils import to_dense_batch

        class RelationAddon(nn.Module):
            def __init__(self):
                super().__init__()
                self.mode = mode
                # ``source`` is the receiver i and ``target`` is the paired
                # index j; keeping both projections matches the old K1 addon
                # initialization surface without making a graph token.
                self.source = nn.Linear(HIDDEN_CHANNELS, PAIR_CHANNELS)
                self.target = nn.Linear(HIDDEN_CHANNELS, PAIR_CHANNELS)
                self.pair_norm = nn.LayerNorm(PAIR_CHANNELS)
                self.query = nn.Parameter(torch.empty(PAIR_CHANNELS))
                nn.init.normal_(self.query, std=PAIR_CHANNELS ** -0.5)
                self.message_norm = nn.LayerNorm(PAIR_CHANNELS)
                self.message_ffn = nn.Sequential(
                    nn.Linear(PAIR_CHANNELS, 2 * PAIR_CHANNELS),
                    nn.SiLU(),
                    nn.Linear(2 * PAIR_CHANNELS, PAIR_CHANNELS),
                )
                self.return_projection = nn.Linear(
                    PAIR_CHANNELS, HIDDEN_CHANNELS, bias=False
                )
                nn.init.zeros_(self.return_projection.weight)

                if mode == MODE_RRWP_PAIR:
                    self.rrwp_projection = nn.Linear(
                        RRWP_STEPS, PAIR_CHANNELS
                    )
                else:
                    self.rrwp_projection = None

                if mode == MODE_TRIPLET_AGGREGATE:
                    self.triplet_value = nn.Linear(
                        PAIR_CHANNELS, PAIR_CHANNELS, bias=False
                    )
                    self.triplet_bias = nn.Linear(PAIR_CHANNELS, 1)
                    self.triplet_gate = nn.Linear(PAIR_CHANNELS, 1)
                    self.triplet_out = nn.Linear(
                        PAIR_CHANNELS, PAIR_CHANNELS, bias=False
                    )
                    nn.init.zeros_(self.triplet_out.weight)
                    self.triplet_norm = nn.LayerNorm(PAIR_CHANNELS)

            @staticmethod
            def _safe_softmax(logits, valid, *, receiver_axis: bool):
                """Softmax with valid rows and no NaN for padded receivers."""
                if receiver_axis:
                    key_mask = valid.unsqueeze(1)
                    row_mask = valid.unsqueeze(-1)
                else:
                    key_mask = valid.unsqueeze(1)
                    row_mask = valid.unsqueeze(-1)
                safe_logits = logits.masked_fill(~key_mask, float("-inf"))
                has_key = valid.any(dim=-1).view(-1, 1, 1)
                safe_logits = torch.where(
                    has_key, safe_logits, torch.zeros_like(safe_logits)
                )
                return torch.softmax(safe_logits, dim=-1) * key_mask * row_mask

            def _pair_state(self, hidden, batch, edge_index):
                dense, valid = to_dense_batch(hidden, batch)
                pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
                source = self.source(dense).unsqueeze(2)
                target = self.target(dense).unsqueeze(1)
                preactivation = source + target
                rrwp = None
                if self.rrwp_projection is not None:
                    if edge_index is None:
                        raise ValueError(
                            "RRWP relation mode requires edge_index inside encode"
                        )
                    rrwp = _rrwp_features(
                        edge_index,
                        batch,
                        valid,
                        dtype=preactivation.dtype,
                    )
                    preactivation = preactivation + self.rrwp_projection(rrwp)
                pair = self.pair_norm(functional.silu(preactivation))
                pair = pair.masked_fill(~pair_valid.unsqueeze(-1), 0.0)
                return pair, pair_valid, valid, rrwp

            def _receiver_assignment(self, pair, pair_valid, valid):
                logits = torch.einsum("bijd,d->bij", pair, self.query)
                logits = logits / math.sqrt(PAIR_CHANNELS)
                assignment = self._safe_softmax(
                    logits, valid, receiver_axis=True
                )
                return assignment * pair_valid

            def _triplet_update(self, pair, pair_valid, valid):
                bias_logits = self.triplet_bias(pair).squeeze(-1)
                weights = self._safe_softmax(
                    bias_logits, valid, receiver_axis=True
                )
                gates = torch.sigmoid(self.triplet_gate(pair).squeeze(-1))
                triplet_assignment = weights * gates * pair_valid
                values = self.triplet_value(pair) * pair_valid.unsqueeze(-1)
                # values[b, j, k] is aggregated over k with weights[b, i, k]
                # for every target pair (i, j): this is vector aggregation.
                aggregate = torch.einsum(
                    "bik,bjkd->bijd", triplet_assignment, values
                )
                updated = self.triplet_norm(
                    pair + self.triplet_out(aggregate)
                )
                return (
                    updated.masked_fill(~pair_valid.unsqueeze(-1), 0.0),
                    triplet_assignment,
                    values,
                    aggregate,
                )

            def compute_update(self, hidden, batch, edge_index=None):
                pair, pair_valid, valid, rrwp = self._pair_state(
                    hidden, batch, edge_index
                )
                pair_before_triplet = pair
                triplet_assignment = None
                triplet_values = None
                triplet_aggregate = None
                if self.mode == MODE_TRIPLET_AGGREGATE:
                    (
                        pair,
                        triplet_assignment,
                        triplet_values,
                        triplet_aggregate,
                    ) = self._triplet_update(pair, pair_valid, valid)

                assignment = self._receiver_assignment(
                    pair, pair_valid, valid
                )
                message = torch.einsum("bij,bijd->bid", assignment, pair)
                message = self.message_norm(
                    message + self.message_ffn(message)
                )
                message = message.masked_fill(~valid.unsqueeze(-1), 0.0)
                return_features_dense = message
                return_features = return_features_dense[valid]
                update = self.return_projection(return_features)
                return update, {
                    "assignment": assignment,
                    "pair_valid": pair_valid,
                    "valid": valid,
                    "pair": pair,
                    "pair_before_triplet": pair_before_triplet,
                    "rrwp": rrwp,
                    "triplet_assignment": triplet_assignment,
                    "triplet_values": triplet_values,
                    "triplet_aggregate": triplet_aggregate,
                    "return_features": return_features,
                    "return_features_dense": return_features_dense,
                }

            def forward(self, hidden, batch, edge_index=None):
                update, _ = self.compute_update(
                    hidden, batch, edge_index=edge_index
                )
                return hidden + update

        return RelationAddon()


def make_encoder(mode: str):
    """Build one frozen-layout K1 plus exactly one layer-6 relation addon."""
    if type(mode) is not str or mode not in MODES:
        raise ValueError(mode)

    import torch.nn as nn

    # This construction must precede addon construction so the base K1 random
    # stream and all shared base tensors remain byte-identical to make_k1().
    from .qm9_neural_atom import make_encoder as make_k1

    class K1RelationResolution(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = make_k1("neural_atom_k1")
            # Keep the runner's stable addon name.  This is a receiver-axis
            # module, not a graph-wide token reduction.
            self.relation_token = _RelationFactory.make(mode)

        @property
        def relation_addon(self):
            """Compatibility view without registering the module twice."""
            return self.relation_token

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base.head(
                self.encode(x, edge_index, edge_attr, batch, random_walk_pe)
            )

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
            expected = (x.shape[0], self.base.rwse_dim)
            if tuple(random_walk_pe.shape) != expected:
                raise ValueError(
                    f"random_walk_pe must have shape {expected}, "
                    f"got {tuple(random_walk_pe.shape)}"
                )
            hidden = self.base._embed_nodes(x)
            hidden = hidden + self.base.rwse_encoder(random_walk_pe.float())
            edge_state = self.base._embed_edges(edge_attr)
            for layer, (edge_update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                edge_state = edge_update(hidden, edge_index, edge_state)
                hidden = block(
                    hidden, edge_index, batch, edge_attr=edge_state
                )
                if str(layer) in self.base.neural_atom_mixers:
                    hidden = self.base.neural_atom_mixers[str(layer)](
                        hidden, batch
                    )
                if layer == TARGET_LAYER:
                    hidden = self.relation_token(
                        hidden, batch, edge_index=edge_index
                    )
            return self.base._pool(hidden, batch)

    return K1RelationResolution()


def _independent_rrwp_expected(edge_index, batch, *, max_nodes, dtype):
    """Small independent RRWP oracle used only by the remote preflight."""
    import torch

    num_graphs = int(batch.max().item()) + 1
    expected = torch.zeros(
        (num_graphs, max_nodes, max_nodes, RRWP_STEPS),
        device=batch.device,
        dtype=dtype,
    )
    for graph_id in range(num_graphs):
        nodes = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        count = int(nodes.numel())
        identity = torch.eye(count, device=batch.device, dtype=dtype)
        adjacency = torch.zeros_like(identity)
        for edge_number in range(edge_index.shape[1]):
            left = int(edge_index[0, edge_number].item())
            right = int(edge_index[1, edge_number].item())
            if left == right or int(batch[left].item()) != graph_id:
                continue
            left_local = int((nodes == left).nonzero().item())
            right_local = int((nodes == right).nonzero().item())
            adjacency[left_local, right_local] = 1.0
            adjacency[right_local, left_local] = 1.0
        degree = adjacency.sum(dim=-1)
        transition = adjacency / degree.clamp_min(1).unsqueeze(-1)
        transition = transition + identity * (degree == 0).unsqueeze(-1)
        powers = [identity]
        for _ in range(1, RRWP_STEPS):
            powers.append(
                transition
                if len(powers) == 1
                else torch.matmul(powers[-1], transition)
            )
        expected[graph_id, :count, :count] = torch.stack(powers, dim=-1)
    return expected


def _remote_rrwp_checks(model, batch, *, dtype):
    import torch

    device = batch.batch.device
    synthetic_batch = torch.tensor(
        [0, 0, 1, 1], device=device, dtype=torch.long
    )
    synthetic_edges = torch.tensor(
        [[0, 1], [1, 0]],
        device=device,
        dtype=torch.long,
    )
    synthetic_valid = torch.ones((2, 2), device=device, dtype=torch.bool)
    actual = _rrwp_features(
        synthetic_edges,
        synthetic_batch,
        synthetic_valid,
        dtype=dtype,
    )
    expected = _independent_rrwp_expected(
        synthetic_edges,
        synthetic_batch,
        max_nodes=2,
        dtype=dtype,
    )
    isolated_batch = torch.tensor([0], device=device, dtype=torch.long)
    isolated_edges = torch.empty((2, 0), device=device, dtype=torch.long)
    isolated_valid = torch.ones((1, 1), device=device, dtype=torch.bool)
    isolated = _rrwp_features(
        isolated_edges,
        isolated_batch,
        isolated_valid,
        dtype=dtype,
    )
    isolated_expected = torch.ones(
        (1, 1, 1, RRWP_STEPS), device=device, dtype=dtype
    )
    return {
        "rrwp_batched_graphs_match_independent_expected": bool(
            torch.allclose(actual, expected, atol=1e-6, rtol=0)
        ),
        "rrwp_isolate_self_transition": bool(
            torch.allclose(isolated, isolated_expected, atol=1e-6, rtol=0)
        ),
    }


def _remote_permutation_check(addon, hidden, batch, edge_index, *, atol, rtol):
    import torch

    node_count = hidden.shape[0]
    permutation = torch.arange(node_count, device=hidden.device)
    for graph_id in range(int(batch.max().item()) + 1):
        nodes = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        permutation[nodes] = nodes.flip(0)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(node_count, device=hidden.device)
    permuted_edges = inverse[edge_index]
    _, original = addon.compute_update(
        hidden, batch, edge_index=edge_index
    )
    _, permuted = addon.compute_update(
        hidden[permutation],
        batch[permutation],
        edge_index=permuted_edges,
    )
    return bool(
        torch.allclose(
            permuted["return_features"],
            original["return_features"][permutation],
            atol=atol,
            rtol=rtol,
        )
    )


def _remote_triplet_loop_check(addon, *, dtype, device):
    import torch

    pair = torch.linspace(
        -0.75,
        0.75,
        steps=3 * 3 * PAIR_CHANNELS,
        device=device,
        dtype=dtype,
    ).reshape(1, 3, 3, PAIR_CHANNELS)
    valid = torch.ones((1, 3), device=device, dtype=torch.bool)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    _, weights, values, aggregate = addon._triplet_update(
        pair, pair_valid, valid
    )
    explicit = torch.zeros_like(aggregate)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                explicit[0, i, j] = (
                    explicit[0, i, j] + weights[0, i, k] * values[0, j, k]
                )
    return bool(torch.allclose(aggregate, explicit, atol=1e-6, rtol=0))


def check_mechanism(model, batch) -> dict:
    """Run the bounded REMOTE-ONLY sidecar preflight and fail closed.

    The caller must run this on the authorized remote preflight environment.
    It intentionally performs small synthetic checks only; it is not a
    training, inference, or benchmark helper.
    """
    import torch

    if not hasattr(model, "relation_token"):
        raise TypeError("model does not expose relation_token")
    addon = model.relation_token
    if addon.mode not in MODES:
        raise ValueError(f"unsupported relation mode: {addon.mode}")
    if not hasattr(batch, "batch") or not hasattr(batch, "edge_index"):
        raise TypeError("batch must expose batch and edge_index")

    node_count = int(batch.num_nodes)
    dtype = addon.source.weight.dtype
    device = batch.batch.device
    probe = torch.linspace(
        -1.0,
        1.0,
        steps=node_count * HIDDEN_CHANNELS,
        device=device,
        dtype=dtype,
    ).reshape(node_count, HIDDEN_CHANNELS)
    update, diagnostics = addon.compute_update(
        probe, batch.batch, edge_index=batch.edge_index
    )
    assignment = diagnostics["assignment"]
    pair_valid = diagnostics["pair_valid"]
    valid = diagnostics["valid"]
    finite_tensors = [update]
    finite_tensors.extend(
        value
        for value in diagnostics.values()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
    )
    finite = all(bool(torch.isfinite(value).all().item()) for value in finite_tensors)
    valid_rows = valid
    row_sums = assignment.sum(dim=-1)
    if bool(valid_rows.any().item()):
        valid_row_assignment = bool(
            torch.allclose(
                row_sums[valid_rows],
                torch.ones_like(row_sums[valid_rows]),
                atol=1e-6,
                rtol=0,
            )
        )
    else:
        valid_row_assignment = True
    padding_assignment = bool(
        torch.count_nonzero(assignment.masked_select(~pair_valid)).item() == 0
    )

    probe_distinct = probe.clone()
    if node_count:
        probe_distinct[0] = probe_distinct[0] + 0.25
    _, distinct_diagnostics = addon.compute_update(
        probe_distinct, batch.batch, edge_index=batch.edge_index
    )
    different_receiver_features = False
    for graph_id in range(diagnostics["valid"].shape[0]):
        receivers = torch.nonzero(
            diagnostics["valid"][graph_id], as_tuple=False
        ).flatten()
        if receivers.numel() < 2:
            continue
        features = distinct_diagnostics["return_features_dense"][graph_id]
        different_receiver_features = not bool(
            torch.allclose(
                features[receivers[0]],
                features[receivers[1]],
                atol=1e-7,
                rtol=1e-6,
            )
        )
        if different_receiver_features:
            break
    probe_changes_return = bool(
        torch.max(
            (
                diagnostics["return_features"]
                - distinct_diagnostics["return_features"]
            ).abs()
        ).item()
        > 1e-8
    )
    receiver_specific = different_receiver_features and probe_changes_return
    other_graph_nodes = batch.batch != 0
    graph0_perturbation_isolated = bool(
        other_graph_nodes.any().item()
        and torch.allclose(
            diagnostics["return_features"][other_graph_nodes],
            distinct_diagnostics["return_features"][other_graph_nodes],
            atol=1e-6,
            rtol=1e-6,
        )
    )

    checks = {
        "target_layer": TARGET_LAYER,
        "hidden_channels": HIDDEN_CHANNELS,
        "pair_channels": PAIR_CHANNELS,
        "relation_addons": 1,
        "zero_return_projection": bool(
            torch.count_nonzero(addon.return_projection.weight).item() == 0
        ),
        "zero_update": bool(torch.count_nonzero(update).item() == 0),
        "valid_row_assignment_sum_one": valid_row_assignment,
        "padding_assignment_zero": padding_assignment,
        "finite": finite,
        "different_receiver_features_within_graph": different_receiver_features,
        "graph0_perturbation_isolated": graph0_perturbation_isolated,
        "receiver_specific_return_features": receiver_specific,
        "permutation_equivariant_before_return": _remote_permutation_check(
            addon,
            probe,
            batch.batch,
            batch.edge_index,
            atol=2e-5,
            rtol=2e-4,
        ),
    }
    if addon.mode == MODE_RRWP_PAIR:
        checks.update(
            _remote_rrwp_checks(model, batch, dtype=dtype)
        )
    if addon.mode == MODE_TRIPLET_AGGREGATE:
        checks["tgt_vector_loop_agreement"] = _remote_triplet_loop_check(
            addon, dtype=dtype, device=device
        )

    required = {
        "zero_return_projection",
        "zero_update",
        "valid_row_assignment_sum_one",
        "padding_assignment_zero",
        "finite",
        "different_receiver_features_within_graph",
        "graph0_perturbation_isolated",
        "receiver_specific_return_features",
        "permutation_equivariant_before_return",
    }
    if addon.mode == MODE_RRWP_PAIR:
        required.update(
            {
                "rrwp_batched_graphs_match_independent_expected",
                "rrwp_isolate_self_transition",
            }
        )
    if addon.mode == MODE_TRIPLET_AGGREGATE:
        required.add("tgt_vector_loop_agreement")
    if not all(checks[name] is True for name in required):
        raise RuntimeError(f"K1 relation sidecar invariant failed: {checks}")
    return checks

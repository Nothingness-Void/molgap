"""Opt-in device-resident training payloads; topology stays a CPU PyG batch.

Heavy node/edge/target tensors are concatenated once on the device in stored dtype
(no casts). Each step collates only topology on CPU exactly like
``Batch.from_data_list`` and gathers payload rows in the sampler's order. The batch
has no ``_slice_dict``/``_inc_dict``: ``to_data_list`` and ``get_example`` are
unsupported, and callers refuse a mode with a ``prepare_batch`` hook.
"""
import torch
from torch_geometric.data import Batch

# Fixed by name: PyG's size-based is_node_attr guess flips for 1-node graphs.
KINDS = {"x": "node", "random_walk_pe": "node", "edge_attr": "edge", "y": "graph", "source_idx": "graph"}
DEVICE_KEYS = {"x", "random_walk_pe", "edge_attr", "y"}


def _ranges(ptr, index, counts, total):
    """Row ids of the concatenated ranges ptr[i]:ptr[i]+counts for each i in index."""
    offsets = torch.cumsum(counts, 0) - counts
    return (torch.repeat_interleave(ptr[index] - offsets, counts, output_size=total)
            + torch.arange(total, device=ptr.device))


class DeviceResidentTrain:
    """One pass over the dataset builds CPU concatenations; ``upload`` moves and frees them."""
    def __init__(self, dataset, rows):
        if getattr(dataset, "transform", None) is not None:
            raise ValueError("Device-resident inputs cannot reproduce a dataset transform")
        graphs = [dataset[i] for i in range(rows)]
        first = graphs[0]
        self.rows = rows
        self.node_ptr = torch.zeros(rows + 1, dtype=torch.long)
        self.edge_ptr = torch.zeros(rows + 1, dtype=torch.long)
        self.node_ptr[1:] = torch.cumsum(torch.tensor([g.num_nodes for g in graphs]), 0)
        self.edge_ptr[1:] = torch.cumsum(torch.tensor([g.edge_index.shape[1] for g in graphs]), 0)
        self.edge_index = torch.cat([g.edge_index for g in graphs], dim=1)
        self.constants, self.cpu, self.fields = {}, {}, {}
        self.order = list(first.keys())
        totals = {"node": int(self.node_ptr[-1]), "edge": int(self.edge_ptr[-1]), "graph": rows}
        for key in self.order:
            if key == "edge_index":
                continue
            value = first[key]
            if not torch.is_tensor(value):
                if any(g[key] != value for g in graphs):
                    raise ValueError("Non-tensor attribute is not uniform: " + key)
                self.constants[key] = value
                continue
            if key not in KINDS:
                raise ValueError("Unsupported tensor attribute for device-resident inputs: " + key)
            joined = torch.cat([g[key] for g in graphs], dim=0)
            if joined.shape[0] != totals[KINDS[key]]:
                raise ValueError("Attribute rows differ from its graph layout: " + key)
            (self.fields if key in DEVICE_KEYS else self.cpu)[key] = joined
        self.estimate = sum(t.numel() * t.element_size() for t in self.fields.values())
        self.device = None

    @staticmethod
    def admit(estimate, device, max_fraction=0.5):
        return bool(torch.cuda.is_available() and estimate <= max_fraction * torch.cuda.mem_get_info(device)[0])

    def upload(self, device):
        self.device = torch.device(device)
        self.fields = {k: t.to(self.device) for k, t in self.fields.items()}
        self.dev_node_ptr, self.dev_edge_ptr = self.node_ptr.to(self.device), self.edge_ptr.to(self.device)
        return self


class _DeviceStream:
    def __init__(self, store, sampler):
        self.store, self.sampler = store, sampler

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        s = self.store
        cuda = s.device.type == "cuda"
        for index in self.sampler:
            index = torch.as_tensor(index, dtype=torch.long)
            size = len(index)
            nodes = s.node_ptr[index + 1] - s.node_ptr[index]
            edges = s.edge_ptr[index + 1] - s.edge_ptr[index]
            ptr = torch.zeros(size + 1, dtype=torch.long)
            ptr[1:] = torch.cumsum(nodes, 0)
            total_nodes, total_edges = int(ptr[-1]), int(edges.sum())
            # One host-to-device copy per batch; row ranges are then built on the device.
            device_index = index.pin_memory().to(s.device, non_blocking=True) if cuda else index
            node_rows = _ranges(s.dev_node_ptr, device_index, s.dev_node_ptr[device_index + 1] - s.dev_node_ptr[device_index], total_nodes)
            edge_rows = _ranges(s.dev_edge_ptr, device_index, s.dev_edge_ptr[device_index + 1] - s.dev_edge_ptr[device_index], total_edges)
            rows = {"node": node_rows, "edge": edge_rows, "graph": device_index}
            batch = Batch()
            for key in s.order:
                if key == "edge_index":
                    cols = s.edge_index[:, _ranges(s.edge_ptr, index, edges, total_edges)]
                    batch[key] = cols + torch.repeat_interleave(ptr[:-1], edges)
                elif key in s.constants:
                    batch[key] = [s.constants[key]] * size
                elif key in s.cpu:
                    batch[key] = s.cpu[key][index]
                else:
                    batch[key] = s.fields[key].index_select(0, rows[KINDS[key]])
            batch.batch = torch.repeat_interleave(torch.arange(size), nodes)
            batch.ptr = ptr
            batch._num_graphs = size
            yield batch


def device_training_stream(store, sampler):
    return _DeviceStream(store, sampler)

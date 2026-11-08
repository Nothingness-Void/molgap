"""Opt-in device-resident training payloads; topology stays a CPU PyG batch.

Heavy node/edge/target tensors are concatenated once on the device in stored dtype
(no casts). Each step collates only topology on CPU exactly like
``Batch.from_data_list`` and gathers payload rows in the sampler's order.
"""
import torch
from torch_geometric.data import Batch


def _ranges(ptr, index, counts):
    """Row ids of the concatenated ranges ptr[i]:ptr[i]+counts for each i in index."""
    offsets = torch.cumsum(counts, 0) - counts
    return torch.repeat_interleave(ptr[index] - offsets, counts) + torch.arange(int(counts.sum()))


class DeviceResidentTrain:
    def __init__(self, dataset, rows, *, device, max_fraction=0.5):
        graphs = [dataset[i] for i in range(rows)]
        first = graphs[0]
        self.device, self.max_fraction, self.rows = torch.device(device), max_fraction, rows
        self.node_ptr = torch.zeros(rows + 1, dtype=torch.long)
        self.edge_ptr = torch.zeros(rows + 1, dtype=torch.long)
        self.node_ptr[1:] = torch.cumsum(torch.tensor([g.num_nodes for g in graphs]), 0)
        self.edge_ptr[1:] = torch.cumsum(torch.tensor([g.edge_index.shape[1] for g in graphs]), 0)
        self.edge_index = torch.cat([g.edge_index for g in graphs], dim=1)
        self.constants, self.cpu, self.fields = {}, {}, {}
        self.order = list(first.keys())
        for key in self.order:
            if key == "edge_index":
                continue
            value = first[key]
            if not torch.is_tensor(value):
                if any(g[key] != value for g in graphs):
                    raise ValueError("Non-tensor attribute is not uniform: " + key)
                self.constants[key] = value
                continue
            kind = "node" if first.is_node_attr(key) else "edge" if first.is_edge_attr(key) else "graph"
            joined = torch.cat([g[key] for g in graphs], dim=0)
            if kind == "graph" and key != "y":
                self.cpu[key] = joined
            else:
                self.fields[key] = (kind, joined.to(self.device))

    @staticmethod
    def estimate_bytes(dataset):
        total = 0
        for i in range(len(dataset)):
            g = dataset[i]
            total += sum(g[k].numel() * g[k].element_size() for k in g.keys()
                         if torch.is_tensor(g[k]) and (k == "y" or g.is_node_attr(k) or
                                                         (g.is_edge_attr(k) and k != "edge_index")))
        return total

    @staticmethod
    def admit(estimate, device, max_fraction=0.5):
        return bool(torch.cuda.is_available() and estimate <= max_fraction * torch.cuda.mem_get_info(device)[0])


class _DeviceStream:
    def __init__(self, store, sampler):
        self.store, self.sampler = store, sampler

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        s = self.store
        for index in self.sampler:
            index = torch.as_tensor(index, dtype=torch.long)
            size = len(index)
            nodes = s.node_ptr[index + 1] - s.node_ptr[index]
            edges = s.edge_ptr[index + 1] - s.edge_ptr[index]
            node_rows, edge_rows = _ranges(s.node_ptr, index, nodes), _ranges(s.edge_ptr, index, edges)
            ptr = torch.zeros(size + 1, dtype=torch.long)
            ptr[1:] = torch.cumsum(nodes, 0)
            batch = Batch()
            for key in s.order:
                if key == "edge_index":
                    batch[key] = s.edge_index[:, edge_rows] + torch.repeat_interleave(ptr[:-1], edges)
                elif key in s.constants:
                    batch[key] = [s.constants[key]] * size
                elif key in s.cpu:
                    batch[key] = s.cpu[key][index]
                else:
                    kind, tensor = s.fields[key]
                    rows = node_rows if kind == "node" else edge_rows if kind == "edge" else index
                    batch[key] = tensor.index_select(0, rows.to(s.device))
            batch.batch = torch.repeat_interleave(torch.arange(size), nodes)
            batch.ptr = ptr
            batch._num_graphs = size
            yield batch


def device_training_stream(store, sampler):
    return _DeviceStream(store, sampler)

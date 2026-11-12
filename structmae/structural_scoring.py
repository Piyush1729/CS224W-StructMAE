"""PageRank scoring for unweighted, undirected PyG graphs (StructMAE Eq. 3)."""
import torch
from torch import Tensor
from torch_geometric.utils import coalesce, to_undirected


def graph_batch(num_nodes: int, batch: Tensor | None, device) -> Tensor:
    """Validate membership; arbitrary nonnegative graph IDs are allowed."""
    if batch is None:
        return torch.zeros(num_nodes, dtype=torch.long, device=device)
    if batch.shape != (num_nodes,) or batch.dtype != torch.long:
        raise ValueError('batch must be a long vector with one entry per node')
    if batch.device != device or (batch < 0).any():
        raise ValueError('batch must be nonnegative and on the input device')
    return batch


@torch.no_grad()
def pagerank_scores(edge_index: Tensor, num_nodes: int, batch: Tensor | None = None,
                    damping: float = 0.85, tol: float = 1e-8,
                    max_iter: int = 1000) -> Tensor:
    """Return float64 PageRank, normalized to sum to one **per graph**.

    Inputs are PyG ``Data.edge_index``, ``Data.num_nodes`` and optionally
    ``Batch.batch``. Edges are made undirected and duplicates removed, matching
    the authors' NetworkX Graph construction. Self-loops are retained. Isolates
    use uniform dangling-mass redistribution within their own graph. No graph
    can influence another graph's teleportation or normalization. Cross-graph
    edges are rejected. Raises RuntimeError on failure to converge.
    """
    if not isinstance(num_nodes, int) or num_nodes < 0:
        raise ValueError('num_nodes must be a nonnegative integer')
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError('edge_index must have shape [2, E] and dtype long')
    if not 0 < damping < 1 or tol <= 0 or max_iter < 1:
        raise ValueError('invalid damping, tolerance or iteration limit')
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError('edge endpoint outside num_nodes')
    membership = graph_batch(num_nodes, batch, edge_index.device)
    if edge_index.numel() and (membership[edge_index[0]] != membership[edge_index[1]]).any():
        raise ValueError('edges must not cross graph boundaries')
    result = torch.empty(num_nodes, dtype=torch.float64, device=edge_index.device)
    edges = coalesce(to_undirected(edge_index, num_nodes=num_nodes), num_nodes=num_nodes)
    for graph_id in membership.unique():
        nodes = (membership == graph_id).nonzero(as_tuple=True)[0]
        n = nodes.numel()
        mapping = membership.new_full((num_nodes,), -1)
        mapping[nodes] = torch.arange(n, device=nodes.device)
        selected = membership[edges[0]] == graph_id
        src, dst = mapping[edges[:, selected]]
        degree = torch.bincount(src, minlength=n).to(torch.float64)
        score = torch.full((n,), 1 / n, dtype=torch.float64, device=nodes.device)
        for _ in range(max_iter):
            incoming = torch.zeros_like(score)
            incoming.index_add_(0, dst, score[src] / degree[src])
            updated = (1 - damping) / n + damping * (incoming + score[degree == 0].sum() / n)
            # Same L1 stopping convention as NetworkX, with tighter tolerance.
            if (updated - score).abs().sum() < n * tol:
                score = updated
                break
            score = updated
        else:
            raise RuntimeError('PageRank did not converge; increase max_iter')
        result[nodes] = score / score.sum()
    return result

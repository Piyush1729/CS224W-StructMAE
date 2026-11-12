import networkx as nx
import pytest
import torch
from torch_geometric.data import Data, Batch
from structmae import pagerank_scores


def edges(pairs):
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()


@pytest.mark.parametrize('pairs,n', [([], 0), ([], 1), ([], 4),
    ([(0, 1), (0, 2), (0, 3)], 4), ([(0, 1), (2, 3)], 5),
    ([(0, 0), (0, 1), (0, 1), (1, 2)], 3)])
def test_matches_networkx(pairs, n):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(pairs)
    expected = nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=1000)
    actual = pagerank_scores(edges(pairs), n)
    torch.testing.assert_close(actual, torch.tensor(list(expected.values()), dtype=torch.float64), atol=1e-7, rtol=1e-6)


def test_star_analytical_center():
    result = pagerank_scores(edges([(0, 1), (0, 2), (0, 3)]), 4)
    # c = (1-d)/4 + d*(1-c), independent closed-form stationary center.
    assert result[0].item() == pytest.approx((0.15 / 4 + 0.85) / 1.85, abs=1e-7)
    assert (result[0] > result[1:]).all()


def test_batch_matches_individual_graphs_including_isolates():
    graphs = [Data(edge_index=edges([(0, 1), (0, 2)]), num_nodes=4),
              Data(edge_index=edges([]), num_nodes=2)]
    batch = Batch.from_data_list(graphs)
    result = pagerank_scores(batch.edge_index, batch.num_nodes, batch.batch)
    expected = torch.cat([pagerank_scores(g.edge_index, g.num_nodes) for g in graphs])
    torch.testing.assert_close(result, expected)
    assert result[:4].sum() == pytest.approx(1.)
    assert result[4:].sum() == pytest.approx(1.)


def test_invalid_edges_and_convergence():
    with pytest.raises(ValueError):
        pagerank_scores(edges([(0, 2)]), 2)
    with pytest.raises(ValueError, match='cross'):
        pagerank_scores(edges([(0, 1)]), 2, torch.tensor([0, 1]))
    with pytest.raises(RuntimeError):
        pagerank_scores(edges([(0, 1), (0, 2)]), 3, max_iter=1)
    with pytest.raises(ValueError):
        pagerank_scores(edges([]), 2, damping=1)

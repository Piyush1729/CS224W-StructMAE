import pytest
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from structmae import GCNEncoder, StructMAE


def graph():
    return Data(x=torch.eye(3).repeat(2, 1), edge_index=torch.tensor(
        [[0,1,1,2,2,3,3,4,4,5,5,0], [1,0,2,1,3,2,4,3,5,4,0,5]]))


def model():
    torch.manual_seed(7)
    return StructMAE(GCNEncoder(3, 8), GCNConv(8, 3), 3, 8)


def test_shapes_gradients_and_optimization():
    data, net = graph(), model()
    assert net.encode(data.x, data.edge_index).shape == (6, 8)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    before = {n:p.detach().clone() for n,p in net.named_parameters()}
    loss = net.pretraining_loss(data.x, data.edge_index, 1, 10,
                               generator=torch.Generator().manual_seed(42))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for prefix in ('encoder.', 'decoder.', 'projection.', 'mask_token'):
        assert any(p.grad.abs().sum() > 0 for n,p in net.named_parameters() if n.startswith(prefix))
    optimizer.step()
    assert any(not torch.equal(p, before[n]) for n,p in net.named_parameters())


def test_batch_and_cached_score_equivalence():
    from structmae import pagerank_scores
    data = Batch.from_data_list([graph(), graph()])
    net = model()
    scores = pagerank_scores(data.edge_index, data.num_nodes, data.batch)
    kwargs = dict(epoch=2, total_epochs=10, batch=data.batch)
    a = net.pretraining_loss(data.x, data.edge_index, **kwargs, generator=torch.Generator().manual_seed(1))
    b = net.pretraining_loss(data.x, data.edge_index, **kwargs, scores=scores, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize('n', [0, 1])
def test_empty_masks_safe_backward(n):
    net = model()
    loss = net.pretraining_loss(torch.ones(n, 3), torch.empty(2, 0, dtype=torch.long), 0, 1)
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())


def test_loss_is_masked_only_and_decoder_latents_remasked():
    class IdentityEncoder(nn.Module):
        def forward(self, x, edge_index):
            return x
    class InspectDecoder(nn.Module):
        def forward(self, x, edge_index):
            self.seen = x.detach().clone()
            return torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    decoder = InspectDecoder()
    net = StructMAE(IdentityEncoder(), decoder, 2, 2, beta=1, loss_power=2)
    x = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [1., 0.]])
    scores = torch.arange(4, dtype=torch.float)
    # Last two nodes are the only targets: cosine=0 on each => loss=1.
    loss = net.pretraining_loss(x, torch.empty(2,0,dtype=torch.long), 1, 1, scores=scores)
    assert loss.item() == pytest.approx(1.)
    assert torch.equal(decoder.seen[2:], torch.zeros(2,2))

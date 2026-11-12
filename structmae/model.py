"""Minimal GraphMAE-style encoder/decoder using the StructMAE-P masker."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from .structural_scoring import pagerank_scores
from .curriculum_masking import CurriculumMasker


class GCNEncoder(nn.Module):
    """Two PyG GCN layers with PReLU, mapping input to hidden channels."""
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.activation = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.conv2(self.activation(self.conv1(x, edge_index)), edge_index)


class StructMAE(nn.Module):
    """Injectable GNN encoder and decoder, each accepting (x, edge_index).

    Decoder input/output widths must be hidden_channels/in_channels. A learned
    input mask token replaces targets; a linear projection and zero latent
    re-mask precede graph decoding. Scaled cosine error is computed only on
    selected nodes. No edge dropout, feature replacement, hidden concatenation,
    or learned scorer is included in this minimal configuration.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 in_channels: int, hidden_channels: int, mask_ratio: float = 0.5,
                 scoring: str = 'pagerank', mode: str = 'curriculum',
                 beta: float = 0.5, loss_power: float = 1.0):
        super().__init__()
        if scoring != 'pagerank' or loss_power <= 0:
            raise ValueError('only pagerank scoring and positive loss_power supported')
        self.encoder, self.decoder = encoder, decoder
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels))
        self.projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.masker = CurriculumMasker(mask_ratio, beta, mode)
        self.loss_power = loss_power

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return unmasked node embeddings (caller controls train/eval mode)."""
        return self.encoder(x, edge_index)

    def pretraining_loss(self, x: Tensor, edge_index: Tensor, epoch: int,
                         total_epochs: int, batch: Tensor | None = None,
                         scores: Tensor | None = None,
                         generator: torch.Generator | None = None) -> Tensor:
        """Return scalar masked-only SCE; empty selection gives differentiable zero.

        Optional cached scores must correspond exactly to this node ordering and
        device. Pass Batch.batch to preserve per-graph mask budgets and PageRank.
        """
        if x.ndim != 2 or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('x must be a finite floating feature matrix')
        if scores is None:
            scores = pagerank_scores(edge_index, x.size(0), batch)
        if scores.shape != (x.size(0),) or scores.device != x.device:
            raise ValueError('scores must align with x')
        selection = self.masker(scores, epoch, total_epochs, batch, generator)
        mask = selection.mask
        if not mask.any():
            return sum((p.sum() * 0 for p in self.parameters()), x.sum() * 0)
        corrupted = torch.where(mask[:, None], self.mask_token, x)
        hidden = self.projection(self.encoder(corrupted, edge_index))
        hidden = hidden.masked_fill(mask[:, None], 0)
        reconstruction = self.decoder(hidden, edge_index)
        if reconstruction.shape != x.shape:
            raise ValueError('decoder output must have the same shape as x')
        cosine = (F.normalize(reconstruction[mask], dim=-1) * F.normalize(x[mask], dim=-1)).sum(-1)
        return (1 - cosine).clamp_min(0).pow(self.loss_power).mean()

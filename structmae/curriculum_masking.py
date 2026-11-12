"""StructMAE Eqs. 6-7: rank, bias uniform priorities, then select targets."""
import math
from dataclasses import dataclass
import torch
from torch import Tensor
from .structural_scoring import graph_batch


@dataclass
class MaskSelection:
    """Boolean node masks, eligible pool, and sampled priorities for inspection."""
    mask: Tensor
    pool: Tensor
    priorities: Tensor


class CurriculumMasker:
    """Select targets independently per graph.

    ``K(t) = ceil(p*n*sqrt(t/T))``; K(0)=0 explicitly. The paper leaves
    rounding unspecified. Following reference graph code, the final target
    count is n-ceil((1-p)*n) = floor(p*n), while the biased pool uses ceil.
    Thus tiny graphs can have no targets. At p=0 and p=1 the API gives none
    and all targets respectively, avoiding the reference topk ratio=1 bug.
    Ties use stable node-index ordering. beta is an additive priority bias,
    not a normalized probability. Generators must match the scores' device.
    """
    def __init__(self, mask_ratio: float = 0.5, beta: float = 0.5,
                 mode: str = 'curriculum'):
        if not 0 <= mask_ratio <= 1 or not math.isfinite(beta) or beta < 0:
            raise ValueError('require mask_ratio in [0,1] and finite beta >= 0')
        if mode not in ('curriculum', 'random'):
            raise ValueError('mode must be curriculum or random')
        self.mask_ratio, self.beta, self.mode = mask_ratio, beta, mode

    def __call__(self, scores: Tensor, epoch: int, total_epochs: int,
                 batch: Tensor | None = None,
                 generator: torch.Generator | None = None) -> MaskSelection:
        if total_epochs <= 0 or not 0 <= epoch <= total_epochs:
            raise ValueError('require total_epochs > 0 and 0 <= epoch <= total_epochs')
        if scores.ndim != 1 or not scores.is_floating_point() or not torch.isfinite(scores).all():
            raise ValueError('scores must be a finite floating vector')
        membership = graph_batch(scores.numel(), batch, scores.device)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        pool = torch.zeros_like(mask)
        priorities = torch.rand(scores.shape, device=scores.device,
                                dtype=scores.dtype, generator=generator)
        for graph_id in membership.unique():
            nodes = (membership == graph_id).nonzero(as_tuple=True)[0]
            n = nodes.numel()
            count = math.floor(self.mask_ratio * n)
            if self.mode == 'curriculum':
                k = math.ceil(self.mask_ratio * n * math.sqrt(epoch / total_epochs))
                ranked = nodes[torch.argsort(scores[nodes], descending=True, stable=True)]
                pool[ranked[:k]] = True
                priorities[ranked[:k]] += self.beta
            ranked = nodes[torch.argsort(priorities[nodes], descending=True, stable=True)]
            mask[ranked[:count]] = True
        return MaskSelection(mask, pool, priorities)

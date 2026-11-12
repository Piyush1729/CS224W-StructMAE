import math
import pytest
import torch
from torch_geometric.data import Data, Batch
from structmae import CurriculumMasker


def rng(seed=42):
    return torch.Generator().manual_seed(seed)


def test_equations_against_independent_reference_selection():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.7, 0.3, 0.6, 0.4, 0.8])
    # At t/T=1/4: K=ceil(0.5*8*0.5)=2, so nodes 1 and 7.
    noise = torch.rand(8, generator=rng())
    noise[[1, 7]] += 0.5
    # Reference code retains lowest ceil((1-p)*n) priorities.
    retained = sorted(range(8), key=lambda i: float(noise[i]))[:4]
    expected = set(range(8)) - set(retained)
    actual = CurriculumMasker()(scores, 1, 4, generator=rng())
    assert set(actual.mask.nonzero().flatten().tolist()) == expected
    assert actual.pool.nonzero().flatten().tolist() == [1, 7]
    torch.testing.assert_close(actual.priorities, noise)


def test_progression_and_high_score_direction():
    scores = torch.arange(100, dtype=torch.float)
    masker = CurriculumMasker(beta=1)
    selections = [masker(scores, t, 100, generator=rng()) for t in (0, 25, 100)]
    assert [s.pool.sum().item() for s in selections] == [0, 25, 50]
    assert selections[-1].mask[:50].sum() == 0
    assert selections[-1].mask[50:].all()
    assert all(s.mask.sum() == 50 for s in selections)


def test_fixed_seed_and_epoch_zero_random_equivalence():
    scores = torch.arange(21, dtype=torch.float)
    c = CurriculumMasker()(scores, 0, 10, generator=rng())
    r = CurriculumMasker(mode='random')(scores, 10, 10, generator=rng())
    assert torch.equal(c.mask, r.mask)
    assert torch.equal(c.mask, CurriculumMasker()(scores, 0, 10, generator=rng()).mask)
    assert not torch.equal(c.mask, CurriculumMasker()(scores, 0, 10, generator=rng(99)).mask)


@pytest.mark.parametrize('n,p', [(0, 0.5), (1, 0.5), (3, 0.5), (5, 0), (5, 1), (10, 0.7), (10, 0.3)])
def test_edge_counts(n, p):
    selection = CurriculumMasker(p)(torch.ones(n), 10, 10, generator=rng())
    assert selection.mask.sum() == math.floor(p*n)
    assert selection.mask.dtype == torch.bool


def test_pyg_batch_has_independent_budgets():
    batch = Batch.from_data_list([Data(x=torch.ones(3, 1)), Data(x=torch.ones(8, 1))])
    selected = CurriculumMasker(beta=1)(torch.arange(11, dtype=torch.float), 4, 4, batch.batch, rng())
    assert selected.mask[:3].sum() == 1
    assert selected.mask[3:].sum() == 4
    assert selected.pool[:3].sum() == 2
    assert selected.pool[3:].sum() == 4


def test_invalid_inputs():
    with pytest.raises(ValueError):
        CurriculumMasker(-0.1)
    with pytest.raises(ValueError):
        CurriculumMasker()(torch.ones(3), 11, 10)
    with pytest.raises(ValueError):
        CurriculumMasker()(torch.tensor([float('nan')]), 0, 1)
    with pytest.raises(ValueError):
        CurriculumMasker()(torch.ones(3), 0, 1, torch.zeros(2, dtype=torch.long))

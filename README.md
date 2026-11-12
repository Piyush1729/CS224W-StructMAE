# StructMAE in PyTorch Geometric

Piyush Gade | Stanford CS224W, Fall 2026 | Project 3 milestone

I implemented StructMAE-P with modular components for PageRank scoring, structure-guided
curriculum masking, and masked-feature reconstruction using PyG GCN layers.
This independent project is intended for later adaptation into
`torch_geometric.contrib`; it is not an upstream PyG PR.

- [Repository](https://github.com/Piyush1729/CS224W-StructMAE)
- [Executed Colab session](https://colab.research.google.com/drive/1JA-mS6bwwKbfHbgyCNB4O1AE9OI_sVV2?usp=sharing)
- [StructMAE paper, IJCAI 2024](https://www.ijcai.org/proceedings/2024/241)
- [Author implementation](https://github.com/LiuChuang0059/StructMAE)
- [GraphMAE paper](https://arxiv.org/abs/2205.10803)

## Install and run

From the repository root, with Python and CPU PyTorch installed:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v
python examples/train_mutag.py
```

Internet is needed for packages and the first MUTAG download. Optional CPU-only
PyTorch installation: `python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`.
No optional compiled PyG extensions are required. Default training is CPU-only.

## Google Colab

Use a fresh CPU runtime and execute:

```python
!git clone https://github.com/Piyush1729/CS224W-StructMAE.git
%cd /content/CS224W-StructMAE
!git checkout 7a29a12375adf7622bb51ac687dfa56904456ae3
%pip install -r requirements.txt
!python -m pytest tests/ -v
!python examples/train_mutag.py
```

The pinned revision is the actual code used in the successful Colab run.
It predates documentation cleanup; omit checkout to inspect the latest docs.
No ZIP upload, Drive mount, or source edits are required. Save the executed
session with outputs visible in Drive. The notebook only executes the Python
project; no implementation is duplicated in a distributed notebook.

## Experiment and evidence

MUTAG: 188 graphs, 3,371 nodes, seven atom-type node features. A fixed seed-42
split uses 150 training and 38 held-out graphs; labels and edge labels are
unused. Six runs compare random and curriculum masking with matched initial
weights and shuffle order: seeds 0/1/2, 30 epochs, hidden width 64, batch size 32,
Adam lr=0.001, mask ratio 0.5, priority bias 0.5, and SCE exponent 1.
Held-out evaluation uses identical fixed random masks (seed 2026) for both methods.

`outputs/metrics.json` records the run's Git SHA, versions, split indices,
archive SHA256, curves, timing, and masking diagnostics. Unmasked sample
embeddings are exported as `.pt` files. Generated data, outputs and caches are
ignored. Retained final local evidence is under `evidence/`; the Colab summary
records the results of my completed run.

| Method | Held-out SCE, mean ± sample SD (3 seeds) |
| --- | --- |
| Random | 0.1874 ± 0.0102 |
| Curriculum | 0.2124 ± 0.0099 |

Random masking achieved lower held-out loss on all three seeds. Curriculum
moved selected-node PageRank percentiles from about 0.496 to 0.634. Different
training target distributions make training SCE an unsuitable direct comparison.
These results demonstrate working training, not improved downstream accuracy.

I ran the tests and training in Colab: **26 passed, one deprecation warning,
9.39 seconds**, plus six completed MUTAG runs. Python 3.13.18, PyTorch
2.14.1+cpu, PyG 2.6.1. See [VERIFICATION.md](VERIFICATION.md) for execution details.

## Method and API

The inspected author revision is `9f723ffc70a40c100e26af7e63d64d571dde43ce`,
especially `StructMAE-P/graphmae/models/edcoder.py` and `main_graph.py`.

- Eq. 3: undirected, unweighted PageRank with damping 0.85 and sum-one
  normalization per graph. Reciprocal/duplicate edges collapse; self-loops
  remain. Isolates use uniform within-graph dangling-mass redistribution.
  Distinct Batch graphs do not share teleportation mass; disconnected
  components within one Data do. Cross-graph edges are rejected.
- Eqs. 6-7: select the highest-PageRank pool of
  `ceil(p*n*sqrt(epoch/total_epochs))` nodes. Add beta to their U(0,1)
  priorities and mask the highest priorities. These are ranking priorities,
  not Bernoulli probabilities. Each graph masks `floor(p*n)` nodes,
  mathematically equivalent to the author's keep-lowest complement rule.
- The paper leaves integer rounding unspecified. Pool size uses ceil;
  target count uses floor. Epoch zero has no biased pool, ratios zero/one
  give none/all, and score ties resolve stably by node index. Tiny graphs
  may have no targets and yield differentiable zero loss.
- A learned input token replaces target features. Two GCN encoder layers
  with PReLU, a bias-free projection, zero latent re-mask and a GCN decoder
  predict features. Masked-only loss is
  `mean((1-cosine(reconstruction, original))**loss_power)`.
  PyTorch normalization handles zero norms with its epsilon convention.
- PyTorch RNG replaces NumPy RNG: equivalent sampling distribution, not
  identical author draws. Edge dropout, feature replacement, hidden
  concatenation and learned scoring are outside this minimal configuration.

```python
from torch_geometric.nn import GCNConv
from structmae import StructMAE, GCNEncoder, pagerank_scores

model = StructMAE(GCNEncoder(7, 64), GCNConv(64, 7),
                  in_channels=7, hidden_channels=64)
# For a single Data, omit batch=; for a PyG Batch, pass data.batch.
scores = pagerank_scores(data.edge_index, data.num_nodes)
loss = model.pretraining_loss(data.x, data.edge_index,
    epoch=1, total_epochs=30, scores=scores)
loss.backward()
z = model.encode(data.x, data.edge_index)
```

| API | Contract |
| --- | --- |
| `pagerank_scores` | edge_index, num_nodes, optional batch; returns float64 node scores |
| `CurriculumMasker` | ratio, beta, mode; call with scores, epoch, T, optional batch/generator; returns mask, pool, priorities |
| `StructMAE` | Encoder/decoder accepting `(x, edge_index)`; input/hidden channel counts required |
| `pretraining_loss` | x, edges, epoch, T, optional batch, cached scores, generator; scalar masked-only SCE |
| `encode` | Unmasked node embeddings; caller sets train/eval mode |

Cached scores must match graph structure, node ordering and device. Core tensors
retain their devices; validation and the example cover CPU. GPU, compilation,
TorchScript, and large-graph performance are unverified. Scoring loops over graphs.

## Scope and report

Tests cover independent NetworkX and analytical PageRank, batching, masking
rules, boundaries, reproducibility, finite gradients, masked-only SCE, latent
re-masking and an optimizer step. Remaining work: learned SBS, PROTEINS,
downstream SVM probing, full benchmark reproduction, API review, PyG fork/PR,
contributor checks and technical blog. No upstream acceptance is assumed.

The report PDF, LaTeX source and official NeurIPS style are in `report/`.
Rebuild with two `pdflatex -interaction=nonstopmode -halt-on-error milestone_report.tex`
runs from that directory in a full LaTeX installation.

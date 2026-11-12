"""CPU MUTAG demonstration with matched random/curriculum reconstruction runs."""
import argparse
import hashlib
import io
import urllib.request
import zipfile
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import sys
import time

# Direct script execution works from a checkout or extracted ZIP.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from structmae import StructMAE, GCNEncoder, pagerank_scores, CurriculumMasker


def ensure_mutag_raw(root):
    """Download official TU files with standard proxy-aware HTTPS, then use PyG.

    PyG's aiohttp downloader may bypass proxy environment variables. Keeping
    transport here also makes the exact archive hash available in the evidence.
    """
    raw = root / 'MUTAG' / 'raw'
    provenance = root / 'MUTAG' / 'archive_sha256.txt'
    if not (raw / 'MUTAG_A.txt').exists():
        url = TUDataset.url + '/MUTAG.zip'
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for entry in archive.infolist():
                name = Path(entry.filename).name
                if name.startswith('MUTAG_') and name.endswith('.txt'):
                    (raw / name).write_bytes(archive.read(entry))
        provenance.write_text(hashlib.sha256(content).hexdigest() + '\n')
    return provenance.read_text().strip() if provenance.exists() else 'pre-existing raw files'


def git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable (ZIP without .git); see evidence/source_commit.txt'


def target_count(batch):
    sizes = torch.bincount(batch.batch).tolist()
    return sum(n - math.ceil(0.5*n) for n in sizes)


def mask_observation(graphs, mode, epoch, total, seed):
    masker = CurriculumMasker(mode=mode)
    generator = torch.Generator().manual_seed(seed)
    selected_ranks, pool, count = [], 0, 0
    for graph in graphs:
        sel = masker(graph.pr, epoch, total, generator=generator)
        scores = graph.pr
        # Average percentiles for tied PageRank scores.
        ranks = ((scores[:, None] > scores[None, :]).sum(1)
                 + 0.5 * (scores[:, None] == scores[None, :]).sum(1)) / len(scores)
        selected_ranks.extend(ranks[sel.mask].tolist())
        pool += int(sel.pool.sum())
        count += int(sel.mask.sum())
    return dict(masked_nodes=count, pool_nodes=pool,
                mean_masked_pagerank_percentile=sum(selected_ranks) / len(selected_ranks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_hash = ensure_mutag_raw(args.data_dir)
    dataset = TUDataset(str(args.data_dir), name='MUTAG')
    graphs = []
    for data in dataset:
        graph = data.clone()
        graph.pr = pagerank_scores(graph.edge_index, graph.num_nodes)
        graphs.append(graph)
    permutation = torch.randperm(len(graphs), generator=torch.Generator().manual_seed(42))
    cut = int(0.8 * len(graphs))
    train = [graphs[i] for i in permutation[:cut]]
    valid = [graphs[i] for i in permutation[cut:]]
    env = dict(python=platform.python_version(), pytorch=torch.__version__,
               pyg=torch_geometric.__version__, git_commit=git_revision(), device='cpu')
    report = dict(environment=env, dataset=dict(name='MUTAG', graphs=len(graphs),
        train_graphs=len(train), validation_graphs=len(valid), features=dataset.num_features,
        nodes=sum(g.num_nodes for g in graphs), split_seed=42, archive_sha256=archive_hash,
        train_indices=permutation[:cut].tolist(), validation_indices=permutation[cut:].tolist()),
        config=dict(epochs=args.epochs, seeds=args.seeds, batch_size=32, hidden=64,
                    lr=0.001, mask_ratio=0.5, beta=0.5, loss_power=1,
                    note='validation uses identical fixed random masks for both methods; no labels used'), runs=[])
    print(json.dumps(dict(environment=env, dataset='MUTAG'), indent=2), flush=True)
    for seed in args.seeds:
        for mode in ('random', 'curriculum'):
            random.seed(seed)
            torch.manual_seed(seed)
            net = StructMAE(GCNEncoder(dataset.num_features, 64),
                            GCNConv(64, dataset.num_features), dataset.num_features, 64,
                            mode=mode)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            train_loader = DataLoader(train, batch_size=32, shuffle=True,
                                      generator=torch.Generator().manual_seed(seed))
            valid_loader = DataLoader(valid, batch_size=32, shuffle=False)
            masking_rng = torch.Generator().manual_seed(seed + 1000)
            curve = []
            start = time.perf_counter()
            for epoch in range(1, args.epochs + 1):
                net.train()
                total_loss, targets = 0., 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    loss = net.pretraining_loss(batch.x, batch.edge_index, epoch, args.epochs,
                        batch=batch.batch, scores=batch.pr, generator=masking_rng)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite training loss')
                    loss.backward()
                    optimizer.step()
                    n = target_count(batch)
                    total_loss += loss.item() * n
                    targets += n
                curve.append(total_loss / targets)
            runtime = time.perf_counter() - start
            net.eval()
            original_mode = net.masker.mode
            net.masker.mode = 'random'
            validation_rng = torch.Generator().manual_seed(2026)
            val_loss, val_targets = 0., 0
            with torch.no_grad():
                for batch in valid_loader:
                    loss = net.pretraining_loss(batch.x, batch.edge_index, 0, args.epochs,
                        batch=batch.batch, scores=batch.pr, generator=validation_rng)
                    n = target_count(batch)
                    val_loss += loss.item() * n
                    val_targets += n
                sample = graphs[0]
                embeddings = net.encode(sample.x, sample.edge_index)
                assert embeddings.shape == (sample.num_nodes, 64) and torch.isfinite(embeddings).all()
                torch.save(embeddings, args.output_dir / f'embeddings_{mode}_seed{seed}.pt')
            net.masker.mode = original_mode
            row = dict(seed=seed, mode=mode, train_sce=curve, final_train_sce=curve[-1],
                       validation_random_mask_sce=val_loss/val_targets, training_seconds=runtime,
                       early_mask=mask_observation(train, mode, 0, args.epochs, seed),
                       late_mask=mask_observation(train, mode, args.epochs, args.epochs, seed),
                       embedding_shape=list(embeddings.shape))
            report['runs'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k != 'train_sce'}), flush=True)
    destination = args.output_dir / 'metrics.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {destination}', flush=True)


if __name__ == '__main__':
    main()

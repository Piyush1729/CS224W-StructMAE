# Verification summary

## Public GitHub-to-Colab execution

I ran the tests and MUTAG training in Colab using public GitHub revision
`7a29a12375adf7622bb51ac687dfa56904456ae3`:

- Python 3.13.18, PyTorch 2.14.1+cpu, PyG 2.6.1; CPU runtime.
- Tests: **26 passed, 1 warning in 9.39 seconds**. The warning concerns
  deprecated `torch.jit.script`; it did not stop execution.
- Six completed MUTAG runs: random/curriculum, seeds 0/1/2, 30 epochs each.
- Results and sample embeddings saved by the example. Held-out SCE agrees
  with local results to the reported four decimal places.

[Executed Colab notebook](https://colab.research.google.com/drive/1JA-mS6bwwKbfHbgyCNB4O1AE9OI_sVV2?usp=sharing)

`evidence/colab_verification.json` summarizes my completed Colab run.
The linked notebook contains the execution outputs; the summary file does
not include the full Colab training curves.

## Original local execution

Local source revision: `af1efbe60b796db5f9fb6ecd46fe588433a12504`.
Python 3.12.14, PyTorch 2.5.1+cpu, PyG 2.6.1, NetworkX 3.4.2,
SciPy 1.14.1, pytest 8.3.5. Tests: 26 passed in 2.52 seconds.
Six runs completed and were repeated with exact equality of 180 training
losses and six validation losses on the same local CPU environment.
The public upload creates a different Git SHA for the source files.

Retained evidence: `pytest.txt`, `experiment.txt`, `metrics.json`,
`reproducibility.txt`, and `source_commit.txt`. Setup, failed-attempt,
duplicate-run and report-build logs have been removed from the deliverable.

| Method | Local held-out SCE, mean ± sample SD |
| --- | --- |
| Random | 0.1874 ± 0.0102 |
| Curriculum | 0.2124 ± 0.0099 |

Random masking had lower held-out reconstruction loss on every seed.
Curriculum shifted masking toward higher-PageRank nodes. No downstream
accuracy improvement is claimed. Learned SBS, PROTEINS, SVM probing,
full benchmark reproduction and an upstream PyG PR/blog remain future work.

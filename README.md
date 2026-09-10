# GraphGPS Upgrade

Project 12 is a fixture-first molecular graph benchmark scaffold that compares uniform GCN, GIN, and PyTorch Geometric `GPSConv` model contracts under a frozen configuration.

## Purpose

The project is for engineers and researchers who need a small, inspectable starting point for a reproducible graph-level benchmark. The release includes graph batching, graph-local positional features, uniform PyG baselines, an official OGB runner, output-shape validation, deterministic fixtures, and fail-closed reporting rules. A private one-epoch Kaggle run verifies the official benchmark path; a full-budget ablation remains pending.

## Architecture

```mermaid
flowchart LR
    A[fixture.toml] --> B[offline preflight]
    B --> C[deterministic synthetic graphs]
    C --> D[GCN / GIN / GPSConv]
    D --> E[graph logits B x 1]
    E --> F[shape and finite-value contracts]
    F --> G[official evaluator]
    G --> H[raw per-seed JSON or CSV]
    H --> I[aggregate JSON and figure]
```

The GPS variant uses PyG `GPSConv` with `GINConv` local message passing, multi-head attention with one head, and graph-local Laplacian eigenvector features. The public forward contract is `forward(x, edge_index, edge_attr, batch) -> [B, 1]`.

## Installation

```text
python -m pip install -e ".[runtime,test]"
```

The exact versions observed during local verification are recorded in `requirements-verified-local.txt` and `compatibility.md`. The local evidence used Python 3.11.9, CPU PyTorch 2.12.1, PyG 2.7.0, OGB 1.3.6, and pytest 9.0.3. Other environments must record their own versions before runtime use.

## Local synthetic validation

```text
python -m compileall -q graphgps_bench tests
python -m pytest -q
python -m graphgps_bench.static_validation --root .
python -m graphgps_bench.preflight --config configs/fixture.toml
```

Verified locally on 2026-09-11: `48 passed`, exit code `0`. PyG emitted three deprecation warnings from its distributed and JIT compatibility layers; no test failed.

## Kaggle synthetic validation

The public notebook and runbook under `notebooks/` are approval-gated. The approved private synthetic GPU smoke verified the synthetic contracts with internet used only to install `torch-geometric==2.7.0`; it did not access OGB data or train. The official benchmark notebook is also gated and must run only in a private Kaggle session after explicit runtime approval.

## Bounded official-path result

Private Kaggle kernel [`ajinkya1225/project-12-graphgps-official-ogb-benchmark`](https://www.kaggle.com/code/ajinkya1225/project-12-graphgps-official-ogb-benchmark) version 3 completed on a Tesla T4. It passed 48 pre-run tests, used the official `ogbg-molhiv` scaffold split and evaluator, and retained nine raw records across seeds 0, 1, and 2. Under the deliberately bounded one-epoch configuration, test ROC-AUC was GCN `0.6383 +/- 0.0181`, GIN `0.6007 +/- 0.0175`, and GPS `0.7052 +/- 0.0360` (mean +/- sample standard deviation). These measured values verify the runtime path; they are not a full-budget comparison and do not establish a general GraphGPS advantage.

## Verified status

- Implemented: offline TOML preflight, deterministic fixtures, real PyG GCN/GIN message passing, OGB categorical encoding, GPSConv adapter, graph-local Laplacian features, shared training/evaluation/checkpoint runner, raw-record aggregation, and figure generation.
- Locally tested: 48 tests passed on CPU; compile, static validation, and preflight passed.
- Kaggle-tested: private Tesla T4 version 3 completed with official OGB loading, training, evaluation, records, and aggregation.
- Benchmark evidence: one epoch, three seeds per model, official scaffold split and evaluator.
- Benchmark-pending: frozen full-budget ablation and scientific review.
- GPU status: bounded official-path run and synthetic smoke verified on Tesla T4.
- Public release: repository published with MIT licensing; generated outputs and checkpoints remain private.

## Reproducibility and safety

Keep the config hash, Python/PyTorch/PyG/OGB versions, device, seed set, GPS recipe, split, and artifact paths with every future run. The default config disables network, OGB download, GPU, and W&B. Do not place datasets, weights, checkpoints, generated results, credentials, or private paths in the source tree.

The device helper accepts explicit `cpu` or `cuda[:index]` and raises when requested CUDA is unavailable; it does not silently switch devices.

## Data, privacy, and security

Only public benchmark metadata is in scope. No patient data or private clinical text is used. The export contains no datasets, model files, checkpoints, credentials, or raw benchmark output. W&B is optional and disabled by default.

## Citation

- Hu et al., “Open Graph Benchmark: Datasets for Machine Learning on Graphs,” arXiv:2005.00687, 2020. See the [official OGB repository](https://github.com/snap-stanford/ogb).
- Wu et al., “MoleculeNet: a benchmark for molecular machine learning,” *Chemical Science*, 9(2):513–530, 2018. See the [OGB graph property documentation](https://ogb.stanford.edu/docs/graphprop/).

## License status

This project is released under the MIT License. The license covers the project code; third-party dependencies and cited
datasets retain their own licenses and terms.

## Roadmap

1. Obtain separate approval for the frozen full-budget ablation.
2. Review full-budget raw records and checkpoint provenance before drawing a scientific comparison.
3. Publish only measured results with complete provenance and limitations.

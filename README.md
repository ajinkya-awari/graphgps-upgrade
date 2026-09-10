# GraphGPS Upgrade

Project 12 is a fixture-first molecular graph benchmark scaffold that compares uniform GCN, GIN, and PyTorch Geometric `GPSConv` model contracts under a frozen configuration.

## Purpose

The project is for engineers and researchers who need a small, inspectable starting point for a reproducible graph-level benchmark. The current release demonstrates graph batching, graph-local positional features, output-shape validation, deterministic fixtures, and fail-closed reporting rules. It does not report a molecular benchmark result.

## Architecture

```mermaid
flowchart LR
    A[fixture.toml] --> B[offline preflight]
    B --> C[deterministic synthetic graphs]
    C --> D[GCN / GIN / GPSConv]
    D --> E[graph logits B x 1]
    E --> F[shape and finite-value contracts]
    F --> G[local JSON or CSV records]
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

Verified locally on 2026-09-10: `41 passed`, exit code `0`. PyG emitted three deprecation warnings from its distributed and JIT compatibility layers; no test failed.

## Kaggle synthetic validation

The notebook and runbook under `notebooks/` are provider-free and use no dataset sources, internet, GPU, weights, or checkpoints. Push only a source-only package with a private kernel metadata file, then run the environment check, compile check, synthetic tests, bounded GPSConv smoke test, and sanitized evidence JSON. Stop at the approval gate before OGB access, GPU work, or training.

## Future official benchmark

The official `ogbg-molhiv` run is gated and has not been executed from this export. A future approved run must preserve the official scaffold split and evaluator, use equal data/optimizer/budget/seed protocols for GCN, GIN, and GPSConv, and retain raw per-seed ROC-AUC records before aggregation. No score, winner, or scientific comparison is included here.

## Verified status

- Implemented: offline TOML preflight, deterministic fixtures, GCN/GIN contracts, real PyG GPSConv adapter, graph-local Laplacian features, device selection guard, and fail-closed record aggregation.
- Locally tested: 41 tests passed on CPU; compile, static validation, and preflight passed.
- Kaggle-tested: pending sanitized provider-free run evidence.
- Synthetic-only: all current model and report evidence.
- Benchmark-pending: official OGB evaluation, training, raw metrics, and aggregate report.
- GPU-pending: no GPU execution has been performed.
- Blocked: Kaggle evidence, official benchmark evidence, and GitHub publication workflow remain pending.

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

1. Diagnose the bounded Kaggle source-package update failure and record private synthetic evidence.
2. Obtain separate approval for the official OGB runtime gate, GPU smoke, and bounded benchmark.
3. Publish only measured results with complete provenance, limitations, and raw per-seed records.

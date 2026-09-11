<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,14,20&height=200&section=header&text=GraphGPS%20Upgrade&fontSize=52&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Fixture-first%20GCN%2FGIN%2FGraphGPS%20benchmark%20on%20ogbg-molhiv&descAlignY=58&descAlign=50&descSize=16" width="100%" />

</div>

# GraphGPS Upgrade

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11.9-blue?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-50%20passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![Kaggle](https://img.shields.io/badge/kaggle-full--budget--ablation%20v1-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ajinkya1225/project-12-graphgps-full-budget-ablation)
[![License](https://img.shields.io/badge/license-MIT-22C55E)](LICENSE)

</div>

<div align="center">

[Purpose](#purpose) &bull; [Architecture](#architecture) &bull; [Installation](#installation) &bull; [Validation](#local-synthetic-validation) &bull; [Results](#full-budget-ablation-result) &bull; [Kaggle History](#kaggle-version-history) &bull; [Bugs](#bugs-that-cost-time) &bull; [Reproducibility](#reproducibility-and-safety) &bull; [Citation](#citation) &bull; [License](#license-status)

</div>

Project 12 is a fixture-first molecular graph benchmark scaffold that compares uniform GCN, GIN, and PyTorch Geometric `GPSConv` model contracts under a frozen configuration.

## Purpose

The project is for engineers and researchers who need a small, inspectable starting point for a reproducible graph-level benchmark. The release includes graph batching, graph-local positional features, uniform PyG baselines, an official OGB runner, output-shape validation, deterministic fixtures, and fail-closed reporting rules. A private 30-epoch-budget Kaggle ablation verifies the complete frozen benchmark workflow.

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

Verified locally on 2026-09-11: `50 passed`, exit code `0`. PyG emitted three deprecation warnings from its distributed and JIT compatibility layers; no test failed.

## Kaggle synthetic validation

The public notebook and runbooks are approval-gated. The private synthetic GPU smoke verified the synthetic contracts without OGB data. The approved full-budget run used internet for documented dependencies and OGB access in a private T4 session. Keep datasets and checkpoints on Kaggle and retrieve only sanitized reports.

## Full-budget ablation result

Private Kaggle kernel [`ajinkya1225/project-12-graphgps-full-budget-ablation`](https://www.kaggle.com/code/ajinkya1225/project-12-graphgps-full-budget-ablation) version 1 completed on a Tesla T4 in 3,785.76 seconds. It passed 49 pre-run tests and used the official `ogbg-molhiv` scaffold split/evaluator, a 30-epoch maximum, patience 5, batch size 32, Adam learning rate 0.001, no scheduler, and seeds 0, 1, and 2. Test ROC-AUC was GCN `0.6932 +/- 0.0081`, GIN `0.6982 +/- 0.0125`, and GPS `0.7482 +/- 0.0084` (mean +/- sample standard deviation). These measured values apply only to this frozen protocol and do not establish SOTA performance or a general GraphGPS advantage.

## Kaggle version history

Both private kernels below retain their datasets and checkpoints on Kaggle; only sanitized reports were retrieved locally. Note that a runtime capped at one epoch is a runtime-path smoke test, not a benchmark result.

| Kernel | Version | Outcome | What happened |
| --- | --- | --- | --- |
| `project-12-graphgps-official-ogb-benchmark` | earlier version | Blocked | Passed compile and 45 tests, then failed before OGB access because the notebook kernel did not import its extracted source tree; the one path repair attempt returned HTTP 409, so no official evidence was produced. |
| `project-12-graphgps-official-ogb-benchmark` | v3 | Completed (bounded) | Path/import issue resolved; passed 48 pre-run tests and ran the official scaffold/evaluator path capped at one epoch. Test ROC-AUC: GCN `0.6383 +/- 0.0181`, GIN `0.6007 +/- 0.0175`, GPS `0.7052 +/- 0.0360` (seeds 0-2). Not a full-budget benchmark. |
| `project-12-graphgps-full-budget-ablation` | v1 | Complete | Passed 49 pre-run tests and ran the approved frozen 30-epoch-maximum, patience-5, 3-seed ablation on Tesla T4 in 3,785.76s. Test ROC-AUC: GCN `0.6932 +/- 0.0081`, GIN `0.6982 +/- 0.0125`, GPS `0.7482 +/- 0.0084`. |

## Bugs that cost time

- **Boolean masks collapsed the evaluator's task dimension.** Indexing `[N, 1]` labels and predictions with an `[N, 1]` boolean mask returned `[N]`, which only broke the official evaluator's shape contract on the real OGB path (fixtures never exercised it). Fixed by selecting valid rows while explicitly preserving the task column; guarded by `test_predict_preserves_single_task_column_shape`.
- **Batched attention leaked across graph boundaries.** The fixture GPS model ran multi-head attention over an entire batch as one sequence, so editing one graph's features changed another graph's output. Fixed by scoping attention to each graph independently; guarded by `test_gps_batch_attention_does_not_cross_graph_boundaries`, which changes only graph 2 and asserts graph 1's logit is unchanged.
- **PyTorch's safe loader rejected the cached OGB serialized object.** OGB 1.3.6's processed `ogbg-molhiv` cache is a serialized PyG object that PyTorch 2.12.1+cpu's default safe-loading mode refused to deserialize. Instead of disabling safe loading outright, the fix registers only the four exact PyG classes involved (`Data`, `DataEdgeAttr`, `DataTensorAttr`, `GlobalStorage`) as explicit safe globals; guarded by `test_pyg_safe_global_names_are_explicit_and_minimal`.
- **A private kernel upload silently hid a packaging failure.** The kernel uploaded and appeared to run, but its runtime couldn't locate the bundled source package, so nothing executed past that point. One path repair was attempted; Kaggle's metadata API returned HTTP 409. The rule going forward: only a retrieved, sanitized evidence file counts as validation, never an upload's "success" status.
- **A case-sensitive claim scanner missed a lowercase bypass.** The report-language guard blocked phrases like `SOTA` but not `sota`, so an unsupported claim could slip through in different casing. Fixed by normalizing both the blocked-phrase list and the scanned text before matching; guarded by `test_report_language_scan_rejects_case_variant_of_blocked_claim`.

## Verified status

- Implemented: offline TOML preflight, deterministic fixtures, real PyG GCN/GIN message passing, OGB categorical encoding, GPSConv adapter, graph-local Laplacian features, shared training/evaluation/checkpoint runner, raw-record aggregation, and figure generation.
- Locally tested: 50 tests passed on CPU; compile, static validation, and preflight passed.
- Kaggle-tested: private Tesla T4 full-budget version 1 completed with official OGB loading, training, evaluation, records, and aggregation.
- Benchmark evidence: maximum 30 epochs, three seeds per model, official scaffold split and evaluator.
- Benchmark status: complete for the documented frozen protocol; broader scientific generalization is outside scope.
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

The documented Project 12 release scope is complete. Any future dataset, architecture, hyperparameter, or seed expansion is a new experiment and must use a separately frozen protocol.

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,14,20&height=100&section=footer" width="100%" />

</div>

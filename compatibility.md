# Compatibility Record

## Verified Full-Budget Environment - 2026-09-11

Private Kaggle kernel `ajinkya1225/project-12-graphgps-full-budget-ablation` version 1 completed on Tesla T4 with Python 3.12, PyTorch `2.10.0+cu128`, PyG `2.7.0`, and OGB `1.3.6`. The frozen package passed 49 pre-run tests and completed nine official model/seed records. Local release verification passes 50 tests on Python 3.11.9 with CPU PyTorch 2.12.1, PyG 2.7.0, OGB 1.3.6, and pytest 9.0.3.

These versions are verified for the recorded environments. A dependency change requires fresh compatibility and benchmark evidence.

Observed during local synthetic verification on 2026-09-10:

| Component | Version | Evidence scope |
|---|---:|---|
| Python | 3.11.9 | Local CPU fixture tests |
| PyTorch | 2.12.1+cpu | Local CPU fixture tests |
| PyTorch Geometric | 2.7.0 | `GPSConv` import, signature, and tiny adapter smoke |
| OGB | 1.3.6 | Package metadata only; no official benchmark run |
| pytest | 9.0.3 | Local test suite |

The observed `GPSConv` constructor accepts `channels`, `conv`, `heads`, `dropout`, normalization, attention type, and attention keyword arguments. The adapter uses the current `GPSConv(channels=..., conv=GINConv(...), heads=1, dropout=..., attn_type="multihead")` form and passes `batch` to its forward method.

This record is not GPU or Kaggle compatibility evidence.

# Compatibility Record

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

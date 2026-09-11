# Kaggle Full-Budget Ablation

Approved on 2026-09-11 for a private Tesla T4 run.

- Task/split: `ogbg-molhiv`, official scaffold split and evaluator.
- Models/seeds: GCN, GIN, GPS; seeds 0, 1, and 2.
- Shared budget: maximum 30 epochs, batch size 32, Adam learning rate 0.001.
- Early stopping: validation ROC-AUC, patience 5; best validation checkpoint selects test evaluation.
- Scheduler: none.
- W&B: disabled.
- Runtime bound: four hours.

Retain raw per-seed JSON/CSV before aggregation. Keep datasets and checkpoints on Kaggle. Download only sanitized evidence, raw metrics, aggregate JSON/PNG, and the run manifest. Do not claim a directional winner until the completed evidence is inspected.

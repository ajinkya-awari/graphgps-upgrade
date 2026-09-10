# Kaggle synthetic validation runbook

This package is source-only. It contains no dataset, weights, checkpoint, credentials, private evidence, or generated benchmark output.

1. Check the private kernel metadata: `enable_gpu=false`, `enable_internet=false`, and all source lists are empty.
2. Push with `kaggle kernels push -p "<KAGGLE_NOTEBOOK_FOLDER>"`.
3. Poll with `kaggle kernels status ajinkya1225/12-graphgps-upgrade-validation`.
4. Read logs with `kaggle kernels logs ajinkya1225/12-graphgps-upgrade-validation`.
5. Download only the sanitized JSON with `kaggle kernels output ajinkya1225/12-graphgps-upgrade-validation -p "<DEDICATED_OUTPUT_FOLDER>"`.
6. Inspect the JSON fields only. Stop before the approval-gated cells.

The bounded sequence records Python, OS, PyTorch, PyG, OGB, CUDA availability, explicit CPU device, source/config hashes, seed, synthetic sample count, test count, exit codes, artifact path, restricted-artifact count, and the fact that no benchmark or GPU run occurred.

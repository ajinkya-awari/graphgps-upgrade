from __future__ import annotations

from pathlib import Path


def write_aggregate_figure(path: str | Path, aggregate: dict[str, dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(aggregate)
    means = [aggregate[label]["mean"] for label in labels]
    deviations = [aggregate[label]["stdev"] for label in labels]
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.bar(labels, means, yerr=deviations, capsize=4, color=("#4c78a8", "#f58518", "#54a24b"))
    axis.set_ylabel("ROC-AUC")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Model")
    figure.savefig(path, dpi=160)
    plt.close(figure)

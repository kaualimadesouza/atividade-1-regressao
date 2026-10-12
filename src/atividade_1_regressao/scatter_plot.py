"""Step 3.1.1: scatter of the response (vertical axis) against the predictor (horizontal axis)."""

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from atividade_1_regressao.datasets import ROOT, Dataset, DatasetChoice, load_dataset
from atividade_1_regressao.style import SERIES_1, SURFACE, style_axes

OUTPUT_FILE = "original_dispersao.png"


def plot_scatter(predictor: pd.Series, response: pd.Series) -> Figure:
    # A bare Figure, not pyplot: no global state and no disk access, the caller decides where it goes.
    fig = Figure(figsize=(6, 4.2), facecolor=SURFACE, layout="tight")
    ax = fig.subplots()
    ax.scatter(predictor, response, s=36, color=SERIES_1, edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.set_xlabel(str(predictor.name))
    ax.set_ylabel(str(response.name))
    style_axes(ax)
    return fig


def save_scatter(dataset: Dataset) -> Path:
    frame = load_dataset(dataset)
    fig = plot_scatter(frame[dataset.value.predictor], frame[dataset.value.response])
    output = dataset.output_dir / OUTPUT_FILE
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    return output


def main() -> None:
    choice = DatasetChoice.ALL  # or DatasetChoice.TREADWEAR, .ALLIGATORS, .CONCPATOG
    for dataset in choice.value:
        print(save_scatter(dataset).relative_to(ROOT))


if __name__ == "__main__":
    main()

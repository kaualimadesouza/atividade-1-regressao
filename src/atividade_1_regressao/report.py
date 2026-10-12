"""Everything the LaTeX report reads from outputs/<dataset>/: figures, OLS summaries and the step 3.2 table."""

from pathlib import Path

from matplotlib.figure import Figure

from atividade_1_regressao.datasets import Dataset, Stage
from atividade_1_regressao.figures import (
    decimal,
    exploration_figure,
    histogram_figure,
    icip_figure,
    qqplot_figure,
    residuals_figure,
    scatter_figure,
)
from atividade_1_regressao.regression import Explored, Fit, explore
from atividade_1_regressao.variables import Combination


def run(datasets: list[Dataset], outputs_dir: Path) -> list[Path]:
    """Write every output of every dataset, one folder each. Returns the paths written."""
    return [path for dataset in datasets for path in write_dataset(dataset, outputs_dir / dataset.slug)]


def write_dataset(dataset: Dataset, folder: Path) -> list[Path]:
    """Both stages plus the 9-option exploration. One fit per stage, reused by every figure."""
    folder.mkdir(parents=True, exist_ok=True)
    fits = {stage: Fit.of(dataset.load(stage)) for stage in Stage}
    stage_paths = [path for stage, fit in fits.items() for path in write_stage(fit, stage, folder)]
    return [*stage_paths, *write_exploration(fits[Stage.ORIGINAL], dataset.chosen, folder)]


def write_stage(fit: Fit, stage: Stage, folder: Path) -> list[Path]:
    """The 5 figures and the OLS summary step 3.1 asks for, named <stage>_<what>."""
    figures = {
        "dispersao": scatter_figure(fit.variables),
        "icip": icip_figure(fit),
        "histograma": histogram_figure(fit),
        "residuos_preditos": residuals_figure(fit),
        "qqplot": qqplot_figure(fit),
    }
    summary = folder / f"{stage.value}_ols.txt"
    summary.write_text(fit.model.summary().as_text())
    return [*(save(figure, folder / f"{stage.value}_{name}.png") for name, figure in figures.items()), summary]


def write_exploration(original: Fit, chosen: Combination, folder: Path) -> list[Path]:
    """Step 3.2: the 3x3 figure and the adjusted R² table of the 9 options."""
    results = explore(original.variables)
    table = folder / "transformacoes.tex"
    table.write_text(exploration_table(original, results, chosen))
    return [save(exploration_figure(results), folder / "transformacoes.png"), table]


def exploration_table(original: Fit, results: list[Explored], chosen: Combination) -> str:
    """LaTeX tabular the report inputs as is: no transform first, then the 9; chosen row in bold."""
    rows = [r"\textbf{Transformação} & \textbf{Aplicada em} & \textbf{$R^2$ ajustado} & \textbf{$n$} \\"]
    rows.append(rf"Nenhuma & -- & {decimal(original.model.rsquared_adj)} & {int(original.model.nobs)} \\")
    for result in results:
        cells = [
            result.combination.transform.value.title,
            result.combination.target.value,
            decimal(result.fit.model.rsquared_adj),
            str(int(result.fit.model.nobs)),
        ]
        if result.combination == chosen:
            cells = [rf"\textbf{{{cell}}}" for cell in cells]
        rows.append(" & ".join(cells) + r" \\")
    body = "\n\\hline\n".join(rows)
    return f"\\begin{{tabular}}{{|l|c|c|c|}}\n\\hline\n{body}\n\\hline\n\\end{{tabular}}\n"


def save(figure: Figure, path: Path) -> Path:
    """PNG at 150 dpi. Returns the path, so callers can list what was written."""
    figure.savefig(path, dpi=150)
    return path

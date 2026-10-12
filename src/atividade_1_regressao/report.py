"""Everything the LaTeX report reads from outputs/<dataset>/: figures, OLS summaries and the step 3.2 table."""

from pathlib import Path

from matplotlib.figure import Figure

from atividade_1_regressao.datasets import OUTPUTS_DIR, ROOT, Dataset
from atividade_1_regressao.figures import (
    comma_decimal,
    exploration_figure,
    histogram_figure,
    icip_figure,
    qqplot_figure,
    residuals_figure,
    scatter_figure,
)
from atividade_1_regressao.regression import Explored, Fit, explore
from atividade_1_regressao.variables import Combination


def write_dataset(dataset: Dataset) -> None:
    """Original stage, chosen-rescaling stage and the 9-option exploration, in outputs/<slug>/."""
    folder = OUTPUTS_DIR / dataset.slug
    folder.mkdir(parents=True, exist_ok=True)
    original = Fit.of(dataset.load())
    write_stage(original, "original", folder)
    write_stage(Fit.of(dataset.chosen.apply(original.variables)), "transformado", folder)
    write_exploration(original, dataset.chosen, folder)


def write_stage(fit: Fit, stage: str, folder: Path) -> None:
    """The 5 figures and the OLS summary step 3.1 asks for, named <stage>_<what>."""
    save(folder / f"{stage}_dispersao.png", scatter_figure(fit.variables))
    save(folder / f"{stage}_icip.png", icip_figure(fit))
    save(folder / f"{stage}_histograma.png", histogram_figure(fit))
    save(folder / f"{stage}_residuos_preditos.png", residuals_figure(fit))
    save(folder / f"{stage}_qqplot.png", qqplot_figure(fit))
    save(folder / f"{stage}_ols.txt", ols_summary(fit))


def ols_summary(fit: Fit) -> str:
    """statsmodels summary without Date/Time, so rerunning leaves committed outputs unchanged."""
    summary = fit.model.summary()
    for row in summary.tables[0]:
        for label, value in zip(row[::2], row[1::2]):
            if label.data.strip() in ("Date:", "Time:"):
                label.data, value.data = "", ""
    return summary.as_text()


def write_exploration(original: Fit, chosen: Combination, folder: Path) -> None:
    """Step 3.2: the 3x3 figure and the adjusted R² table of the 9 options."""
    results = explore(original.variables)
    save(folder / "transformacoes.png", exploration_figure(results))
    save(folder / "transformacoes.tex", exploration_table(original, results, chosen))


def exploration_table(original: Fit, results: list[Explored], chosen: Combination) -> str:
    """LaTeX tabular the report inputs as is: no transform first, then the 9; chosen row in bold."""
    rows = [r"\textbf{Transformação} & \textbf{Aplicada em} & \textbf{$R^2$ ajustado} & \textbf{$n$} \\"]
    rows.append(rf"Nenhuma & -- & {comma_decimal(original.model.rsquared_adj)} & {int(original.model.nobs)} \\")
    for result in results:
        cells = [
            result.combination.transform.title,
            result.combination.target.value,
            comma_decimal(result.fit.model.rsquared_adj),
            str(int(result.fit.model.nobs)),
        ]
        if result.combination == chosen:
            cells = [rf"\textbf{{\boldmath {cell}}}" for cell in cells]  # \boldmath: math parts too
        rows.append(" & ".join(cells) + r" \\")
    body = "\n\\hline\n".join(rows)
    return f"\\begin{{tabular}}{{|l|c|c|c|}}\n\\hline\n{body}\n\\hline\n\\end{{tabular}}\n"


def save(path: Path, content: Figure | str) -> None:
    """Write a figure (PNG, 150 dpi) or a text file, and print where it went."""
    if isinstance(content, Figure):
        content.savefig(path, dpi=150)
    else:
        path.write_text(content)
    print(path.relative_to(ROOT))

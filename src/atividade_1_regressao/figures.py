"""Figure builders shared by the report and the guide: each takes data and returns a bare Figure, never touching disk."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

from atividade_1_regressao.regression import Explored, Fit
from atividade_1_regressao.style import SERIES_1, SERIES_2, SURFACE, scatter, styled, zero_line
from atividade_1_regressao.variables import TRANSFORMS, Target, Variables


@styled
def scatter_figure(variables: Variables) -> Figure:
    """Y vs X, one dot per row. Axis names come from the series names."""
    fig = Figure(figsize=(6.4, 4.2), layout="tight")
    ax = fig.subplots()
    scatter(ax, variables.x, variables.y)
    ax.set_xlabel(str(variables.x.name))
    ax.set_ylabel(str(variables.y.name))
    return fig


@styled
def icip_figure(fit: Fit) -> Figure:
    """Data, fitted line, 95% band for the mean (IC, narrow) and for a new point (IP, wide)."""
    x, y = fit.variables.x, fit.variables.y
    grid = np.linspace(x.min(), x.max(), 100)
    bands = fit.bands(grid)
    fig = Figure(figsize=(7, 4.4), layout="tight")
    ax = fig.subplots()
    ax.fill_between(grid, bands["obs_ci_lower"], bands["obs_ci_upper"], color=SERIES_2, alpha=0.10, linewidth=0, label="IP 95%: nova observação")
    ax.fill_between(grid, bands["mean_ci_lower"], bands["mean_ci_upper"], color=SERIES_2, alpha=0.30, linewidth=0, label="IC 95%: média")
    ax.plot(grid, bands["mean"], color=SERIES_2, linewidth=2, label="reta de regressão")
    scatter(ax, x, y, label="observações")
    ax.set_xlabel(str(x.name))
    ax.set_ylabel(str(y.name))
    ax.legend(loc="best")
    return fig


@styled
def histogram_figure(fit: Fit) -> Figure:
    """Residual histogram over the normal curve they should follow (mean 0, residual std)."""
    residuals = fit.model.resid.to_numpy()
    spread = np.sqrt(fit.model.mse_resid)
    grid = np.linspace(residuals.min() - spread, residuals.max() + spread, 200)
    fig = Figure(figsize=(6.4, 4.2), layout="tight")
    ax = fig.subplots()
    ax.hist(residuals, bins=10, density=True, color=SERIES_1, edgecolor=SURFACE, linewidth=1.5, label="resíduos")
    ax.plot(grid, stats.norm.pdf(grid, loc=0, scale=spread), color=SERIES_2, linewidth=2, label="normal de referência")
    ax.set_xlabel("resíduo")
    ax.set_ylabel("densidade")
    ax.legend(loc="best")  # skewed residuals put the tallest bar in any fixed corner
    return fig


@styled
def residuals_figure(fit: Fit) -> Figure:
    """Residual vs predicted value. Good fit: shapeless cloud around 0, same width everywhere."""
    fig = Figure(figsize=(6.4, 4.2), layout="tight")
    ax = fig.subplots()
    zero_line(ax)
    scatter(ax, fit.model.fittedvalues, fit.model.resid)
    ax.set_xlabel("valor predito ŷ")
    ax.set_ylabel("resíduo e")
    return fig


def draw_qq(ax: Axes, sample: np.ndarray) -> None:
    """Sorted sample vs normal quantiles on `ax`, plus the line a normal sample would sit on."""
    (theoretical, ordered), _ = stats.probplot(sample, dist="norm")
    # Same reference as sm.qqplot(line="s"): mean + std * theoretical quantile
    ax.plot(theoretical, sample.mean() + sample.std(ddof=1) * theoretical, color=SERIES_2, linewidth=2, zorder=2)
    scatter(ax, theoretical, ordered)


@styled
def qqplot_figure(fit: Fit) -> Figure:
    """QQ-plot of the residuals. Dots on the line -> residuals look normal."""
    fig = Figure(figsize=(6.4, 4.2), layout="tight")
    ax = fig.subplots()
    draw_qq(ax, fit.model.resid.to_numpy())
    ax.set_xlabel("quantis teóricos da normal")
    ax.set_ylabel("quantis dos resíduos")
    return fig


@styled
def exploration_figure(results: list[Explored]) -> Figure:
    """3x3 grid, one scatter + line per rescaling. Row = transform, column = X, Y or both."""
    fig = Figure(figsize=(9.5, 8.5), layout="tight")
    axes = fig.subplots(len(TRANSFORMS), len(Target))
    for ax, result in zip(axes.flat, results, strict=True):
        fit, combination = result.fit, result.combination
        scatter(ax, fit.variables.x, fit.variables.y)
        ax.plot(fit.variables.x, fit.model.fittedvalues, color=SERIES_2, linewidth=2)
        ax.set_xlabel(str(fit.variables.x.name))
        ax.set_ylabel(str(fit.variables.y.name))
        ax.set_title(
            f"{combination.transform.title} em {combination.target.value}\n"
            f"R² ajustado {comma_decimal(fit.model.rsquared_adj)}, n = {int(fit.model.nobs)}"
        )
    return fig


def comma_decimal(value: float) -> str:
    """4 decimals, Brazilian comma: 0.9583 -> "0,9583"."""
    return f"{value:.4f}".replace(".", ",")

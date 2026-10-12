"""Generate every figure used by docs/guia_regressao_iniciantes.md into docs/img/.

Run from the repo root with `uv run python docs/scripts/make_figures.py`.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from atividade_1_regressao.figures import draw_qq, histogram_figure, icip_figure, qqplot_figure, residuals_figure, scatter_figure
from atividade_1_regressao.regression import Fit
from atividade_1_regressao.style import INK, INK_SECONDARY, MUTED, RC, SERIES_1, SERIES_2, scatter, zero_line
from atividade_1_regressao.variables import Variables

DOCS = Path(__file__).resolve().parents[1]
IMG = DOCS / "img"
HOSPITAL = DOCS / "data" / "hospitalinfct_reg1and2.txt"


@dataclass(frozen=True)
class Scenario:
    """A made-up dataset for the residual-pattern figure."""

    title: str
    x: np.ndarray
    y: np.ndarray


def save(fig: Figure, name: str) -> None:
    """Write docs/img/<name> at 150 dpi, close the figure and print the path."""
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=150)
    plt.close(fig)
    print(f"docs/img/{name}")


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares (intercept, slope), quick version for the synthetic examples."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def plot_normal() -> None:
    """Fig 01: the bell curve with the 68% and 95% areas shaded."""
    z = np.linspace(-4, 4, 400)
    density = stats.norm.pdf(z)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.fill_between(z, density, where=np.abs(z) <= 1.96, color=SERIES_1, alpha=0.12, linewidth=0)
    ax.fill_between(z, density, where=np.abs(z) <= 1, color=SERIES_1, alpha=0.25, linewidth=0)
    ax.plot(z, density, color=SERIES_1, linewidth=2)
    ax.annotate("68% entre -1 e +1\ndesvio padrão", xy=(0, 0.2), ha="center", color=INK, fontsize=9)
    ax.annotate("95% entre -1.96 e +1.96", xy=(0, 0.03), ha="center", color=INK, fontsize=9)
    ticks = [-3, -1.96, -1, 0, 1, 1.96, 3]
    ax.set_xticks(ticks, labels=[f"{tick:g}" for tick in ticks])
    ax.set_xlabel("distância da média, em desvios padrão (z)")
    ax.set_ylabel("densidade")
    ax.set_title("Distribuição normal: o formato de sino")
    save(fig, "01_normal.png")


def plot_least_squares() -> None:
    """Fig 02: five points, the fitted line and one residual labelled."""
    hours = np.array([1, 2, 3, 4, 5], dtype=float)
    grade = np.array([2, 4, 5, 4, 6], dtype=float)
    b0, b1 = fit_line(hours, grade)
    fitted = b0 + b1 * hours
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.vlines(hours, fitted, grade, color=MUTED, linewidth=1.5, zorder=2)
    grid = np.linspace(0.5, 5.5, 50)
    ax.plot(grid, b0 + b1 * grid, color=SERIES_2, linewidth=2, label=f"reta ŷ = {b0:.1f} + {b1:.1f}x")
    scatter(ax, hours, grade, label="dados observados (y)")
    ax.annotate(
        "resíduo e = y - ŷ = -1.0",
        xy=(4, 4.5),
        xytext=(4.25, 3.1),
        color=INK,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1},
    )
    ax.set_xlabel("horas de estudo (x)")
    ax.set_ylabel("nota (y)")
    ax.set_title("Mínimos quadrados: a reta que minimiza a soma dos resíduos²")
    ax.legend(loc="upper left")
    save(fig, "02_minimos_quadrados.png")


def plot_hospital(fit: Fit) -> None:
    """Figs 03 to 07 with the report's own builders, so the guide shows what the analysis produces."""
    pages = [
        (scatter_figure(fit.variables), "58 hospitais: internação vs risco de infecção", "03_dispersao_hospital.png"),
        (icip_figure(fit), "Reta, intervalo de confiança (IC) e de predição (IP)", "04_ic_ip_hospital.png"),
        (histogram_figure(fit), "Histograma dos resíduos (hospitais)", "05_histograma_residuos.png"),
        (residuals_figure(fit), "Resíduos vs valores preditos (hospitais)", "06_residuos_vs_preditos.png"),
        (qqplot_figure(fit), "QQ-plot dos resíduos (hospitais)", "07_qqplot.png"),
    ]
    for fig, title, name in pages:
        fig.axes[0].set_title(title)
        save(fig, name)


def residual_scenarios(rng: np.random.Generator) -> list[Scenario]:
    """Four synthetic cases: fine, curve, funnel, one outlier."""
    x = np.sort(rng.uniform(0, 10, 60))
    outlier_y = 2 + 0.8 * x + rng.normal(0, 1, x.size)
    outlier_y[30] += 9
    return [
        Scenario("Tudo certo", x, 2 + 0.8 * x + rng.normal(0, 1, x.size)),
        Scenario("Curva (não linear)", x, 0.15 * x**2 + rng.normal(0, 0.8, x.size)),
        Scenario("Funil (variância cresce)", x, 2 + 0.8 * x + rng.normal(0, 0.1 + 0.35 * x, x.size)),
        Scenario("Um outlier", x, outlier_y),
    ]


def plot_residual_patterns(scenarios: list[Scenario]) -> None:
    """Fig 08: each scenario as data + line (top) and residual vs fitted (bottom)."""
    fig, axes = plt.subplots(2, len(scenarios), figsize=(14, 6.2))
    for column, scenario in enumerate(scenarios):
        b0, b1 = fit_line(scenario.x, scenario.y)
        fitted = b0 + b1 * scenario.x
        top, bottom = axes[0, column], axes[1, column]
        scatter(top, scenario.x, scenario.y)
        top.plot(scenario.x, fitted, color=SERIES_2, linewidth=2)
        top.set_title(scenario.title)
        top.set_xlabel("x")
        zero_line(bottom)
        scatter(bottom, fitted, scenario.y - fitted)
        bottom.set_xlabel("valor predito ŷ")
    axes[0, 0].set_ylabel("dados + reta ajustada\n(y vs x)")
    axes[1, 0].set_ylabel("resíduo vs predito\n(e vs ŷ)")
    save(fig, "08_padroes_residuos.png")


def plot_qq_patterns(rng: np.random.Generator) -> None:
    """Fig 09: QQ-plots of normal, outlier, skewed and heavy-tailed samples."""
    with_outlier = rng.normal(0, 1, 80)
    with_outlier[0] = 5.5
    samples = [
        ("Normal", rng.normal(0, 1, 80)),
        ("Um outlier", with_outlier),
        ("Assimétrica à direita", rng.exponential(1, 80)),
        ("Caudas pesadas", rng.standard_t(2.5, 80)),
    ]
    fig, axes = plt.subplots(1, len(samples), figsize=(14, 3.6))
    for ax, (title, sample) in zip(axes, samples, strict=True):
        draw_qq(ax, sample)
        ax.set_title(title)
        ax.set_xlabel("quantis teóricos")
    axes[0].set_ylabel("quantis da amostra")
    save(fig, "09_padroes_qqplot.png")


def plot_independence(rng: np.random.Generator) -> None:
    """Fig 10: residuals in collection order, independent vs trend vs serial correlation."""
    order = np.arange(1, 51)
    autocorrelated = np.zeros(order.size)
    for i in range(1, order.size):
        autocorrelated[i] = 0.85 * autocorrelated[i - 1] + rng.normal(0, 0.6)
    series = [
        ("Independentes", rng.normal(0, 1, order.size)),
        ("Tendência no tempo", np.linspace(-2, 2, order.size) + rng.normal(0, 0.6, order.size)),
        ("Correlação serial positiva", autocorrelated),
    ]
    fig, axes = plt.subplots(1, len(series), figsize=(13, 3.4))
    for ax, (title, residuals) in zip(axes, series, strict=True):
        zero_line(ax)
        ax.plot(order, residuals, color=MUTED, linewidth=1, zorder=2)
        scatter(ax, order, residuals)
        ax.set_title(title)
        ax.set_xlabel("ordem de coleta")
    axes[0].set_ylabel("resíduo")
    save(fig, "10_independencia.png")


def plot_transformations(rng: np.random.Generator) -> None:
    """Fig 11: one exponential-ish dataset under the original scale and the 9 rescalings."""
    x = np.sort(rng.uniform(1, 10, 50))
    y = 10 ** (0.5 + 0.15 * x + rng.normal(0, 0.08, x.size))
    combos = [
        ("Original", x, y),
        ("√X", np.sqrt(x), y),
        ("√Y", x, np.sqrt(y)),
        ("√X e √Y", np.sqrt(x), np.sqrt(y)),
        ("X²", x**2, y),
        ("Y²", x, y**2),
        ("X² e Y²", x**2, y**2),
        ("log10 X", np.log10(x), y),
        ("log10 Y", x, np.log10(y)),
        ("log10 X e log10 Y", np.log10(x), np.log10(y)),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, (title, tx, ty) in zip(axes.flat, combos, strict=True):
        b0, b1 = fit_line(tx, ty)
        scatter(ax, tx, ty)
        ax.plot(tx, b0 + b1 * tx, color=SERIES_2, linewidth=2)
        ax.set_title(title)
    fig.supxlabel("X (ou X transformado)", color=INK_SECONDARY, fontsize=10)
    fig.supylabel("Y (ou Y transformado)", color=INK_SECONDARY, fontsize=10)
    save(fig, "11_transformacoes.png")


def main() -> None:
    """Regenerate every guide figure in docs/img/, same random seed every time."""
    IMG.mkdir(exist_ok=True)
    plt.rcParams.update(RC)
    rng = np.random.default_rng(501)
    hospital = pd.read_csv(HOSPITAL, sep="\t")
    stay = hospital["Stay"].rename("tempo médio de internação, em dias (Stay)")
    risk = hospital["InfctRsk"].rename("risco de infecção, em % (InfctRsk)")
    plot_normal()
    plot_least_squares()
    plot_hospital(Fit.of(Variables(stay, risk)))
    plot_residual_patterns(residual_scenarios(rng))
    plot_qq_patterns(rng)
    plot_independence(rng)
    plot_transformations(rng)


if __name__ == "__main__":
    main()

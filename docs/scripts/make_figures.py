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
import statsmodels.formula.api as smf
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

DOCS = Path(__file__).resolve().parents[1]
IMG = DOCS / "img"
HOSPITAL = DOCS / "data" / "hospitalinfct_reg1and2.txt"

# Validated categorical pair (dataviz reference palette, slots 1 and 2) plus neutral chrome
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


@dataclass(frozen=True)
class Scenario:
    title: str
    x: np.ndarray
    y: np.ndarray


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig: Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=150)
    plt.close(fig)
    print(f"docs/img/{name}")


def scatter(ax: Axes, x: np.ndarray, y: np.ndarray, label: str | None = None) -> None:
    ax.scatter(x, y, s=36, color=SERIES_1, edgecolor=SURFACE, linewidth=1, zorder=3, label=label)


def zero_line(ax: Axes) -> None:
    ax.axhline(0, color=INK_SECONDARY, linewidth=1, zorder=2)


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def plot_normal() -> None:
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


def load_hospital() -> pd.DataFrame:
    return pd.read_csv(HOSPITAL, sep="\t")


def plot_hospital_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    scatter(ax, df["Stay"].to_numpy(), df["InfctRsk"].to_numpy())
    ax.set_xlabel("tempo médio de internação, em dias (Stay)")
    ax.set_ylabel("risco de infecção, em % (InfctRsk)")
    ax.set_title("58 hospitais: internação vs risco de infecção")
    save(fig, "03_dispersao_hospital.png")


def plot_hospital_intervals(df: pd.DataFrame, model: RegressionResultsWrapper) -> None:
    grid = pd.DataFrame({"Stay": np.linspace(df["Stay"].min(), df["Stay"].max(), 100)})
    bands = model.get_prediction(grid).summary_frame(alpha=0.05)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    stay = grid["Stay"].to_numpy()
    ax.fill_between(stay, bands["obs_ci_lower"], bands["obs_ci_upper"], color=SERIES_2, alpha=0.10, linewidth=0, label="IP 95%: novo hospital")
    ax.fill_between(stay, bands["mean_ci_lower"], bands["mean_ci_upper"], color=SERIES_2, alpha=0.30, linewidth=0, label="IC 95%: média dos hospitais")
    ax.plot(stay, bands["mean"], color=SERIES_2, linewidth=2, label="reta de regressão")
    scatter(ax, df["Stay"].to_numpy(), df["InfctRsk"].to_numpy(), label="hospitais observados")
    ax.set_xlabel("tempo médio de internação, em dias (Stay)")
    ax.set_ylabel("risco de infecção, em % (InfctRsk)")
    ax.set_title("Reta, intervalo de confiança (IC) e de predição (IP)")
    ax.legend(loc="upper left")
    save(fig, "04_ic_ip_hospital.png")


def plot_residual_histogram(model: RegressionResultsWrapper) -> None:
    residuals = model.resid.to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(residuals, bins=10, density=True, color=SERIES_1, edgecolor=SURFACE, linewidth=1.5, label="resíduos")
    grid = np.linspace(residuals.min() - 1, residuals.max() + 1, 200)
    normal = stats.norm.pdf(grid, loc=0, scale=residuals.std(ddof=2))
    ax.plot(grid, normal, color=SERIES_2, linewidth=2, label="normal de referência")
    ax.set_xlabel("resíduo (pontos percentuais de risco)")
    ax.set_ylabel("densidade")
    ax.set_title("Histograma dos resíduos (hospitais)")
    ax.legend(loc="upper left")
    save(fig, "05_histograma_residuos.png")


def plot_residuals_vs_fitted(model: RegressionResultsWrapper) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    zero_line(ax)
    scatter(ax, model.fittedvalues.to_numpy(), model.resid.to_numpy())
    ax.set_xlabel("valor predito ŷ")
    ax.set_ylabel("resíduo e")
    ax.set_title("Resíduos vs valores preditos (hospitais)")
    save(fig, "06_residuos_vs_preditos.png")


def qq_points(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (theoretical, ordered), _ = stats.probplot(sample, dist="norm")
    return theoretical, ordered


def draw_qq(ax: Axes, sample: np.ndarray) -> None:
    theoretical, ordered = qq_points(sample)
    # Same reference as sm.qqplot(line="s"): mean + std * theoretical quantile
    ax.plot(theoretical, sample.mean() + sample.std(ddof=1) * theoretical, color=SERIES_2, linewidth=2, zorder=2)
    scatter(ax, theoretical, ordered)


def plot_qqplot(model: RegressionResultsWrapper) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    draw_qq(ax, model.resid.to_numpy())
    ax.set_xlabel("quantis teóricos da normal")
    ax.set_ylabel("quantis dos resíduos")
    ax.set_title("QQ-plot dos resíduos (hospitais)")
    save(fig, "07_qqplot.png")


def residual_scenarios(rng: np.random.Generator) -> list[Scenario]:
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
    IMG.mkdir(exist_ok=True)
    apply_style()
    rng = np.random.default_rng(501)
    hospital = load_hospital()
    model = smf.ols("InfctRsk ~ Stay", data=hospital).fit()
    plot_normal()
    plot_least_squares()
    plot_hospital_scatter(hospital)
    plot_hospital_intervals(hospital, model)
    plot_residual_histogram(model)
    plot_residuals_vs_fitted(model)
    plot_qqplot(model)
    plot_residual_patterns(residual_scenarios(rng))
    plot_qq_patterns(rng)
    plot_independence(rng)
    plot_transformations(rng)


if __name__ == "__main__":
    main()

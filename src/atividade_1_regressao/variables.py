"""The X/Y pair of a dataset and the 9 rescalings of step 3.2: sqrt, square or log10 on X, Y or both."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from itertools import product

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Variables:
    """X (predictor) and Y (response), same rows, same order."""

    x: pd.Series
    y: pd.Series


@dataclass(frozen=True)
class TransformSpec:
    """What a transform is called, what it computes and which values it accepts."""

    title: str  # mathtext, so the same string renders in matplotlib titles and in the LaTeX table
    label: str  # ASCII on purpose: it names the column in OLS summaries that LaTeX includes verbatim
    function: Callable[[pd.Series], pd.Series]
    defined: Callable[[pd.Series], pd.Series]


class Transform(Enum):
    """The 3 rescalings of step 3.2."""

    SQRT = TransformSpec("Raiz quadrada", "sqrt({})", np.sqrt, lambda v: v >= 0)
    SQUARE = TransformSpec("Quadrado", "{}^2", np.square, lambda v: v.notna())
    LOG10 = TransformSpec(r"$\log_{10}$", "log10({})", np.log10, lambda v: v > 0)

    def apply(self, series: pd.Series) -> pd.Series:
        """Transformed copy, renamed: groove -> log10(groove). Caller drops out-of-domain rows first."""
        return self.value.function(series).rename(self.value.label.format(series.name))


class Target(Enum):
    """Which side gets the transform: X only, Y only or both."""

    X = "$X$"
    Y = "$Y$"
    BOTH = "$X$ e $Y$"


@dataclass(frozen=True)
class Combination:
    """One of the 9 options: a transform plus where it goes (e.g. log10 on Y)."""

    transform: Transform
    target: Target

    def apply(self, variables: Variables) -> Variables:
        """Transform X and/or Y. Rows the transform can't take (log10 of 0) are dropped, not fixed."""
        on_x, on_y = self.target is not Target.Y, self.target is not Target.X
        keep = pd.Series(True, index=variables.x.index)
        if on_x:
            keep &= self.transform.value.defined(variables.x)
        if on_y:
            keep &= self.transform.value.defined(variables.y)
        x, y = variables.x[keep], variables.y[keep]
        return Variables(self.transform.apply(x) if on_x else x, self.transform.apply(y) if on_y else y)


# Same order as the transformation table of the report: transform first, then where it applies
COMBINATIONS = tuple(Combination(transform, target) for transform, target in product(Transform, Target))

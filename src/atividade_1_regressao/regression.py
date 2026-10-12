"""Least-squares fit of Y on X, and the step 3.2 comparison of the 9 rescaled fits."""

from dataclasses import dataclass

import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from atividade_1_regressao.variables import COMBINATIONS, Combination, Variables


@dataclass(frozen=True)
class Fit:
    """The data and the least-squares line fitted to it."""

    variables: Variables
    model: RegressionResultsWrapper

    @classmethod
    def of(cls, variables: Variables) -> "Fit":
        """Fit y = b0 + b1 * x by ordinary least squares."""
        return cls(variables, sm.OLS(variables.y, sm.add_constant(variables.x)).fit())


@dataclass(frozen=True)
class Explored:
    """One of the 9 options and its fit."""

    combination: Combination
    fit: Fit


def explore(original: Variables) -> list[Explored]:
    """Fit all 9 options on the original data, in COMBINATIONS order."""
    return [Explored(combination, Fit.of(combination.apply(original))) for combination in COMBINATIONS]

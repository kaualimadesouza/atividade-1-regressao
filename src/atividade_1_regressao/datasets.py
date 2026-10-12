"""The three datasets of Activity 1: where each file lives, which columns are X and Y, and the rescaling chosen."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from atividade_1_regressao.variables import LOG10, Combination, Target, Variables

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class Dataset:
    """One data file: how to read it, which columns are X and Y, and its chosen rescaling."""

    slug: str
    file_name: str
    separator: str
    predictor: str
    response: str
    chosen: Combination  # the step 3.2 pick, justified in the report

    def load(self) -> Variables:
        """Read the file and return its original X and Y."""
        frame = pd.read_csv(DATA_DIR / self.file_name, sep=self.separator)
        return Variables(frame[self.predictor], frame[self.response])


DATASETS = (
    Dataset(
        "treadwear", "treadwear.txt", "\t",
        predictor="mileage", response="groove", chosen=Combination(LOG10, Target.Y),
    ),
    Dataset(
        "alligators", "alligator.txt", r"\s+",
        predictor="length", response="weight", chosen=Combination(LOG10, Target.Y),
    ),
    Dataset(
        "concpatog", "ConcPatog.txt", "\t",
        predictor="CPatog1", response="CPatog2", chosen=Combination(LOG10, Target.BOTH),
    ),
)

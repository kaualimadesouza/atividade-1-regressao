"""The three datasets of Activity 1: where each file lives, which columns are X and Y, and the rescaling chosen."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from atividade_1_regressao.variables import Combination, Target, Transform, Variables

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"


class Stage(Enum):
    """Raw data (step 3.1) or data after the chosen rescaling (step 3.3). Value = output file prefix."""

    ORIGINAL = "original"
    TRANSFORMED = "transformado"


@dataclass(frozen=True)
class Dataset:
    """One data file: how to read it, which columns are X and Y, and its chosen rescaling."""

    slug: str
    file_name: str
    separator: str
    predictor: str
    response: str
    chosen: Combination  # the step 3.2 pick, justified in the report

    def load(self, stage: Stage) -> Variables:
        """Read the file. TRANSFORMED -> apply `chosen` on top of the original X and Y."""
        frame = pd.read_csv(DATA_DIR / self.file_name, sep=self.separator)
        original = Variables(frame[self.predictor], frame[self.response])
        return original if stage is Stage.ORIGINAL else self.chosen.apply(original)


DATASETS = (
    Dataset("treadwear", "treadwear.txt", "\t", "mileage", "groove", Combination(Transform.LOG10, Target.Y)),
    Dataset("alligators", "alligator.txt", r"\s+", "length", "weight", Combination(Transform.LOG10, Target.Y)),
    Dataset("concpatog", "ConcPatog.txt", "\t", "CPatog1", "CPatog2", Combination(Transform.LOG10, Target.BOTH)),
)

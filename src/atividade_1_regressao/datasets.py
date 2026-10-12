"""The three datasets of Activity 1: where each file lives and which columns are X and Y."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class DatasetSpec:
    file_name: str
    separator: str
    predictor: str
    response: str


class Dataset(Enum):
    TREADWEAR = DatasetSpec("treadwear.txt", "\t", predictor="mileage", response="groove")
    ALLIGATORS = DatasetSpec("alligator.txt", r"\s+", predictor="length", response="weight")
    CONCPATOG = DatasetSpec("ConcPatog.txt", "\t", predictor="CPatog1", response="CPatog2")

    @property
    def output_dir(self) -> Path:
        return OUTPUTS_DIR / self.name.lower()


class DatasetChoice(Enum):
    # Separate from Dataset so ALL never reaches load_dataset as if it were a file.
    TREADWEAR = (Dataset.TREADWEAR,)
    ALLIGATORS = (Dataset.ALLIGATORS,)
    CONCPATOG = (Dataset.CONCPATOG,)
    ALL = tuple(Dataset)


def load_dataset(dataset: Dataset) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / dataset.value.file_name, sep=dataset.value.separator)

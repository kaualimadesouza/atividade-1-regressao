"""Write the report's outputs for every dataset: `uv run python -m atividade_1_regressao`."""

from atividade_1_regressao.datasets import DATASETS
from atividade_1_regressao.report import write_dataset


def main() -> None:
    """Regenerate outputs/ for the 3 datasets and print each file written."""
    for dataset in DATASETS:
        write_dataset(dataset)


if __name__ == "__main__":
    main()

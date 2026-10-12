"""Write the report's outputs for every dataset: `uv run python -m atividade_1_regressao`."""

from atividade_1_regressao.datasets import DATASETS, OUTPUTS_DIR, ROOT
from atividade_1_regressao.report import run


def main() -> None:
    """Regenerate outputs/ for the 3 datasets and print each file written."""
    for path in run(list(DATASETS), OUTPUTS_DIR):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

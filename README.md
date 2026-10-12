# ACH2036: Atividade 1, Modelos de Regressão

> **Novo em regressão linear?** Comece pelo
> [guia para iniciantes](docs/guia_regressao_iniciantes.md). Ele ensina do zero tudo que
> esta atividade usa (reta, R², testes, intervalos, resíduos, premissas LINE e
> transformações), com exemplos do STAT501 e código Python testado.

Dupla: Kauã Lima de Souza e Victor Yodono. EACH/USP, Prof. Marcelo de Souza Lauretto,
2º semestre de 2026.

Regressão linear simples, análise de resíduos e transformações de escala sobre três
datasets. Enunciado: [`docs/Ativ1_Enunciado1005.pdf`](docs/Ativ1_Enunciado1005.pdf).
Entrega: relatório em PDF, sem código, pelo e-disciplinas até **13/10** (só um da dupla
envia).

## Estrutura

| Pasta | O que tem |
| --- | --- |
| [`docs/`](docs) | Enunciado, [guia para iniciantes](docs/guia_regressao_iniciantes.md), figuras do guia (`img/`), script que as gera (`scripts/`) e o dado do exemplo (`data/`) |
| [`data/`](data) | Datasets da atividade (tabela abaixo) |
| [`src/atividade_1_regressao/`](src/atividade_1_regressao) | Scripts da análise: `datasets.py` (enum dos 3 datasets e leitura), `style.py` (cores das figuras), um script por etapa |
| [`outputs/`](outputs) | Figuras e tabelas geradas, uma pasta por dataset; o relatório lê daqui |
| [`entrega_final/`](entrega_final) | Relatório LaTeX e o PDF compilado |

## Datasets

| Arquivo | Origem | n | Y (resposta) | X (preditora) |
| --- | --- | --- | --- | --- |
| `data/treadwear.txt` | [STAT501, Lição 4.4](https://online.stat.psu.edu/stat501/Lesson04.html) | 9 | `groove` | `mileage` |
| `data/alligator.txt` | [STAT501, Lição 4.7](https://online.stat.psu.edu/stat501/Lesson04.html) | 25 | `weight` | `length` |
| `data/ConcPatog.txt` | Drive compartilhado ACH2036_2026 | 105 | `CPatog2` | `CPatog1` |

`alligator.txt` é separado por espaços; os outros dois, por tab. O ConcPatog tem 3 linhas
com concentração 0, que quebram o `log10` (ver seção 6.6 do guia).

## Como rodar

```bash
uv sync                                          # Python + dependências + pacote em .venv
uv run python -m atividade_1_regressao.scatter_plot   # scatter dos 3 datasets (ou o escolhido no main)
uv run python docs/scripts/make_figures.py       # regera as figuras do guia
cd entrega_final && pdflatex atividade1_MQAM_2026.tex   # compila o relatório (rode 2x)
```

O relatório inclui sozinho cada figura e cada resumo OLS salvo em `outputs/<dataset>/`,
com os nomes da seção 7 do guia. Por padrão cada script gera os 3 datasets; para gerar só um, troque a constante
`choice = DatasetChoice.ALL` no `main` dele (ex.: `DatasetChoice.TREADWEAR`). Enquanto o arquivo não existe, o PDF mostra
um quadro com o caminho esperado. Texto em vermelho é o que ainda falta escrever.

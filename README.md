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
| [`src/atividade_1_regressao/`](src/atividade_1_regressao) | A análise: `datasets.py` (os 3 datasets e seus estágios), `variables.py` (X, Y e as 9 transformações), `regression.py` (ajuste OLS e comparação das 9), `figures.py` (os gráficos), `style.py` (o visual, compartilhado com o guia) e `report.py` (grava tudo em `outputs/`) |
| [`outputs/`](outputs) | Figuras e tabelas geradas, uma pasta por dataset; o relatório lê daqui |
| [`entrega_final/`](entrega_final) | Relatório LaTeX em duas versões, 1 coluna (`atividade1_MQAM_2026`) e 2 colunas (`_2colunas`), com os PDFs. O texto fica só em `conteudo.tex` e o visual comum em `preambulo.tex`, então uma edição vale para as duas |

## Datasets

| Arquivo | Origem | n | Y (resposta) | X (preditora) |
| --- | --- | --- | --- | --- |
| `data/treadwear.txt` | [STAT501, Lição 4.4](https://online.stat.psu.edu/stat501/Lesson04.html) | 9 | `groove` | `mileage` |
| `data/alligator.txt` | [STAT501, Lição 4.7](https://online.stat.psu.edu/stat501/Lesson04.html) | 25 | `weight` | `length` |
| `data/ConcPatog.txt` | Drive compartilhado ACH2036_2026 | 105 | `CPatog2` | `CPatog1` |

`alligator.txt` é separado por espaços; os outros dois, por tab. Dois datasets têm zeros em X,
que quebram o `log10` (ver seção 6.6 do guia): o Treadwear tem `mileage = 0` na primeira linha e
o ConcPatog tem 3 linhas com `CPatog1 = 0`. Nas combinações com `log10` de X essas linhas saem
do ajuste (o `n` de cada combinação aparece na tabela), e o relatório explica por quê.

## Como rodar

```bash
uv sync                                          # Python + dependências + pacote em .venv
uv run python -m atividade_1_regressao          # gera tudo dos 3 datasets em outputs/
uv run python docs/scripts/make_figures.py       # regera as figuras do guia
cd entrega_final && pdflatex atividade1_MQAM_2026.tex   # compila o relatório (rode 2x)
pdflatex atividade1_MQAM_2026_2colunas.tex               # a versão em 2 colunas (rode 2x)
```

Para cada dataset, o comando grava em `outputs/<dataset>/` a dispersão, o resumo OLS, a reta
com IC e IP, o histograma, os resíduos vs preditos e o QQ-plot dos estágios `original` e
`transformado`, com os nomes da seção 7 do guia, mais `transformacoes.png` (as 9 combinações)
e `transformacoes.tex` (o R² ajustado delas). A combinação do estágio `transformado` é o campo
`chosen` de cada dataset em `datasets.py`.

O relatório inclui sozinho cada arquivo de `outputs/`, e a compilação para com erro se algum
faltar.

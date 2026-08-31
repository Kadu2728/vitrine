# Resultados de avaliacao

## Estado atual: NAO MEDIDO

Nao ha metrica de deteccao neste repositorio porque **nao ha detector**. A Fase
1 entrega apenas o dominio -- geometria, agrupamento, share e vazios -- e o
dominio nao detecta nada: ele recebe caixas prontas e calcula.

Precisao, recall e mAP@50 sobre o SKU-110K entram na Fase 2, junto com o
`YoloDetector`. Ate la esta secao permanece com a palavra "nao medido", e nao
com uma estimativa.

## O que ja esta medido na Fase 1

Estes numeros vieram de execucao real e sao reproduziveis pelos comandos
listados.

| Metrica | Valor | Data | Comando |
|---|---|---|---|
| Testes da suite rapida | 151 passando | 2026-08-31 | `uv run pytest` |
| Tempo da suite rapida | 3,9 s | 2026-08-31 | `uv run pytest` |
| Cobertura de `vitrine.domain` | 100% de linhas e de ramos | 2026-08-31 | `uv run pytest --cov=vitrine.domain --cov-report=term-missing` |
| Invariantes de propriedade | 18 propriedades x 500 exemplos, 58 s | 2026-08-31 | `HYPOTHESIS_PROFILE=thorough uv run pytest tests/property` |

Ambiente da medicao: Windows 11, CPython 3.13.3, pytest 8.x, coverage com o
tracer em Python puro (`COVERAGE_CORE=pytrace`), maquina de desenvolvimento do
autor. Os tempos variam com a maquina; a contagem de testes e a cobertura, nao.

## Como a avaliacao da Fase 2 sera feita

Registrado aqui antes de existir, para que o metodo nao seja escolhido depois
de ver o resultado:

- **Dataset**: SKU-110K, split de validacao, sem nenhum reaproveitamento do
  split de treino.
- **Metricas**: precisao, recall e mAP@50, classe unica.
- **Registro**: data, comando exato, versao do peso e hash do arquivo de peso.
- **Se o resultado for ruim**, o numero ruim vai para esta tabela do mesmo
  jeito. Um mAP baixo documentado vale mais que um numero alto sem
  procedencia.

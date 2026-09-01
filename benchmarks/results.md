# Resultados de avaliacao

## SKU-110K: NAO MEDIDO

Precisao, recall e mAP@50 sobre o conjunto de validacao do SKU-110K: **nao
medido**, em 2026-08-31.

O motivo nao e falta de codigo. A maquina de medicao existe, esta testada e roda
(`vitrine benchmark`, ver abaixo). O que falta e o peso: **nao existe checkpoint
YOLO oficial pre-treinado em SKU-110K**. O Ultralytics distribui o
`SKU-110K.yaml` -- uma configuracao de *dataset* para treinar -- e pesos
pre-treinados em COCO, que e outra coisa: COCO nao tem a classe "produto de
gondola" e nao foi treinado em prateleira densa.

Rodar a avaliacao de verdade exige uma destas tres coisas, e a escolha ainda nao
foi feita:

1. **Treinar** sobre o SKU-110K: baixar o dataset (na ordem de 13 GB) e treinar,
   o que precisa de GPU e horas.
2. **Usar um peso de terceiro** treinado em SKU-110K, com a procedencia e a
   licenca verificadas antes.
3. **Medir o COCO como linha de base**, deixando claro que e um piso e nao um
   resultado -- serve para dimensionar quanto o treino especifico melhora.

A opcao 1 esta preparada em `notebooks/treino_sku110k.ipynb`, para rodar no
Google Colab com GPU gratuita: o dataset baixa para o disco do Google, o treino
usa a GPU do Google, e so o peso de ~6 MB desce para a maquina de quem rodou. A
ultima celula imprime o bloco de procedencia pronto para substituir esta secao.

Enquanto nenhuma delas acontecer, este arquivo continua dizendo "nao medido".
Nao ha estimativa aqui, e nao havera.

### Como a medicao sera feita

Registrado antes de existir numero, para que o metodo nao seja escolhido depois
de ver o resultado:

- **Dataset**: SKU-110K, split de validacao, sem nenhum uso do split de treino.
- **Correspondencia**: gulosa em ordem decrescente de confianca, IoU >= 0.5,
  cada anotacao casando com no maximo uma predicao. Predicao duplicada sobre o
  mesmo produto conta como falso positivo.
- **AP**: interpolacao em todos os pontos (VOC 2010+), nao a de 11 pontos.
- **Escala**: as anotacoes sao convertidas de normalizado para pixel usando o
  tamanho da imagem **ja reduzida** por `--max-size`, para que a reducao nao
  desalinhe a comparacao.
- **Registro**: data, comando exato, versao do Ultralytics e sha256 do peso --
  campos que o proprio relatorio ja carrega em `detector`.

Comando:

```bash
vitrine benchmark ./sku110k --split val --detector yolo --weights <peso>.pt --json
```

---

## Verificacao da maquina de medicao (2026-08-31)

O que **esta** medido: que o avaliador computa corretamente o que promete. Sobre
um conjunto sintetico de resultado conhecido -- retangulos desenhados em
coordenadas exatas, com anotacoes derivadas das mesmas contas -- o
`ContourDetector` recupera as caixas pixel a pixel, e o avaliador reporta acerto
total.

| Metrica | Valor |
|---|---|
| Imagens | 2 |
| Produtos anotados | 12 |
| Precisao | 1.0000 |
| Recall | 1.0000 |
| AP@0.5 | 1.0000 |

Isto **nao e um resultado de deteccao**. E o teste de que o instrumento marca
zero corretamente antes de ser usado para medir alguma coisa. Reproduzivel por
`tests/unit/test_benchmark.py::TestComandoBenchmark::test_avaliacao_de_ponta_a_ponta`.

---

## Qualidade de codigo (2026-09-01)

Numeros de execucao real, reproduziveis pelos comandos da tabela.

| Metrica | Valor | Comando |
|---|---|---|
| Suite completa | 358 passando, 1 pulado, 1 desmarcado | `uv run pytest` |
| Laco interno (sem disco nem banco) | 221 passando | `uv run pytest -m "not slow and not integration"` |
| Cobertura de `vitrine.domain` | 100% de linhas e de ramos | `uv run pytest --cov=src/vitrine/domain` |
| Cobertura do pacote | 95% de linhas, excluindo `vision/yolo.py` | `uv run pytest --cov=vitrine --cov-config=.coveragerc-ci` |
| CI (Python 3.12 e 3.13) | verde | https://github.com/Kadu2728/vitrine/actions |
| Invariantes de propriedade | 18 propriedades x 500 exemplos | `HYPOTHESIS_PROFILE=thorough uv run pytest tests/property` |

### O orcamento de 5 segundos: nao cumprido, e por que

A regra do projeto pede suite rapida abaixo de 5 s.

**Na CI, com 7,5 s, ela quase e cumprida.** Nesta maquina, nao: a suite completa
leva de 24 s a 36 s e o laco interno cerca de 13 s. A diferenca de quase cinco
vezes para o mesmo codigo mostra onde esta o problema.

O que foi feito a respeito, com medicao:

- **Marcador `integration`**, separando o que toca disco, banco e a CLI inteira.
  O laco interno roda 221 dos 358 testes. Eles continuam rodando por padrao --
  marcar nao e esconder.
- **Spinner do Rich desligado fora de terminal.** Sozinho, isso levou os testes
  de CLI de lote de 12,5 s para 3,3 s. Era desperdicio real: uma thread de
  atualizacao por execucao, escrevendo ruido em log de CI.
- **`pytest-xdist` testado e descartado**: 5 s viraram 8,3 s com 4 workers e
  19 s com 8. Subir processo no Windows custa mais que o ganho.

O que **nao** foi feito: apagar teste para o numero caber. A suite cresceu de
matematica pura para cobrir IO de imagem, subprocesso, SQLite e mais de 80
invocacoes de CLI; o custo e real e esta declarado em vez de disfarcado.

### Ambiente e ressalvas de medicao

- Windows 11, CPython 3.13.3 (o `pyproject` declara `>=3.12`; a CI cobre 3.12 e
  3.13, mas o interpretador 3.12 baixado pelo `uv` e bloqueado por uma politica
  de Application Control nesta maquina).
- Cobertura com o tracer em Python puro (`COVERAGE_CORE=pytrace`): a DLL do
  tracer em C tambem e bloqueada pela mesma politica.
- **A CI roda a suite completa em 7,5 s** (Linux, runner do GitHub), contra
  24 s a 36 s nesta maquina. Isso confirma o diagnostico: o custo esta no
  ambiente local, nao na suite. **O numero de tempo que vale e o da CI.**
- **Os tempos desta maquina tem variancia patologica**, aparentemente por
  interceptacao na criacao de processo e de arquivo. A mesma suite ja mediu
  2,75 s e 413 s em execucoes consecutivas. Cada teste de historico cria tres
  arquivos (SQLite em modo WAL), e e ai que o custo aparece. **A medicao
  confiavel de tempo tem de vir da CI**, nao daqui.

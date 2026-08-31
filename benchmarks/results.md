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

## Qualidade de codigo (2026-08-31)

Numeros de execucao real, reproduziveis pelos comandos da tabela.

| Metrica | Valor | Comando |
|---|---|---|
| Testes da suite rapida | 279 passando, 1 pulado | `uv run pytest` |
| Tempo da suite rapida | 4,7 s / 5,0 s / 6,6 s em tres execucoes | `uv run pytest` |
| Cobertura de `vitrine.domain` | 100% de linhas e de ramos | `uv run pytest --cov=vitrine.domain` |
| Cobertura do pacote inteiro | 92% de linhas | `uv run pytest --cov=vitrine` |
| Invariantes de propriedade | 18 propriedades x 500 exemplos, 58 s | `HYPOTHESIS_PROFILE=thorough uv run pytest tests/property` |

O teste pulado e o de `vitrine/yolo.py`, que exige o extra `yolo`. Por isso esse
modulo aparece com 0% de cobertura no relatorio: ele e exercitado apenas por
`uv run pytest -m slow`, com o extra instalado.

### Ambiente e ressalvas de medicao

- Windows 11, CPython 3.13.3 (o `pyproject` declara `>=3.12`; a CI cobre 3.12 e
  3.13, mas o interpretador 3.12 baixado pelo `uv` e bloqueado por uma politica
  de Application Control nesta maquina).
- Cobertura com o tracer em Python puro (`COVERAGE_CORE=pytrace`): a DLL do
  tracer em C tambem e bloqueada pela mesma politica.
- **Os tempos desta maquina tem variancia patologica.** A mesma suite ja mediu
  2,75 s e 413 s em execucoes consecutivas, aparentemente por interceptacao na
  criacao de processo. Os numeros acima sao de execucoes consecutivas sem
  outlier, mas devem ser lidos como ordem de grandeza. A medicao confiavel de
  tempo virá da CI.
- `pytest-xdist` foi testado e **descartado por medicao**: com 4 workers a suite
  passou de ~5 s para 8,3 s, e com 8 workers para 19 s. O custo de subir
  processo no Windows domina uma suite deste tamanho.

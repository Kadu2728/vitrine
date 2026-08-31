# Vitrine

**Auditoria de execução em ponto de venda por visão computacional.**

> **Status: Fase 1 de 4 — domínio puro.**
> A matemática está pronta e testada. **Ainda não existe detector, imagem, CLI
> nem persistência.** O que existe é uma biblioteca que transforma caixas
> delimitadoras em um relatório de gôndola, e que pode ser verificada de ponta a
> ponta sem carregar um único peso de modelo. A lista completa do que não
> funciona está em [O que ainda não existe](#o-que-ainda-não-existe).

---

## O problema

Trabalho como promotor de merchandising. A rotina é essa: chego na loja com uma
prancheta, conto quantos produtos meus estão na gôndola, anoto se tem espaço
vazio, tiro uma foto, mando no grupo do WhatsApp e alguém joga aquilo numa
planilha no fim do dia. Na loja seguinte, tudo de novo.

A indústria de bens de consumo paga — em dinheiro, em bonificação, em acordo
comercial — para que seus produtos ocupem um determinado espaço de prateleira. E
não tem como verificar se aquele espaço está sendo ocupado. A informação existe:
está na foto que eu tirei. Só que ela morre ali, como imagem, sem virar número.
Quando vira número, vira o número que eu digitei na prancheta, que ninguém
consegue auditar depois.

Este projeto pega a foto e devolve o número. Quantos produtos estão expostos,
como o espaço está dividido, onde tem buraco na prateleira, e como isso mudou
desde a última visita. O problema não é hipotético e o sistema não é uma demo:
é a ferramenta que eu queria ter.

---

## Por que não tem interface web

Porque a interface não é o difícil aqui, e fingir que é seria desonesto.

Este projeto **não tem front-end, não tem API HTTP e não tem Docker Compose com
cinco serviços**. É decisão de arquitetura, não limitação. A saída visual é a
imagem anotada — ela *é* a UI. A integração com qualquer outro sistema se faz
pelo `--json`, com schema versionado — ele *é* a API. Uma camada web em cima
disso adicionaria superfície de manutenção sem adicionar capacidade.

O efeito colateral é o ponto: sem tela bonita, não há onde esconder problema. A
qualidade do código, dos testes e do contrato de saída é o produto inteiro.

---

## O que faz e o que não faz

**Faz** (Fase 1, funcionando e testado):

- Agrupa detecções soltas em prateleiras, com limiar relativo ao tamanho do
  produto — funciona igual em foto de 800 px ou de 4000 px.
- Calcula **share por contagem** e **share por área linear**, lado a lado.
- Calcula **ocupação** por prateleira e por região.
- Detecta **espaço vazio** com limiar relativo à largura mediana do produto
  daquela prateleira.
- Remove detecções duplicadas antes de contar.
- Emite um relatório Pydantic serializável, com schema versionado e os
  parâmetros da análise ecoados na saída.

**Não faz, e não vai fazer:**

- **Não identifica SKU nem marca.** Produto é classe única. Distinguir uma
  marca da outra é um problema de classificação fina que exige dataset
  rotulado por SKU, e está fora do escopo. Sem isso, "share of shelf" só faz
  sentido entre *regiões espaciais* da gôndola, e é assim que ele é calculado.
- **Não detecta prateleira vazia.** O sistema infere prateleiras a partir dos
  produtos. Onde não há produto, não há prateleira, e portanto não há alerta.
  Ruptura total de uma prateleira inteira é um ponto cego real deste método.
- **Não tem interface web.** Ver acima.

---

## Como funciona

```
detecções (caixas)
      │
      ├─ dedup ............. remove a mesma garrafa contada três vezes
      │
      ├─ shelves ........... agrupa em prateleiras (clusterização 1D)
      │
      ├─ share ............. contagem, área linear e ocupação por região
      │
      └─ gaps .............. espaços vazios que caberia produto
      │
      ▼
  ShareReport (JSON com schema versionado)
```

A regra de dependência é rígida: `domain/` não importa nada de `vision/`,
`storage/`, `batch/` ou `render/`, e não conhece imagem, arquivo nem modelo. Isso
não é uma promessa do README — é
[um teste](tests/unit/test_architecture.py) que quebra a suíte se alguém violar.

### As quatro decisões que importam

**1. Agrupamento em prateleiras.** Prateleira é inferência, não dado.
Clusterização aglomerativa 1D por *single linkage* sobre o centro vertical, com
limiar `τ = 0.5 × mediana(altura dos produtos)`. Relativo, nunca absoluto em
pixels: um limiar em pixels quebraria assim que a resolução mudasse.

Descartei k-means (exige saber quantas prateleiras existem — que é justamente o
que não se sabe), DBSCAN (em 1D degenera para o mesmo algoritmo, com dois
hiperparâmetros a mais e uma dependência externa) e histograma com detecção de
picos (depende do tamanho do bin, e bin é uma constante em pixels).

Single linkage puro sofre de *chaining*: uma escada de produtos com centros
deslizando de pouco em pouco — o que acontece em toda gôndola fotografada em
ângulo — funde duas prateleiras num cluster só, silenciosamente. Por isso todo
cluster com dispersão vertical acima de `1.5 × mediana(altura)` é reparticionado
na sua maior lacuna interna. Onde o método quebra está documentado em
[`shelves.py`](src/vitrine/domain/shelves.py).

**2. Share of shelf tem duas definições, e elas discordam.** Share por contagem
de produtos e share por área linear ocupada não dão o mesmo número — e chegam a
inverter a ordem entre regiões. Duas embalagens grandes contra três pequenas: por
contagem, as pequenas ganham; por área, as grandes. As duas leituras são
legítimas e respondem a perguntas diferentes. Publicar só uma seria escolher a
resposta mais conveniente, então o relatório traz as duas.

O share linear é calculado sobre a **união** das projeções horizontais, nunca
pela soma das larguras: em gôndola cheia os produtos se sobrepõem na projeção 2D,
e a soma ingênua produziria share acima de 100%.

**3. O denominador do share é uma decisão, não um dado.** `ocupado / total` exige
dizer o que é `total`, e a prateleira física não é detectada. São duas escolhas
defensáveis, com números diferentes:

- `envelope` (padrão): da borda esquerda do primeiro produto à direita do
  último. Ignora vazio nas pontas e tende a superestimar a ocupação.
- `explicit`: limites informados por quem conhece a gôndola. Enxerga vazio nas
  extremidades, ao custo de depender de um parâmetro humano.

O relatório **sempre** carrega qual foi usado, para que o número seja auditável.

**4. Espaço vazio é relativo, não absoluto.** Vazio não é ausência de caixa —
entre dois produtos vizinhos sempre sobram alguns pixels. Vazio é um intervalo
onde caberia mais um produto *daquela prateleira*. O limiar é a largura mediana
local: uma prateleira de latas e uma de caixas de sabão em pó têm noções
diferentes de "grande".

---

## O contrato de saída

Saída real, gerada pela gôndola de exemplo dos testes (recorte; o JSON completo
tem mais campos):

```json
{
  "schema_version": "1.0",
  "status": "ok",
  "total_detections": 7,
  "shelf_count": 2,
  "regions": [
    { "region": "esquerda", "count": 3, "count_share": 0.4285714285714286,
      "occupied_length": 205.0, "linear_share": 0.5540540540540541,
      "occupancy": 0.6612903225806451 },
    { "region": "direita",  "count": 4, "count_share": 0.5714285714285714,
      "occupied_length": 165.0, "linear_share": 0.4459459459459459,
      "occupancy": 0.532258064516129 }
  ]
}
```

Repare no que este exemplo mostra: **por contagem a direita ganha (4 contra 3),
por área linear a esquerda ganha (0,554 contra 0,446)**. É exatamente o caso que
justifica publicar as duas métricas em vez de escolher uma.

`occupancy` não é share e não soma 1,0 — é ocupado dividido pela largura da
região. É a única das três métricas afirmável sem definir regiões, e por isso é o
que o relatório traz por padrão.

---

## Resultados

Métricas de detecção — precisão, recall, mAP@50 sobre o SKU-110K: **não
medido.** Não existe detector neste repositório ainda. O número entra em
[`benchmarks/results.md`](benchmarks/results.md) na Fase 2, com data e comando,
seja ele bom ou ruim.

O que está medido hoje, de execução real em 2026-08-31:

| Métrica | Valor |
|---|---|
| Testes da suíte rápida | 151 passando |
| Tempo da suíte rápida | 3,9 s |
| Cobertura de `vitrine.domain` | **100%** de linhas e de ramos |
| Invariantes de propriedade | 18 propriedades × 500 exemplos, 58 s |

Comandos e ambiente em [`benchmarks/results.md`](benchmarks/results.md).

---

## Como rodar

Ainda não há CLI — ela chega na Fase 2. O que existe é a biblioteca:

```bash
uv sync
```

```python
from vitrine import BoundingBox, Detection, RegionSet, analyze_detections

detections = [
    Detection(box=BoundingBox(x1=0, y1=0, x2=80, y2=100)),
    Detection(box=BoundingBox(x1=90, y1=0, x2=170, y2=100)),
    Detection(box=BoundingBox(x1=200, y1=0, x2=230, y2=100)),
]

report = analyze_detections(
    detections,
    regions=RegionSet.from_cuts((0.0, 0.5, 1.0), ("esquerda", "direita")),
)
print(report.model_dump_json(indent=2))
```

---

## Testes

Suíte rápida — o ciclo de desenvolvimento, sem modelo, abaixo de 5 segundos:

```bash
uv run pytest
```

Invariantes em profundidade — 500 exemplos por propriedade, cerca de um minuto:

```bash
HYPOTHESIS_PROFILE=thorough uv run pytest tests/property
```

Cobertura do domínio:

```bash
uv run pytest --cov=vitrine.domain --cov-report=term-missing
```

Tipos e lint:

```bash
uv run mypy src/ tests/ && uv run ruff check . && uv run ruff format --check .
```

### As invariantes

Um exemplo escrito à mão prova que uma conta está certa. Uma invariante prova que
ela continua certa para entradas que ninguém pensou em escrever. As oito:

| | Invariante |
|---|---|
| P1 | A soma dos shares vale 1,0 — por contagem e por área, no total e em cada prateleira |
| P2 | Reordenar as detecções de entrada não muda um byte da saída |
| P3 | Escalar todas as coordenadas não muda proporção alguma |
| P4 | Transladar a foto inteira não muda nada — pega o bug de comparar coordenada absoluta com limiar |
| P5 | O agrupamento é uma partição total: nada se perde, nada se duplica, cada detecção cai em exatamente uma região |
| P6 | Com caixas sobrepostas o share continua em [0, 1] — o que força união de intervalos em vez de soma de larguras |
| P7 | Vazios são disjuntos, ordenados e não tocam produto nenhum |
| P8 | Duas execuções sobre a mesma entrada produzem o mesmo JSON, inclusive em outro processo com `PYTHONHASHSEED` diferente |

As estratégias do hypothesis usam coordenadas inteiras e fatores de escala em
potências de dois, de propósito: assim P3 e P4 são verificadas com **igualdade
exata** em vez de tolerância. O raciocínio está em
[`strategies.py`](tests/property/strategies.py).

---

## O que ainda não existe

Sem eufemismo, e sem `TODO` escondido no código:

- **Detector.** Não há `Detector` Protocol, não há `FakeDetector`, não há
  `YoloDetector`. Fase 2.
- **Qualquer coisa que toque em imagem.** Nada de OpenCV, Pillow ou NumPy — o
  `pyproject` da Fase 1 declara Pydantic e mais nada, e
  [um teste](tests/unit/test_architecture.py) impede que isso mude por descuido.
- **Correção de perspectiva.** Fase 2, com quatro pontos informados à mão.
  Detecção automática do retângulo da gôndola está fora do MVP: é um projeto
  inteiro sozinho.
- **CLI.** Fase 2.
- **Renderização da imagem anotada e o GIF de demonstração.** Fase 2 — o GIF
  virá de execução real, não de mockup.
- **Lote, SQLite e histórico por PDV.** Fase 3.
- **Métricas de detecção.** Fase 2. Hoje: não medido.

Limitações do que **já** existe, que não vão embora com mais código:

- Prateleira totalmente vazia é invisível para o agrupamento.
- Gôndola inclinada ou com perspectiva não corrigida degrada o agrupamento; o
  relatório sinaliza via `spread_ratio` e um aviso, mas não corrige.
- Prateleiras de alturas muito diferentes na mesma foto usam uma mediana de
  altura global, que é o denominador errado para ambas.
- Produto deitado distorce a altura mediana e portanto o limiar da foto inteira.

---

## Licença e ética

Código sob licença MIT.

**SKU-110K** é distribuído para uso acadêmico e não comercial. Quando o detector
entrar (Fase 2), o dataset será usado dentro desses termos, e nenhum peso
treinado sobre ele será redistribuído comercialmente a partir deste repositório.

**Nenhuma imagem neste repositório identifica loja, rede ou marca de cliente
real.** Os testes usam detecções construídas à mão — retângulos com coordenadas
escolhidas para que o resultado seja conferível no papel — e não imagens. Quando
houver foto real de campo, ela entra anonimizada: sem fachada, sem placa de
preço legível, sem crachá, sem rosto, e com o identificador de loja substituído
por um código opaco. O procedimento será descrito aqui antes de a primeira foto
entrar.

Nada neste projeto faz scraping de imagem de terceiros.

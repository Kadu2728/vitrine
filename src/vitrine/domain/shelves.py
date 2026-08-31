"""Agrupamento de deteccoes em prateleiras.

Prateleira e inferencia, nao dado. O detector devolve uma lista solta de
caixas; a nocao de "mesma prateleira" precisa ser reconstruida.

Metodo
------
Clusterizacao aglomerativa 1D por *single linkage* sobre o centro vertical das
caixas, com limiar relativo ``tau = shelf_gap_ratio * mediana(altura)``, mais
uma guarda contra encadeamento.

Por que este metodo e nao outro:

- **k-means 1D** exige ``k``. O numero de prateleiras e justamente o que nao se
  sabe.
- **DBSCAN** em 1D degenera para o mesmo single linkage por lacuna, com dois
  hiperparametros a mais e uma dependencia externa que o dominio nao pode ter.
- **Histograma + deteccao de picos** depende do tamanho do bin, e bin e uma
  constante em pixels: quebraria as invariantes de escala e translacao.

O limiar e relativo a altura mediana, nunca absoluto em pixels. Duas
prateleiras distintas separam seus centros por aproximadamente uma altura de
produto; produtos da mesma prateleira mantem os centros dentro de meia altura,
mesmo variando de tamanho. Dai o padrao ``0.5``.

A guarda contra encadeamento
----------------------------
Single linkage puro sofre de *chaining*: uma escada de produtos com centros
deslizando de pouco em pouco -- o que acontece em toda gondola fotografada em
angulo -- funde duas prateleiras num cluster so, silenciosamente. Por isso,
apos a formacao dos clusters, todo cluster cuja dispersao vertical exceda
``max_shelf_spread_ratio * mediana(altura)`` e reparticionado na sua maior
lacuna interna, recursivamente.

Onde este metodo quebra
-----------------------
Documentado aqui e no README, porque limitacao declarada vale mais que promessa
vaga:

1. **Gondola inclinada ou perspectiva nao corrigida.** O centro vertical de uma
   mesma prateleira varia com ``x``. A guarda de dispersao evita o pior caso --
   uma prateleira unica -- mas o corte fica arbitrario. Sinalizado via
   ``spread_ratio``. Correcao real: regressao robusta por prateleira e
   clusterizacao sobre o residuo.
2. **Prateleiras de alturas muito diferentes na mesma foto.** A mediana global
   e o denominador errado para ambas. Mediana local seria melhor e e circular:
   depende da prateleira que ainda nao existe.
3. **Produto deitado.** Distorce a altura mediana e portanto ``tau`` para a
   foto inteira.
4. **Prateleira totalmente vazia e invisivel.** O metodo infere prateleiras a
   partir de produtos; onde nao ha produto, nao ha prateleira, e portanto nao
   ha alerta de ruptura. Este e um ponto cego real de um sistema que se propoe
   a detectar ruptura, e nao ha como contorna-lo sem detectar a prateleira
   fisica na imagem.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from vitrine.domain import geometry
from vitrine.domain.models import Detection, Shelf

DEFAULT_SHELF_GAP_RATIO = 0.5
"""Lacuna entre centros verticais, em multiplos da altura mediana, que separa
duas prateleiras."""

DEFAULT_MAX_SPREAD_RATIO = 1.5
"""Dispersao vertical maxima tolerada dentro de uma prateleira, em multiplos da
altura mediana global."""

_Item = tuple[float, Detection]
"""Deteccao pareada com o seu centro vertical, para ordenar uma vez so."""


def group_into_shelves(
    detections: Sequence[Detection],
    *,
    gap_ratio: float = DEFAULT_SHELF_GAP_RATIO,
    max_spread_ratio: float = DEFAULT_MAX_SPREAD_RATIO,
) -> tuple[Shelf, ...]:
    """Agrupa deteccoes em prateleiras, da mais alta para a mais baixa.

    O resultado e uma particao total das deteccoes de entrada: nenhuma se
    perde, nenhuma aparece duas vezes (invariante P5).

    Args:
        detections: deteccoes em qualquer ordem.
        gap_ratio: limiar de separacao, em multiplos da altura mediana.
        max_spread_ratio: dispersao maxima antes de reparticionar um cluster.

    Returns:
        Prateleiras indexadas de cima para baixo, cada uma com as deteccoes
        ordenadas da esquerda para a direita.

    Raises:
        ValueError: se algum dos limiares nao for positivo.
    """
    if gap_ratio <= 0.0:
        raise ValueError(f"gap_ratio precisa ser positivo; recebido {gap_ratio!r}")
    if max_spread_ratio <= 0.0:
        raise ValueError(f"max_spread_ratio precisa ser positivo; recebido {max_spread_ratio!r}")
    if not detections:
        return ()

    median_height = geometry.median([d.box.height for d in detections])
    tau = gap_ratio * median_height
    max_spread = max_spread_ratio * median_height

    items: list[_Item] = sorted(
        ((d.box.center_y, d) for d in detections),
        key=lambda item: (item[0], item[1].sort_key),
    )

    clusters: list[list[_Item]] = []
    for cluster in _split_by_gap(items, tau):
        clusters.extend(_enforce_spread(cluster, max_spread))

    return tuple(
        Shelf(
            index=index,
            detections=tuple(sorted((item[1] for item in cluster), key=lambda d: d.sort_key)),
        )
        for index, cluster in enumerate(clusters)
    )


def _split_by_gap(items: Sequence[_Item], tau: float) -> list[list[_Item]]:
    """Corta a sequencia ordenada onde a lacuna entre centros excede ``tau``."""
    clusters: list[list[_Item]] = [[items[0]]]
    for previous, current in pairwise(items):
        if current[0] - previous[0] > tau:
            clusters.append([current])
        else:
            clusters[-1].append(current)
    return clusters


def _enforce_spread(cluster: list[_Item], max_spread: float) -> list[list[_Item]]:
    """Reparte um cluster esticado demais na sua maior lacuna interna.

    Desempate deterministico: entre lacunas de mesmo tamanho, escolhe a mais
    proxima do meio -- corte equilibrado -- e, persistindo o empate, a de menor
    indice. Termina sempre: cada corte produz duas partes estritamente menores.
    """
    if len(cluster) < 2:
        return [cluster]

    spread = cluster[-1][0] - cluster[0][0]
    if spread <= max_spread:
        return [cluster]

    middle = len(cluster) / 2.0
    best_index = max(
        range(len(cluster) - 1),
        key=lambda i: (cluster[i + 1][0] - cluster[i][0], -abs((i + 1) - middle), -i),
    )
    left = cluster[: best_index + 1]
    right = cluster[best_index + 1 :]
    return _enforce_spread(left, max_spread) + _enforce_spread(right, max_spread)

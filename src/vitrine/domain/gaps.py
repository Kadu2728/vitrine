"""Deteccao de espaco vazio dentro de uma prateleira.

Vazio nao e ausencia de caixa em termos absolutos -- entre dois produtos
vizinhos sempre ha alguns pixels livres. Vazio e um intervalo horizontal largo
o bastante *em relacao ao produto daquela prateleira*: onde caberia mais um
item e nao ha nenhum.

Por isso o limiar e a largura mediana dos produtos da propria prateleira, e nao
uma constante global. Uma prateleira de latas e uma de caixas de sabao em po
tem nocoes diferentes de "grande".
"""

from __future__ import annotations

from vitrine.domain import geometry
from vitrine.domain.models import Gap, Shelf, ShelfExtent

DEFAULT_GAP_MIN_WIDTH_RATIO = 1.0
"""Um vazio conta quando cabe ao menos um produto mediano dentro dele."""


def find_gaps(
    shelf: Shelf,
    extent: ShelfExtent,
    *,
    min_width_ratio: float = DEFAULT_GAP_MIN_WIDTH_RATIO,
) -> tuple[Gap, ...]:
    """Encontra os vazios de uma prateleira dentro da extensao considerada.

    Os vazios sao o complemento da uniao das projecoes horizontais das caixas.
    Por construcao nao intersectam nenhuma deteccao, sao disjuntos entre si e
    vem ordenados por ``x`` (invariante P7).

    A extensao importa: com ``kind="envelope"`` so aparecem vazios internos,
    porque as bordas coincidem com o primeiro e o ultimo produto. Vazio nas
    pontas da gondola so e visivel com extensao explicita.

    Args:
        shelf: prateleira analisada.
        extent: o que conta como largura total da prateleira.
        min_width_ratio: largura minima do vazio, em multiplos da largura
            mediana do produto desta prateleira.

    Returns:
        Vazios em ordem crescente de ``x``.

    Raises:
        ValueError: se ``min_width_ratio`` nao for positivo.
    """
    if min_width_ratio <= 0.0:
        raise ValueError(f"min_width_ratio precisa ser positivo; recebido {min_width_ratio!r}")

    median_width = shelf.median_width
    minimum = min_width_ratio * median_width
    free = geometry.complement(
        [d.box.x_interval for d in shelf.detections],
        (extent.x_min, extent.x_max),
    )

    return tuple(
        Gap(
            x_start=start,
            x_end=end,
            y_top=shelf.y_top,
            y_bottom=shelf.y_bottom,
            width_ratio=(end - start) / median_width,
        )
        for start, end in free
        if (end - start) >= minimum
    )

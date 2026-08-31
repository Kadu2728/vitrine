"""Remocao de deteccoes duplicadas.

Um detector real com NMS mal calibrado entrega a mesma garrafa duas ou tres
vezes. Como "quantos produtos estao expostos" e a primeira pergunta que o
sistema responde, contar duplicata infla exatamente o numero que mais importa.

Isto e geometria pura, nao pos-processamento de modelo: mora no dominio e e
testavel sem carregar peso nenhum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vitrine.domain.models import Detection

DEFAULT_DEDUP_IOU = 0.9
"""Limiar conservador: so remove caixas quase coincidentes.

Deliberadamente alto. Em gondola cheia, produtos vizinhos legitimamente se
sobrepoem na projecao 2D; um limiar baixo apagaria produto real, que e um erro
pior que contar duplicata -- ruptura falsa manda promotor viajar a toa.
"""


def deduplicate(
    detections: Sequence[Detection],
    *,
    iou_threshold: float = DEFAULT_DEDUP_IOU,
) -> tuple[Detection, ...]:
    """Remove deteccoes que se sobrepoem acima de ``iou_threshold``.

    Guloso, mantendo sempre a de maior confianca. A ordenacao inicial e por
    ``(-confianca, sort_key)``, que e uma ordem total: portanto o resultado nao
    depende da ordem em que o detector devolveu as caixas (invariante P2).

    Args:
        detections: deteccoes candidatas, em qualquer ordem.
        iou_threshold: IoU a partir do qual duas caixas sao a mesma coisa.

    Returns:
        Deteccoes mantidas, ordenadas por ``sort_key``.

    Raises:
        ValueError: se ``iou_threshold`` estiver fora de ``(0, 1]``.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold precisa estar em (0, 1]; recebido {iou_threshold!r}")

    ordered = sorted(detections, key=lambda d: (-d.confidence, d.sort_key))
    kept: list[Detection] = []
    for candidate in ordered:
        if all(candidate.box.iou(other.box) < iou_threshold for other in kept):
            kept.append(candidate)
    return tuple(sorted(kept, key=lambda d: d.sort_key))

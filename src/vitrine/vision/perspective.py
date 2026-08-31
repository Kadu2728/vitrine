"""Correcao de perspectiva por quatro pontos informados.

Fotografar gondola de frente e raro: o corredor e estreito, o promotor esta de
lado, e a imagem sai com as prateleiras convergindo. Isso quebra o agrupamento,
que assume que produtos da mesma prateleira compartilham a faixa vertical.

**Deteccao automatica do retangulo da gondola esta fora do MVP.** E um projeto
inteiro sozinho -- segmentacao de plano, deteccao de linhas, escolha entre
candidatos -- e consumiria a semana que deve ir para o resto. Entra depois, e
somente se for medido que ajuda.

O que existe aqui e a versao honesta: quem tirou a foto informa os quatro cantos
da area util, e a homografia leva esse quadrilatero a um retangulo. A imagem e
retificada **antes** da deteccao, nunca as caixas depois -- transformar caixas
por homografia produz quadrilateros, e o dominio so entende retangulo alinhado
ao eixo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from vitrine.errors import PerspectiveError

if TYPE_CHECKING:
    from numpy.typing import NDArray

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]

MIN_OUTPUT_SIDE = 16
"""Menor lado aceitavel do retangulo de saida, em pixels."""


def order_corners(points: Quad) -> Quad:
    """Ordena os quatro cantos no sentido horario a partir do superior-esquerdo.

    O usuario informa os cantos na ordem que quiser; o resultado precisa ser o
    mesmo. A ordenacao usa soma e diferenca das coordenadas, que e estavel para
    quadrilateros convexos: o canto superior-esquerdo minimiza ``x + y``, o
    inferior-direito maximiza, e a diagonal secundaria separa os outros dois
    por ``y - x``.

    Args:
        points: os quatro cantos, em qualquer ordem.

    Returns:
        Os mesmos pontos em ordem horaria comecando pelo superior-esquerdo.
    """
    por_soma = sorted(range(4), key=lambda i: (points[i][0] + points[i][1], points[i]))
    indice_se, indice_id = por_soma[0], por_soma[-1]
    restantes = sorted(
        (i for i in range(4) if i not in (indice_se, indice_id)),
        key=lambda i: (points[i][1] - points[i][0], points[i]),
    )
    return (
        points[indice_se],
        points[restantes[0]],
        points[indice_id],
        points[restantes[1]],
    )


def rectify(
    pixels: NDArray[np.uint8],
    corners: Quad,
    *,
    output_size: tuple[int, int] | None = None,
) -> NDArray[np.uint8]:
    """Aplica a homografia que leva ``corners`` a um retangulo.

    O tamanho do retangulo de saida, quando nao informado, vem do proprio
    quadrilatero: a maior das duas bordas horizontais e a maior das duas
    verticais. Isso preserva o detalhe do lado mais proximo da camera, que e o
    que tem mais informacao.

    Args:
        pixels: imagem BGR uint8.
        corners: os quatro cantos da area util, em qualquer ordem.
        output_size: ``(largura, altura)`` forcados, se desejado.

    Returns:
        A imagem retificada.

    Raises:
        PerspectiveError: se os pontos forem repetidos, colineares, estiverem
            fora da imagem ou produzirem um retangulo degenerado.
    """
    _validate(pixels, corners)
    ordered = order_corners(corners)

    if output_size is None:
        largura, altura = _infer_size(ordered)
    else:
        largura, altura = output_size

    if largura < MIN_OUTPUT_SIDE or altura < MIN_OUTPUT_SIDE:
        raise PerspectiveError(
            f"Os quatro pontos produzem uma area de {largura}x{altura} pixels, "
            f"pequena demais para analisar.",
            "Confira a ordem e a unidade dos pontos: sao coordenadas em pixels "
            "da imagem original, no formato x,y.",
        )

    origem = np.array(ordered, dtype=np.float32)
    destino = np.array(
        [(0.0, 0.0), (largura - 1.0, 0.0), (largura - 1.0, altura - 1.0), (0.0, altura - 1.0)],
        dtype=np.float32,
    )
    matriz = cv2.getPerspectiveTransform(origem, destino)
    return cast(
        "NDArray[np.uint8]",
        cv2.warpPerspective(pixels, matriz, (largura, altura), flags=cv2.INTER_LINEAR),
    )


def _validate(pixels: NDArray[np.uint8], corners: Quad) -> None:
    """Recusa quadrilateros que nao servem, com a dica correspondente."""
    if len(corners) != 4:
        raise PerspectiveError(
            f"Sao necessarios exatamente 4 pontos; recebidos {len(corners)}.",
            "Informe os quatro cantos da area util: --perspective x1,y1 x2,y2 x3,y3 x4,y4",
        )
    if len(set(corners)) != 4:
        raise PerspectiveError(
            "Ha pontos repetidos entre os quatro cantos informados.",
            "Cada canto precisa ser distinto dos outros tres.",
        )

    altura, largura = pixels.shape[:2]
    fora = [p for p in corners if not (0 <= p[0] <= largura and 0 <= p[1] <= altura)]
    if fora:
        raise PerspectiveError(
            f"Pontos fora da imagem de {largura}x{altura}: {fora}.",
            "As coordenadas sao em pixels da imagem ja carregada. Se voce usou "
            "--max-size, informe os pontos no tamanho reduzido ou desative a reducao.",
        )

    if _area(order_corners(corners)) < 1.0:
        raise PerspectiveError(
            "Os quatro pontos sao colineares ou formam area nula.",
            "Escolha cantos que realmente delimitem a gondola, nao pontos numa mesma linha.",
        )


def _infer_size(ordered: Quad) -> tuple[int, int]:
    """Deduz o tamanho de saida a partir das bordas do quadrilatero."""
    superior_esquerdo, superior_direito, inferior_direito, inferior_esquerdo = ordered
    largura = max(
        _distance(superior_esquerdo, superior_direito),
        _distance(inferior_esquerdo, inferior_direito),
    )
    altura = max(
        _distance(superior_esquerdo, inferior_esquerdo),
        _distance(superior_direito, inferior_direito),
    )
    return round(largura), round(altura)


def _distance(a: Point, b: Point) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def _area(ordered: Quad) -> float:
    """Area do quadrilatero pela formula do cadarco."""
    total = 0.0
    for index, (x, y) in enumerate(ordered):
        next_x, next_y = ordered[(index + 1) % 4]
        total += x * next_y - next_x * y
    return abs(total) / 2.0

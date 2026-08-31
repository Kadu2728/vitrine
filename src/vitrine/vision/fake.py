"""Detector de teste: devolve exatamente o que voce mandou ele devolver.

Nao e um mock improvisado dentro de um arquivo de teste. E uma implementacao de
primeira classe do protocolo ``Detector``, e por isso qualquer coisa construida
sobre ele -- pipeline, render, CLI -- pode ser exercitada de verdade, do
carregamento da imagem ate o JSON final, em milissegundos e sem baixar peso
nenhum.

Existe tambem ``from_grid``, que fabrica uma gondola sintetica de resultado
conhecido: N prateleiras com M produtos cada, coordenadas calculaveis no papel.
E o que sustenta os testes de ponta a ponta sem fixture de imagem pesada no Git.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vitrine.domain.models import BoundingBox, Detection, DetectorInfo

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


class FakeDetector:
    """Detector deterministico que ignora os pixels e devolve caixas fixas.

    Args:
        detections: o que ``detect`` vai devolver, sempre.
        name: identificacao que aparece no relatorio.
    """

    def __init__(self, detections: Sequence[Detection] = (), *, name: str = "fake") -> None:
        self._detections = tuple(detections)
        self._name = name

    @property
    def info(self) -> DetectorInfo:
        """Identificacao deste detector, marcada claramente como sintetica."""
        return DetectorInfo(
            name=self._name,
            version="synthetic",
            weights=None,
            weights_sha256=None,
            confidence_threshold=0.0,
            iou_threshold=0.0,
        )

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        """Devolve as deteccoes configuradas, ignorando a imagem."""
        del image
        return self._detections

    @classmethod
    def from_grid(
        cls,
        *,
        rows: int,
        columns: int,
        box_width: float = 60.0,
        box_height: float = 90.0,
        gap_x: float = 20.0,
        gap_y: float = 60.0,
        origin: tuple[float, float] = (20.0, 20.0),
        name: str = "fake-grid",
    ) -> FakeDetector:
        """Fabrica uma gondola regular de resultado conhecido.

        A prateleira ``r`` fica em ``y = origin_y + r * (box_height + gap_y)`` e
        o produto ``c`` em ``x = origin_x + c * (box_width + gap_x)``. Como toda
        coordenada e uma conta de uma linha, o resultado esperado de qualquer
        metrica pode ser derivado no papel.

        Args:
            rows: numero de prateleiras.
            columns: produtos por prateleira.
            box_width: largura de cada produto.
            box_height: altura de cada produto.
            gap_x: espaco horizontal entre produtos vizinhos.
            gap_y: espaco vertical entre prateleiras.
            origin: canto superior esquerdo do primeiro produto.
            name: identificacao do detector.

        Returns:
            Um detector que devolve essa grade.

        Raises:
            ValueError: se ``rows`` ou ``columns`` nao forem positivos.
        """
        if rows <= 0 or columns <= 0:
            raise ValueError(f"rows e columns precisam ser positivos; recebido {rows}x{columns}")

        origin_x, origin_y = origin
        detections = [
            Detection(
                box=BoundingBox(
                    x1=origin_x + column * (box_width + gap_x),
                    y1=origin_y + row * (box_height + gap_y),
                    x2=origin_x + column * (box_width + gap_x) + box_width,
                    y2=origin_y + row * (box_height + gap_y) + box_height,
                )
            )
            for row in range(rows)
            for column in range(columns)
        ]
        return cls(detections, name=name)

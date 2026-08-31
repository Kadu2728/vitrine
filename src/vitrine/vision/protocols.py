"""O contrato entre o mundo dos modelos e o resto do sistema.

Esta e a peca mais importante do projeto, e vale explicar por que.

Um modelo de deteccao e uma dependencia **lenta, pesada e nao-deterministica**:
carrega centenas de megabytes, demora segundos por imagem e pode devolver
resultados ligeiramente diferentes entre versoes. Se a logica de negocio
importar o Ultralytics diretamente, nada abaixo dela pode ser testado sem pagar
esse custo -- e na pratica isso significa que nao sera testado.

Com o protocolo, a direcao da dependencia inverte: o pipeline depende de
``Detector``, e ``YoloDetector`` e apenas *uma* implementacao possivel. A suite
rapida inteira roda com ``FakeDetector`` e ``ContourDetector``, em milissegundos
e sem baixar um unico peso.

Uma implementacao de ``Detector`` precisa de duas coisas:

- ``info``: quem e, com que peso e com que limiares. Vai no relatorio, porque
  comparar duas visitas ao mesmo PDV so faz sentido se o detector for o mesmo.
- ``detect``: pixels entram, deteccoes saem. Nada de caminho de arquivo -- quem
  carrega a imagem, corrige EXIF e retifica perspectiva e o pipeline, nao o
  detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from vitrine.domain.models import Detection, DetectorInfo

BgrImage = "NDArray[np.uint8]"
"""Imagem em BGR uint8, convencao do OpenCV, com shape ``(altura, largura, 3)``."""


@runtime_checkable
class Detector(Protocol):
    """Qualquer coisa capaz de encontrar produtos numa imagem.

    ``runtime_checkable`` para que a suite verifique conformidade estrutural,
    mas a verificacao que importa e a do mypy: se uma implementacao nao bater
    com esta assinatura, o erro aparece antes de rodar.
    """

    @property
    def info(self) -> DetectorInfo:
        """Identificacao e limiares deste detector, para o relatorio."""
        ...

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        """Encontra produtos numa imagem BGR uint8.

        Args:
            image: pixels em BGR, shape ``(altura, largura, 3)``, ja com EXIF
                corrigido e perspectiva retificada pelo pipeline.

        Returns:
            Deteccoes em coordenadas de pixel da imagem recebida, em qualquer
            ordem -- o dominio ordena.

        Raises:
            DetectorError: se a inferencia falhar.
        """
        ...

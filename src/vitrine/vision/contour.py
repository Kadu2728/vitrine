"""Detector por contorno: encontra retangulos de alto contraste, sem rede neural.

Isto **nao e um detector de varejo** e o nome do arquivo nao tenta disfarcar.
Ele encontra blocos retangulares que se destacam do fundo. Numa foto de gondola
real, com iluminacao de supermercado, embalagem brilhante e produto encostado em
produto, o resultado e ruim -- e essa limitacao esta escrita aqui, no ``--help``
e no README.

Existe por dois motivos concretos:

1. **Testes de ponta a ponta reais.** Com ele a suite exercita o caminho
   completo -- carregar imagem, retificar, detectar, calcular, desenhar -- sobre
   uma imagem sintetica gerada em runtime cujo resultado e conhecido
   matematicamente. Sem mock, sem peso, em milissegundos.
2. **Demonstracao sem dependencia pesada.** Permite ver a ferramenta funcionando
   ponta a ponta sem instalar torch, o que importa para quem so quer avaliar o
   projeto.

Metodo: escala de cinza, limiar de Otsu, contornos externos, retangulo
envolvente, filtro por area e por proporcao. Deterministico -- nenhuma etapa
usa aleatoriedade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from vitrine.domain.models import BoundingBox, Detection, DetectorInfo
from vitrine.errors import DetectorError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

DEFAULT_MIN_AREA_RATIO = 0.0005
"""Area minima de um produto, em fracao da area da imagem."""

DEFAULT_MAX_AREA_RATIO = 0.5
"""Area maxima: acima disto o contorno e o fundo ou a propria gondola."""

DEFAULT_MAX_ASPECT = 12.0
"""Proporcao maxima entre lados. Acima disto e prateleira, moldura ou risco."""


class ContourDetector:
    """Encontra retangulos de alto contraste por limiar e contorno.

    Args:
        min_area_ratio: area minima aceitavel, em fracao da area da imagem.
        max_area_ratio: area maxima aceitavel, em fracao da area da imagem.
        max_aspect: razao maxima entre o lado maior e o menor.
        invert: se ``True``, procura formas escuras em fundo claro.
    """

    def __init__(
        self,
        *,
        min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
        max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
        max_aspect: float = DEFAULT_MAX_ASPECT,
        invert: bool = False,
    ) -> None:
        if not 0.0 < min_area_ratio < max_area_ratio <= 1.0:
            raise DetectorError(
                f"Faixa de area invalida: min={min_area_ratio} max={max_area_ratio}.",
                "Use 0 < min < max <= 1, por exemplo --min-area 0.0005 e --max-area 0.5.",
            )
        if max_aspect <= 1.0:
            raise DetectorError(
                f"max_aspect precisa ser maior que 1; recebido {max_aspect}.",
                "Um valor tipico e 12, que aceita produto alongado mas recusa prateleira.",
            )
        self._min_area_ratio = min_area_ratio
        self._max_area_ratio = max_area_ratio
        self._max_aspect = max_aspect
        self._invert = invert

    @property
    def info(self) -> DetectorInfo:
        """Identificacao deste detector; sem peso, porque nao ha modelo."""
        return DetectorInfo(
            name="contour",
            version=f"opencv-{cv2.__version__}",
            weights=None,
            weights_sha256=None,
            confidence_threshold=0.0,
            iou_threshold=0.0,
        )

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        """Encontra retangulos candidatos na imagem.

        Args:
            image: pixels BGR uint8.

        Returns:
            Deteccoes com confianca fixa em 1.0 -- este metodo nao produz
            pontuacao, e inventar uma seria pior que declarar a ausencia.

        Raises:
            DetectorError: se a imagem nao estiver no formato esperado.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise DetectorError(
                f"Esperava imagem BGR com 3 canais; recebido shape {image.shape}.",
                "Carregue a imagem com vitrine.vision.image.load_image.",
            )

        altura, largura = image.shape[:2]
        area_imagem = float(altura * largura)
        cinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        modo = cv2.THRESH_BINARY_INV if self._invert else cv2.THRESH_BINARY
        _, binaria = cv2.threshold(cinza, 0, 255, modo + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        deteccoes: list[Detection] = []
        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)
            if w < 1 or h < 1:
                continue
            fracao = (w * h) / area_imagem
            if not self._min_area_ratio <= fracao <= self._max_area_ratio:
                continue
            if max(w, h) / min(w, h) > self._max_aspect:
                continue
            deteccoes.append(
                Detection(
                    box=BoundingBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h))
                )
            )

        return tuple(sorted(deteccoes, key=lambda d: d.sort_key))

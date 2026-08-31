"""De uma imagem a um relatorio: a costura entre visao e dominio.

O pipeline e fino de proposito. Ele nao calcula nada -- carrega, retifica,
delega a deteccao ao protocolo e entrega as caixas ao dominio. Toda a
matematica mora em ``vitrine.domain``, que nao sabe que imagens existem.

Ordem das etapas, e por que ela e essa:

1. **Carregar** com correcao de EXIF. Antes de tudo, porque orientacao errada
   invalida qualquer etapa seguinte sem dar erro.
2. **Retificar** a perspectiva, se houver quatro pontos. Antes da deteccao,
   nunca depois: transformar caixas por homografia produziria quadrilateros, e
   o dominio so entende retangulo alinhado ao eixo.
3. **Detectar** via protocolo. O pipeline nao sabe se do outro lado ha uma rede
   neural ou uma lista fixa.
4. **Analisar** no dominio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vitrine.domain.dedup import DEFAULT_DEDUP_IOU
from vitrine.domain.gaps import DEFAULT_GAP_MIN_WIDTH_RATIO
from vitrine.domain.models import ImageMeta
from vitrine.domain.share import analyze_detections
from vitrine.domain.shelves import DEFAULT_MAX_SPREAD_RATIO, DEFAULT_SHELF_GAP_RATIO
from vitrine.vision.image import DEFAULT_MAX_SIZE, load_image
from vitrine.vision.perspective import rectify

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from vitrine.domain.models import Detection, RegionSet, ShareReport
    from vitrine.vision.perspective import Quad
    from vitrine.vision.protocols import Detector


@dataclass(frozen=True)
class AnalysisResult:
    """Relatorio mais os pixels analisados.

    O relatorio sozinho nao basta para desenhar, e enfiar um ``ndarray`` dentro
    de um modelo Pydantic serializavel seria pior: ``ShareReport`` precisa
    continuar sendo JSON puro. Por isso os dois viajam juntos aqui, e separados
    na saida.
    """

    report: ShareReport
    pixels: NDArray[np.uint8]
    """Imagem efetivamente analisada: com EXIF corrigido, reduzida e retificada."""

    detections: tuple[Detection, ...]
    """Deteccoes cruas, antes da deduplicacao do dominio."""


def analyze_image(
    path: Path,
    detector: Detector,
    *,
    perspective: Quad | None = None,
    max_size: int | None = DEFAULT_MAX_SIZE,
    regions: RegionSet | None = None,
    extent: tuple[float, float] | None = None,
    shelf_gap_ratio: float = DEFAULT_SHELF_GAP_RATIO,
    max_shelf_spread_ratio: float = DEFAULT_MAX_SPREAD_RATIO,
    gap_min_width_ratio: float = DEFAULT_GAP_MIN_WIDTH_RATIO,
    dedup_iou: float = DEFAULT_DEDUP_IOU,
    source: str | None = None,
) -> AnalysisResult:
    """Analisa uma foto de gondola e devolve o relatorio.

    Args:
        path: caminho da imagem.
        detector: qualquer implementacao do protocolo ``Detector``.
        perspective: quatro cantos da area util, em pixels da imagem ja
            carregada; ``None`` desliga a retificacao.
        max_size: maior lado apos a reducao; ``None`` mantem o tamanho.
        regions: particao da gondola para o share; ``None`` usa regiao unica.
        extent: limites horizontais explicitos; ``None`` usa o envelope.
        shelf_gap_ratio: ver ``domain.shelves``.
        max_shelf_spread_ratio: ver ``domain.shelves``.
        gap_min_width_ratio: ver ``domain.gaps``.
        dedup_iou: ver ``domain.dedup``.
        source: identificador da origem; o padrao e o nome do arquivo.

    Returns:
        O relatorio, os pixels analisados e as deteccoes cruas.

    Raises:
        ImageLoadError: se a imagem nao puder ser lida.
        PerspectiveError: se os quatro pontos forem invalidos.
        DetectorError: se a inferencia falhar.
    """
    imagem = load_image(path, max_size=max_size)
    pixels = imagem.pixels

    if perspective is not None:
        pixels = rectify(pixels, perspective)

    deteccoes = detector.detect(pixels)

    altura, largura = pixels.shape[:2]
    meta = ImageMeta(
        name=imagem.name,
        width=int(largura),
        height=int(altura),
        exif_rotated=imagem.exif_rotated,
        downscale=imagem.downscale,
        rectified=perspective is not None,
    )

    report = analyze_detections(
        deteccoes,
        regions=regions,
        extent=extent,
        shelf_gap_ratio=shelf_gap_ratio,
        max_shelf_spread_ratio=max_shelf_spread_ratio,
        gap_min_width_ratio=gap_min_width_ratio,
        dedup_iou=dedup_iou,
        source=source if source is not None else imagem.name,
        image=meta,
        detector=detector.info,
    )

    return AnalysisResult(report=report, pixels=pixels, detections=deteccoes)

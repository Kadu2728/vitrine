"""Vitrine -- auditoria de execucao em ponto de venda por visao computacional.

A biblioteca vem primeiro; a CLI e um consumidor fino dela. Na Fase 1 apenas o
dominio existe, e o ponto de entrada e ``analyze_detections``::

    from vitrine import BoundingBox, Detection, analyze_detections

    detections = [Detection(box=BoundingBox(x1=0, y1=0, x2=80, y2=100))]
    report = analyze_detections(detections)
    print(report.model_dump_json(indent=2))

A partir da Fase 2 existe tambem o caminho completo, imagem para relatorio::

    from pathlib import Path
    from vitrine import analyze_image
    from vitrine.vision.contour import ContourDetector

    resultado = analyze_image(Path("foto.jpg"), ContourDetector())
    print(resultado.report.model_dump_json(indent=2))

O detector e injetado, nunca importado pela logica: ``analyze_image`` aceita
qualquer coisa que satisfaca o protocolo ``Detector``.
"""

from vitrine.domain.dedup import deduplicate
from vitrine.domain.gaps import find_gaps
from vitrine.domain.models import (
    SCHEMA_VERSION,
    AnalysisParams,
    BoundingBox,
    Detection,
    DetectorInfo,
    Gap,
    ImageMeta,
    Region,
    RegionSet,
    RegionShare,
    ShareReport,
    Shelf,
    ShelfExtent,
    ShelfReport,
)
from vitrine.domain.share import analyze_detections
from vitrine.domain.shelves import group_into_shelves
from vitrine.errors import (
    DetectorError,
    ImageLoadError,
    PerspectiveError,
    UsageError,
    VitrineError,
)
from vitrine.pipeline import AnalysisResult, analyze_image
from vitrine.render.annotate import annotate
from vitrine.vision.contour import ContourDetector
from vitrine.vision.fake import FakeDetector
from vitrine.vision.protocols import Detector

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "AnalysisParams",
    "AnalysisResult",
    "BoundingBox",
    "ContourDetector",
    "Detection",
    "Detector",
    "DetectorError",
    "DetectorInfo",
    "FakeDetector",
    "Gap",
    "ImageLoadError",
    "ImageMeta",
    "PerspectiveError",
    "Region",
    "RegionSet",
    "RegionShare",
    "ShareReport",
    "Shelf",
    "ShelfExtent",
    "ShelfReport",
    "UsageError",
    "VitrineError",
    "__version__",
    "analyze_detections",
    "analyze_image",
    "annotate",
    "deduplicate",
    "find_gaps",
    "group_into_shelves",
]

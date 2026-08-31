"""Vitrine -- auditoria de execucao em ponto de venda por visao computacional.

A biblioteca vem primeiro; a CLI e um consumidor fino dela. Na Fase 1 apenas o
dominio existe, e o ponto de entrada e ``analyze_detections``::

    from vitrine import BoundingBox, Detection, analyze_detections

    detections = [Detection(box=BoundingBox(x1=0, y1=0, x2=80, y2=100))]
    report = analyze_detections(detections)
    print(report.model_dump_json(indent=2))

``analyze_image`` -- imagem para relatorio -- entra na Fase 2, quando o
detector real for plugado atras do protocolo. Ate la ele nao existe, e nao ha
stub fingindo que existe.
"""

from vitrine.domain.dedup import deduplicate
from vitrine.domain.gaps import find_gaps
from vitrine.domain.models import (
    SCHEMA_VERSION,
    AnalysisParams,
    BoundingBox,
    Detection,
    Gap,
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

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "AnalysisParams",
    "BoundingBox",
    "Detection",
    "Gap",
    "Region",
    "RegionSet",
    "RegionShare",
    "ShareReport",
    "Shelf",
    "ShelfExtent",
    "ShelfReport",
    "__version__",
    "analyze_detections",
    "deduplicate",
    "find_gaps",
    "group_into_shelves",
]

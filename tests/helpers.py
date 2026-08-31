"""Construtores curtos usados pelos testes.

Ficam num modulo proprio, e nao no ``conftest``, para que os testes os importem
explicitamente em vez de depender do mecanismo de descoberta do pytest.
"""

from __future__ import annotations

from vitrine import BoundingBox, Detection


def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    """Atalho para montar caixas nos testes."""
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def detection(x1: float, y1: float, x2: float, y2: float, confidence: float = 1.0) -> Detection:
    """Atalho para montar deteccoes nos testes."""
    return Detection(box=box(x1, y1, x2, y2), confidence=confidence)

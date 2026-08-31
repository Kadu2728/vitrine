"""Estrategias do hypothesis para o dominio.

Duas escolhas deliberadas, ambas para tornar as invariantes *afiadas* em vez de
aproximadas:

**Coordenadas inteiras.** Geradas como inteiros e convertidas para float. Somas
e diferencas ficam exatas ate 2^53, entao a invariante de translacao pode ser
verificada com igualdade estrita em vez de tolerancia. Com coordenadas
arbitrarias, o teste passaria a medir erro de arredondamento em vez de logica.

**Fatores de escala potencia de dois.** ``x * 8`` e exato em binario; ``x *
1.1`` nao e. Com potencias de dois, escalar a entrada produz *exatamente* os
mesmos shares, e a invariante vira uma igualdade em vez de um "quase". Para
fatores arbitrarios a invariancia continua valendo, so que a menos de
arredondamento -- e esta limitacao esta declarada aqui em vez de escondida numa
tolerancia generosa.
"""

from __future__ import annotations

from hypothesis import strategies as st

from vitrine import BoundingBox, Detection, RegionSet

POWERS_OF_TWO = (0.125, 0.25, 0.5, 2.0, 4.0, 8.0)
"""Fatores de escala exatamente representaveis em ponto flutuante."""

CONFIDENCES = (0.25, 0.5, 0.75, 1.0)
"""Confiancas exatas, para que a ordenacao da deduplicacao nao dependa de
arredondamento."""


@st.composite
def boxes(draw: st.DrawFn) -> BoundingBox:
    """Caixa com coordenadas inteiras e dimensoes positivas."""
    x1 = draw(st.integers(min_value=0, max_value=1000))
    y1 = draw(st.integers(min_value=0, max_value=1000))
    width = draw(st.integers(min_value=1, max_value=200))
    height = draw(st.integers(min_value=1, max_value=200))
    return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x1 + width), y2=float(y1 + height))


@st.composite
def detections(draw: st.DrawFn) -> Detection:
    """Deteccao com caixa inteira e confianca exata."""
    return Detection(box=draw(boxes()), confidence=draw(st.sampled_from(CONFIDENCES)))


def detection_lists(min_size: int = 1, max_size: int = 12) -> st.SearchStrategy[list[Detection]]:
    """Lista de deteccoes de tamanho controlado."""
    return st.lists(detections(), min_size=min_size, max_size=max_size)


@st.composite
def region_sets(draw: st.DrawFn) -> RegionSet:
    """Particao de [0, 1] em cortes de oitavos, exatos em binario."""
    inner = draw(
        st.lists(
            st.sampled_from([1, 2, 3, 4, 5, 6, 7]),
            min_size=0,
            max_size=3,
            unique=True,
        )
    )
    cuts = (0.0, *(value / 8 for value in sorted(inner)), 1.0)
    return RegionSet.from_cuts(cuts)

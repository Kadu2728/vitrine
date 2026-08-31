"""Testes da deteccao de vazios."""

from __future__ import annotations

import pytest

from helpers import detection
from vitrine import Shelf, ShelfExtent, find_gaps


def envelope(shelf: Shelf) -> ShelfExtent:
    return ShelfExtent(kind="envelope", x_min=shelf.x_min, x_max=shelf.x_max)


class TestVaziosInternos:
    def test_prateleira_cheia_nao_tem_vazio(self) -> None:
        shelf = Shelf(
            index=0,
            detections=(detection(0, 0, 50, 100), detection(50, 0, 100, 100)),
        )
        assert find_gaps(shelf, envelope(shelf)) == ()

    def test_vazio_pequeno_e_ignorado(self) -> None:
        # Largura mediana 50; o vao de 10 nao cabe um produto.
        shelf = Shelf(
            index=0,
            detections=(detection(0, 0, 50, 100), detection(60, 0, 110, 100)),
        )
        assert find_gaps(shelf, envelope(shelf)) == ()

    def test_vazio_de_um_produto_e_reportado(self) -> None:
        # Largura mediana 50; o vao de 50 cabe exatamente um produto.
        shelf = Shelf(
            index=0,
            detections=(detection(0, 0, 50, 100), detection(100, 0, 150, 100)),
        )
        gaps = find_gaps(shelf, envelope(shelf))
        assert len(gaps) == 1
        assert (gaps[0].x_start, gaps[0].x_end) == (50.0, 100.0)
        assert gaps[0].width == 50.0
        assert gaps[0].width_ratio == pytest.approx(1.0)

    def test_coordenadas_verticais_vem_da_prateleira(self) -> None:
        shelf = Shelf(
            index=0,
            detections=(detection(0, 10, 50, 100), detection(200, 0, 250, 110)),
        )
        gap = find_gaps(shelf, envelope(shelf))[0]
        assert gap.y_top == 0.0
        assert gap.y_bottom == 110.0

    def test_varios_vazios_saem_ordenados_e_disjuntos(self) -> None:
        shelf = Shelf(
            index=0,
            detections=(
                detection(0, 0, 50, 100),
                detection(150, 0, 200, 100),
                detection(300, 0, 350, 100),
            ),
        )
        gaps = find_gaps(shelf, envelope(shelf))
        assert [(g.x_start, g.x_end) for g in gaps] == [(50.0, 150.0), (200.0, 300.0)]

    def test_caixas_sobrepostas_nao_geram_vazio_fantasma(self) -> None:
        shelf = Shelf(
            index=0,
            detections=(detection(0, 0, 100, 100), detection(40, 0, 140, 100)),
        )
        assert find_gaps(shelf, envelope(shelf)) == ()


class TestLimiarRelativo:
    def test_o_limiar_acompanha_o_tamanho_do_produto(self) -> None:
        # O mesmo vao absoluto de 60: e vazio numa prateleira de latas
        # (mediana 20) e nao e numa de caixas grandes (mediana 100).
        latas = Shelf(
            index=0,
            detections=(detection(0, 0, 20, 50), detection(80, 0, 100, 50)),
        )
        caixas = Shelf(
            index=0,
            detections=(detection(0, 0, 100, 200), detection(160, 0, 260, 200)),
        )
        assert len(find_gaps(latas, envelope(latas))) == 1
        assert len(find_gaps(caixas, envelope(caixas))) == 0

    def test_min_width_ratio_configuravel(self) -> None:
        shelf = Shelf(
            index=0,
            detections=(detection(0, 0, 50, 100), detection(80, 0, 130, 100)),
        )
        assert find_gaps(shelf, envelope(shelf)) == ()
        assert len(find_gaps(shelf, envelope(shelf), min_width_ratio=0.5)) == 1

    def test_min_width_ratio_invalido(self) -> None:
        shelf = Shelf(index=0, detections=(detection(0, 0, 50, 100),))
        with pytest.raises(ValueError, match="min_width_ratio"):
            find_gaps(shelf, envelope(shelf), min_width_ratio=0.0)


class TestExtentMudaOQueEVisivel:
    def test_envelope_esconde_vazio_nas_pontas(self) -> None:
        shelf = Shelf(
            index=0,
            detections=(detection(400, 0, 450, 100), detection(460, 0, 510, 100)),
        )
        assert find_gaps(shelf, envelope(shelf)) == ()

    def test_extent_explicito_revela_vazio_nas_pontas(self) -> None:
        # A mesma prateleira, agora sabendo que a gondola vai de 0 a 1000.
        shelf = Shelf(
            index=0,
            detections=(detection(400, 0, 450, 100), detection(460, 0, 510, 100)),
        )
        extent = ShelfExtent(kind="explicit", x_min=0.0, x_max=1000.0)
        gaps = find_gaps(shelf, extent)
        assert [(g.x_start, g.x_end) for g in gaps] == [(0.0, 400.0), (510.0, 1000.0)]

"""Testes do pipeline e da renderizacao.

Estes sao os testes de ponta a ponta: arquivo em disco, carregamento, deteccao
real por contorno, calculo no dominio, imagem anotada. Sem mock e sem modelo --
o resultado esperado sai da geometria da imagem sintetica.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from helpers import synthetic_shelf, write_image
from vitrine import ContourDetector, FakeDetector, RegionSet, analyze_image, annotate
from vitrine.errors import ImageLoadError, PerspectiveError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def gondola_sintetica(tmp_path: Path) -> Path:
    """Duas prateleiras de tres produtos, gravadas em disco."""
    pixels, _ = synthetic_shelf(rows=2, columns=3)
    return write_image(tmp_path / "gondola.png", pixels)


class TestAnalyzeImage:
    def test_ponta_a_ponta_com_deteccao_real(self, gondola_sintetica: Path) -> None:
        resultado = analyze_image(gondola_sintetica, ContourDetector())
        report = resultado.report
        assert report.status == "ok"
        assert report.total_detections == 6
        assert report.shelf_count == 2
        assert all(shelf.detection_count == 3 for shelf in report.shelves)

    def test_geometria_bate_com_a_imagem_desenhada(self, gondola_sintetica: Path) -> None:
        # Produtos de 40 px com vaos de 20 px: envelope 40*3 + 20*2 = 160,
        # ocupado 120. Ocupacao = 120/160 = 0.75.
        shelf = analyze_image(gondola_sintetica, ContourDetector()).report.shelves[0]
        assert shelf.extent.width == 160.0
        assert shelf.occupied_length == 120.0
        assert shelf.occupancy == pytest.approx(0.75)
        assert shelf.median_product_width == 40.0

    def test_vaos_de_meio_produto_nao_viram_ruptura(self, gondola_sintetica: Path) -> None:
        # Vao de 20 px contra largura mediana de 40: metade de um produto.
        report = analyze_image(gondola_sintetica, ContourDetector()).report
        assert all(shelf.gaps == () for shelf in report.shelves)

    def test_ruptura_de_verdade_e_detectada(self, tmp_path: Path) -> None:
        # Produtos de 40 px com vao de 120: cabem tres produtos no buraco.
        pixels, _ = synthetic_shelf(rows=1, columns=2, box_width=40, gap_x=120)
        caminho = write_image(tmp_path / "ruptura.png", pixels)
        report = analyze_image(caminho, ContourDetector()).report
        gaps = report.shelves[0].gaps
        assert len(gaps) == 1
        assert gaps[0].width == 120.0
        assert gaps[0].width_ratio == pytest.approx(3.0)

    def test_procedencia_vai_para_o_relatorio(self, gondola_sintetica: Path) -> None:
        report = analyze_image(gondola_sintetica, ContourDetector()).report
        assert report.image is not None
        assert report.image.name == "gondola.png"
        assert report.image.rectified is False
        assert report.image.exif_rotated is False
        assert report.detector is not None
        assert report.detector.name == "contour"
        assert report.source == "gondola.png"

    def test_source_pode_ser_sobrescrito(self, gondola_sintetica: Path) -> None:
        resultado = analyze_image(gondola_sintetica, ContourDetector(), source="LOJA_12")
        assert resultado.report.source == "LOJA_12"

    def test_regioes_atravessam_ate_o_relatorio(self, gondola_sintetica: Path) -> None:
        regions = RegionSet.from_cuts((0.0, 0.5, 1.0), ("esquerda", "direita"))
        report = analyze_image(gondola_sintetica, ContourDetector(), regions=regions).report
        assert [r.region for r in report.regions] == ["esquerda", "direita"]
        assert sum(r.count_share for r in report.regions) == pytest.approx(1.0)

    def test_retificacao_marca_o_relatorio(self, gondola_sintetica: Path) -> None:
        cantos = ((10.0, 10.0), (170.0, 10.0), (170.0, 160.0), (10.0, 160.0))
        resultado = analyze_image(gondola_sintetica, ContourDetector(), perspective=cantos)
        assert resultado.report.image is not None
        assert resultado.report.image.rectified is True
        assert resultado.pixels.shape[:2] == (150, 160)

    def test_detector_injetado_dispensa_a_imagem(self, gondola_sintetica: Path) -> None:
        """A prova de que a R2 funciona: o resultado vem do detector, nao dos pixels."""
        detector = FakeDetector.from_grid(rows=3, columns=2)
        report = analyze_image(gondola_sintetica, detector).report
        assert report.shelf_count == 3
        assert report.total_detections == 6
        assert report.detector is not None
        assert report.detector.name == "fake-grid"

    def test_imagem_sem_produto(self, tmp_path: Path) -> None:
        vazia = np.zeros((120, 120, 3), dtype=np.uint8)
        caminho = write_image(tmp_path / "vazia.png", vazia)
        report = analyze_image(caminho, ContourDetector()).report
        assert report.status == "no_detections"
        assert report.shelves == ()

    def test_erro_de_imagem_sobe_com_dica(self, tmp_path: Path) -> None:
        with pytest.raises(ImageLoadError) as exc:
            analyze_image(tmp_path / "fantasma.jpg", ContourDetector())
        assert exc.value.hint

    def test_erro_de_perspectiva_sobe_com_dica(self, gondola_sintetica: Path) -> None:
        cantos = ((0.0, 0.0), (0.0, 0.0), (10.0, 10.0), (0.0, 10.0))
        with pytest.raises(PerspectiveError) as exc:
            analyze_image(gondola_sintetica, ContourDetector(), perspective=cantos)
        assert exc.value.hint

    def test_e_deterministico(self, gondola_sintetica: Path) -> None:
        primeiro = analyze_image(gondola_sintetica, ContourDetector()).report
        segundo = analyze_image(gondola_sintetica, ContourDetector()).report
        assert primeiro.model_dump_json() == segundo.model_dump_json()


class TestAnnotate:
    def test_nao_modifica_a_imagem_original(self, gondola_sintetica: Path) -> None:
        resultado = analyze_image(gondola_sintetica, ContourDetector())
        antes = resultado.pixels.copy()
        annotate(resultado.pixels, resultado.report)
        assert np.array_equal(resultado.pixels, antes)

    def test_preserva_as_dimensoes(self, gondola_sintetica: Path) -> None:
        resultado = analyze_image(gondola_sintetica, ContourDetector())
        anotada = annotate(resultado.pixels, resultado.report)
        assert anotada.shape == resultado.pixels.shape

    def test_desenha_alguma_coisa(self, gondola_sintetica: Path) -> None:
        resultado = analyze_image(gondola_sintetica, ContourDetector())
        anotada = annotate(resultado.pixels, resultado.report)
        assert not np.array_equal(anotada, resultado.pixels)

    def test_pinta_o_vazio_de_vermelho(self, tmp_path: Path) -> None:
        pixels, _ = synthetic_shelf(rows=1, columns=2, box_width=40, gap_x=120)
        caminho = write_image(tmp_path / "ruptura.png", pixels)
        resultado = analyze_image(caminho, ContourDetector())
        anotada = annotate(resultado.pixels, resultado.report)

        gap = resultado.report.shelves[0].gaps[0]
        centro_x = round((gap.x_start + gap.x_end) / 2)
        centro_y = round((gap.y_top + gap.y_bottom) / 2)
        azul, verde, vermelho = anotada[centro_y, centro_x]
        # O fundo era preto; a marcacao do vazio e vermelha em BGR.
        assert int(vermelho) > int(azul)
        assert int(vermelho) > int(verde)

    def test_prateleiras_recebem_cores_diferentes(self, gondola_sintetica: Path) -> None:
        resultado = analyze_image(gondola_sintetica, ContourDetector())
        anotada = annotate(resultado.pixels, resultado.report)
        cores = set()
        for shelf in resultado.report.shelves:
            caixa = shelf.boxes[0]
            cores.add(tuple(int(c) for c in anotada[round(caixa.y1), round(caixa.x1) + 2]))
        assert len(cores) == len(resultado.report.shelves)

    def test_imagem_sem_deteccao_nao_quebra(self, tmp_path: Path) -> None:
        vazia = np.zeros((120, 120, 3), dtype=np.uint8)
        caminho = write_image(tmp_path / "vazia.png", vazia)
        resultado = analyze_image(caminho, ContourDetector())
        assert annotate(resultado.pixels, resultado.report).shape == resultado.pixels.shape

    def test_e_deterministico(self, gondola_sintetica: Path) -> None:
        resultado = analyze_image(gondola_sintetica, ContourDetector())
        primeira = annotate(resultado.pixels, resultado.report)
        segunda = annotate(resultado.pixels, resultado.report)
        assert np.array_equal(primeira, segunda)

    def test_desenha_a_regua_quando_ha_regioes(self, gondola_sintetica: Path) -> None:
        """A divisao usada no share precisa ficar visivel, nao implicita."""
        regions = RegionSet.from_cuts((0.0, 0.5, 1.0), ("minha", "concorrencia"))
        resultado = analyze_image(gondola_sintetica, ContourDetector(), regions=regions)
        com_regua = annotate(resultado.pixels, resultado.report, show_regions=True)
        sem_regua = annotate(resultado.pixels, resultado.report, show_regions=False)
        assert not np.array_equal(com_regua, sem_regua)

    def test_rodape_pode_ser_desligado(self, gondola_sintetica: Path) -> None:
        resultado = analyze_image(gondola_sintetica, ContourDetector())
        com = annotate(resultado.pixels, resultado.report, show_footer=True)
        sem = annotate(resultado.pixels, resultado.report, show_footer=False)
        assert not np.array_equal(com, sem)
        # O rodape ocupa a faixa inferior; o topo precisa ficar identico.
        assert np.array_equal(com[:50], sem[:50])

    def test_tres_regioes_com_nomes_longos_nao_estouram(self, gondola_sintetica: Path) -> None:
        regions = RegionSet.from_cuts(
            (0.0, 0.3, 0.7, 1.0), ("marca_muito_longa", "concorrente_a", "concorrente_b")
        )
        resultado = analyze_image(gondola_sintetica, ContourDetector(), regions=regions)
        anotada = annotate(resultado.pixels, resultado.report)
        assert anotada.shape == resultado.pixels.shape

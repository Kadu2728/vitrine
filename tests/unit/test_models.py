"""Testes dos contratos: validacao, geometria derivada e particao de regioes."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from helpers import box, detection
from vitrine import BoundingBox, Detection, Region, RegionSet, Shelf, ShelfExtent


class TestBoundingBox:
    def test_geometria_derivada(self) -> None:
        b = box(10, 20, 50, 120)
        assert b.width == 40.0
        assert b.height == 100.0
        assert b.center_x == 30.0
        assert b.center_y == 70.0
        assert b.area == 4000.0
        assert b.x_interval == (10.0, 50.0)
        assert b.y_interval == (20.0, 120.0)

    def test_largura_zero_e_rejeitada(self) -> None:
        with pytest.raises(ValidationError, match="largura precisa ser positiva"):
            BoundingBox(x1=10, y1=0, x2=10, y2=100)

    def test_altura_invertida_e_rejeitada(self) -> None:
        with pytest.raises(ValidationError, match="altura precisa ser positiva"):
            BoundingBox(x1=0, y1=100, x2=10, y2=50)

    def test_nao_finito_e_rejeitado(self) -> None:
        with pytest.raises(ValidationError, match="numero finito"):
            BoundingBox(x1=0, y1=0, x2=math.inf, y2=100)

    def test_imutavel(self) -> None:
        with pytest.raises(ValidationError):
            box(0, 0, 10, 10).x1 = 5.0

    def test_campo_extra_e_rejeitado(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(x1=0, y1=0, x2=1, y2=1, sku="123")  # type: ignore[call-arg]

    def test_iou_identicas(self) -> None:
        assert box(0, 0, 10, 10).iou(box(0, 0, 10, 10)) == 1.0

    def test_iou_disjuntas(self) -> None:
        assert box(0, 0, 10, 10).iou(box(50, 50, 60, 60)) == 0.0

    def test_iou_meia_sobreposicao(self) -> None:
        # Interseccao 50, uniao 100 + 100 - 50 = 150.
        assert box(0, 0, 10, 10).iou(box(5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_escala_e_translacao(self) -> None:
        b = box(10, 20, 30, 40)
        assert b.scaled(2.0) == box(20, 40, 60, 80)
        assert b.translated(5, -5) == box(15, 15, 35, 35)

    def test_escala_nao_positiva_levanta(self) -> None:
        with pytest.raises(ValueError, match="fator de escala"):
            box(0, 0, 10, 10).scaled(0.0)


class TestDetection:
    def test_padroes(self) -> None:
        d = Detection(box=box(0, 0, 10, 10))
        assert d.confidence == 1.0
        assert d.label == "object"

    def test_confianca_fora_da_faixa(self) -> None:
        with pytest.raises(ValidationError):
            Detection(box=box(0, 0, 10, 10), confidence=1.5)

    def test_sort_key_e_ordem_total(self) -> None:
        a = detection(0, 0, 10, 10)
        b = detection(0, 0, 10, 20)
        c = detection(5, 0, 10, 10)
        assert sorted([c, b, a], key=lambda d: d.sort_key) == [a, b, c]


class TestShelf:
    def test_metricas_derivadas(self) -> None:
        shelf = Shelf(
            index=0,
            detections=(
                detection(0, 0, 80, 100),
                detection(90, 10, 170, 110),
                detection(200, 0, 230, 100),
            ),
        )
        assert shelf.count == 3
        assert shelf.x_min == 0.0
        assert shelf.x_max == 230.0
        assert shelf.y_top == 0.0
        assert shelf.y_bottom == 110.0
        assert shelf.median_width == 80.0
        assert shelf.median_height == 100.0
        # 80 + 80 + 30, sem sobreposicao.
        assert shelf.occupied_length == 190.0
        # Centros verticais 50, 60, 50 -> dispersao 10 sobre altura mediana 100.
        assert shelf.spread_ratio == pytest.approx(0.1)

    def test_prateleira_vazia_e_rejeitada(self) -> None:
        with pytest.raises(ValidationError):
            Shelf(index=0, detections=())

    def test_ocupacao_conta_sobreposicao_uma_vez(self) -> None:
        shelf = Shelf(index=0, detections=(detection(0, 0, 100, 50), detection(50, 0, 150, 50)))
        assert shelf.occupied_length == 150.0


class TestRegion:
    def test_regiao_invertida_e_rejeitada(self) -> None:
        with pytest.raises(ValidationError, match="end precisa ser maior que start"):
            Region(name="vazia", start=0.6, end=0.2)

    def test_regiao_degenerada_e_rejeitada(self) -> None:
        with pytest.raises(ValidationError, match="end precisa ser maior que start"):
            Region(name="ponto", start=0.5, end=0.5)


class TestRegionSet:
    def test_whole_e_regiao_unica(self) -> None:
        region_set = RegionSet.whole()
        assert len(region_set.regions) == 1
        assert region_set.regions[0].name == "total"

    def test_from_cuts_nomeia_por_padrao(self) -> None:
        region_set = RegionSet.from_cuts((0.0, 0.4, 1.0))
        assert [r.name for r in region_set.regions] == ["r1", "r2"]
        assert region_set.regions[0].span == pytest.approx(0.4)

    def test_from_cuts_com_nomes(self) -> None:
        region_set = RegionSet.from_cuts((0.0, 0.5, 1.0), ("esquerda", "direita"))
        assert [r.name for r in region_set.regions] == ["esquerda", "direita"]

    def test_buraco_na_particao_e_rejeitado(self) -> None:
        with pytest.raises(ValidationError, match="contiguas"):
            RegionSet(
                regions=(
                    Region(name="a", start=0.0, end=0.3),
                    Region(name="b", start=0.5, end=1.0),
                )
            )

    def test_particao_incompleta_e_rejeitada(self) -> None:
        with pytest.raises(ValidationError, match=r"cobrir ate 1\.0"):
            RegionSet(regions=(Region(name="a", start=0.0, end=0.8),))

    def test_sobreposicao_e_rejeitada(self) -> None:
        with pytest.raises(ValidationError, match="contiguas"):
            RegionSet(
                regions=(
                    Region(name="a", start=0.0, end=0.6),
                    Region(name="b", start=0.4, end=1.0),
                )
            )

    def test_nomes_duplicados_sao_rejeitados(self) -> None:
        with pytest.raises(ValidationError, match="unicos"):
            RegionSet(
                regions=(
                    Region(name="a", start=0.0, end=0.5),
                    Region(name="a", start=0.5, end=1.0),
                )
            )

    def test_poucos_cortes(self) -> None:
        with pytest.raises(ValueError, match="dois cortes"):
            RegionSet.from_cuts((0.0,))

    def test_nomes_em_quantidade_errada(self) -> None:
        with pytest.raises(ValueError, match="exigem"):
            RegionSet.from_cuts((0.0, 0.5, 1.0), ("so_um",))

    def test_locate_e_funcao_total(self) -> None:
        region_set = RegionSet.from_cuts((0.0, 0.5, 1.0), ("esquerda", "direita"))
        assert region_set.locate(0.0).name == "esquerda"
        assert region_set.locate(0.4999).name == "esquerda"
        # A fronteira pertence a regiao da direita.
        assert region_set.locate(0.5).name == "direita"
        assert region_set.locate(1.0).name == "direita"
        # Fora da faixa, preso na borda.
        assert region_set.locate(-3.0).name == "esquerda"
        assert region_set.locate(9.0).name == "direita"


class TestShelfExtent:
    def test_janela_de_regiao(self) -> None:
        extent = ShelfExtent(kind="envelope", x_min=100.0, x_max=300.0)
        assert extent.width == 200.0
        assert extent.window(Region(name="meio", start=0.25, end=0.75)) == (150.0, 250.0)

    def test_extensao_degenerada_e_rejeitada(self) -> None:
        with pytest.raises(ValidationError, match="extensao vazia"):
            ShelfExtent(kind="explicit", x_min=100.0, x_max=100.0)

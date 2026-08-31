"""Invariantes do dominio, verificadas com hypothesis.

Estas propriedades sao o contrato matematico do sistema. Um exemplo escrito a
mao prova que uma conta esta certa; uma invariante prova que a conta continua
certa para entradas que ninguem pensou em escrever.
"""

from __future__ import annotations

import math
from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st
from strategies import POWERS_OF_TWO, detection_lists, region_sets

from vitrine import Detection, RegionSet, ShareReport, analyze_detections, deduplicate
from vitrine.domain.shelves import group_into_shelves

TOLERANCE = 1e-9


def share_fingerprint(report: ShareReport) -> list[tuple[str, int, float, float, float]]:
    """Extrai apenas as grandezas adimensionais do relatorio.

    Comprimentos absolutos mudam sob escala e sao irrelevantes aqui; o que
    precisa ser invariante e a proporcao.
    """
    rows = [(r.region, r.count, r.count_share, r.linear_share, r.occupancy) for r in report.regions]
    for shelf in report.shelves:
        rows.extend(
            (f"{shelf.index}:{r.region}", r.count, r.count_share, r.linear_share, r.occupancy)
            for r in shelf.regions
        )
    return rows


class TestP1SomaDosShares:
    """A soma dos shares vale 1.0 sempre que houver algo a medir."""

    @given(detections=detection_lists(), regions=region_sets())
    def test_shares_somam_um(self, detections: list[Detection], regions: RegionSet) -> None:
        report = analyze_detections(detections, regions=regions)
        assert report.status == "ok"
        assert math.isclose(sum(r.count_share for r in report.regions), 1.0, abs_tol=TOLERANCE)
        assert math.isclose(sum(r.linear_share for r in report.regions), 1.0, abs_tol=TOLERANCE)

    @given(detections=detection_lists(), regions=region_sets())
    def test_shares_somam_um_em_cada_prateleira(
        self, detections: list[Detection], regions: RegionSet
    ) -> None:
        report = analyze_detections(detections, regions=regions)
        for shelf in report.shelves:
            assert math.isclose(sum(r.count_share for r in shelf.regions), 1.0, abs_tol=TOLERANCE)
            assert math.isclose(sum(r.linear_share for r in shelf.regions), 1.0, abs_tol=TOLERANCE)

    @given(regions=region_sets())
    def test_sem_deteccoes_a_soma_e_zero_e_nao_nan(self, regions: RegionSet) -> None:
        report = analyze_detections([], regions=regions)
        assert report.status == "no_detections"
        assert sum(r.count_share for r in report.regions) == 0.0


class TestP2OrdemDaEntrada:
    """Reordenar as deteccoes nao muda absolutamente nada na saida."""

    @given(detections=detection_lists(), regions=region_sets(), data=st.data())
    def test_permutacao_produz_json_identico(
        self, detections: list[Detection], regions: RegionSet, data: st.DataObject
    ) -> None:
        embaralhada = data.draw(st.permutations(detections))
        original = analyze_detections(detections, regions=regions).model_dump_json()
        permutada = analyze_detections(embaralhada, regions=regions).model_dump_json()
        assert original == permutada


class TestP3Escala:
    """Escalar todas as coordenadas nao muda proporcao alguma."""

    @given(
        detections=detection_lists(),
        regions=region_sets(),
        factor=st.sampled_from(POWERS_OF_TWO),
    )
    def test_escala_preserva_os_shares(
        self, detections: list[Detection], regions: RegionSet, factor: float
    ) -> None:
        escaladas = [
            Detection(box=d.box.scaled(factor), confidence=d.confidence, label=d.label)
            for d in detections
        ]
        original = analyze_detections(detections, regions=regions)
        escalado = analyze_detections(escaladas, regions=regions)
        assert share_fingerprint(original) == share_fingerprint(escalado)

    @given(detections=detection_lists(), factor=st.sampled_from(POWERS_OF_TWO))
    def test_escala_preserva_o_agrupamento(
        self, detections: list[Detection], factor: float
    ) -> None:
        escaladas = [
            Detection(box=d.box.scaled(factor), confidence=d.confidence, label=d.label)
            for d in detections
        ]
        original = group_into_shelves(detections)
        escalado = group_into_shelves(escaladas)
        assert [s.count for s in original] == [s.count for s in escalado]


class TestP4Translacao:
    """Deslocar a foto inteira nao muda nada.

    Distinta da invariante de escala: pega o bug em que alguem comparou uma
    coordenada absoluta com um limiar, em vez de comparar uma distancia.
    """

    @given(
        detections=detection_lists(),
        regions=region_sets(),
        dx=st.integers(min_value=-5000, max_value=5000),
        dy=st.integers(min_value=-5000, max_value=5000),
    )
    def test_translacao_preserva_os_shares(
        self, detections: list[Detection], regions: RegionSet, dx: int, dy: int
    ) -> None:
        movidas = [
            Detection(
                box=d.box.translated(float(dx), float(dy)),
                confidence=d.confidence,
                label=d.label,
            )
            for d in detections
        ]
        original = analyze_detections(detections, regions=regions)
        movido = analyze_detections(movidas, regions=regions)
        assert share_fingerprint(original) == share_fingerprint(movido)


class TestP5Particao:
    """O agrupamento e uma particao total: nada se perde, nada se duplica."""

    @given(detections=detection_lists())
    def test_agrupamento_conserva_as_deteccoes(self, detections: list[Detection]) -> None:
        shelves = group_into_shelves(detections)
        saida = [d for shelf in shelves for d in shelf.detections]
        assert sorted(d.sort_key for d in saida) == sorted(d.sort_key for d in detections)

    @given(detections=detection_lists())
    def test_indices_sao_contiguos_e_ordenados(self, detections: list[Detection]) -> None:
        shelves = group_into_shelves(detections)
        assert [s.index for s in shelves] == list(range(len(shelves)))
        centros = [sum(d.box.center_y for d in s.detections) / s.count for s in shelves]
        assert centros == sorted(centros)

    @given(detections=detection_lists(), regions=region_sets())
    def test_cada_deteccao_cai_em_exatamente_uma_regiao(
        self, detections: list[Detection], regions: RegionSet
    ) -> None:
        report = analyze_detections(detections, regions=regions)
        for shelf in report.shelves:
            assert sum(r.count for r in shelf.regions) == shelf.detection_count


class TestP6ShareLinearComSobreposicao:
    """Com caixas sobrepostas o share continua em [0, 1].

    Somar larguras produziria valores acima de 1.0 em gondola cheia, onde
    produtos vizinhos se sobrepoem na projecao. A resposta certa e a uniao de
    intervalos, e esta propriedade e o que impede a regressao.
    """

    @given(detections=detection_lists(), regions=region_sets())
    def test_shares_e_ocupacao_ficam_na_faixa(
        self, detections: list[Detection], regions: RegionSet
    ) -> None:
        report = analyze_detections(detections, regions=regions)
        for shelf in report.shelves:
            assert 0.0 <= shelf.occupancy <= 1.0
            assert shelf.occupied_length <= shelf.extent.width + TOLERANCE
            for share in shelf.regions:
                assert 0.0 <= share.count_share <= 1.0
                assert 0.0 <= share.linear_share <= 1.0
                assert 0.0 <= share.occupancy <= 1.0

    @given(detections=detection_lists())
    def test_ocupado_nunca_passa_da_soma_ingenua(self, detections: list[Detection]) -> None:
        """A uniao nunca excede a soma das larguras, e e menor quando ha sobreposicao."""
        shelves = group_into_shelves(deduplicate(detections))
        for shelf_report, shelf in zip(
            analyze_detections(detections).shelves, shelves, strict=True
        ):
            soma_ingenua = sum(d.box.width for d in shelf.detections)
            assert shelf_report.occupied_length <= soma_ingenua + TOLERANCE


class TestP7Vazios:
    """Vazios sao disjuntos, ordenados e nao tocam produto nenhum."""

    @given(detections=detection_lists(), regions=region_sets())
    def test_vazios_nao_intersectam_deteccao(
        self, detections: list[Detection], regions: RegionSet
    ) -> None:
        report = analyze_detections(detections, regions=regions)
        shelves = group_into_shelves(deduplicate(detections))
        for shelf_report, shelf in zip(report.shelves, shelves, strict=True):
            for gap in shelf_report.gaps:
                for detection in shelf.detections:
                    assert gap.x_end <= detection.box.x1 or gap.x_start >= detection.box.x2

    @given(detections=detection_lists())
    def test_vazios_sao_disjuntos_e_ordenados(self, detections: list[Detection]) -> None:
        for shelf in analyze_detections(detections).shelves:
            bordas = [(gap.x_start, gap.x_end) for gap in shelf.gaps]
            assert bordas == sorted(bordas)
            for anterior, seguinte in pairwise(bordas):
                assert anterior[1] <= seguinte[0]

    @given(detections=detection_lists())
    def test_vazio_cabe_dentro_da_prateleira(self, detections: list[Detection]) -> None:
        for shelf in analyze_detections(detections).shelves:
            for gap in shelf.gaps:
                assert shelf.extent.x_min <= gap.x_start < gap.x_end <= shelf.extent.x_max


class TestP8Determinismo:
    """Duas execucoes sobre a mesma entrada produzem o mesmo JSON."""

    @given(detections=detection_lists(), regions=region_sets())
    def test_json_identico_entre_chamadas(
        self, detections: list[Detection], regions: RegionSet
    ) -> None:
        primeira = analyze_detections(detections, regions=regions).model_dump_json()
        segunda = analyze_detections(detections, regions=regions).model_dump_json()
        assert primeira == segunda


class TestDeduplicacao:
    """A deduplicacao e idempotente e nunca inventa deteccao."""

    @given(detections=detection_lists())
    def test_idempotente(self, detections: list[Detection]) -> None:
        uma_vez = deduplicate(detections)
        duas_vezes = deduplicate(uma_vez)
        assert uma_vez == duas_vezes

    @given(detections=detection_lists())
    def test_nunca_cresce_e_e_subconjunto(self, detections: list[Detection]) -> None:
        mantidas = deduplicate(detections)
        assert len(mantidas) <= len(detections)
        originais = [d.sort_key for d in detections]
        assert all(d.sort_key in originais for d in mantidas)

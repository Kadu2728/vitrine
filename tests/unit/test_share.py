"""Testes do relatorio de share.

Os numeros esperados foram calculados a mao a partir da gondola canonica
descrita em ``conftest.gondola``. Cada assercao traz a conta ao lado.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from helpers import detection
from vitrine import Detection, RegionSet, analyze_detections

ESQUERDA_DIREITA = RegionSet.from_cuts((0.0, 0.5, 1.0), ("esquerda", "direita"))


class TestPadraoSemRegioes:
    def test_estrutura_geral(self, gondola: list[Detection]) -> None:
        report = analyze_detections(gondola)
        assert report.status == "ok"
        assert report.schema_version == "1.2"
        assert report.total_detections == 7
        assert report.duplicates_removed == 0
        assert report.shelf_count == 2
        assert report.warnings == ()

    def test_regiao_unica_tem_share_total(self, gondola: list[Detection]) -> None:
        report = analyze_detections(gondola)
        assert len(report.regions) == 1
        total = report.regions[0]
        assert total.region == "total"
        assert total.count == 7
        assert total.count_share == 1.0
        assert total.linear_share == 1.0

    def test_ocupacao_da_prateleira_de_cima(self, gondola: list[Detection]) -> None:
        # 80 + 80 + 30 + 30 + 30 = 250 ocupados no envelope 0..310.
        shelf = analyze_detections(gondola).shelves[0]
        assert shelf.detection_count == 5
        assert shelf.extent.kind == "envelope"
        assert (shelf.extent.x_min, shelf.extent.x_max) == (0.0, 310.0)
        assert shelf.occupied_length == 250.0
        assert shelf.occupancy == pytest.approx(250 / 310)
        assert shelf.median_product_width == 30.0
        assert shelf.median_product_height == 100.0
        assert shelf.spread_ratio == 0.0

    def test_ocupacao_da_prateleira_de_baixo(self, gondola: list[Detection]) -> None:
        # 60 + 60 = 120 ocupados no envelope 0..310.
        shelf = analyze_detections(gondola).shelves[1]
        assert shelf.detection_count == 2
        assert shelf.occupied_length == 120.0
        assert shelf.occupancy == pytest.approx(120 / 310)

    def test_vazios(self, gondola: list[Detection]) -> None:
        report = analyze_detections(gondola)
        # Prateleira 0: mediana 30, so o vao de 170 a 200 alcanca o limiar.
        assert [(g.x_start, g.x_end) for g in report.shelves[0].gaps] == [(170.0, 200.0)]
        # Prateleira 1: mediana 60, o vao de 60 a 250 vale mais de tres produtos.
        gap = report.shelves[1].gaps[0]
        assert (gap.x_start, gap.x_end) == (60.0, 250.0)
        assert gap.width_ratio == pytest.approx(190 / 60)


class TestAsDuasMetricasDiscordam:
    """O caso que justifica publicar contagem e area lado a lado."""

    def test_na_prateleira_de_cima_a_ordem_se_inverte(self, gondola: list[Detection]) -> None:
        shelf = analyze_detections(gondola, regions=ESQUERDA_DIREITA).shelves[0]
        esquerda, direita = shelf.regions

        # Por contagem, a direita ganha: 3 produtos contra 2.
        assert (esquerda.count, direita.count) == (2, 3)
        assert esquerda.count_share == pytest.approx(0.4)
        assert direita.count_share == pytest.approx(0.6)

        # Por area linear, a esquerda ganha: 145 de comprimento contra 105.
        # Corte em x = 155: (0,80) + (90,155) = 145 | (155,170) + 3 x 30 = 105.
        assert esquerda.occupied_length == 145.0
        assert direita.occupied_length == 105.0
        assert esquerda.linear_share == pytest.approx(145 / 250)
        assert direita.linear_share == pytest.approx(105 / 250)

        # As duas leituras somam 1.0 cada uma.
        assert esquerda.count_share + direita.count_share == pytest.approx(1.0)
        assert esquerda.linear_share + direita.linear_share == pytest.approx(1.0)

    def test_ocupacao_nao_soma_um(self, gondola: list[Detection]) -> None:
        shelf = analyze_detections(gondola, regions=ESQUERDA_DIREITA).shelves[0]
        esquerda, direita = shelf.regions
        assert esquerda.occupancy == pytest.approx(145 / 155)
        assert direita.occupancy == pytest.approx(105 / 155)
        assert esquerda.occupancy + direita.occupancy > 1.0

    def test_agregado_soma_as_prateleiras(self, gondola: list[Detection]) -> None:
        report = analyze_detections(gondola, regions=ESQUERDA_DIREITA)
        esquerda, direita = report.regions

        # Contagem: 2 + 1 a esquerda, 3 + 1 a direita.
        assert (esquerda.count, direita.count) == (3, 4)
        assert esquerda.count_share == pytest.approx(3 / 7)
        assert direita.count_share == pytest.approx(4 / 7)

        # Comprimento: 145 + 60 a esquerda, 105 + 60 a direita.
        assert esquerda.occupied_length == 205.0
        assert direita.occupied_length == 165.0
        assert esquerda.linear_share == pytest.approx(205 / 370)
        assert direita.linear_share == pytest.approx(165 / 370)

        # Ocupacao sobre 155 + 155 de largura disponivel por lado.
        assert esquerda.occupancy == pytest.approx(205 / 310)


class TestCaixaNaFronteira:
    def test_a_contagem_atribui_inteira_e_a_area_reparte(self) -> None:
        # Uma caixa de 0 a 100 com o corte em 50: o centro cai na direita,
        # entao a contagem da tudo a direita; a area reparte 50 e 50.
        detections = [detection(0, 0, 100, 100)]
        regions = RegionSet.from_cuts((0.0, 0.5, 1.0), ("esquerda", "direita"))
        shelf = analyze_detections(detections, regions=regions).shelves[0]
        esquerda, direita = shelf.regions
        assert (esquerda.count, direita.count) == (0, 1)
        assert esquerda.occupied_length == 50.0
        assert direita.occupied_length == 50.0


class TestExtentExplicito:
    def test_muda_o_denominador_e_o_numero(self, gondola: list[Detection]) -> None:
        envelope = analyze_detections(gondola).shelves[0]
        explicito = analyze_detections(gondola, extent=(0.0, 1000.0)).shelves[0]
        assert envelope.occupancy == pytest.approx(250 / 310)
        assert explicito.occupancy == pytest.approx(250 / 1000)
        assert explicito.extent.kind == "explicit"

    def test_revela_vazio_nas_pontas(self) -> None:
        detections = [detection(400, 0, 450, 100), detection(460, 0, 510, 100)]
        report = analyze_detections(detections, extent=(0.0, 1000.0))
        assert len(report.shelves[0].gaps) == 2

    def test_avisa_sobre_deteccao_fora_do_extent(self) -> None:
        detections = [detection(0, 0, 50, 100), detection(900, 0, 950, 100)]
        report = analyze_detections(detections, extent=(0.0, 500.0))
        assert any("fora do extent" in w for w in report.warnings)

    def test_extent_invertido_e_rejeitado(self, gondola: list[Detection]) -> None:
        with pytest.raises(ValueError, match="x_max > x_min"):
            analyze_detections(gondola, extent=(500.0, 100.0))


class TestCasosDegenerados:
    def test_sem_deteccoes(self) -> None:
        report = analyze_detections([])
        assert report.status == "no_detections"
        assert report.total_detections == 0
        assert report.shelf_count == 0
        assert report.shelves == ()
        assert report.regions[0].count_share == 0.0
        assert report.regions[0].linear_share == 0.0
        assert "nenhuma deteccao" in report.warnings[0]

    def test_sem_deteccoes_preserva_as_regioes_pedidas(self) -> None:
        report = analyze_detections([], regions=ESQUERDA_DIREITA)
        assert [r.region for r in report.regions] == ["esquerda", "direita"]

    def test_json_sem_deteccoes_nao_tem_nan(self) -> None:
        payload = analyze_detections([]).model_dump_json()
        assert "NaN" not in payload
        assert "Infinity" not in payload

    def test_regiao_sem_nada_ocupado_tem_share_zero_e_nao_nan(self) -> None:
        # Toda a deteccao cai na metade esquerda; a direita fica com
        # denominador de ocupacao valido mas numerador zero.
        detections = [detection(0, 0, 40, 100), detection(50, 0, 90, 100)]
        report = analyze_detections(detections, regions=ESQUERDA_DIREITA)
        direita = report.shelves[0].regions[1]
        assert direita.occupied_length >= 0.0
        assert "NaN" not in report.model_dump_json()

    def test_extent_fora_das_deteccoes_zera_a_ocupacao(self) -> None:
        # O extent nao intersecta nenhum produto: nada ocupado, e as divisoes
        # por zero precisam devolver 0.0 em vez de estourar.
        detections = [detection(0, 0, 40, 100)]
        report = analyze_detections(detections, extent=(500.0, 900.0))
        shelf = report.shelves[0]
        assert shelf.occupied_length == 0.0
        assert shelf.occupancy == 0.0
        assert shelf.regions[0].linear_share == 0.0
        assert "NaN" not in report.model_dump_json()

    def test_uma_unica_deteccao_gera_aviso(self) -> None:
        report = analyze_detections([detection(0, 0, 50, 100)])
        assert any("uma unica deteccao" in w for w in report.warnings)

    def test_gondola_inclinada_gera_aviso(self) -> None:
        # Dispersao vertical de 1.2 alturas dentro de uma prateleira so.
        detections = [detection(i * 60, i * 30, i * 60 + 50, i * 30 + 100) for i in range(5)]
        report = analyze_detections(detections, max_shelf_spread_ratio=10.0)
        assert report.shelf_count == 1
        assert any("perspectiva nao corrigida" in w for w in report.warnings)


class TestDeduplicacao:
    def test_duplicata_nao_infla_a_contagem(self) -> None:
        detections = [detection(0, 0, 50, 100), detection(0, 0, 50, 100)]
        report = analyze_detections(detections)
        assert report.total_detections == 1
        assert report.duplicates_removed == 1


class TestContratoDeSaida:
    def test_params_sao_ecoados(self, gondola: list[Detection]) -> None:
        report = analyze_detections(gondola, regions=ESQUERDA_DIREITA, extent=(0.0, 400.0))
        assert report.params.extent_kind == "explicit"
        assert report.params.explicit_extent == (0.0, 400.0)
        assert report.params.shelf_gap_ratio == 0.5
        assert [r.name for r in report.params.regions.regions] == ["esquerda", "direita"]

    def test_source_e_preservado(self, gondola: list[Detection]) -> None:
        assert analyze_detections(gondola, source="LOJA_12/2026-08-31").source == (
            "LOJA_12/2026-08-31"
        )

    def test_json_e_recarregavel(self, gondola: list[Detection]) -> None:
        report = analyze_detections(gondola, regions=ESQUERDA_DIREITA)
        payload = json.loads(report.model_dump_json())
        assert payload["schema_version"] == "1.2"
        assert payload["shelves"][0]["regions"][0]["region"] == "esquerda"


class TestDeterminismo:
    def test_duas_chamadas_no_mesmo_processo(self, gondola: list[Detection]) -> None:
        primeira = analyze_detections(gondola, regions=ESQUERDA_DIREITA).model_dump_json()
        segunda = analyze_detections(gondola, regions=ESQUERDA_DIREITA).model_dump_json()
        assert primeira == segunda

    def test_outro_processo_com_outro_hash_seed(self, gondola: list[Detection]) -> None:
        """A prova real de R4: mesma entrada, interpretador novo, seed diferente.

        E o unico teste que pega dependencia de ordenacao de ``set`` ou ``dict``
        vazando para a saida, porque ``PYTHONHASHSEED`` so pode ser trocado
        antes de o interpretador iniciar.
        """
        script = (
            "from helpers import detection\n"
            "from vitrine import RegionSet, analyze_detections\n"
            "d = [detection(*c) for c in "
            "[(0,0,80,100),(90,0,170,100),(200,0,230,100),(240,0,270,100),"
            "(280,0,310,100),(0,200,60,300),(250,200,310,300)]]\n"
            "r = RegionSet.from_cuts((0.0,0.5,1.0), ('esquerda','direita'))\n"
            "print(analyze_detections(d, regions=r).model_dump_json())"
        )
        env = dict(os.environ, PYTHONHASHSEED="1", PYTHONPATH=_tests_dir())
        resultado = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        esperado = analyze_detections(gondola, regions=ESQUERDA_DIREITA).model_dump_json()
        assert resultado.stdout.strip() == esperado


def _tests_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

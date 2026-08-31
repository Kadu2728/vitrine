"""Testes do agrupamento em prateleiras.

Todos os casos sao montados a mao com centros verticais escolhidos para cair de
um lado ou do outro do limiar, de modo que a resposta correta seja conferivel
sem rodar nada.
"""

from __future__ import annotations

import pytest

from helpers import detection
from vitrine import Detection, group_into_shelves


class TestCasosBasicos:
    def test_entrada_vazia(self) -> None:
        assert group_into_shelves([]) == ()

    def test_uma_deteccao_e_uma_prateleira(self) -> None:
        shelves = group_into_shelves([detection(0, 0, 50, 100)])
        assert len(shelves) == 1
        assert shelves[0].count == 1

    def test_duas_prateleiras_bem_separadas(self) -> None:
        # Altura mediana 100 -> tau = 50. Centros em 50 e 250, lacuna 200 > 50.
        detections = [
            detection(0, 0, 50, 100),
            detection(60, 0, 110, 100),
            detection(0, 200, 50, 300),
            detection(60, 200, 110, 300),
        ]
        shelves = group_into_shelves(detections)
        assert [s.count for s in shelves] == [2, 2]
        assert shelves[0].y_top == 0.0
        assert shelves[1].y_top == 200.0

    def test_indices_vao_de_cima_para_baixo(self) -> None:
        detections = [detection(0, 400, 50, 500), detection(0, 0, 50, 100)]
        shelves = group_into_shelves(detections)
        assert shelves[0].index == 0
        assert shelves[0].y_top == 0.0
        assert shelves[1].index == 1
        assert shelves[1].y_top == 400.0

    def test_deteccoes_saem_da_esquerda_para_a_direita(self) -> None:
        detections = [detection(200, 0, 250, 100), detection(0, 0, 50, 100)]
        shelves = group_into_shelves(detections)
        assert [d.box.x1 for d in shelves[0].detections] == [0.0, 200.0]

    def test_alturas_diferentes_na_mesma_prateleira(self) -> None:
        # Centros 50, 55, 60: alturas 100, 90 e 80 com bases alinhadas.
        detections = [
            detection(0, 0, 50, 100),
            detection(60, 10, 110, 100),
            detection(120, 20, 170, 100),
        ]
        assert len(group_into_shelves(detections)) == 1


class TestLimiar:
    def test_lacuna_logo_abaixo_do_limiar_mantem_junto(self) -> None:
        # tau = 0.5 * 100 = 50; lacuna entre centros = 49.
        detections = [detection(0, 0, 50, 100), detection(60, 49, 110, 149)]
        assert len(group_into_shelves(detections)) == 1

    def test_lacuna_logo_acima_do_limiar_separa(self) -> None:
        # Lacuna entre centros = 51 > 50.
        detections = [detection(0, 0, 50, 100), detection(60, 51, 110, 151)]
        assert len(group_into_shelves(detections)) == 2

    def test_limiar_e_relativo_ao_tamanho_do_produto(self) -> None:
        # A mesma configuracao geometrica com produtos dez vezes menores
        # produz o mesmo agrupamento. Um limiar absoluto em pixels falharia.
        grandes = [detection(0, 0, 50, 100), detection(60, 200, 110, 300)]
        pequenos = [detection(0, 0, 5, 10), detection(6, 20, 11, 30)]
        assert len(group_into_shelves(grandes)) == len(group_into_shelves(pequenos)) == 2

    def test_gap_ratio_invalido(self) -> None:
        with pytest.raises(ValueError, match="gap_ratio"):
            group_into_shelves([detection(0, 0, 10, 10)], gap_ratio=0.0)

    def test_max_spread_ratio_invalido(self) -> None:
        with pytest.raises(ValueError, match="max_spread_ratio"):
            group_into_shelves([detection(0, 0, 10, 10)], max_spread_ratio=-1.0)


class TestGuardaContraEncadeamento:
    def test_escada_seria_uma_prateleira_so_sem_a_guarda(self) -> None:
        # Cinco produtos de altura 100 com centros em 50, 90, 130, 170, 210.
        # Cada lacuna vale 40 < tau = 50, entao o single linkage puro juntaria
        # tudo. A dispersao total e 160 > 1.5 * 100 = 150, e a guarda reparte.
        detections = [detection(i * 60, i * 40, i * 60 + 50, i * 40 + 100) for i in range(5)]
        sem_guarda = group_into_shelves(detections, max_spread_ratio=100.0)
        com_guarda = group_into_shelves(detections)
        assert len(sem_guarda) == 1
        assert len(com_guarda) == 2

    def test_reparticao_e_deterministica_com_lacunas_iguais(self) -> None:
        # Lacunas todas iguais: o desempate escolhe o corte mais proximo do
        # meio e, persistindo o empate, o de menor indice.
        detections = [detection(i * 60, i * 40, i * 60 + 50, i * 40 + 100) for i in range(5)]
        primeira = group_into_shelves(detections)
        segunda = group_into_shelves(list(reversed(detections)))
        assert [s.count for s in primeira] == [s.count for s in segunda] == [2, 3]

    def test_prateleira_bem_comportada_nao_e_repartida(self) -> None:
        detections = [detection(i * 60, 0, i * 60 + 50, 100) for i in range(6)]
        assert len(group_into_shelves(detections)) == 1


class TestParticao:
    def test_nenhuma_deteccao_se_perde_ou_duplica(self, gondola: list[Detection]) -> None:
        shelves = group_into_shelves(gondola)
        saida = [d for shelf in shelves for d in shelf.detections]
        assert len(saida) == len(gondola)
        assert sorted(d.sort_key for d in saida) == sorted(d.sort_key for d in gondola)

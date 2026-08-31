"""Testes das primitivas 1D. Todos os resultados sao conferiveis de cabeca."""

from __future__ import annotations

import pytest

from vitrine.domain import geometry


class TestMergeIntervals:
    def test_disjuntos_permanecem_separados(self) -> None:
        assert geometry.merge_intervals([(0, 10), (20, 30)]) == ((0, 10), (20, 30))

    def test_sobrepostos_viram_um(self) -> None:
        assert geometry.merge_intervals([(0, 10), (5, 20)]) == ((0, 20),)

    def test_adjacentes_viram_um(self) -> None:
        assert geometry.merge_intervals([(0, 10), (10, 20)]) == ((0, 20),)

    def test_contido_desaparece(self) -> None:
        assert geometry.merge_intervals([(0, 100), (10, 20)]) == ((0, 100),)

    def test_ordem_de_entrada_nao_importa(self) -> None:
        direta = geometry.merge_intervals([(0, 10), (5, 20), (40, 50)])
        invertida = geometry.merge_intervals([(40, 50), (5, 20), (0, 10)])
        assert direta == invertida

    def test_degenerados_sao_descartados(self) -> None:
        assert geometry.merge_intervals([(10, 10), (20, 5)]) == ()

    def test_vazio(self) -> None:
        assert geometry.merge_intervals([]) == ()


class TestCoveredLength:
    def test_sem_sobreposicao_e_a_soma(self) -> None:
        assert geometry.covered_length([(0, 10), (20, 30)]) == 20.0

    def test_com_sobreposicao_conta_uma_vez(self) -> None:
        # A razao de este modulo existir: a soma ingenua daria 30.
        assert geometry.covered_length([(0, 20), (10, 30)]) == 30.0


class TestClip:
    def test_recorta_bordas(self) -> None:
        assert geometry.clip([(0, 100)], (20, 80)) == ((20, 80),)

    def test_descarta_o_que_fica_fora(self) -> None:
        assert geometry.clip([(0, 10), (200, 300)], (20, 80)) == ()


class TestComplement:
    def test_buraco_no_meio(self) -> None:
        assert geometry.complement([(0, 40), (60, 100)], (0, 100)) == ((40, 60),)

    def test_bordas_livres_aparecem(self) -> None:
        assert geometry.complement([(40, 60)], (0, 100)) == ((0, 40), (60, 100))

    def test_envelope_nao_produz_vazio_nas_pontas(self) -> None:
        # Com extent = envelope das caixas, so sobram vazios internos.
        assert geometry.complement([(0, 40), (60, 100)], (0, 100)) == ((40, 60),)

    def test_totalmente_ocupado(self) -> None:
        assert geometry.complement([(0, 100)], (0, 100)) == ()

    def test_totalmente_livre(self) -> None:
        assert geometry.complement([], (0, 100)) == ((0, 100),)

    def test_janela_degenerada(self) -> None:
        assert geometry.complement([(0, 10)], (50, 50)) == ()


class TestOverlapLength:
    def test_parcial(self) -> None:
        assert geometry.overlap_length((0, 10), (5, 20)) == 5.0

    def test_disjunto_e_zero(self) -> None:
        assert geometry.overlap_length((0, 10), (20, 30)) == 0.0


class TestMedian:
    def test_impar(self) -> None:
        assert geometry.median([3, 1, 2]) == 2.0

    def test_par_e_a_media_dos_centrais(self) -> None:
        assert geometry.median([1, 2, 3, 4]) == 2.5

    def test_vazio_levanta(self) -> None:
        with pytest.raises(ValueError, match="sequencia vazia"):
            geometry.median([])

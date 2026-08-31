"""Testes das metricas de avaliacao.

Cada caso tem a conta ao lado. Metrica sem teste e numero sem procedencia, e o
proposito destas funcoes e justamente dar procedencia a um numero.
"""

from __future__ import annotations

import pytest

from helpers import box, detection
from vitrine.eval.metrics import average_precision, evaluate, match


class TestMatch:
    def test_acerto_perfeito(self) -> None:
        predicoes = [detection(0, 0, 10, 10)]
        anotadas = [box(0, 0, 10, 10)]
        assert match(predicoes, anotadas) == (True,)

    def test_iou_abaixo_do_limiar_e_erro(self) -> None:
        # Interseccao 20, uniao 180: IoU = 0.111.
        predicoes = [detection(0, 0, 10, 10)]
        anotadas = [box(8, 0, 18, 10)]
        assert match(predicoes, anotadas) == (False,)

    def test_iou_exatamente_no_limiar_conta_como_acerto(self) -> None:
        # Interseccao 50, uniao 150: IoU = 1/3. Com limiar 1/3, acerta.
        predicoes = [detection(0, 0, 10, 10)]
        anotadas = [box(5, 0, 15, 10)]
        assert match(predicoes, anotadas, iou_threshold=1 / 3) == (True,)

    def test_cada_anotacao_casa_com_uma_predicao_so(self) -> None:
        """Duas predicoes sobre o mesmo produto: uma acerta, a outra e falso positivo.

        E o comportamento que penaliza detector com NMS mal calibrado, que e
        justamente o defeito que inflaria a contagem de produtos.
        """
        predicoes = [
            detection(0, 0, 10, 10, confidence=0.9),
            detection(1, 1, 11, 11, confidence=0.6),
        ]
        anotadas = [box(0, 0, 10, 10)]
        assert match(predicoes, anotadas) == (True, False)

    def test_a_de_maior_confianca_tem_prioridade(self) -> None:
        predicoes = [
            detection(1, 1, 11, 11, confidence=0.4),
            detection(0, 0, 10, 10, confidence=0.95),
        ]
        anotadas = [box(0, 0, 10, 10)]
        assert match(predicoes, anotadas) == (False, True)

    def test_resultado_alinhado_com_a_ordem_de_entrada(self) -> None:
        predicoes = [detection(100, 100, 110, 110), detection(0, 0, 10, 10)]
        anotadas = [box(0, 0, 10, 10)]
        assert match(predicoes, anotadas) == (False, True)

    def test_sem_anotacao_tudo_e_falso_positivo(self) -> None:
        assert match([detection(0, 0, 10, 10)], []) == (False,)

    def test_sem_predicao(self) -> None:
        assert match([], [box(0, 0, 10, 10)]) == ()

    def test_limiar_invalido(self) -> None:
        with pytest.raises(ValueError, match="iou_threshold"):
            match([], [], iou_threshold=0.0)


class TestAveragePrecision:
    def test_deteccao_perfeita_vale_um(self) -> None:
        pontuacoes = [(0.9, True), (0.8, True), (0.7, True)]
        assert average_precision(pontuacoes, ground_truth=3) == pytest.approx(1.0)

    def test_tudo_errado_vale_zero(self) -> None:
        assert average_precision([(0.9, False), (0.8, False)], ground_truth=2) == 0.0

    def test_sem_anotacao_vale_zero_e_nao_nan(self) -> None:
        assert average_precision([(0.9, False)], ground_truth=0) == 0.0

    def test_sem_predicao_vale_zero(self) -> None:
        assert average_precision([], ground_truth=5) == 0.0

    def test_metade_encontrada_com_precisao_total(self) -> None:
        # Duas predicoes certas de quatro anotacoes: recall para em 0.5 com
        # precisao 1.0, entao a area vale 0.5.
        pontuacoes = [(0.9, True), (0.8, True)]
        assert average_precision(pontuacoes, ground_truth=4) == pytest.approx(0.5)

    def test_caso_conferido_no_papel(self) -> None:
        """Um falso positivo no meio da ordenacao.

        Ordem por confianca: acerto, erro, acerto. Com 2 anotacoes:

        - posicao 1: TP=1, recall 0.5, precisao 1.000
        - posicao 2: TP=1, recall 0.5, precisao 0.500
        - posicao 3: TP=2, recall 1.0, precisao 0.667

        Envoltoria decrescente: [1.000, 0.667, 0.667].
        Area = (0.5 - 0.0) * 1.0 + (0.5 - 0.5) * 0.667 + (1.0 - 0.5) * 0.667
             = 0.5 + 0 + 0.3333 = 0.8333
        """
        pontuacoes = [(0.9, True), (0.8, False), (0.7, True)]
        assert average_precision(pontuacoes, ground_truth=2) == pytest.approx(0.8333, abs=1e-4)

    def test_a_envoltoria_e_aplicada(self) -> None:
        """Sem a envoltoria decrescente, a area sairia menor que a definicao."""
        pontuacoes = [(0.9, False), (0.8, True)]
        # posicao 1: recall 0, precisao 0. posicao 2: recall 1.0, precisao 0.5.
        # Envoltoria: [0.5, 0.5]. Area = 1.0 * 0.5 = 0.5.
        assert average_precision(pontuacoes, ground_truth=1) == pytest.approx(0.5)


class TestEvaluate:
    def test_deteccao_perfeita(self) -> None:
        amostras = [([detection(0, 0, 10, 10, confidence=0.9)], [box(0, 0, 10, 10)])]
        resultado = evaluate(amostras)
        assert resultado.images == 1
        assert resultado.ground_truth == 1
        assert resultado.true_positives == 1
        assert resultado.false_positives == 0
        assert resultado.false_negatives == 0
        assert resultado.precision == 1.0
        assert resultado.recall == 1.0
        assert resultado.f1 == 1.0
        assert resultado.average_precision == pytest.approx(1.0)

    def test_metade_dos_produtos_perdida(self) -> None:
        amostras = [
            (
                [detection(0, 0, 10, 10, confidence=0.9)],
                [box(0, 0, 10, 10), box(50, 0, 60, 10)],
            )
        ]
        resultado = evaluate(amostras)
        assert resultado.recall == pytest.approx(0.5)
        assert resultado.precision == pytest.approx(1.0)
        assert resultado.false_negatives == 1
        assert resultado.f1 == pytest.approx(2 / 3)

    def test_predicao_a_mais_derruba_a_precisao(self) -> None:
        amostras = [
            (
                [
                    detection(0, 0, 10, 10, confidence=0.9),
                    detection(500, 500, 510, 510, confidence=0.8),
                ],
                [box(0, 0, 10, 10)],
            )
        ]
        resultado = evaluate(amostras)
        assert resultado.precision == pytest.approx(0.5)
        assert resultado.recall == pytest.approx(1.0)
        assert resultado.false_positives == 1

    def test_confianca_abaixo_do_limiar_nao_entra_na_precisao(self) -> None:
        amostras = [
            (
                [
                    detection(0, 0, 10, 10, confidence=0.9),
                    detection(500, 500, 510, 510, confidence=0.1),
                ],
                [box(0, 0, 10, 10)],
            )
        ]
        resultado = evaluate(amostras, confidence_threshold=0.25)
        assert resultado.predictions == 1
        assert resultado.precision == pytest.approx(1.0)

    def test_varias_imagens_sao_agregadas(self) -> None:
        amostras = [
            ([detection(0, 0, 10, 10, confidence=0.9)], [box(0, 0, 10, 10)]),
            ([], [box(0, 0, 10, 10)]),
        ]
        resultado = evaluate(amostras)
        assert resultado.images == 2
        assert resultado.ground_truth == 2
        assert resultado.recall == pytest.approx(0.5)

    def test_conjunto_vazio_nao_produz_nan(self) -> None:
        resultado = evaluate([])
        assert resultado.precision == 0.0
        assert resultado.recall == 0.0
        assert resultado.f1 == 0.0
        assert resultado.average_precision == 0.0
        assert "NaN" not in resultado.model_dump_json()

    def test_limiar_de_confianca_invalido(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            evaluate([], confidence_threshold=1.5)

    def test_resultado_registra_os_limiares(self) -> None:
        resultado = evaluate([], iou_threshold=0.75, confidence_threshold=0.4)
        assert resultado.iou_threshold == 0.75
        assert resultado.confidence_threshold == 0.4

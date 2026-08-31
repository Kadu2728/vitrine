"""Testes da deduplicacao de deteccoes."""

from __future__ import annotations

import pytest

from helpers import detection
from vitrine import deduplicate


class TestDeduplicate:
    def test_entrada_vazia(self) -> None:
        assert deduplicate([]) == ()

    def test_caixas_distintas_sobrevivem(self) -> None:
        detections = [detection(0, 0, 50, 100), detection(200, 0, 250, 100)]
        assert len(deduplicate(detections)) == 2

    def test_caixas_identicas_viram_uma(self) -> None:
        detections = [detection(0, 0, 50, 100), detection(0, 0, 50, 100)]
        assert len(deduplicate(detections)) == 1

    def test_mantem_a_de_maior_confianca(self) -> None:
        fraca = detection(0, 0, 50, 100, confidence=0.4)
        forte = detection(1, 0, 51, 100, confidence=0.9)
        mantidas = deduplicate([fraca, forte])
        assert len(mantidas) == 1
        assert mantidas[0].confidence == 0.9

    def test_produtos_vizinhos_legitimos_nao_sao_apagados(self) -> None:
        # Sobreposicao de 20% em gondola cheia e normal e precisa sobreviver:
        # apagar produto real geraria ruptura falsa.
        detections = [detection(0, 0, 100, 100), detection(80, 0, 180, 100)]
        assert len(deduplicate(detections)) == 2

    def test_ordem_de_entrada_nao_muda_o_resultado(self) -> None:
        detections = [
            detection(0, 0, 50, 100, confidence=0.5),
            detection(1, 1, 51, 101, confidence=0.7),
            detection(200, 0, 250, 100, confidence=0.6),
        ]
        assert deduplicate(detections) == deduplicate(list(reversed(detections)))

    def test_saida_sai_ordenada(self) -> None:
        detections = [detection(200, 0, 250, 100), detection(0, 0, 50, 100)]
        assert [d.box.x1 for d in deduplicate(detections)] == [0.0, 200.0]

    def test_limiar_configuravel(self) -> None:
        detections = [detection(0, 0, 100, 100), detection(20, 0, 120, 100)]
        # IoU aproximadamente 0.67: sobrevive ao padrao, cai num limiar baixo.
        assert len(deduplicate(detections)) == 2
        assert len(deduplicate(detections, iou_threshold=0.5)) == 1

    @pytest.mark.parametrize("threshold", [0.0, -0.5, 1.5])
    def test_limiar_invalido(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="iou_threshold"):
            deduplicate([detection(0, 0, 10, 10)], iou_threshold=threshold)

"""Testes da pagina de demonstracao.

Pulados quando o Gradio nao esta instalado -- ele mora no grupo ``demo``, nao
nas dependencias base, justamente porque a pagina nao faz parte do produto.

O que se verifica aqui e o contrato entre a pagina e a biblioteca: que ela
consome apenas a API publica, que o resumo sai coerente e que erro previsivel
vira mensagem, nunca excecao vazando para o navegador.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from helpers import synthetic_shelf, write_image

pytest.importorskip("gradio", reason="grupo 'demo' nao instalado")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from gradio_app import analisar


def _foto(tmp_path: Path) -> str:
    pixels, _ = synthetic_shelf(rows=2, columns=3)
    return str(write_image(tmp_path / "gondola.png", pixels))


class TestAnalisar:
    def test_sem_imagem_orienta_em_vez_de_quebrar(self) -> None:
        imagem, texto, payload = analisar(None, False, False, 0.5, "a", "b", 2000)
        assert imagem is None
        assert payload is None
        assert "Envie uma foto" in texto

    def test_caminho_inexistente_vira_mensagem_com_dica(self) -> None:
        imagem, texto, _ = analisar("nao_existe.jpg", False, False, 0.5, "a", "b", 2000)
        assert imagem is None
        assert "nao encontrada" in texto
        assert "Verifique o caminho" in texto

    def test_analise_completa(self, tmp_path: Path) -> None:
        imagem, texto, payload = analisar(_foto(tmp_path), False, False, 0.5, "a", "b", 2000)
        assert imagem is not None
        assert payload is not None
        assert payload["total_detections"] == 6
        assert "6 produtos" in texto

    def test_regioes_aparecem_no_resumo(self, tmp_path: Path) -> None:
        _, texto, payload = analisar(
            _foto(tmp_path), False, True, 0.5, "minha_marca", "concorrencia", 2000
        )
        assert payload is not None
        assert [r["region"] for r in payload["regions"]] == ["minha_marca", "concorrencia"]
        assert "Share por contagem" in texto
        assert "costumam discordar" in texto

    def test_imagem_sem_produto_sugere_inverter(self, tmp_path: Path) -> None:
        pixels, _ = synthetic_shelf(rows=1, columns=2)
        caminho = str(write_image(tmp_path / "clara.png", 255 - pixels))
        _, texto, _ = analisar(caminho, False, False, 0.5, "a", "b", 2000)
        assert "Inverter polaridade" in texto


class TestAviso:
    """A ressalva precisa estar na tela, nao so no README.

    Nao ha teste montando o ``gr.Blocks`` aqui de proposito: a construcao da
    pagina abre sockets e um event loop que o Gradio nao fecha, e o
    ``filterwarnings = error`` da suite transforma esse vazamento de terceiro
    numa falha. Silenciar a categoria inteira esconderia vazamento nosso de
    verdade. A montagem da pagina e verificada subindo o app --
    ``uv run python app/gradio_app.py`` -- que e evidencia mais forte que um
    teste de fumaca.
    """

    def test_declara_a_limitacao_do_detector(self) -> None:
        from gradio_app import AVISO

        assert "funciona mal em foto de supermercado real" in AVISO

    def test_declara_que_as_metricas_nao_foram_medidas(self) -> None:
        from gradio_app import AVISO

        assert "nao medidas" in AVISO

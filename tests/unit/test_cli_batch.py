"""Testes de CLI dos comandos de lote e histórico.

Cobrem o que o usuário vê: códigos de saída, canais, mensagens com dica e o
ciclo completo — processar uma pasta, gravar no histórico e consultar a
evolução.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from helpers import synthetic_shelf, write_image
from vitrine.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, app

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration
"""Estes testes criam arquivos, bancos e invocam a CLI inteira.

Continuam rodando por padrao -- marcar nao e esconder. A marca existe para
que quem esta mexendo no dominio possa rodar
``pytest -m 'not slow and not integration'`` e ter resposta em menos de dois
segundos.
"""


runner = CliRunner()


def povoar(pasta: Path, quantidade: int = 3) -> Path:
    """Cria uma pasta com N fotos sintéticas válidas."""
    pasta.mkdir(parents=True, exist_ok=True)
    for indice in range(quantidade):
        pixels, _ = synthetic_shelf(rows=2, columns=3)
        write_image(pasta / f"foto_{indice:02d}.png", pixels)
    return pasta


class TestAjuda:
    def test_batch_explica_a_retomada(self) -> None:
        saida = runner.invoke(app, ["batch", "--help"]).output
        assert "resumivel" in saida
        assert "Ctrl+C" in saida

    def test_batch_mostra_como_ler_o_log(self) -> None:
        assert "jq" in runner.invoke(app, ["batch", "--help"]).output

    def test_history_explica_a_data_usada(self) -> None:
        assert "data da foto" in runner.invoke(app, ["history", "--help"]).output

    def test_os_quatro_comandos_aparecem(self) -> None:
        saida = runner.invoke(app, ["--help"]).output
        for comando in ("analyze", "batch", "benchmark", "history"):
            assert comando in saida


class TestBatch:
    def test_processa_a_pasta(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app,
            [
                "batch",
                str(povoar(tmp_path / "fotos", 3)),
                "--out",
                str(tmp_path / "out"),
                "--json",
            ],
        )
        assert resultado.exit_code == EXIT_OK
        payload = json.loads(resultado.stdout)
        assert payload["total"] == 3
        assert payload["processed"] == 3
        assert payload["failed"] == 0

    def test_segunda_execucao_pula(self, tmp_path: Path) -> None:
        pasta = str(povoar(tmp_path / "fotos", 2))
        saida = str(tmp_path / "out")
        runner.invoke(app, ["batch", pasta, "--out", saida, "--json"])
        segunda = runner.invoke(app, ["batch", pasta, "--out", saida, "--json"])
        assert json.loads(segunda.stdout)["skipped"] == 2

    def test_no_resume_refaz(self, tmp_path: Path) -> None:
        pasta = str(povoar(tmp_path / "fotos", 2))
        saida = str(tmp_path / "out")
        runner.invoke(app, ["batch", pasta, "--out", saida, "--json"])
        segunda = runner.invoke(app, ["batch", pasta, "--out", saida, "--no-resume", "--json"])
        assert json.loads(segunda.stdout)["processed"] == 2

    def test_tabela_no_terminal(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app, ["batch", str(povoar(tmp_path / "fotos", 2)), "--out", str(tmp_path / "out")]
        )
        assert resultado.exit_code == EXIT_OK
        assert "Processadas agora" in resultado.output

    def test_imagem_ruim_nao_derruba_o_lote(self, tmp_path: Path) -> None:
        pasta = povoar(tmp_path / "fotos", 2)
        (pasta / "ruim.png").write_text("nao e imagem", encoding="utf-8")
        resultado = runner.invoke(
            app, ["batch", str(pasta), "--out", str(tmp_path / "out"), "--json"]
        )
        assert resultado.exit_code == EXIT_OK
        payload = json.loads(resultado.stdout)
        assert payload["processed"] == 2
        assert payload["failed"] == 1
        assert payload["failures"][0]["image"] == "ruim.png"

    def test_gera_manifesto_e_log(self, tmp_path: Path) -> None:
        saida = tmp_path / "out"
        runner.invoke(app, ["batch", str(povoar(tmp_path / "fotos", 2)), "--out", str(saida)])
        assert (saida / "manifest.jsonl").is_file()
        assert (saida / "vitrine.jsonl").is_file()

    def test_pasta_inexistente(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["batch", str(tmp_path / "fantasma")])
        assert resultado.exit_code == EXIT_FAILURE
        assert "nao e uma pasta" in resultado.output

    def test_detector_desconhecido(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app, ["batch", str(povoar(tmp_path / "fotos", 1)), "--detector", "magico"]
        )
        assert resultado.exit_code == EXIT_USAGE
        assert "contour" in resultado.output

    def test_weights_com_contour(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app, ["batch", str(povoar(tmp_path / "fotos", 1)), "--weights", "p.pt"]
        )
        assert resultado.exit_code == EXIT_USAGE
        assert "--detector yolo" in resultado.output


class TestCicloCompleto:
    """Processar uma pasta, gravar no histórico e consultar a evolução."""

    def test_batch_grava_e_history_le(self, tmp_path: Path) -> None:
        banco = tmp_path / "vitrine.db"
        lote = runner.invoke(
            app,
            [
                "batch",
                str(povoar(tmp_path / "fotos", 2)),
                "--out",
                str(tmp_path / "out"),
                "--store-id",
                "LOJA_12",
                "--db",
                str(banco),
                "--json",
            ],
        )
        assert lote.exit_code == EXIT_OK
        assert banco.is_file()

        historico = runner.invoke(
            app, ["history", "--store-id", "LOJA_12", "--db", str(banco), "--json"]
        )
        assert historico.exit_code == EXIT_OK
        visitas = json.loads(historico.stdout)
        assert len(visitas) == 2
        assert all(v["store_id"] == "LOJA_12" for v in visitas)

    def test_history_em_tabela(self, tmp_path: Path) -> None:
        banco = tmp_path / "vitrine.db"
        runner.invoke(
            app,
            [
                "batch",
                str(povoar(tmp_path / "fotos", 1)),
                "--out",
                str(tmp_path / "out"),
                "--store-id",
                "LOJA_12",
                "--db",
                str(banco),
            ],
        )
        resultado = runner.invoke(app, ["history", "--store-id", "LOJA_12", "--db", str(banco)])
        assert resultado.exit_code == EXIT_OK
        assert "LOJA_12" in resultado.output
        assert "Ocupacao" in resultado.output

    def test_sem_store_id_nao_cria_banco(self, tmp_path: Path) -> None:
        banco = tmp_path / "vitrine.db"
        runner.invoke(
            app,
            [
                "batch",
                str(povoar(tmp_path / "fotos", 1)),
                "--out",
                str(tmp_path / "out"),
                "--db",
                str(banco),
            ],
        )
        assert not banco.exists()


class TestHistory:
    def test_banco_inexistente_orienta(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app, ["history", "--store-id", "LOJA_1", "--db", str(tmp_path / "nada.db")]
        )
        assert resultado.exit_code == EXIT_FAILURE
        assert "vitrine batch" in resultado.output

    def test_pdv_sem_historico_lista_os_conhecidos(self, tmp_path: Path) -> None:
        banco = tmp_path / "vitrine.db"
        runner.invoke(
            app,
            [
                "batch",
                str(povoar(tmp_path / "fotos", 1)),
                "--out",
                str(tmp_path / "out"),
                "--store-id",
                "LOJA_12",
                "--db",
                str(banco),
            ],
        )
        resultado = runner.invoke(app, ["history", "--store-id", "LOJA_99", "--db", str(banco)])
        assert resultado.exit_code == EXIT_OK
        assert "Sem historico" in resultado.output
        assert "LOJA_12" in resultado.output

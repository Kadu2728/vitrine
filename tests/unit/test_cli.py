"""Testes da CLI.

Sem front-end, a CLI e a interface do produto -- entao ela e testada como
interface: codigos de saida, canal de cada saida, texto de ajuda e, sobretudo,
as mensagens de erro. Uma mensagem que nao diz o que fazer e um defeito, e aqui
isso falha o teste.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from typer.testing import CliRunner

from helpers import synthetic_shelf, write_image
from vitrine.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _gondola(tmp_path: Path) -> Path:
    pixels, _ = synthetic_shelf(rows=2, columns=3)
    return write_image(tmp_path / "gondola.png", pixels)


class TestAjuda:
    def test_ajuda_geral_descreve_o_produto(self) -> None:
        resultado = runner.invoke(app, ["--help"])
        assert resultado.exit_code == EXIT_OK
        assert "ponto de venda" in resultado.output

    def test_ajuda_do_analyze_traz_exemplo_real(self) -> None:
        resultado = runner.invoke(app, ["analyze", "--help"])
        assert resultado.exit_code == EXIT_OK
        assert "vitrine analyze foto.jpg" in resultado.output

    def test_ajuda_declara_a_unidade_dos_parametros(self) -> None:
        saida = runner.invoke(app, ["analyze", "--help"]).output
        assert "pixels" in saida
        assert "fracao" in saida

    def test_ajuda_avisa_que_contour_e_ruim_em_foto_real(self) -> None:
        """A limitacao do detector padrao precisa estar onde o usuario ve."""
        assert "foto de loja real" in runner.invoke(app, ["analyze", "--help"]).output


class TestExecucaoBemSucedida:
    def test_tabela_no_terminal(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["analyze", str(_gondola(tmp_path))])
        assert resultado.exit_code == EXIT_OK
        assert "6" in resultado.output
        assert "prateleira" in resultado.output

    def test_json_em_stdout_e_valido(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["analyze", str(_gondola(tmp_path)), "--json"])
        assert resultado.exit_code == EXIT_OK
        payload = json.loads(resultado.stdout)
        assert payload["schema_version"] == "1.1"
        assert payload["total_detections"] == 6

    def test_json_nao_se_mistura_com_a_saida_humana(self, tmp_path: Path) -> None:
        """``--json`` precisa ser canalizavel: nada de tabela em stdout."""
        resultado = runner.invoke(app, ["analyze", str(_gondola(tmp_path)), "--json"])
        json.loads(resultado.stdout)

    def test_regioes_nomeadas(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app,
            [
                "analyze",
                str(_gondola(tmp_path)),
                "--json",
                "--cuts",
                "0,0.5,1",
                "--region-names",
                "minha,concorrencia",
            ],
        )
        assert resultado.exit_code == EXIT_OK
        payload = json.loads(resultado.stdout)
        assert [r["region"] for r in payload["regions"]] == ["minha", "concorrencia"]

    def test_grava_os_artefatos(self, tmp_path: Path) -> None:
        saida = tmp_path / "resultado"
        resultado = runner.invoke(app, ["analyze", str(_gondola(tmp_path)), "--out", str(saida)])
        assert resultado.exit_code == EXIT_OK
        assert (saida / "gondola.json").is_file()
        assert (saida / "gondola.anotada.jpg").is_file()
        payload = json.loads((saida / "gondola.json").read_text(encoding="utf-8"))
        assert payload["shelf_count"] == 2

    def test_perspectiva_pelos_quatro_pontos(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app,
            [
                "analyze",
                str(_gondola(tmp_path)),
                "--json",
                "--perspective",
                "10,10",
                "170,10",
                "170,160",
                "10,160",
            ],
        )
        assert resultado.exit_code == EXIT_OK
        assert json.loads(resultado.stdout)["image"]["rectified"] is True

    def test_imagem_sem_produto_orienta_o_usuario(self, tmp_path: Path) -> None:
        caminho = write_image(tmp_path / "vazia.png", np.zeros((120, 120, 3), dtype=np.uint8))
        resultado = runner.invoke(app, ["analyze", str(caminho)])
        assert resultado.exit_code == EXIT_OK
        assert "Nenhum produto detectado" in resultado.output
        assert "--detector yolo" in resultado.output


class TestInversaoDePolaridade:
    """Produto escuro em fundo claro e o caso comum; sem --invert nao ha deteccao."""

    @staticmethod
    def _gondola_escura_em_fundo_claro(tmp_path: Path) -> Path:
        pixels, _ = synthetic_shelf(rows=2, columns=3)
        invertida = 255 - pixels
        return write_image(tmp_path / "clara.png", invertida)

    def test_sem_invert_nao_encontra_nada(self, tmp_path: Path) -> None:
        caminho = self._gondola_escura_em_fundo_claro(tmp_path)
        resultado = runner.invoke(app, ["analyze", str(caminho), "--json"])
        assert json.loads(resultado.stdout)["status"] == "no_detections"

    def test_com_invert_encontra_os_produtos(self, tmp_path: Path) -> None:
        caminho = self._gondola_escura_em_fundo_claro(tmp_path)
        resultado = runner.invoke(app, ["analyze", str(caminho), "--invert", "--json"])
        payload = json.loads(resultado.stdout)
        assert payload["status"] == "ok"
        assert payload["total_detections"] == 6

    def test_a_dica_de_saida_vazia_menciona_invert(self, tmp_path: Path) -> None:
        caminho = self._gondola_escura_em_fundo_claro(tmp_path)
        resultado = runner.invoke(app, ["analyze", str(caminho)])
        assert "--invert" in resultado.output

    def test_invert_nao_se_aplica_ao_yolo(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app, ["analyze", str(_gondola(tmp_path)), "--detector", "yolo", "--invert"]
        )
        assert resultado.exit_code == EXIT_USAGE
        assert "contorno" in resultado.output


class TestErrosDeUso:
    """Codigo 1, e sempre com uma dica acionavel."""

    def test_detector_desconhecido(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["analyze", str(_gondola(tmp_path)), "--detector", "magico"])
        assert resultado.exit_code == EXIT_USAGE
        assert "contour" in resultado.output

    def test_weights_com_detector_sem_modelo(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["analyze", str(_gondola(tmp_path)), "--weights", "peso.pt"])
        assert resultado.exit_code == EXIT_USAGE
        assert "--detector yolo" in resultado.output

    def test_cuts_nao_numericos(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["analyze", str(_gondola(tmp_path)), "--cuts", "a,b"])
        assert resultado.exit_code == EXIT_USAGE
        assert "--cuts 0,0.4,1" in resultado.output

    def test_cuts_que_nao_formam_particao(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["analyze", str(_gondola(tmp_path)), "--cuts", "0,0.5,0.9"])
        assert resultado.exit_code == EXIT_USAGE
        assert "terminar em 1" in resultado.output

    def test_nomes_sem_cuts(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app, ["analyze", str(_gondola(tmp_path)), "--region-names", "a,b"]
        )
        assert resultado.exit_code == EXIT_USAGE
        assert "--cuts" in resultado.output

    def test_nomes_em_quantidade_errada(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app,
            ["analyze", str(_gondola(tmp_path)), "--cuts", "0,0.5,1", "--region-names", "so_um"],
        )
        assert resultado.exit_code == EXIT_USAGE
        assert "2 nomes" in resultado.output

    def test_ponto_de_perspectiva_malformado(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app,
            [
                "analyze",
                str(_gondola(tmp_path)),
                "--perspective",
                "10",
                "170,10",
                "170,160",
                "10,160",
            ],
        )
        assert resultado.exit_code == EXIT_USAGE
        assert "x,y" in resultado.output

    def test_extent_invertido(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["analyze", str(_gondola(tmp_path)), "--extent", "900,100"])
        assert resultado.exit_code == EXIT_USAGE
        assert "x_max maior que x_min" in resultado.output


class TestFalhasDeProcessamento:
    """Codigo 2: a entrada era plausivel, o processamento e que falhou."""

    def test_imagem_inexistente(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["analyze", str(tmp_path / "fantasma.jpg")])
        assert resultado.exit_code == EXIT_FAILURE
        assert "nao encontrada" in resultado.output

    def test_perspectiva_fora_da_imagem(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app,
            [
                "analyze",
                str(_gondola(tmp_path)),
                "--perspective",
                "0,0",
                "9000,0",
                "9000,9000",
                "0,9000",
            ],
        )
        assert resultado.exit_code == EXIT_FAILURE
        assert "fora da imagem" in resultado.output


class TestTratamentoDeErro:
    def test_nunca_mostra_stack_trace(self, tmp_path: Path) -> None:
        resultado = runner.invoke(
            app,
            ["analyze", str(tmp_path / "fantasma.jpg"), "--log-file", str(tmp_path / "v.log")],
        )
        assert "Traceback" not in resultado.output
        assert 'File "' not in resultado.output

    def test_o_traceback_completo_vai_para_o_log(self, tmp_path: Path) -> None:
        log = tmp_path / "vitrine.log"
        runner.invoke(app, ["analyze", str(tmp_path / "fantasma.jpg"), "--log-file", str(log)])
        assert log.is_file()
        assert "Traceback" in log.read_text(encoding="utf-8")

    def test_toda_mensagem_de_erro_traz_uma_dica(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["analyze", str(tmp_path / "fantasma.jpg")])
        assert "erro:" in resultado.output
        assert "->" in resultado.output

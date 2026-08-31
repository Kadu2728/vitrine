"""Testes do carregamento de dataset e do comando de avaliacao.

O conjunto de validacao e gerado em runtime: retangulos desenhados em posicoes
conhecidas e um arquivo de rotulos derivado das mesmas contas. Como o
``ContourDetector`` recupera exatamente esses retangulos, a avaliacao sobre esse
conjunto tem resultado esperado igual a 1.0 -- o que torna o teste uma
verificacao da maquina de medicao, nao do detector.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from helpers import synthetic_shelf, write_image
from vitrine.cli import EXIT_FAILURE, EXIT_OK, app
from vitrine.errors import VitrineError
from vitrine.eval.dataset import Sample, load_split

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def build_dataset(root: Path, *, images: int = 3, split: str = "val") -> Path:
    """Monta um dataset YOLO sintetico com anotacoes exatas."""
    pasta_imagens = root / "images" / split
    pasta_rotulos = root / "labels" / split
    pasta_imagens.mkdir(parents=True)
    pasta_rotulos.mkdir(parents=True)

    for indice in range(images):
        pixels, caixas = synthetic_shelf(rows=2, columns=3)
        altura, largura = pixels.shape[:2]
        write_image(pasta_imagens / f"foto_{indice:03d}.png", pixels)
        linhas = [
            f"0 {(b.x1 + b.x2) / 2 / largura:.6f} {(b.y1 + b.y2) / 2 / altura:.6f} "
            f"{b.width / largura:.6f} {b.height / altura:.6f}"
            for b in caixas
        ]
        (pasta_rotulos / f"foto_{indice:03d}.txt").write_text(
            "\n".join(linhas) + "\n", encoding="utf-8"
        )
    return root


class TestLoadSplit:
    def test_carrega_imagens_e_rotulos(self, tmp_path: Path) -> None:
        raiz = build_dataset(tmp_path / "ds", images=3)
        amostras = load_split(raiz, "val")
        assert len(amostras) == 3
        assert all(len(a.normalized) == 6 for a in amostras)

    def test_ordem_e_estavel(self, tmp_path: Path) -> None:
        raiz = build_dataset(tmp_path / "ds", images=4)
        nomes = [a.image_path.name for a in load_split(raiz, "val")]
        assert nomes == sorted(nomes)

    def test_imagem_sem_rotulo_e_anotacao_vazia_valida(self, tmp_path: Path) -> None:
        raiz = build_dataset(tmp_path / "ds", images=1)
        (raiz / "labels" / "val" / "foto_000.txt").unlink()
        assert load_split(raiz, "val")[0].normalized == ()

    def test_pasta_de_imagens_ausente(self, tmp_path: Path) -> None:
        with pytest.raises(VitrineError, match="pasta de imagens") as exc:
            load_split(tmp_path, "val")
        assert "formato YOLO" in exc.value.hint

    def test_pasta_de_rotulos_ausente(self, tmp_path: Path) -> None:
        (tmp_path / "images" / "val").mkdir(parents=True)
        with pytest.raises(VitrineError, match="pasta de rotulos"):
            load_split(tmp_path, "val")

    def test_split_sem_imagem(self, tmp_path: Path) -> None:
        (tmp_path / "images" / "val").mkdir(parents=True)
        (tmp_path / "labels" / "val").mkdir(parents=True)
        with pytest.raises(VitrineError, match="nenhuma imagem"):
            load_split(tmp_path, "val")

    def test_linha_em_branco_e_ignorada(self, tmp_path: Path) -> None:
        raiz = build_dataset(tmp_path / "ds", images=1)
        rotulo = raiz / "labels" / "val" / "foto_000.txt"
        rotulo.write_text("\n0 0.5 0.5 0.1 0.1\n\n   \n", encoding="utf-8")
        assert len(load_split(raiz, "val")[0].normalized) == 1

    def test_rotulo_com_campos_faltando(self, tmp_path: Path) -> None:
        raiz = build_dataset(tmp_path / "ds", images=1)
        (raiz / "labels" / "val" / "foto_000.txt").write_text("0 0.5 0.5\n", encoding="utf-8")
        with pytest.raises(VitrineError, match="campo"):
            load_split(raiz, "val")

    def test_rotulo_nao_numerico(self, tmp_path: Path) -> None:
        raiz = build_dataset(tmp_path / "ds", images=1)
        (raiz / "labels" / "val" / "foto_000.txt").write_text("0 a b c d\n", encoding="utf-8")
        with pytest.raises(VitrineError, match="nao numericas"):
            load_split(raiz, "val")


class TestSampleBoxes:
    def test_converte_normalizado_para_pixel(self) -> None:
        amostra = Sample(image_path=None, normalized=((0.5, 0.5, 0.5, 0.25),))  # type: ignore[arg-type]
        (caixa,) = amostra.boxes(width=200, height=400)
        assert (caixa.x1, caixa.x2) == (50.0, 150.0)
        assert (caixa.y1, caixa.y2) == (150.0, 250.0)

    def test_acompanha_a_reducao_da_imagem(self) -> None:
        """O ponto do formato normalizado: metade do tamanho, metade das coordenadas."""
        amostra = Sample(image_path=None, normalized=((0.5, 0.5, 0.5, 0.25),))  # type: ignore[arg-type]
        grande = amostra.boxes(width=200, height=400)[0]
        pequena = amostra.boxes(width=100, height=200)[0]
        assert pequena.x1 == grande.x1 / 2
        assert pequena.width == grande.width / 2

    def test_caixa_degenerada_e_descartada(self) -> None:
        amostra = Sample(image_path=None, normalized=((0.5, 0.5, 0.0, 0.5),))  # type: ignore[arg-type]
        assert amostra.boxes(width=100, height=100) == ()


class TestComandoBenchmark:
    def test_avaliacao_de_ponta_a_ponta(self, tmp_path: Path) -> None:
        raiz = build_dataset(tmp_path / "ds", images=2)
        resultado = runner.invoke(app, ["benchmark", str(raiz), "--detector", "contour", "--json"])
        assert resultado.exit_code == EXIT_OK
        payload = json.loads(resultado.stdout)
        assert payload["images"] == 2
        assert payload["ground_truth"] == 12
        # O detector por contorno recupera exatamente os retangulos desenhados,
        # entao a maquina de medicao precisa reportar acerto total.
        assert payload["precision"] == 1.0
        assert payload["recall"] == 1.0
        assert payload["average_precision"] == 1.0

    def test_tabela_no_terminal(self, tmp_path: Path) -> None:
        raiz = build_dataset(tmp_path / "ds", images=1)
        resultado = runner.invoke(app, ["benchmark", str(raiz), "--detector", "contour"])
        assert resultado.exit_code == EXIT_OK
        assert "Recall" in resultado.output
        assert "AP@0.5" in resultado.output

    def test_limit_reduz_o_conjunto(self, tmp_path: Path) -> None:
        raiz = build_dataset(tmp_path / "ds", images=4)
        resultado = runner.invoke(
            app, ["benchmark", str(raiz), "--detector", "contour", "--limit", "2", "--json"]
        )
        assert json.loads(resultado.stdout)["images"] == 2

    def test_dataset_inexistente(self, tmp_path: Path) -> None:
        resultado = runner.invoke(app, ["benchmark", str(tmp_path / "nada")])
        assert resultado.exit_code == EXIT_FAILURE
        assert "pasta de imagens" in resultado.output

    def test_ajuda_diz_o_que_acontece_com_numero_ruim(self) -> None:
        assert "inclusive se for ruim" in runner.invoke(app, ["benchmark", "--help"]).output

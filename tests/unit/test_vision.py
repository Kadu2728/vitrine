"""Testes da camada de visao: carregamento, perspectiva e detectores.

Nenhuma imagem versionada. Todas as fixtures sao geradas em runtime e o
resultado esperado sai de uma conta, nao de conferencia visual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import pytest
from PIL import Image

from helpers import synthetic_shelf, write_image
from vitrine import BoundingBox, ContourDetector, Detection, Detector, FakeDetector
from vitrine.errors import DetectorError, ImageLoadError, PerspectiveError
from vitrine.vision.image import load_image
from vitrine.vision.perspective import order_corners, rectify

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


class TestConformidadeComOProtocolo:
    """Toda implementacao precisa satisfazer ``Detector`` estruturalmente."""

    @pytest.mark.parametrize(
        "detector",
        [FakeDetector(), ContourDetector()],
        ids=["fake", "contour"],
    )
    def test_satisfaz_o_protocolo(self, detector: object) -> None:
        assert isinstance(detector, Detector)

    @pytest.mark.parametrize(
        "detector",
        [FakeDetector(), ContourDetector()],
        ids=["fake", "contour"],
    )
    def test_info_e_serializavel(self, detector: Detector) -> None:
        payload = detector.info.model_dump_json()
        assert "name" in payload


class TestFakeDetector:
    def test_devolve_o_que_foi_configurado(self) -> None:
        deteccoes = (Detection(box=BoundingBox(x1=0, y1=0, x2=10, y2=10)),)
        detector = FakeDetector(deteccoes)
        imagem = np.zeros((5, 5, 3), dtype=np.uint8)
        assert detector.detect(imagem) == deteccoes

    def test_from_grid_tem_geometria_calculavel(self) -> None:
        detector = FakeDetector.from_grid(
            rows=2, columns=3, box_width=60, box_height=90, gap_x=20, gap_y=60, origin=(20, 20)
        )
        deteccoes = detector.detect(np.zeros((1, 1, 3), dtype=np.uint8))
        assert len(deteccoes) == 6
        # Primeiro produto da segunda prateleira: x = 20, y = 20 + 90 + 60 = 170.
        segunda = [d for d in deteccoes if d.box.y1 == 170.0]
        assert len(segunda) == 3
        assert segunda[0].box.x1 == 20.0

    def test_grid_invalido(self) -> None:
        with pytest.raises(ValueError, match="positivos"):
            FakeDetector.from_grid(rows=0, columns=3)


class TestContourDetector:
    def test_recupera_as_caixas_exatas(self) -> None:
        imagem, esperadas = synthetic_shelf(rows=2, columns=3)
        achadas = ContourDetector().detect(imagem)
        assert [d.box for d in achadas] == sorted(esperadas, key=lambda b: (b.x1, b.y1))

    def test_saida_ordenada_e_deterministica(self) -> None:
        imagem, _ = synthetic_shelf(rows=3, columns=4)
        detector = ContourDetector()
        assert detector.detect(imagem) == detector.detect(imagem)

    def test_imagem_vazia_nao_gera_deteccao(self) -> None:
        vazia = np.zeros((100, 100, 3), dtype=np.uint8)
        assert ContourDetector().detect(vazia) == ()

    def test_filtro_de_area_descarta_o_que_e_grande_demais(self) -> None:
        imagem = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(imagem, (2, 2), (97, 97), (255, 255, 255), thickness=-1)
        # Ocupa mais de 50% da imagem: e fundo ou gondola, nao produto.
        assert ContourDetector().detect(imagem) == ()

    def test_filtro_de_proporcao_descarta_prateleira(self) -> None:
        imagem = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.rectangle(imagem, (10, 100), (390, 104), (255, 255, 255), thickness=-1)
        assert ContourDetector().detect(imagem) == ()

    def test_recusa_imagem_que_nao_e_bgr(self) -> None:
        with pytest.raises(DetectorError, match="3 canais"):
            ContourDetector().detect(np.zeros((10, 10), dtype=np.uint8))

    @pytest.mark.parametrize(
        ("kwargs", "trecho"),
        [
            ({"min_area_ratio": 0.6, "max_area_ratio": 0.5}, "Faixa de area"),
            ({"max_aspect": 0.5}, "max_aspect"),
        ],
    )
    def test_parametros_invalidos(self, kwargs: dict[str, float], trecho: str) -> None:
        with pytest.raises(DetectorError, match=trecho):
            ContourDetector(**kwargs)  # type: ignore[arg-type]


class TestLoadImage:
    def test_carrega_e_converte_para_bgr(self, tmp_path: Path) -> None:
        pixels, _ = synthetic_shelf(rows=1, columns=2)
        caminho = write_image(tmp_path / "gondola.png", pixels)
        carregada = load_image(caminho)
        assert carregada.name == "gondola.png"
        assert carregada.pixels.shape == pixels.shape
        assert carregada.downscale == 1.0
        assert carregada.exif_rotated is False

    def test_reduz_imagem_grande_e_registra_o_fator(self, tmp_path: Path) -> None:
        grande = np.zeros((400, 800, 3), dtype=np.uint8)
        caminho = write_image(tmp_path / "grande.png", grande)
        carregada = load_image(caminho, max_size=200)
        assert carregada.width == 200
        assert carregada.height == 100
        assert carregada.downscale == pytest.approx(0.25)

    def test_max_size_none_mantem_o_tamanho(self, tmp_path: Path) -> None:
        grande = np.zeros((400, 800, 3), dtype=np.uint8)
        caminho = write_image(tmp_path / "grande.png", grande)
        assert load_image(caminho, max_size=None).width == 800

    def test_imagem_menor_que_o_limite_nao_e_ampliada(self, tmp_path: Path) -> None:
        pequena = np.zeros((50, 50, 3), dtype=np.uint8)
        caminho = write_image(tmp_path / "pequena.png", pequena)
        carregada = load_image(caminho, max_size=2000)
        assert (carregada.width, carregada.height) == (50, 50)
        assert carregada.downscale == 1.0

    def test_orientacao_exif_e_aplicada(self, tmp_path: Path) -> None:
        """Foto de celular deitada com metadado de rotacao.

        Se a orientacao for ignorada, a gondola chega deitada e o agrupamento
        por centro vertical produz lixo sem levantar erro nenhum. E a falha
        silenciosa mais cara do sistema, e por isso ela tem teste proprio.
        """
        caminho = tmp_path / "girada.jpg"
        imagem = Image.new("RGB", (120, 60), color=(10, 20, 30))
        exif = imagem.getexif()
        exif[274] = 6  # Orientation: girar 90 graus no sentido horario.
        imagem.save(caminho, exif=exif)

        carregada = load_image(caminho)
        assert carregada.exif_rotated is True
        # Largura e altura trocam de lugar.
        assert (carregada.width, carregada.height) == (60, 120)

    def test_arquivo_inexistente(self, tmp_path: Path) -> None:
        with pytest.raises(ImageLoadError, match="nao encontrada") as exc:
            load_image(tmp_path / "nao_existe.jpg")
        assert "Verifique o caminho" in exc.value.hint

    def test_pasta_em_vez_de_arquivo(self, tmp_path: Path) -> None:
        with pytest.raises(ImageLoadError, match="nao e um arquivo") as exc:
            load_image(tmp_path)
        assert "batch" in exc.value.hint

    def test_arquivo_que_nao_e_imagem(self, tmp_path: Path) -> None:
        caminho = tmp_path / "texto.jpg"
        caminho.write_text("isto nao e uma imagem", encoding="utf-8")
        with pytest.raises(ImageLoadError, match="decodificar"):
            load_image(caminho)

    def test_arquivo_truncado(self, tmp_path: Path) -> None:
        pixels, _ = synthetic_shelf(rows=2, columns=3)
        caminho = write_image(tmp_path / "inteira.png", pixels)
        bruto = caminho.read_bytes()
        caminho.write_bytes(bruto[: len(bruto) // 2])
        with pytest.raises(ImageLoadError):
            load_image(caminho)

    def test_limite_de_megapixels(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("vitrine.vision.image.ABSOLUTE_MAX_PIXELS", 100)
        pixels, _ = synthetic_shelf(rows=1, columns=1)
        caminho = write_image(tmp_path / "grande.png", pixels)
        with pytest.raises(ImageLoadError, match="megapixels") as exc:
            load_image(caminho)
        assert "Redimensione" in exc.value.hint

    def test_max_size_invalido(self, tmp_path: Path) -> None:
        pixels, _ = synthetic_shelf(rows=1, columns=1)
        caminho = write_image(tmp_path / "g.png", pixels)
        with pytest.raises(ImageLoadError, match="max_size"):
            load_image(caminho, max_size=0)


class TestOrderCorners:
    def test_ordena_a_partir_de_qualquer_entrada(self) -> None:
        cantos = ((0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0))
        esperado = cantos
        for rotacao in range(4):
            girado = cantos[rotacao:] + cantos[:rotacao]
            assert order_corners(girado) == esperado  # type: ignore[arg-type]

    def test_quadrilatero_irregular(self) -> None:
        cantos = ((10.0, 12.0), (90.0, 4.0), (96.0, 70.0), (4.0, 78.0))
        superior_esquerdo, superior_direito, inferior_direito, inferior_esquerdo = order_corners(
            cantos
        )
        assert superior_esquerdo == (10.0, 12.0)
        assert superior_direito == (90.0, 4.0)
        assert inferior_direito == (96.0, 70.0)
        assert inferior_esquerdo == (4.0, 78.0)


class TestRectify:
    def test_identidade_preserva_a_imagem(self) -> None:
        pixels, _ = synthetic_shelf(rows=1, columns=2)
        altura, largura = pixels.shape[:2]
        cantos = (
            (0.0, 0.0),
            (float(largura - 1), 0.0),
            (float(largura - 1), float(altura - 1)),
            (0.0, float(altura - 1)),
        )
        resultado = rectify(pixels, cantos)
        assert resultado.shape[:2] == (altura - 1, largura - 1)

    def test_desfaz_uma_distorcao_conhecida(self) -> None:
        """Aplica uma homografia conhecida e verifica que rectify a desfaz.

        O retangulo desenhado volta a ocupar aproximadamente a mesma fracao da
        imagem que ocupava antes da distorcao -- a checagem certa aqui e de
        area relativa, porque a reamostragem nao devolve pixels identicos.
        """
        original = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(original, (60, 40), (240, 160), (255, 255, 255), thickness=-1)
        area_original = float((original[:, :, 0] > 127).sum()) / original[:, :, 0].size

        origem = np.array([[0, 0], [299, 0], [299, 199], [0, 199]], dtype=np.float32)
        destino = np.array([[30, 10], [270, 0], [299, 199], [0, 180]], dtype=np.float32)
        matriz = cv2.getPerspectiveTransform(origem, destino)
        distorcida = cast("NDArray[np.uint8]", cv2.warpPerspective(original, matriz, (300, 200)))

        cantos = ((30.0, 10.0), (270.0, 0.0), (299.0, 199.0), (0.0, 180.0))
        corrigida = rectify(distorcida, cantos)
        area_corrigida = float((corrigida[:, :, 0] > 127).sum()) / corrigida[:, :, 0].size

        assert area_corrigida == pytest.approx(area_original, abs=0.03)

    def test_tamanho_de_saida_forcado(self) -> None:
        pixels, _ = synthetic_shelf(rows=1, columns=2)
        cantos = ((0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0))
        assert rectify(pixels, cantos, output_size=(120, 90)).shape[:2] == (90, 120)

    def test_pontos_repetidos(self) -> None:
        pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        cantos = ((0.0, 0.0), (0.0, 0.0), (99.0, 99.0), (0.0, 99.0))
        with pytest.raises(PerspectiveError, match="repetidos"):
            rectify(pixels, cantos)

    def test_pontos_fora_da_imagem(self) -> None:
        pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        cantos = ((0.0, 0.0), (500.0, 0.0), (500.0, 99.0), (0.0, 99.0))
        with pytest.raises(PerspectiveError, match="fora da imagem") as exc:
            rectify(pixels, cantos)
        assert "max-size" in exc.value.hint

    def test_pontos_colineares(self) -> None:
        pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        cantos = ((0.0, 50.0), (30.0, 50.0), (60.0, 50.0), (90.0, 50.0))
        with pytest.raises(PerspectiveError, match="colineares"):
            rectify(pixels, cantos)

    def test_area_pequena_demais(self) -> None:
        pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        cantos = ((10.0, 10.0), (14.0, 10.0), (14.0, 14.0), (10.0, 14.0))
        with pytest.raises(PerspectiveError, match="pequena demais"):
            rectify(pixels, cantos)

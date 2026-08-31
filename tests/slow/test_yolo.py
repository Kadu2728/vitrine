"""Testes do detector real. Nao rodam por padrao.

Marcados como ``slow`` porque exigem o extra ``yolo`` instalado e, na primeira
execucao, o download de um peso. Rode com::

    uv run pytest -m slow

Estes testes verificam o **contrato**: que ``YoloDetector`` satisfaz o protocolo,
devolve caixas validas e registra a procedencia. Eles nao verificam qualidade de
deteccao -- isso e trabalho do ``vitrine benchmark``, e o numero vai para
``benchmarks/results.md``, nao para uma assercao aqui. Um teste que afirmasse
"o modelo encontra 5 produtos" estaria fixando um resultado de ML num arquivo de
teste, que e a maneira mais rapida de tornar a suite mentirosa.
"""

from __future__ import annotations

import pytest

from helpers import synthetic_shelf
from vitrine import Detector
from vitrine.errors import DetectorError

pytestmark = pytest.mark.slow

ultralytics = pytest.importorskip(
    "ultralytics",
    reason="extra 'yolo' nao instalado; use: uv pip install 'vitrine-shelf[yolo]'",
)


@pytest.fixture(scope="module")
def detector() -> object:
    from vitrine.vision.yolo import YoloDetector

    return YoloDetector()


class TestContrato:
    def test_satisfaz_o_protocolo(self, detector: object) -> None:
        assert isinstance(detector, Detector)

    def test_info_registra_procedencia(self, detector: Detector) -> None:
        info = detector.info
        assert info.name == "yolo"
        assert info.version
        assert info.weights
        assert 0.0 <= info.confidence_threshold <= 1.0

    def test_devolve_caixas_validas(self, detector: Detector) -> None:
        imagem, _ = synthetic_shelf(rows=2, columns=3, box_width=120, box_height=180)
        deteccoes = detector.detect(imagem)
        # Quantas ele encontra e assunto do benchmark; o que se exige aqui e
        # que tudo que sair seja geometricamente valido e ordenado.
        assert all(d.box.width > 0 and d.box.height > 0 for d in deteccoes)
        assert list(deteccoes) == sorted(deteccoes, key=lambda d: d.sort_key)

    def test_e_deterministico_no_mesmo_dispositivo(self, detector: Detector) -> None:
        imagem, _ = synthetic_shelf(rows=1, columns=4, box_width=120, box_height=180)
        assert detector.detect(imagem) == detector.detect(imagem)


class TestFalhas:
    def test_peso_inexistente(self) -> None:
        from vitrine.vision.yolo import YoloDetector

        with pytest.raises(DetectorError, match="carregar o peso") as exc:
            YoloDetector("peso_que_nao_existe_em_lugar_nenhum.pt")
        assert ".pt" in exc.value.hint

    @pytest.mark.parametrize(("campo", "valor"), [("confidence", 1.5), ("iou", -0.1)])
    def test_limiares_invalidos(self, campo: str, valor: float) -> None:
        from vitrine.vision.yolo import YoloDetector

        with pytest.raises(DetectorError):
            YoloDetector(**{campo: valor})  # type: ignore[arg-type]


class TestPipelineComModeloReal:
    def test_analise_completa(self, detector: Detector, tmp_path: object) -> None:
        from pathlib import Path

        from helpers import write_image
        from vitrine import analyze_image

        assert isinstance(tmp_path, Path)
        imagem, _ = synthetic_shelf(rows=2, columns=3, box_width=120, box_height=180)
        caminho = write_image(tmp_path / "gondola.png", imagem)
        report = analyze_image(caminho, detector).report
        assert report.detector is not None
        assert report.detector.name == "yolo"
        assert report.status in {"ok", "no_detections"}

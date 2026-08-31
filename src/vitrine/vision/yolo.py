"""Detector real, baseado em YOLO via Ultralytics.

Este e o unico modulo do projeto autorizado a mencionar o Ultralytics -- ha
[um teste](../../../tests/unit/test_architecture.py) que falha se qualquer outro
arquivo o citar. O import acontece **dentro** do construtor, nao no topo do
modulo: assim ``import vitrine`` continua custando milissegundos mesmo com o
extra instalado, e quem nao instalou o extra recebe uma mensagem util em vez de
um ``ModuleNotFoundError``.

Instalacao::

    uv pip install 'vitrine-shelf[yolo]'

Sobre determinismo (regra R4): a inferencia em modo de avaliacao nao usa
aleatoriedade, entao repetir a mesma imagem no mesmo dispositivo com o mesmo
peso produz as mesmas caixas. O que **nao** e garantido e a igualdade entre
dispositivos diferentes (CPU contra GPU) ou entre versoes de biblioteca --
aritmetica de ponto flutuante em ordens diferentes. Por isso o relatorio
registra a versao e o hash do peso: dois numeros so sao comparaveis se essa
procedencia bater.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vitrine.domain.models import BoundingBox, Detection, DetectorInfo
from vitrine.errors import DetectorError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

DEFAULT_WEIGHTS = "yolov8n.pt"
"""Peso padrao. Ver a ressalva sobre SKU-110K no README: um peso treinado em
COCO nao e um detector de gondola, e esta escolha e um ponto de partida, nao um
resultado."""

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.5


class YoloDetector:
    """Implementacao de ``Detector`` sobre um modelo YOLO do Ultralytics.

    Args:
        weights: caminho de um arquivo ``.pt`` local ou nome de um peso que o
            Ultralytics saiba resolver.
        confidence: confianca minima para aceitar uma deteccao.
        iou: limiar de IoU do NMS interno do modelo.
        imgsz: lado usado na inferencia; ``None`` deixa o padrao do modelo.
        device: ``"cpu"``, ``"cuda"`` ou ``None`` para deteccao automatica.

    Raises:
        DetectorError: se o extra nao estiver instalado ou o peso nao carregar.
    """

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise DetectorError(
                f"Confianca precisa estar entre 0 e 1; recebido {confidence}.",
                "Use --conf 0.25 como ponto de partida.",
            )
        if not 0.0 <= iou <= 1.0:
            raise DetectorError(
                f"IoU precisa estar entre 0 e 1; recebido {iou}.",
                "Use --iou 0.5 como ponto de partida.",
            )

        self._weights = str(weights)
        self._confidence = confidence
        self._iou = iou
        self._imgsz = imgsz
        self._device = device
        self._version, self._model = self._load()

    def _load(self) -> tuple[str, Any]:
        """Importa o Ultralytics e carrega o peso, traduzindo as falhas."""
        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "O detector YOLO exige o extra 'yolo', que nao esta instalado.",
                "Instale com: uv pip install 'vitrine-shelf[yolo]' "
                "-- ou use --detector contour, que nao precisa de modelo.",
            ) from exc

        try:
            model = YOLO(self._weights)
        except (OSError, ValueError, RuntimeError) as exc:
            raise DetectorError(
                f"Nao consegui carregar o peso {self._weights!r}: {exc}",
                "Confira o caminho do arquivo .pt. Para baixar um peso padrao, "
                "deixe --weights vazio e garanta acesso a internet.",
            ) from exc

        return str(ultralytics.__version__), model

    @property
    def info(self) -> DetectorInfo:
        """Identificacao completa, incluindo o hash do peso quando ele e local."""
        return DetectorInfo(
            name="yolo",
            version=self._version,
            weights=self._weights,
            weights_sha256=_sha256(Path(self._weights)),
            confidence_threshold=self._confidence,
            iou_threshold=self._iou,
        )

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        """Roda a inferencia e converte a saida do modelo em deteccoes.

        Args:
            image: pixels BGR uint8, ja retificados pelo pipeline.

        Returns:
            Deteccoes ordenadas, com a confianca reportada pelo modelo.

        Raises:
            DetectorError: se a inferencia falhar.
        """
        opcoes: dict[str, Any] = {
            "conf": self._confidence,
            "iou": self._iou,
            "verbose": False,
        }
        if self._imgsz is not None:
            opcoes["imgsz"] = self._imgsz
        if self._device is not None:
            opcoes["device"] = self._device

        try:
            resultados = self._model.predict(source=image, **opcoes)
        except (RuntimeError, ValueError) as exc:
            raise DetectorError(
                f"A inferencia do YOLO falhou: {exc}",
                "Se for falta de memoria, reduza --max-size ou force --device cpu.",
            ) from exc

        deteccoes: list[Detection] = []
        for resultado in resultados:
            caixas = getattr(resultado, "boxes", None)
            if caixas is None:
                continue
            coordenadas = caixas.xyxy.tolist()
            confiancas = caixas.conf.tolist()
            for (x1, y1, x2, y2), confianca in zip(coordenadas, confiancas, strict=True):
                # O modelo devolve caixas degeneradas de vez em quando; o
                # dominio recusaria com ValidationError e derrubaria a analise
                # inteira por causa de uma caixa ruim.
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                deteccoes.append(
                    Detection(
                        box=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=min(1.0, max(0.0, float(confianca))),
                    )
                )

        return tuple(sorted(deteccoes, key=lambda d: d.sort_key))


def _sha256(path: Path) -> str | None:
    """Hash do arquivo de peso, ou ``None`` se ele nao for um arquivo local."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()

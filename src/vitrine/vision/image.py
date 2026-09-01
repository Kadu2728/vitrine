"""Carregamento de imagem: EXIF, limite de tamanho e mensagens uteis.

Duas armadilhas moram aqui, e as duas sao silenciosas.

**Orientacao EXIF.** Foto de celular quase sempre vem gravada na horizontal com
um metadado dizendo "gire 90 graus". Bibliotecas que ignoram esse metadado
entregam a gondola deitada -- e o agrupamento em prateleiras, que clusteriza
centros *verticais*, produz lixo com aparencia de resultado. Nao ha erro, nao ha
excecao: so um numero errado. Por isso a orientacao e sempre aplicada, e o
relatorio registra se foi.

**Tamanho.** Uma foto de 12 MP nao melhora a deteccao na mesma proporcao em que
piora o tempo de inferencia. A imagem e reduzida ate caber em ``max_size``, e o
fator aplicado fica registrado -- sem isso, coordenadas de saida nao teriam como
voltar para o referencial da foto original.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from vitrine.errors import ImageLoadError

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

DEFAULT_MAX_SIZE = 2000
"""Maior lado, em pixels, apos a reducao. Acima disto o custo cresce mais que a
qualidade da deteccao."""

ABSOLUTE_MAX_PIXELS = 80_000_000
"""Teto de seguranca contra imagem absurda ou bomba de descompressao."""

EXIF_DATETIME_ORIGINAL = 36867
"""Tag EXIF com o instante em que o obturador disparou."""

_EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


@dataclass(frozen=True)
class LoadedImage:
    """Uma imagem pronta para deteccao, com o historico do que foi feito nela."""

    pixels: NDArray[np.uint8]
    """Pixels em BGR uint8, shape ``(altura, largura, 3)``."""

    name: str
    """Nome do arquivo de origem, sem o caminho."""

    exif_rotated: bool
    """Se a orientacao EXIF precisou ser aplicada."""

    downscale: float
    """Fator aplicado: ``1.0`` significa tamanho original."""

    captured_at: str
    """Instante da captura em ISO 8601.

    Vem do EXIF quando a camera gravou; senao, da data de modificacao do
    arquivo. Historico por ponto de venda se ordena por isto, e nao pela hora
    em que alguem rodou o lote: um lote processado com uma semana de atraso
    embaralharia a serie temporal inteira.
    """

    @property
    def width(self) -> int:
        """Largura em pixels."""
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        """Altura em pixels."""
        return int(self.pixels.shape[0])


def load_image(path: Path, *, max_size: int | None = DEFAULT_MAX_SIZE) -> LoadedImage:
    """Le uma imagem do disco, corrige a orientacao e reduz se necessario.

    Args:
        path: caminho do arquivo.
        max_size: maior lado tolerado; ``None`` desliga a reducao.

    Returns:
        A imagem carregada, com o registro do que foi aplicado.

    Raises:
        ImageLoadError: arquivo ausente, formato nao reconhecido, imagem
            corrompida ou grande demais.
    """
    if not path.exists():
        raise ImageLoadError(
            f"Imagem nao encontrada: {path}",
            "Verifique o caminho. Em nomes com espaco, use aspas.",
        )
    if not path.is_file():
        raise ImageLoadError(
            f"{path} nao e um arquivo.",
            "Para processar uma pasta inteira, use 'vitrine batch' (Fase 3).",
        )

    try:
        with Image.open(path) as raw:
            _guard_pixel_count(path, raw.size)
            captured_at = _captured_at(raw, path)
            oriented = ImageOps.exif_transpose(raw)
            rotated = oriented is not raw and oriented.size != raw.size
            rgb = (oriented if oriented is not None else raw).convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ImageLoadError(
            f"Nao consegui decodificar {path.name} como imagem.",
            "Confirme que o arquivo nao esta corrompido e que a extensao "
            "corresponde ao conteudo (JPEG, PNG, WEBP, BMP ou TIFF).",
        ) from exc
    except OSError as exc:
        raise ImageLoadError(
            f"Falha ao ler {path.name}: {exc}",
            "O arquivo pode estar truncado ou em uso por outro programa.",
        ) from exc

    pixels = cast("NDArray[np.uint8]", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    pixels, downscale = _fit(pixels, max_size)

    return LoadedImage(
        pixels=pixels,
        name=path.name,
        exif_rotated=rotated,
        downscale=downscale,
        captured_at=captured_at,
    )


def _captured_at(imagem: Image.Image, path: Path) -> str:
    """Instante da captura, em ISO 8601.

    Prefere o EXIF, que e o unico registro do momento real da foto. Cai para a
    data de modificacao do arquivo quando a camera nao gravou ou quando o
    metadado esta corrompido -- caso comum em imagem que passou por aplicativo
    de mensagem, que costuma remover o EXIF.
    """
    try:
        bruto = imagem.getexif().get(EXIF_DATETIME_ORIGINAL)
    except (OSError, ValueError, AttributeError):
        bruto = None

    if isinstance(bruto, str):
        try:
            return datetime.strptime(bruto.strip(), _EXIF_DATETIME_FORMAT).isoformat()
        except ValueError:
            pass

    return datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0).isoformat()


def _guard_pixel_count(path: Path, size: tuple[int, int]) -> None:
    """Recusa imagens absurdamente grandes antes de alocar memoria."""
    width, height = size
    if width * height > ABSOLUTE_MAX_PIXELS:
        raise ImageLoadError(
            f"Imagem {width}x{height} excede o limite de "
            f"{ABSOLUTE_MAX_PIXELS // 1_000_000} megapixels.",
            "Redimensione o arquivo antes de processar.",
        )


def _fit(pixels: NDArray[np.uint8], max_size: int | None) -> tuple[NDArray[np.uint8], float]:
    """Reduz a imagem para caber em ``max_size``, devolvendo o fator aplicado.

    Usa ``INTER_AREA``, que e a interpolacao correta para reducao: preserva a
    borda dos produtos melhor que a bilinear e evita serrilhado que o detector
    interpretaria como textura.
    """
    if max_size is None:
        return pixels, 1.0
    if max_size <= 0:
        raise ImageLoadError(
            f"max_size precisa ser positivo; recebido {max_size}.",
            "Use --max-size 2000 ou omita a opcao para o padrao.",
        )

    height, width = pixels.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return pixels, 1.0

    factor = max_size / longest
    new_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    resized = cast("NDArray[np.uint8]", cv2.resize(pixels, new_size, interpolation=cv2.INTER_AREA))
    return resized, factor

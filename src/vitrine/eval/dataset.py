"""Leitura de dataset no formato YOLO, que e o formato do SKU-110K convertido.

Estrutura esperada::

    raiz/
      images/val/foto_0001.jpg
      labels/val/foto_0001.txt

Cada linha de rotulo e ``classe cx cy w h``, com as quatro coordenadas
normalizadas em ``[0, 1]`` relativas ao tamanho da imagem.

**As anotacoes ficam normalizadas ate o ultimo momento**, e sao convertidas para
pixels so quando a imagem e carregada. Nao e detalhe: o pipeline reduz imagens
grandes, e uma anotacao convertida com o tamanho original ficaria deslocada em
relacao as predicoes feitas sobre a imagem reduzida. O IoU despencaria e a
metrica reportaria um detector ruim onde ha apenas um erro de escala.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vitrine.domain.models import BoundingBox
from vitrine.errors import VitrineError

if TYPE_CHECKING:
    from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

_FIELDS_PER_LABEL = 5


@dataclass(frozen=True)
class Sample:
    """Uma imagem do conjunto e as suas anotacoes, ainda normalizadas."""

    image_path: Path
    normalized: tuple[tuple[float, float, float, float], ...]
    """Anotacoes como ``(cx, cy, w, h)`` em fracao do tamanho da imagem."""

    def boxes(self, width: int, height: int) -> tuple[BoundingBox, ...]:
        """Converte as anotacoes para pixels do tamanho informado.

        Args:
            width: largura da imagem efetivamente analisada.
            height: altura da imagem efetivamente analisada.

        Returns:
            As caixas em coordenadas absolutas, descartando as degeneradas.
        """
        caixas: list[BoundingBox] = []
        for cx, cy, w, h in self.normalized:
            x1 = (cx - w / 2) * width
            x2 = (cx + w / 2) * width
            y1 = (cy - h / 2) * height
            y2 = (cy + h / 2) * height
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            caixas.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
        return tuple(caixas)


def load_split(root: Path, split: str) -> tuple[Sample, ...]:
    """Carrega um split do dataset, em ordem estavel de nome de arquivo.

    Args:
        root: raiz do dataset, com ``images/`` e ``labels/`` dentro.
        split: nome do split, por exemplo ``val``.

    Returns:
        As amostras ordenadas por nome, para que a avaliacao seja reproduzivel.

    Raises:
        VitrineError: se a estrutura de pastas nao existir ou estiver vazia.
    """
    imagens = root / "images" / split
    rotulos = root / "labels" / split

    if not imagens.is_dir():
        raise VitrineError(
            f"Nao encontrei a pasta de imagens {imagens}.",
            "O dataset precisa estar no formato YOLO: raiz/images/<split> e "
            "raiz/labels/<split>. Confira --dataset e --split.",
        )
    if not rotulos.is_dir():
        raise VitrineError(
            f"Nao encontrei a pasta de rotulos {rotulos}.",
            "Cada imagem precisa de um .txt de mesmo nome em raiz/labels/<split>.",
        )

    arquivos = sorted(p for p in imagens.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not arquivos:
        raise VitrineError(
            f"A pasta {imagens} nao tem nenhuma imagem reconhecida.",
            f"Formatos aceitos: {', '.join(IMAGE_SUFFIXES)}.",
        )

    return tuple(
        Sample(image_path=arquivo, normalized=_read_labels(rotulos / f"{arquivo.stem}.txt"))
        for arquivo in arquivos
    )


def _read_labels(path: Path) -> tuple[tuple[float, float, float, float], ...]:
    """Le um arquivo de rotulos YOLO.

    Arquivo ausente significa imagem sem nenhum produto, que e uma anotacao
    valida -- e a convencao do proprio formato. Arquivo malformado, nao: isso e
    erro, e silenciar seria inflar artificialmente a precisao do detector.
    """
    if not path.is_file():
        return ()

    anotacoes: list[tuple[float, float, float, float]] = []
    for numero, linha in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        conteudo = linha.strip()
        if not conteudo:
            continue
        partes = conteudo.split()
        if len(partes) < _FIELDS_PER_LABEL:
            raise VitrineError(
                f"{path.name}, linha {numero}: esperava 'classe cx cy w h', "
                f"encontrei {len(partes)} campo(s).",
                "Cada linha precisa de cinco numeros separados por espaco.",
            )
        try:
            cx, cy, w, h = (float(valor) for valor in partes[1:5])
        except ValueError as exc:
            raise VitrineError(
                f"{path.name}, linha {numero}: coordenadas nao numericas.",
                "As quatro coordenadas sao fracoes de 0 a 1.",
            ) from exc
        anotacoes.append((cx, cy, w, h))

    return tuple(anotacoes)

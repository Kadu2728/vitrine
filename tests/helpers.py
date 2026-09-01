"""Construtores curtos usados pelos testes.

Ficam num modulo proprio, e nao no ``conftest``, para que os testes os importem
explicitamente em vez de depender do mecanismo de descoberta do pytest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from vitrine import BoundingBox, Detection

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    """Atalho para montar caixas nos testes."""
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def detection(x1: float, y1: float, x2: float, y2: float, confidence: float = 1.0) -> Detection:
    """Atalho para montar deteccoes nos testes."""
    return Detection(box=box(x1, y1, x2, y2), confidence=confidence)


def synthetic_shelf(
    *,
    rows: int = 2,
    columns: int = 3,
    box_width: int = 40,
    box_height: int = 60,
    gap_x: int = 20,
    gap_y: int = 30,
    margin: int = 15,
) -> tuple[NDArray[np.uint8], list[BoundingBox]]:
    """Desenha uma gondola sintetica e devolve as caixas esperadas.

    Retangulos brancos sobre fundo preto, em posicoes calculadas por uma conta
    de uma linha. Cumpre a regra de nao versionar imagem pesada: a fixture e
    gerada em runtime e o resultado correto e conhecido matematicamente, nao
    conferido de olho.

    O retangulo vai de ``(x, y)`` a ``(x + w - 1, y + h - 1)`` inclusive, que e
    exatamente o que ``cv2.boundingRect`` devolve como ``(x, y, w, h)``.
    """
    largura = margin * 2 + columns * box_width + (columns - 1) * gap_x
    altura = margin * 2 + rows * box_height + (rows - 1) * gap_y
    imagem: NDArray[np.uint8] = np.zeros((altura, largura, 3), dtype=np.uint8)

    esperadas: list[BoundingBox] = []
    for row in range(rows):
        for column in range(columns):
            x = margin + column * (box_width + gap_x)
            y = margin + row * (box_height + gap_y)
            cv2.rectangle(
                imagem,
                (x, y),
                (x + box_width - 1, y + box_height - 1),
                (255, 255, 255),
                thickness=-1,
            )
            esperadas.append(
                BoundingBox(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + box_width),
                    y2=float(y + box_height),
                )
            )

    return imagem, esperadas


def write_image(path: Path, pixels: NDArray[np.uint8]) -> Path:
    """Grava pixels BGR num arquivo, para exercitar o carregamento de verdade."""
    if not cv2.imwrite(str(path), pixels):
        raise RuntimeError(f"nao consegui gravar {path}")
    return path


def corrido(texto: str) -> str:
    """Normaliza o espaco em branco da saida do Rich.

    O Rich quebra linha conforme a largura do terminal, entao uma frase pode
    chegar partida no meio -- e a largura muda entre a maquina de quem
    desenvolve e o runner da CI, onde os caminhos temporarios sao mais longos.
    Assertiva sobre texto de saida precisa ser imune a isso, senao o teste mede
    o terminal em vez do comportamento.
    """
    return " ".join(texto.split())

"""Gera os artefatos de demonstracao a partir de uma execucao real.

Nada aqui e mockup: a imagem de entrada e sintetica, mas o caminho percorrido e
o mesmo de uma foto de loja -- carregar, detectar, agrupar, medir, desenhar. Os
numeros impressos sao os numeros que o sistema produz.

A gondola sintetica tem uma ruptura plantada na prateleira do meio, porque
mostrar o sistema encontrando o problema e mais util que mostra-lo contando
produto em prateleira cheia.

Uso::

    uv run python examples/demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vitrine import ContourDetector, RegionSet, analyze_image, annotate

if TYPE_CHECKING:
    from numpy.typing import NDArray

AQUI = Path(__file__).parent

FUNDO = (238, 238, 238)
PRATELEIRA = (170, 170, 170)
PRODUTOS: tuple[tuple[int, int, int], ...] = (
    (140, 86, 75),
    (79, 129, 189),
    (89, 158, 96),
    (57, 87, 197),
    (168, 120, 110),
    (60, 160, 200),
)

LARGURA, ALTURA = 900, 620
MARGEM = 40
ALTURA_PRATELEIRA = 170
PRODUTO_W, PRODUTO_H = 74, 120
VAO = 14


def desenhar_gondola() -> NDArray[np.uint8]:
    """Desenha tres prateleiras, com uma ruptura de tres produtos no meio."""
    imagem: NDArray[np.uint8] = np.full((ALTURA, LARGURA, 3), FUNDO, dtype=np.uint8)
    rng = np.random.default_rng(20260831)

    for prateleira in range(3):
        base = MARGEM + prateleira * ALTURA_PRATELEIRA + PRODUTO_H
        cv2.rectangle(
            imagem, (MARGEM - 12, base), (LARGURA - MARGEM + 12, base + 9), PRATELEIRA, -1
        )

        for coluna in range(9):
            # A ruptura: nada nas colunas 3, 4 e 5 da prateleira do meio.
            if prateleira == 1 and coluna in (3, 4, 5):
                continue
            x = MARGEM + coluna * (PRODUTO_W + VAO)
            # Folga de 4 px entre a base do produto e a prateleira: sem ela,
            # o detector por contorno funde a fileira inteira num blob so.
            y = base - PRODUTO_H - 4
            cor = PRODUTOS[(prateleira * 9 + coluna) % len(PRODUTOS)]
            cv2.rectangle(imagem, (x, y), (x + PRODUTO_W, y + PRODUTO_H), cor, -1)
            # Uma faixa mais clara no rotulo, para a embalagem nao ficar chapada.
            faixa = int(rng.integers(24, 40))
            cv2.rectangle(
                imagem,
                (x + 8, y + PRODUTO_H // 2 - faixa // 2),
                (x + PRODUTO_W - 8, y + PRODUTO_H // 2 + faixa // 2),
                tuple(min(255, c + 55) for c in cor),
                -1,
            )
    return imagem


def main() -> None:
    """Gera a entrada, roda a analise e grava os artefatos."""
    entrada = AQUI / "gondola.png"
    cv2.imwrite(str(entrada), desenhar_gondola())

    regioes = RegionSet.from_cuts((0.0, 0.5, 1.0), ("minha_marca", "concorrencia"))
    resultado = analyze_image(
        entrada,
        ContourDetector(invert=True),
        regions=regioes,
        source="LOJA_DEMO",
    )
    cv2.imwrite(str(AQUI / "gondola.anotada.png"), annotate(resultado.pixels, resultado.report))
    (AQUI / "gondola.json").write_text(resultado.report.model_dump_json(indent=2), encoding="utf-8")

    report = resultado.report
    print(f"produtos ......... {report.total_detections}")
    print(f"prateleiras ...... {report.shelf_count}")
    for shelf in report.shelves:
        vazios = ", ".join(f"{g.width:.0f}px ({g.width_ratio:.1f} produtos)" for g in shelf.gaps)
        print(
            f"  prateleira {shelf.index}: {shelf.detection_count} produtos, "
            f"ocupacao {shelf.occupancy:.0%}" + (f", vazio: {vazios}" if vazios else "")
        )
    for share in report.regions:
        print(f"  {share.region}: contagem {share.count_share:.0%} | area {share.linear_share:.0%}")
    for aviso in report.warnings:
        print(f"  aviso: {aviso}")


if __name__ == "__main__":
    main()

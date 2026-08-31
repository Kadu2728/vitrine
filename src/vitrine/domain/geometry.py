"""Primitivas geometricas 1D usadas por todo o dominio.

Este modulo opera sobre floats simples. Nao conhece Pydantic, imagem, modelo
nem unidade de medida: um intervalo pode estar em pixels, em milimetros ou em
coordenadas normalizadas, e o resultado continua correto.

Convencao de coordenadas adotada em todo o projeto (convencao de imagem):
``x`` cresce para a direita, ``y`` cresce para baixo.

A razao de este modulo existir separado e a invariante P6: com caixas
sobrepostas, somar larguras produz ocupacao maior que a prateleira. A resposta
certa e sempre a *uniao* de intervalos, e ela precisa estar num lugar so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

Interval = tuple[float, float]
"""Intervalo fechado-aberto ``[start, end)`` com ``end > start``."""


def length(interval: Interval) -> float:
    """Comprimento de um intervalo; zero se degenerado ou invertido."""
    start, end = interval
    return max(0.0, end - start)


def merge_intervals(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    """Une intervalos sobrepostos ou adjacentes.

    Devolve intervalos disjuntos, ordenados por inicio. Intervalos vazios ou
    invertidos sao descartados. O resultado independe da ordem de entrada, o
    que sustenta a invariante P2.
    """
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return ()

    merged: list[Interval] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            if end > last_end:
                merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return tuple(merged)


def clip(intervals: Iterable[Interval], window: Interval) -> tuple[Interval, ...]:
    """Recorta intervalos a uma janela, descartando o que sobra fora dela."""
    low, high = window
    clipped: list[Interval] = []
    for start, end in intervals:
        new_start, new_end = max(start, low), min(end, high)
        if new_end > new_start:
            clipped.append((new_start, new_end))
    return tuple(clipped)


def covered_length(intervals: Iterable[Interval]) -> float:
    """Comprimento total coberto pela uniao dos intervalos.

    Diferente de ``sum(length(i) for i in intervals)`` sempre que houver
    sobreposicao. E esta diferenca que impede o share de passar de 1.0.
    """
    return sum(length(interval) for interval in merge_intervals(intervals))


def complement(intervals: Iterable[Interval], window: Interval) -> tuple[Interval, ...]:
    """Intervalos livres dentro de ``window``, isto e, os vazios.

    Por construcao o resultado e disjunto, ordenado e nao intersecta nenhum
    intervalo de entrada -- a invariante P7.
    """
    low, high = window
    if high <= low:
        return ()

    occupied = merge_intervals(clip(merge_intervals(intervals), window))
    gaps: list[Interval] = []
    cursor = low
    for start, end in occupied:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < high:
        gaps.append((cursor, high))
    return tuple(gaps)


def overlap_length(first: Interval, second: Interval) -> float:
    """Comprimento da interseccao entre dois intervalos; zero se disjuntos."""
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def median(values: Sequence[float]) -> float:
    """Mediana de uma sequencia nao vazia.

    Implementada aqui em vez de ``statistics.median`` para manter a mensagem de
    erro do dominio e garantir que a ordenacao seja a mesma em toda parte.

    Raises:
        ValueError: se ``values`` estiver vazia.
    """
    if not values:
        raise ValueError("mediana indefinida para sequencia vazia")

    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0

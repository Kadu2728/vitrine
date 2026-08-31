"""Desenha o relatorio sobre a imagem.

Sem front-end, **a imagem anotada e a interface do produto**. Quem olha o
resultado precisa entender em dois segundos o que o sistema viu, e precisa
conseguir discordar: se o sistema errou uma prateleira, isso tem que saltar aos
olhos, nao ficar escondido dentro de um JSON.

Decisoes de leitura visual:

- **Cor por prateleira**, nao por confianca. O agrupamento e a inferencia mais
  fragil do sistema; deixa-lo visivel e o que permite auditar o resultado de
  relance.
- **Vazio em vermelho translucido com faixa diagonal.** Preenchimento solido
  esconderia a prateleira; contorno sozinho se confunde com produto.
- **Regua de regioes no topo**, quando ha mais de uma regiao, para que a divisao
  usada no share seja visivel em vez de implicita.
- **Rodape com os numeros principais**, para que a imagem se sustente sozinha ao
  ser encaminhada num grupo de WhatsApp -- que e como estes arquivos circulam de
  verdade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from vitrine.domain.models import ShareReport

SHELF_COLORS: tuple[tuple[int, int, int], ...] = (
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (213, 94, 0),
    (204, 121, 167),
    (0, 114, 178),
    (230, 159, 0),
)
"""Paleta segura para daltonismo (Okabe-Ito), em BGR, ciclica por indice de
prateleira. Fixa e ordenada: a mesma prateleira recebe sempre a mesma cor."""

GAP_COLOR = (60, 60, 220)
"""Vermelho do vazio, em BGR."""

TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (32, 32, 32)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(
    pixels: NDArray[np.uint8],
    report: ShareReport,
    *,
    show_regions: bool = True,
    show_footer: bool = True,
) -> NDArray[np.uint8]:
    """Devolve uma copia da imagem com o relatorio desenhado por cima.

    Args:
        pixels: imagem BGR uint8 -- a mesma que foi analisada.
        report: relatorio correspondente aquela imagem.
        show_regions: desenha a regua de regioes quando ha mais de uma.
        show_footer: desenha a faixa de resumo no rodape.

    Returns:
        Uma nova imagem; a original nao e modificada.
    """
    canvas: NDArray[np.uint8] = pixels.copy()
    escala = _text_scale(canvas)
    espessura = max(1, round(escala * 2))

    for shelf in report.shelves:
        cor = SHELF_COLORS[shelf.index % len(SHELF_COLORS)]
        for gap in shelf.gaps:
            _draw_gap(canvas, gap.x_start, gap.y_top, gap.x_end, gap.y_bottom)
        _draw_shelf_band(canvas, shelf.y_top, shelf.y_bottom, shelf.extent.x_min, cor)

    for shelf in report.shelves:
        cor = SHELF_COLORS[shelf.index % len(SHELF_COLORS)]
        if show_regions and len(shelf.regions) > 1:
            _draw_region_ruler(canvas, report, shelf.index, cor, escala)

    _draw_boxes(canvas, report, espessura)

    if show_footer:
        _draw_footer(canvas, report, escala)

    return canvas


def _draw_boxes(canvas: NDArray[np.uint8], report: ShareReport, espessura: int) -> None:
    """Desenha as caixas dos produtos, coloridas pela prateleira a que pertencem.

    A cor comunica o agrupamento, que e a inferencia mais fragil do sistema.
    Duas prateleiras com a mesma cor, ou uma prateleira com duas cores, sao um
    erro visivel de relance -- que e exatamente o objetivo.
    """
    altura, largura = canvas.shape[:2]
    for shelf in report.shelves:
        cor = SHELF_COLORS[shelf.index % len(SHELF_COLORS)]
        for caixa in shelf.boxes:
            x0 = _clamp(round(caixa.x1), 0, largura - 1)
            y0 = _clamp(round(caixa.y1), 0, altura - 1)
            x1 = _clamp(round(caixa.x2), 0, largura - 1)
            y1 = _clamp(round(caixa.y2), 0, altura - 1)
            if x1 > x0 and y1 > y0:
                cv2.rectangle(canvas, (x0, y0), (x1, y1), cor, thickness=espessura)


def _draw_shelf_band(
    canvas: NDArray[np.uint8],
    y_top: float,
    y_bottom: float,
    x_min: float,
    cor: tuple[int, int, int],
) -> None:
    """Marca a faixa vertical ocupada pela prateleira com uma barra lateral."""
    altura = canvas.shape[0]
    topo = _clamp(round(y_top), 0, altura - 1)
    base = _clamp(round(y_bottom), 0, altura - 1)
    largura_barra = max(3, canvas.shape[1] // 200)
    inicio = _clamp(round(x_min), 0, canvas.shape[1] - 1)
    cv2.rectangle(canvas, (inicio, topo), (inicio + largura_barra, base), cor, thickness=-1)


def _draw_gap(
    canvas: NDArray[np.uint8],
    x_start: float,
    y_top: float,
    x_end: float,
    y_bottom: float,
) -> None:
    """Preenche o vazio com vermelho translucido e faixa diagonal."""
    altura, largura = canvas.shape[:2]
    x0 = _clamp(round(x_start), 0, largura - 1)
    x1 = _clamp(round(x_end), 0, largura - 1)
    y0 = _clamp(round(y_top), 0, altura - 1)
    y1 = _clamp(round(y_bottom), 0, altura - 1)
    if x1 <= x0 or y1 <= y0:
        return

    sobreposicao = canvas.copy()
    cv2.rectangle(sobreposicao, (x0, y0), (x1, y1), GAP_COLOR, thickness=-1)
    cv2.addWeighted(sobreposicao, 0.28, canvas, 0.72, 0.0, dst=canvas)

    # Hachura a 45 graus, recortada ao retangulo. O recorte e feito pelo
    # cv2.clipLine em vez de na mao: limitar cada coordenada separadamente
    # deforma as diagonais num leque em vez de manter as linhas paralelas.
    altura_vao = y1 - y0
    passo = max(10, altura_vao // 6)
    janela = (x0, y0, x1 - x0, altura_vao)
    for deslocamento in range(x0 - altura_vao, x1, passo):
        dentro, inicio, fim = cv2.clipLine(
            janela, (deslocamento, y0), (deslocamento + altura_vao, y1)
        )
        if dentro:
            cv2.line(canvas, inicio, fim, GAP_COLOR, thickness=1, lineType=cv2.LINE_AA)

    cv2.rectangle(canvas, (x0, y0), (x1, y1), GAP_COLOR, thickness=2)


def _draw_region_ruler(
    canvas: NDArray[np.uint8],
    report: ShareReport,
    shelf_index: int,
    cor: tuple[int, int, int],
    escala: float,
) -> None:
    """Desenha as fronteiras de regiao sobre a faixa da prateleira."""
    shelf = report.shelves[shelf_index]
    altura = canvas.shape[0]
    topo = _clamp(round(shelf.y_top), 0, altura - 1)
    base = _clamp(round(shelf.y_bottom), 0, altura - 1)

    for region in report.params.regions.regions[1:]:
        x = shelf.extent.x_min + region.start * shelf.extent.width
        coluna = _clamp(round(x), 0, canvas.shape[1] - 1)
        for y in range(topo, base, 12):
            cv2.line(canvas, (coluna, y), (coluna, min(base, y + 6)), cor, thickness=1)

    for share, region in zip(shelf.regions, report.params.regions.regions, strict=True):
        centro = shelf.extent.x_min + (region.start + region.end) / 2 * shelf.extent.width
        rotulo = f"{share.region} {share.linear_share:.0%}"
        _draw_label(canvas, rotulo, (round(centro), topo), escala * 0.8, cor)


def _draw_footer(canvas: NDArray[np.uint8], report: ShareReport, escala: float) -> None:
    """Faixa inferior com os numeros que sustentam a imagem sozinha."""
    altura, largura = canvas.shape[:2]
    faixa = max(28, round(altura * 0.06))
    topo = altura - faixa

    sobreposicao = canvas.copy()
    cv2.rectangle(sobreposicao, (0, topo), (largura, altura), PANEL_COLOR, thickness=-1)
    cv2.addWeighted(sobreposicao, 0.78, canvas, 0.22, 0.0, dst=canvas)

    vazios = sum(len(shelf.gaps) for shelf in report.shelves)
    ocupacao = _media_ponderada(report)
    texto = (
        f"{report.total_detections} produtos | {report.shelf_count} prateleiras | "
        f"ocupacao {ocupacao:.0%} | {vazios} vazio(s)"
    )
    cv2.putText(
        canvas,
        texto,
        (12, altura - round(faixa * 0.32)),
        FONT,
        escala * 0.85,
        TEXT_COLOR,
        max(1, round(escala * 1.6)),
        cv2.LINE_AA,
    )


def _draw_label(
    canvas: NDArray[np.uint8],
    texto: str,
    posicao: tuple[int, int],
    escala: float,
    cor: tuple[int, int, int],
) -> None:
    """Texto com tarja atras, para continuar legivel sobre embalagem clara."""
    (largura_texto, altura_texto), linha_base = cv2.getTextSize(
        texto, FONT, escala, max(1, round(escala * 2))
    )
    x = _clamp(posicao[0] - largura_texto // 2, 0, canvas.shape[1] - largura_texto - 1)
    y = _clamp(posicao[1], altura_texto + linha_base + 2, canvas.shape[0] - 1)
    cv2.rectangle(
        canvas,
        (x - 3, y - altura_texto - linha_base - 2),
        (x + largura_texto + 3, y),
        cor,
        thickness=-1,
    )
    cv2.putText(
        canvas,
        texto,
        (x, y - linha_base),
        FONT,
        escala,
        (20, 20, 20),
        max(1, round(escala * 1.6)),
        cv2.LINE_AA,
    )


def _media_ponderada(report: ShareReport) -> float:
    """Ocupacao media das prateleiras, ponderada pela largura de cada uma."""
    if not report.shelves:
        return 0.0
    ocupado = sum(shelf.occupied_length for shelf in report.shelves)
    disponivel = sum(shelf.extent.width for shelf in report.shelves)
    return ocupado / disponivel if disponivel > 0 else 0.0


def _text_scale(canvas: NDArray[np.uint8]) -> float:
    """Escala de fonte proporcional a imagem, para o texto nao sumir nem estourar."""
    return max(0.4, min(1.6, float(canvas.shape[1]) / 1200.0))


def _clamp(valor: int, minimo: int, maximo: int) -> int:
    return max(minimo, min(maximo, valor))

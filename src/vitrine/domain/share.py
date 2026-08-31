"""Share of shelf e ocupacao -- o ponto de entrada do dominio.

Tres metricas distintas, deliberadamente exibidas lado a lado:

``count_share``
    Fracao das *deteccoes* cujo centro cai na regiao. Responde "quantos itens
    meus contra quantos deles".

``linear_share``
    Fracao do *comprimento ocupado* que esta na regiao. Responde "quanto de
    espaco fisico". Calculado sobre a uniao das projecoes horizontais, jamais
    pela soma das larguras -- com caixas sobrepostas a soma passaria de 1.0.

``occupancy``
    Comprimento ocupado dividido pela largura da regiao. Nao e share: nao soma
    1.0 entre regioes. E a unica das tres afirmavel sem definir regioes, e por
    isso e o que o relatorio traz por padrao.

As duas primeiras discordam com frequencia, e chegam a inverter a ordem entre
regioes -- duas embalagens grandes contra tres pequenas. Publicar so uma delas
seria escolher a resposta mais conveniente.

O denominador do share linear e uma decisao, nao um dado: ver ``ShelfExtent``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vitrine.domain import geometry
from vitrine.domain.dedup import DEFAULT_DEDUP_IOU, deduplicate
from vitrine.domain.gaps import DEFAULT_GAP_MIN_WIDTH_RATIO, find_gaps
from vitrine.domain.models import (
    AnalysisParams,
    Detection,
    DetectorInfo,
    ImageMeta,
    RegionSet,
    RegionShare,
    ShareReport,
    Shelf,
    ShelfExtent,
    ShelfReport,
)
from vitrine.domain.shelves import (
    DEFAULT_MAX_SPREAD_RATIO,
    DEFAULT_SHELF_GAP_RATIO,
    group_into_shelves,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

SPREAD_WARNING_RATIO = 1.0
"""Acima disto a prateleira e suspeita: provavel gondola inclinada."""


def analyze_detections(
    detections: Sequence[Detection],
    *,
    regions: RegionSet | None = None,
    extent: tuple[float, float] | None = None,
    shelf_gap_ratio: float = DEFAULT_SHELF_GAP_RATIO,
    max_shelf_spread_ratio: float = DEFAULT_MAX_SPREAD_RATIO,
    gap_min_width_ratio: float = DEFAULT_GAP_MIN_WIDTH_RATIO,
    dedup_iou: float = DEFAULT_DEDUP_IOU,
    source: str | None = None,
    image: ImageMeta | None = None,
    detector: DetectorInfo | None = None,
) -> ShareReport:
    """Transforma deteccoes soltas num relatorio completo de gondola.

    Esta e a fronteira do dominio: nao ha imagem, arquivo nem modelo daqui para
    dentro. A pipeline de visao (Fase 2) apenas produz as deteccoes e chama
    esta funcao.

    Args:
        detections: caixas detectadas, em qualquer ordem.
        regions: particao da prateleira; o padrao e uma regiao unica, caso em
            que os shares valem 1.0 e o numero informativo e ``occupancy``.
        extent: limites horizontais explicitos ``(x_min, x_max)``. Omitido, usa
            o envelope das deteccoes de cada prateleira.
        shelf_gap_ratio: ver ``domain.shelves``.
        max_shelf_spread_ratio: ver ``domain.shelves``.
        gap_min_width_ratio: ver ``domain.gaps``.
        dedup_iou: ver ``domain.dedup``.
        source: identificador livre da origem, ecoado na saida.
        image: procedencia da imagem, preenchida pela camada de visao.
        detector: procedencia das deteccoes, preenchida pela camada de visao.

    Returns:
        O relatorio. Com entrada vazia, ``status`` vale ``no_detections`` e
        todos os shares sao ``0.0`` -- nunca ``NaN``.

    Raises:
        ValueError: se ``extent`` for degenerado ou invertido.
    """
    if extent is not None and extent[1] <= extent[0]:
        raise ValueError(f"extent precisa ter x_max > x_min; recebido {extent!r}")

    region_set = regions if regions is not None else RegionSet.whole()
    params = AnalysisParams(
        shelf_gap_ratio=shelf_gap_ratio,
        max_shelf_spread_ratio=max_shelf_spread_ratio,
        gap_min_width_ratio=gap_min_width_ratio,
        dedup_iou=dedup_iou,
        extent_kind="explicit" if extent is not None else "envelope",
        explicit_extent=extent,
        regions=region_set,
    )

    kept = deduplicate(detections, iou_threshold=dedup_iou)
    duplicates_removed = len(detections) - len(kept)

    if not kept:
        return ShareReport(
            status="no_detections",
            source=source,
            image=image,
            detector=detector,
            total_detections=0,
            duplicates_removed=duplicates_removed,
            shelf_count=0,
            shelves=(),
            regions=tuple(_empty_share(region.name) for region in region_set.regions),
            params=params,
            warnings=("nenhuma deteccao: nao ha o que medir",),
        )

    shelves = group_into_shelves(
        kept,
        gap_ratio=shelf_gap_ratio,
        max_spread_ratio=max_shelf_spread_ratio,
    )
    reports = tuple(
        _analyze_shelf(shelf, region_set, extent, gap_min_width_ratio) for shelf in shelves
    )

    return ShareReport(
        status="ok",
        source=source,
        image=image,
        detector=detector,
        total_detections=len(kept),
        duplicates_removed=duplicates_removed,
        shelf_count=len(reports),
        shelves=reports,
        regions=_aggregate(reports, region_set),
        params=params,
        warnings=_collect_warnings(shelves, reports, extent),
    )


def _analyze_shelf(
    shelf: Shelf,
    region_set: RegionSet,
    extent: tuple[float, float] | None,
    gap_min_width_ratio: float,
) -> ShelfReport:
    """Mede uma prateleira: ocupacao, shares por regiao e vazios."""
    shelf_extent = (
        ShelfExtent(kind="explicit", x_min=extent[0], x_max=extent[1])
        if extent is not None
        else ShelfExtent(kind="envelope", x_min=shelf.x_min, x_max=shelf.x_max)
    )

    occupied_union = geometry.merge_intervals([d.box.x_interval for d in shelf.detections])
    total_occupied = geometry.covered_length(
        geometry.clip(occupied_union, (shelf_extent.x_min, shelf_extent.x_max))
    )

    counts = _count_by_region(shelf, shelf_extent, region_set)
    region_shares: list[RegionShare] = []
    for region in region_set.regions:
        window = shelf_extent.window(region)
        occupied = geometry.covered_length(geometry.clip(occupied_union, window))
        region_shares.append(
            RegionShare(
                region=region.name,
                count=counts[region.name],
                count_share=_ratio(counts[region.name], shelf.count),
                occupied_length=occupied,
                linear_share=_ratio(occupied, total_occupied),
                occupancy=_ratio(occupied, geometry.length(window)),
            )
        )

    return ShelfReport(
        index=shelf.index,
        y_top=shelf.y_top,
        y_bottom=shelf.y_bottom,
        extent=shelf_extent,
        detection_count=shelf.count,
        median_product_width=shelf.median_width,
        median_product_height=shelf.median_height,
        occupied_length=total_occupied,
        occupancy=_ratio(total_occupied, shelf_extent.width),
        spread_ratio=shelf.spread_ratio,
        boxes=tuple(d.box for d in shelf.detections),
        regions=tuple(region_shares),
        gaps=find_gaps(shelf, shelf_extent, min_width_ratio=gap_min_width_ratio),
    )


def _count_by_region(
    shelf: Shelf,
    extent: ShelfExtent,
    region_set: RegionSet,
) -> dict[str, int]:
    """Atribui cada deteccao a exatamente uma regiao, pelo centro horizontal.

    Uma caixa que cruza a fronteira pertence inteira a regiao do seu centro --
    e por isso que a contagem discorda do share linear, que reparte a caixa
    proporcionalmente. As duas leituras sao legitimas e a diferenca esta
    documentada.
    """
    counts = dict.fromkeys((region.name for region in region_set.regions), 0)
    for detection in shelf.detections:
        fraction = (detection.box.center_x - extent.x_min) / extent.width
        counts[region_set.locate(fraction).name] += 1
    return counts


def _aggregate(reports: Sequence[ShelfReport], region_set: RegionSet) -> tuple[RegionShare, ...]:
    """Soma as prateleiras numa visao unica da gondola."""
    total_count = sum(report.detection_count for report in reports)
    total_occupied = sum(report.occupied_length for report in reports)

    shares: list[RegionShare] = []
    for index, region in enumerate(region_set.regions):
        count = sum(report.regions[index].count for report in reports)
        occupied = sum(report.regions[index].occupied_length for report in reports)
        available = sum(geometry.length(report.extent.window(region)) for report in reports)
        shares.append(
            RegionShare(
                region=region.name,
                count=count,
                count_share=_ratio(count, total_count),
                occupied_length=occupied,
                linear_share=_ratio(occupied, total_occupied),
                occupancy=_ratio(occupied, available),
            )
        )
    return tuple(shares)


def _collect_warnings(
    shelves: Sequence[Shelf],
    reports: Sequence[ShelfReport],
    extent: tuple[float, float] | None,
) -> tuple[str, ...]:
    """Avisos sobre confiabilidade do resultado, nao sobre erro de execucao."""
    warnings: list[str] = []
    for report in reports:
        if report.spread_ratio > SPREAD_WARNING_RATIO:
            warnings.append(
                f"prateleira {report.index}: dispersao vertical de "
                f"{report.spread_ratio:.2f} alturas de produto; suspeite de gondola "
                f"inclinada ou perspectiva nao corrigida"
            )
        if report.detection_count == 1:
            warnings.append(
                f"prateleira {report.index}: uma unica deteccao, a largura mediana "
                f"do produto e o proprio item e a deteccao de vazio e pouco confiavel"
            )

    if extent is not None:
        outside = sum(
            1
            for shelf in shelves
            for detection in shelf.detections
            if detection.box.center_x < extent[0] or detection.box.center_x > extent[1]
        )
        if outside:
            warnings.append(
                f"{outside} deteccao(oes) com centro fora do extent informado "
                f"{extent}; foram atribuidas a regiao da borda mais proxima"
            )
    return tuple(warnings)


def _empty_share(name: str) -> RegionShare:
    """Regiao sem nenhuma medida, usada no caso ``no_detections``."""
    return RegionShare(
        region=name,
        count=0,
        count_share=0.0,
        occupied_length=0.0,
        linear_share=0.0,
        occupancy=0.0,
    )


def _ratio(numerator: float, denominator: float) -> float:
    """Divisao segura presa em ``[0, 1]``.

    O ``min`` nao e paranoia: recorte de intervalos em ponto flutuante produz
    ``1.0000000000000002`` com facilidade, e o contrato promete ``<= 1.0``.
    """
    if denominator <= 0.0:
        return 0.0
    return min(1.0, max(0.0, numerator / denominator))

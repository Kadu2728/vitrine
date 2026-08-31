"""Contratos de dados do dominio.

Todos os modelos sao imutaveis (``frozen``) e recusam campos extras. Nenhum
deles conhece imagem, arquivo ou modelo de deteccao: um ``Detection`` e apenas
uma caixa com uma confianca, venha ela do YOLO, de um CSV ou de um teste.

Convencao de coordenadas: ``x`` cresce para a direita, ``y`` cresce para baixo.
As coordenadas nao tem unidade -- pixels, milimetros ou fracoes funcionam
igualmente, desde que consistentes dentro de uma mesma analise.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vitrine.domain import geometry

if TYPE_CHECKING:
    from vitrine.domain.geometry import Interval

SCHEMA_VERSION: Final[Literal["1.0"]] = "1.0"
"""Versao do schema JSON de saida. Muda quando o contrato quebra."""

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class BoundingBox(BaseModel):
    """Caixa delimitadora com largura e altura estritamente positivas.

    Caixas degeneradas sao rejeitadas na construcao: uma caixa de largura zero
    nao representa produto nenhum e envenenaria a mediana de largura usada como
    limiar em ``shelves`` e ``gaps``.
    """

    model_config = _FROZEN

    x1: float = Field(description="Borda esquerda.")
    y1: float = Field(description="Borda superior.")
    x2: float = Field(description="Borda direita, estritamente maior que x1.")
    y2: float = Field(description="Borda inferior, estritamente maior que y1.")

    @model_validator(mode="after")
    def _validate_geometry(self) -> Self:
        for name, value in (("x1", self.x1), ("y1", self.y1), ("x2", self.x2), ("y2", self.y2)):
            if not math.isfinite(value):
                raise ValueError(f"{name} precisa ser um numero finito; recebido {value!r}")
        if self.x2 <= self.x1:
            raise ValueError(f"largura precisa ser positiva; x1={self.x1} x2={self.x2}")
        if self.y2 <= self.y1:
            raise ValueError(f"altura precisa ser positiva; y1={self.y1} y2={self.y2}")
        return self

    @property
    def width(self) -> float:
        """Largura da caixa."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Altura da caixa."""
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        """Coordenada horizontal do centro."""
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        """Coordenada vertical do centro. Base do agrupamento em prateleiras."""
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        """Area da caixa."""
        return self.width * self.height

    @property
    def x_interval(self) -> Interval:
        """Projecao horizontal, usada em share linear e deteccao de vazios."""
        return (self.x1, self.x2)

    @property
    def y_interval(self) -> Interval:
        """Projecao vertical."""
        return (self.y1, self.y2)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over Union com outra caixa, em ``[0, 1]``."""
        intersection = geometry.overlap_length(
            self.x_interval, other.x_interval
        ) * geometry.overlap_length(self.y_interval, other.y_interval)
        if intersection == 0.0:
            return 0.0
        union = self.area + other.area - intersection
        return intersection / union

    def scaled(self, factor: float) -> BoundingBox:
        """Caixa escalada em torno da origem por ``factor > 0``."""
        if factor <= 0.0:
            raise ValueError(f"fator de escala precisa ser positivo; recebido {factor!r}")
        return BoundingBox(
            x1=self.x1 * factor,
            y1=self.y1 * factor,
            x2=self.x2 * factor,
            y2=self.y2 * factor,
        )

    def translated(self, dx: float, dy: float) -> BoundingBox:
        """Caixa deslocada por ``(dx, dy)``."""
        return BoundingBox(x1=self.x1 + dx, y1=self.y1 + dy, x2=self.x2 + dx, y2=self.y2 + dy)


class Detection(BaseModel):
    """Uma deteccao de produto.

    O rotulo e classe unica por decisao de escopo -- este sistema conta e mede
    ocupacao, nao identifica SKU nem marca. O campo existe para manter o
    contrato aberto, nao porque haja classificacao.
    """

    model_config = _FROZEN

    box: BoundingBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    label: str = Field(default="object", min_length=1)

    @property
    def sort_key(self) -> tuple[float, float, float, float, str, float]:
        """Chave de ordenacao total e estavel.

        Garante que duas execucoes sobre o mesmo conjunto de deteccoes -- em
        qualquer ordem de entrada -- produzam a mesma saida (invariantes P2 e
        P8).
        """
        box = self.box
        return (box.x1, box.y1, box.x2, box.y2, self.label, self.confidence)


class Shelf(BaseModel):
    """Um grupo de deteccoes inferido como pertencente a mesma prateleira.

    Prateleira e inferencia, nao dado observado. Ver ``domain.shelves`` para o
    metodo, os parametros e -- principalmente -- onde ele quebra.
    """

    model_config = _FROZEN

    index: int = Field(ge=0, description="Posicao vertical: 0 e a prateleira mais alta.")
    detections: tuple[Detection, ...] = Field(min_length=1)

    @property
    def count(self) -> int:
        """Numero de deteccoes na prateleira."""
        return len(self.detections)

    @property
    def x_min(self) -> float:
        """Borda esquerda da deteccao mais a esquerda."""
        return min(d.box.x1 for d in self.detections)

    @property
    def x_max(self) -> float:
        """Borda direita da deteccao mais a direita."""
        return max(d.box.x2 for d in self.detections)

    @property
    def y_top(self) -> float:
        """Topo da deteccao mais alta."""
        return min(d.box.y1 for d in self.detections)

    @property
    def y_bottom(self) -> float:
        """Base da deteccao mais baixa."""
        return max(d.box.y2 for d in self.detections)

    @property
    def median_width(self) -> float:
        """Largura mediana dos produtos desta prateleira."""
        return geometry.median([d.box.width for d in self.detections])

    @property
    def median_height(self) -> float:
        """Altura mediana dos produtos desta prateleira."""
        return geometry.median([d.box.height for d in self.detections])

    @property
    def occupied_length(self) -> float:
        """Comprimento horizontal ocupado, contando sobreposicao uma vez so."""
        return geometry.covered_length([d.box.x_interval for d in self.detections])

    @property
    def spread_ratio(self) -> float:
        """Dispersao vertical dos centros dividida pela altura mediana.

        Metrica de confianca do agrupamento. Valor alto indica gondola
        inclinada, perspectiva nao corrigida ou produtos de alturas muito
        diferentes -- casos em que o agrupamento e pouco confiavel.
        """
        centers = [d.box.center_y for d in self.detections]
        return (max(centers) - min(centers)) / self.median_height


class Region(BaseModel):
    """Faixa vertical de uma prateleira, expressa em fracao da sua extensao.

    Fracao e nao pixel: assim a mesma definicao de regiao serve para qualquer
    foto, em qualquer resolucao, e para prateleiras de larguras diferentes.
    """

    model_config = _FROZEN

    name: str = Field(min_length=1)
    start: float = Field(ge=0.0, le=1.0)
    end: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if self.end <= self.start:
            raise ValueError(
                f"regiao {self.name!r}: end precisa ser maior que start; "
                f"recebido start={self.start} end={self.end}"
            )
        return self

    @property
    def span(self) -> float:
        """Largura da regiao em fracao da prateleira."""
        return self.end - self.start


class RegionSet(BaseModel):
    """Particao de ``[0, 1]`` em regioes contiguas, sem buraco nem sobreposicao.

    A exigencia de particao total nao e burocracia: sem ela a soma dos shares
    nao vale 1.0 e a palavra "share" perde o sentido. Para medir apenas um
    trecho da gondola, use ``ShelfExtent`` explicito e redefina o que conta como
    prateleira.
    """

    model_config = _FROZEN

    regions: tuple[Region, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_partition(self) -> Self:
        names = [region.name for region in self.regions]
        if len(set(names)) != len(names):
            raise ValueError(f"nomes de regiao precisam ser unicos; recebido {names}")

        cursor = 0.0
        for region in self.regions:
            if not math.isclose(region.start, cursor, abs_tol=1e-9):
                raise ValueError(
                    f"regioes precisam ser contiguas e ordenadas: esperado start={cursor} "
                    f"na regiao {region.name!r}, recebido {region.start}"
                )
            cursor = region.end
        if not math.isclose(cursor, 1.0, abs_tol=1e-9):
            raise ValueError(f"regioes precisam cobrir ate 1.0; cobertura termina em {cursor}")
        return self

    @classmethod
    def whole(cls) -> RegionSet:
        """Particao trivial: a prateleira inteira como uma regiao unica.

        E o padrao. Sem regioes informadas o relatorio traz ocupacao e
        contagem, que sao afirmaveis sem classificacao de SKU; share entre
        partes so faz sentido quando o usuario diz quais sao as partes.
        """
        return cls(regions=(Region(name="total", start=0.0, end=1.0),))

    @classmethod
    def from_cuts(cls, cuts: tuple[float, ...], names: tuple[str, ...] | None = None) -> RegionSet:
        """Constroi a particao a partir de cortes, por exemplo ``(0.0, 0.4, 1.0)``.

        Args:
            cuts: fronteiras crescentes comecando em 0.0 e terminando em 1.0.
            names: nomes das regioes; se omitido, usa ``r1``, ``r2``, ...

        Returns:
            A particao correspondente aos cortes.

        Raises:
            ValueError: se houver menos de dois cortes ou se a quantidade de
                nomes nao bater com a quantidade de regioes.
        """
        if len(cuts) < 2:
            raise ValueError(f"sao necessarios ao menos dois cortes; recebido {cuts}")
        expected = len(cuts) - 1
        if names is None:
            names = tuple(f"r{i + 1}" for i in range(expected))
        if len(names) != expected:
            raise ValueError(f"{expected} regioes exigem {expected} nomes; recebido {len(names)}")
        return cls(
            regions=tuple(
                Region(name=name, start=start, end=end)
                for name, start, end in zip(names, cuts[:-1], cuts[1:], strict=True)
            )
        )

    def locate(self, fraction: float) -> Region:
        """Regiao que contem ``fraction``, com as pontas presas em ``[0, 1]``.

        A fronteira pertence a regiao da direita, exceto em 1.0, que pertence a
        ultima regiao. Isso torna a atribuicao por contagem uma funcao total:
        toda deteccao cai em exatamente uma regiao.
        """
        clamped = min(max(fraction, 0.0), 1.0)
        for region in self.regions:
            if clamped < region.end:
                return region
        return self.regions[-1]


class ShelfExtent(BaseModel):
    """O denominador do share linear, declarado explicitamente.

    Nao existe deteccao da prateleira fisica, portanto ``ocupado / total`` exige
    dizer o que e ``total``. Duas escolhas produzem dois numeros diferentes e
    ambas sao defensaveis:

    - ``envelope``: da borda esquerda do primeiro produto a direita do ultimo.
      Ignora vazio nas extremidades e tende a superestimar a ocupacao.
    - ``explicit``: limites informados por quem conhece a gondola. Enxerga vazio
      nas pontas, ao custo de depender de um parametro humano.

    O relatorio sempre carrega esta escolha, para que o numero seja auditavel.
    """

    model_config = _FROZEN

    kind: Literal["envelope", "explicit"]
    x_min: float
    x_max: float

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if self.x_max <= self.x_min:
            raise ValueError(f"extensao vazia: x_min={self.x_min} x_max={self.x_max}")
        return self

    @property
    def width(self) -> float:
        """Largura da extensao considerada como prateleira."""
        return self.x_max - self.x_min

    def window(self, region: Region) -> Interval:
        """Janela absoluta correspondente a uma regiao."""
        return (
            self.x_min + region.start * self.width,
            self.x_min + region.end * self.width,
        )


class RegionShare(BaseModel):
    """Resultado por regiao, com as duas metricas lado a lado.

    ``count_share`` e ``linear_share`` respondem a perguntas diferentes e
    frequentemente discordam -- inclusive invertendo a ordem entre regioes.
    Exibir apenas uma delas seria escolher a resposta mais conveniente.
    """

    model_config = _FROZEN

    region: str
    count: int = Field(ge=0, description="Deteccoes cujo centro cai na regiao.")
    count_share: float = Field(ge=0.0, le=1.0, description="count / total de deteccoes.")
    occupied_length: float = Field(ge=0.0, description="Comprimento ocupado dentro da regiao.")
    linear_share: float = Field(ge=0.0, le=1.0, description="ocupado / total ocupado.")
    occupancy: float = Field(ge=0.0, le=1.0, description="ocupado / largura da regiao.")


class Gap(BaseModel):
    """Um vazio: intervalo horizontal livre dentro de uma prateleira.

    Vazio nao e ausencia de caixa em termos absolutos, e um intervalo largo o
    bastante em relacao ao produto daquela prateleira. As coordenadas sao
    absolutas para que o vazio seja desenhavel sem recalculo.
    """

    model_config = _FROZEN

    x_start: float
    x_end: float
    y_top: float
    y_bottom: float
    width_ratio: float = Field(gt=0.0, description="Largura do vazio / largura mediana.")

    @property
    def width(self) -> float:
        """Largura absoluta do vazio."""
        return self.x_end - self.x_start


class ShelfReport(BaseModel):
    """Relatorio de uma prateleira."""

    model_config = _FROZEN

    index: int = Field(ge=0)
    y_top: float
    y_bottom: float
    extent: ShelfExtent
    detection_count: int = Field(ge=1)
    median_product_width: float = Field(gt=0.0)
    median_product_height: float = Field(gt=0.0)
    occupied_length: float = Field(ge=0.0)
    occupancy: float = Field(ge=0.0, le=1.0)
    spread_ratio: float = Field(ge=0.0)
    regions: tuple[RegionShare, ...] = Field(min_length=1)
    gaps: tuple[Gap, ...]


class AnalysisParams(BaseModel):
    """Parametros usados na analise, ecoados na saida.

    Sem isto o JSON nao e auditavel: dois relatorios com numeros diferentes
    poderiam vir do mesmo conjunto de deteccoes apenas por um limiar distinto.
    """

    model_config = _FROZEN

    shelf_gap_ratio: float = Field(gt=0.0)
    max_shelf_spread_ratio: float = Field(gt=0.0)
    gap_min_width_ratio: float = Field(gt=0.0)
    dedup_iou: float = Field(gt=0.0, le=1.0)
    extent_kind: Literal["envelope", "explicit"]
    explicit_extent: tuple[float, float] | None = None
    regions: RegionSet


class ShareReport(BaseModel):
    """Saida completa da analise de dominio. Este e o contrato do ``--json``.

    ``status`` existe para que o caso degenerado -- nenhuma deteccao -- tenha
    representacao honesta em vez de ``NaN`` vazando pelo JSON. Quando ele vale
    ``no_detections``, todos os shares sao ``0.0`` e nao somam 1.0.
    """

    model_config = _FROZEN

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    status: Literal["ok", "no_detections"]
    source: str | None = None
    total_detections: int = Field(ge=0, description="Deteccoes apos deduplicacao.")
    duplicates_removed: int = Field(ge=0)
    shelf_count: int = Field(ge=0)
    shelves: tuple[ShelfReport, ...]
    regions: tuple[RegionShare, ...] = Field(min_length=1, description="Agregado por regiao.")
    params: AnalysisParams
    warnings: tuple[str, ...] = ()

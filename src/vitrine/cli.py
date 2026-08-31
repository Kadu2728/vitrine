"""A interface do Vitrine.

Sem front-end, a CLI **e** o produto: ela recebe o mesmo cuidado que uma tela
receberia. Isso significa, concretamente:

- ``--help`` legivel por quem nunca viu o projeto, com exemplo real e a unidade
  de cada parametro;
- saida humana em tabela Rich e saida de maquina em ``--json``, com schema
  versionado -- e assim que a ferramenta se integra a outra coisa, sem HTTP;
- codigos de saida uteis: ``0`` sucesso, ``1`` erro de uso, ``2`` falha de
  processamento;
- mensagens de erro que dizem **o que fazer**, nao so o que aconteceu;
- nenhum stack trace na cara do usuario -- o traceback completo vai para o
  arquivo de log, e a mensagem diz onde ele esta.

A CLI e um consumidor fino da biblioteca. Nenhuma regra de negocio mora aqui.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import cv2
import typer
from rich.console import Console
from rich.table import Table

from vitrine.domain.models import RegionSet, ShareReport
from vitrine.errors import UsageError, VitrineError
from vitrine.eval.dataset import load_split
from vitrine.eval.metrics import evaluate
from vitrine.pipeline import analyze_image
from vitrine.render.annotate import annotate
from vitrine.vision.image import DEFAULT_MAX_SIZE, load_image

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from vitrine.vision.perspective import Quad
    from vitrine.vision.protocols import Detector

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FAILURE = 2

app = typer.Typer(
    name="vitrine",
    help=(
        "Auditoria de execucao em ponto de venda por visao computacional.\n\n"
        "Analisa uma foto de gondola e responde: quantos produtos estao expostos, "
        "como o espaco esta dividido e onde ha ruptura."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()
"""Saida humana normal, em stdout: ``vitrine analyze foto.jpg | less`` funciona."""

err_console = Console(stderr=True)
"""Erros e avisos, em stderr, para nao contaminar o que sai canalizado."""


@app.callback()
def main() -> None:
    """Vitrine: auditoria de execucao em ponto de venda.

    Este callback existe para que o app mantenha subcomandos mesmo tendo apenas
    um hoje: sem ele, o Typer promove ``analyze`` a raiz e ``vitrine analyze
    foto.jpg`` deixa de funcionar. ``batch``, ``history`` e ``benchmark`` entram
    nas fases seguintes.
    """


@app.command()
def analyze(
    image: Annotated[
        Path,
        typer.Argument(
            help="Foto da gondola (JPEG, PNG, WEBP, BMP ou TIFF).",
            show_default=False,
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Pasta onde gravar a imagem anotada e o relatorio JSON.",
            show_default=False,
        ),
    ] = None,
    detector_name: Annotated[
        str,
        typer.Option(
            "--detector",
            help=(
                "[b]contour[/b]: sem modelo, encontra retangulos de alto contraste. "
                "Bom para imagem sintetica e demonstracao; ruim em foto de loja real. "
                "[b]yolo[/b]: modelo real, exige o extra 'yolo' instalado."
            ),
        ),
    ] = "contour",
    weights: Annotated[
        str | None,
        typer.Option(
            "--weights",
            help=(
                "Arquivo .pt do modelo, quando --detector yolo. "
                "Fica registrado no relatorio com o hash sha256."
            ),
            show_default=False,
        ),
    ] = None,
    confidence: Annotated[
        float,
        typer.Option("--conf", help="Confianca minima da deteccao, de 0 a 1.", min=0.0, max=1.0),
    ] = 0.25,
    perspective: Annotated[
        tuple[str, str, str, str] | None,
        typer.Option(
            "--perspective",
            help=(
                "Quatro cantos da area util, em pixels da imagem ja carregada, "
                "no formato x,y -- em qualquer ordem. "
                "Exemplo: --perspective 120,80 900,60 940,700 100,720"
            ),
            show_default=False,
        ),
    ] = None,
    cuts: Annotated[
        str | None,
        typer.Option(
            "--cuts",
            help=(
                "Fronteiras das regioes em fracao da prateleira, de 0 a 1, separadas por virgula. "
                "Exemplo: --cuts 0,0.4,1 divide em duas regioes. "
                "Omitido, o relatorio traz ocupacao e contagem sem share entre partes."
            ),
            show_default=False,
        ),
    ] = None,
    region_names: Annotated[
        str | None,
        typer.Option(
            "--region-names",
            help=(
                "Nomes das regioes, separados por virgula. "
                "Exemplo: --region-names minha,concorrencia"
            ),
            show_default=False,
        ),
    ] = None,
    extent: Annotated[
        str | None,
        typer.Option(
            "--extent",
            help=(
                "Limites horizontais da gondola em pixels, no formato x_min,x_max. "
                "Omitido, usa o envelope dos produtos -- que ignora vazio nas pontas."
            ),
            show_default=False,
        ),
    ] = None,
    invert: Annotated[
        bool,
        typer.Option(
            "--invert",
            help=(
                "Para --detector contour: procurar produtos escuros em fundo claro, "
                "em vez de claros em fundo escuro. Se a analise nao encontrar nada, "
                "esta e a primeira coisa a tentar."
            ),
        ),
    ] = False,
    max_size: Annotated[
        int,
        typer.Option("--max-size", help="Maior lado da imagem em pixels apos reducao.", min=1),
    ] = DEFAULT_MAX_SIZE,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Imprime o relatorio JSON em stdout e suprime a tabela."),
    ] = False,
    log_file: Annotated[
        Path,
        typer.Option(
            "--log-file", help="Arquivo onde gravar o traceback completo em caso de erro."
        ),
    ] = Path("vitrine.log"),
) -> None:
    """Analisa uma foto de gondola.

    [b]Exemplos[/b]

      Analise simples, tabela no terminal:

        vitrine analyze foto.jpg

      Com correcao de perspectiva e regioes nomeadas, gravando os artefatos:

        vitrine analyze foto.jpg --out ./resultado
        --perspective 120,80 900,60 940,700 100,720
        --cuts 0,0.5,1 --region-names minha,concorrencia

      Integracao com outra ferramenta, sem HTTP:

        vitrine analyze foto.jpg --json | jq '.regions[].linear_share'
    """
    try:
        regions = _parse_regions(cuts, region_names)
        corners = _parse_perspective(perspective)
        limits = _parse_extent(extent)
        detector = _build_detector(detector_name, weights, confidence, invert=invert)

        resultado = analyze_image(
            image,
            detector,
            perspective=corners,
            max_size=max_size,
            regions=regions,
            extent=limits,
        )
    except UsageError as erro:
        _report_error(erro, log_file)
        raise typer.Exit(EXIT_USAGE) from erro
    except VitrineError as erro:
        _report_error(erro, log_file)
        raise typer.Exit(EXIT_FAILURE) from erro

    if out is not None:
        _write_artifacts(resultado.report, resultado.pixels, out, image.stem)

    if as_json:
        sys.stdout.write(resultado.report.model_dump_json(indent=2) + "\n")
    else:
        _print_report(resultado.report, out)

    raise typer.Exit(EXIT_OK)


@app.command()
def benchmark(
    dataset: Annotated[
        Path,
        typer.Argument(
            help="Raiz do dataset no formato YOLO, com images/<split> e labels/<split>.",
            show_default=False,
        ),
    ],
    split: Annotated[
        str, typer.Option("--split", help="Split a avaliar, por exemplo val ou test.")
    ] = "val",
    detector_name: Annotated[str, typer.Option("--detector", help="'contour' ou 'yolo'.")] = "yolo",
    weights: Annotated[
        str | None,
        typer.Option("--weights", help="Arquivo .pt do modelo.", show_default=False),
    ] = None,
    confidence: Annotated[
        float,
        typer.Option(
            "--conf",
            help="Limiar de confianca em que precisao e recall sao reportados.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.25,
    iou: Annotated[
        float,
        typer.Option(
            "--iou",
            help="IoU minimo para considerar uma deteccao correta.",
            min=0.01,
            max=1.0,
        ),
    ] = 0.5,
    invert: Annotated[
        bool,
        typer.Option("--invert", help="Para --detector contour: produtos escuros em fundo claro."),
    ] = False,
    max_size: Annotated[
        int, typer.Option("--max-size", help="Maior lado da imagem em pixels apos reducao.", min=1)
    ] = DEFAULT_MAX_SIZE,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Avaliar apenas as N primeiras imagens.", show_default=False),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Imprime o resultado em JSON em stdout.")
    ] = False,
    log_file: Annotated[
        Path,
        typer.Option(
            "--log-file", help="Arquivo onde gravar o traceback completo em caso de erro."
        ),
    ] = Path("vitrine.log"),
) -> None:
    """Mede precisao, recall e AP de um detector sobre um dataset anotado.

    As anotacoes vem normalizadas e sao convertidas usando o tamanho da imagem
    [b]ja reduzida[/b], de modo que --max-size nao desalinha a comparacao.

    [b]Exemplo[/b]

      vitrine benchmark ./sku110k --split val --detector yolo --weights modelo.pt

    O numero que sair daqui vai para benchmarks/results.md com data e comando --
    inclusive se for ruim.
    """
    try:
        amostras = load_split(dataset, split)
        if limit is not None:
            amostras = amostras[:limit]
        detector = _build_detector(detector_name, weights, confidence, invert=invert)

        pares = []
        # A barra vai para stderr: stdout precisa continuar limpo para --json.
        with typer.progressbar(amostras, label="Avaliando", file=sys.stderr) as barra:
            for amostra in barra:
                imagem = load_image(amostra.image_path, max_size=max_size)
                predicoes = detector.detect(imagem.pixels)
                pares.append((predicoes, amostra.boxes(imagem.width, imagem.height)))

        resultado = evaluate(pares, iou_threshold=iou, confidence_threshold=confidence)
    except UsageError as erro:
        _report_error(erro, log_file)
        raise typer.Exit(EXIT_USAGE) from erro
    except VitrineError as erro:
        _report_error(erro, log_file)
        raise typer.Exit(EXIT_FAILURE) from erro

    if as_json:
        sys.stdout.write(resultado.model_dump_json(indent=2) + "\n")
    else:
        tabela = Table(title=f"Avaliacao: {dataset.name} / {split}", header_style="bold")
        tabela.add_column("Metrica")
        tabela.add_column("Valor", justify="right")
        tabela.add_row("Imagens", str(resultado.images))
        tabela.add_row("Produtos anotados", str(resultado.ground_truth))
        tabela.add_row("Predicoes", str(resultado.predictions))
        tabela.add_row("Precisao", f"{resultado.precision:.4f}")
        tabela.add_row("Recall", f"{resultado.recall:.4f}")
        tabela.add_row("F1", f"{resultado.f1:.4f}")
        tabela.add_row(f"AP@{resultado.iou_threshold:g}", f"{resultado.average_precision:.4f}")
        console.print(tabela)
        console.print(
            f"[dim]detector={detector_name} conf={confidence} iou={iou} max_size={max_size}[/dim]"
        )

    raise typer.Exit(EXIT_OK)


def _build_detector(
    name: str, weights: str | None, confidence: float, *, invert: bool = False
) -> Detector:
    """Constroi o detector pedido, adiando o import pesado ate ser necessario."""
    escolha = name.strip().lower()
    if escolha == "contour":
        if weights is not None:
            raise UsageError(
                "--weights nao se aplica ao detector 'contour', que nao usa modelo.",
                "Remova --weights, ou use --detector yolo.",
            )
        from vitrine.vision.contour import ContourDetector

        return ContourDetector(invert=invert)

    if escolha == "yolo":
        if invert:
            raise UsageError(
                "--invert nao se aplica ao detector 'yolo'.",
                "A opcao existe apenas para o detector por contorno.",
            )
        from vitrine.vision.yolo import DEFAULT_WEIGHTS, YoloDetector

        return YoloDetector(weights or DEFAULT_WEIGHTS, confidence=confidence)

    raise UsageError(
        f"Detector desconhecido: {name!r}.",
        "Os valores aceitos sao 'contour' e 'yolo'.",
    )


def _parse_regions(cuts: str | None, names: str | None) -> RegionSet | None:
    """Converte ``--cuts`` e ``--region-names`` numa particao validada."""
    if cuts is None:
        if names is not None:
            raise UsageError(
                "--region-names foi informado sem --cuts.",
                "Regioes precisam de fronteiras: adicione --cuts 0,0.5,1",
            )
        return None

    try:
        valores = tuple(float(parte) for parte in cuts.split(","))
    except ValueError as exc:
        raise UsageError(
            f"Nao consegui ler --cuts {cuts!r} como numeros.",
            "Use fracoes separadas por virgula, por exemplo --cuts 0,0.4,1",
        ) from exc

    rotulos = tuple(nome.strip() for nome in names.split(",")) if names is not None else None
    if rotulos is not None and len(rotulos) != len(valores) - 1:
        raise UsageError(
            f"{len(valores) - 1} regioes exigem {len(valores) - 1} nomes; "
            f"recebidos {len(rotulos)}.",
            f"Com --cuts {cuts} informe {len(valores) - 1} nomes separados por virgula.",
        )

    try:
        return RegionSet.from_cuts(valores, rotulos)
    except ValueError as exc:
        raise UsageError(
            f"As regioes informadas nao formam uma particao valida: {exc}",
            "As fronteiras precisam comecar em 0, terminar em 1 e ser crescentes.",
        ) from exc


def _parse_perspective(pontos: tuple[str, str, str, str] | None) -> Quad | None:
    """Converte quatro strings ``x,y`` em pontos."""
    if pontos is None:
        return None

    convertidos: list[tuple[float, float]] = []
    for bruto in pontos:
        partes = bruto.split(",")
        if len(partes) != 2:
            raise UsageError(
                f"Ponto de perspectiva invalido: {bruto!r}.",
                "Cada canto e um par x,y sem espaco, por exemplo 120,80",
            )
        try:
            convertidos.append((float(partes[0]), float(partes[1])))
        except ValueError as exc:
            raise UsageError(
                f"Coordenadas nao numericas em {bruto!r}.",
                "Use numeros em pixels, por exemplo --perspective 120,80 900,60 940,700 100,720",
            ) from exc

    primeiro, segundo, terceiro, quarto = convertidos
    return (primeiro, segundo, terceiro, quarto)


def _parse_extent(extent: str | None) -> tuple[float, float] | None:
    """Converte ``x_min,x_max`` num par de limites."""
    if extent is None:
        return None
    partes = extent.split(",")
    if len(partes) != 2:
        raise UsageError(
            f"--extent precisa de dois valores; recebido {extent!r}.",
            "Use --extent x_min,x_max, por exemplo --extent 0,1600",
        )
    try:
        minimo, maximo = float(partes[0]), float(partes[1])
    except ValueError as exc:
        raise UsageError(
            f"Nao consegui ler --extent {extent!r} como numeros.",
            "Use dois numeros em pixels separados por virgula.",
        ) from exc
    if maximo <= minimo:
        raise UsageError(
            f"--extent precisa ter x_max maior que x_min; recebido {extent!r}.",
            "Inverta os valores: --extent 0,1600",
        )
    return (minimo, maximo)


def _write_artifacts(report: ShareReport, pixels: NDArray[np.uint8], out: Path, stem: str) -> None:
    """Grava a imagem anotada e o relatorio JSON na pasta de saida."""
    out.mkdir(parents=True, exist_ok=True)
    caminho_json = out / f"{stem}.json"
    caminho_imagem = out / f"{stem}.anotada.jpg"

    caminho_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    anotada = annotate(pixels, report)
    if not cv2.imwrite(str(caminho_imagem), anotada):
        raise VitrineError(
            f"Nao consegui gravar a imagem anotada em {caminho_imagem}.",
            "Verifique permissao de escrita na pasta de saida.",
        )


def _print_report(report: ShareReport, out: Path | None) -> None:
    """Imprime a tabela de share por prateleira e o resumo."""
    if report.status == "no_detections":
        console.print("[yellow]Nenhum produto detectado nesta imagem.[/yellow]")
        console.print(
            "[dim]Com --detector contour: se os produtos sao mais escuros que o "
            "fundo, tente --invert. Em foto de loja real, prefira --detector yolo. "
            "Se o enquadramento inclui prateleira vizinha, recorte com "
            "--perspective.[/dim]"
        )
        return

    tabela = Table(title="Share por prateleira", title_style="bold", header_style="bold")
    tabela.add_column("Prat.", justify="right")
    tabela.add_column("Produtos", justify="right")
    tabela.add_column("Ocupacao", justify="right")
    tabela.add_column("Vazios", justify="right")
    for nome in (share.region for share in report.shelves[0].regions):
        tabela.add_column(f"{nome}\ncont. | area", justify="right")

    for shelf in report.shelves:
        linha = [
            str(shelf.index),
            str(shelf.detection_count),
            f"{shelf.occupancy:.0%}",
            str(len(shelf.gaps)),
        ]
        linha.extend(
            f"{share.count_share:.0%} | {share.linear_share:.0%}" for share in shelf.regions
        )
        tabela.add_row(*linha)

    console.print(tabela)
    console.print(
        f"[bold]{report.total_detections}[/bold] produtos em "
        f"[bold]{report.shelf_count}[/bold] prateleira(s); "
        f"[bold]{sum(len(s.gaps) for s in report.shelves)}[/bold] vazio(s)."
    )
    if report.duplicates_removed:
        console.print(
            f"[dim]{report.duplicates_removed} deteccao(oes) duplicada(s) removida(s).[/dim]"
        )
    for aviso in report.warnings:
        err_console.print(f"[yellow]aviso:[/yellow] {aviso}")
    if out is not None:
        console.print(f"[dim]Artefatos gravados em {out}[/dim]")


def _report_error(erro: VitrineError, log_file: Path) -> None:
    """Mostra mensagem e dica ao usuario; manda o traceback para o log."""
    err_console.print(f"[red]erro:[/red] {erro.message}")
    err_console.print(f"[cyan]->[/cyan] {erro.hint}")
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as arquivo:
            arquivo.write(traceback.format_exc())
            arquivo.write("\n")
    except OSError:
        # Nao conseguir escrever o log nao pode transformar um erro tratado
        # numa falha diferente e mais confusa para o usuario.
        return
    err_console.print(f"[dim]Detalhes tecnicos em {log_file}[/dim]")


if __name__ == "__main__":  # pragma: no cover
    app()

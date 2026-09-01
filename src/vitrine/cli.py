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

import json
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import cv2
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from vitrine import logs
from vitrine.batch.runner import BatchOptions, DetectorSpec, run_batch
from vitrine.domain.models import RegionSet, ShareReport
from vitrine.errors import UsageError, VitrineError
from vitrine.eval.dataset import load_split
from vitrine.eval.metrics import evaluate
from vitrine.pipeline import analyze_image
from vitrine.render.annotate import annotate
from vitrine.storage.repository import Repository
from vitrine.vision.image import DEFAULT_MAX_SIZE, load_image

if TYPE_CHECKING:
    import logging
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from vitrine.batch.runner import BatchSummary, ImageOutcome
    from vitrine.storage.repository import Visit
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


@app.command()
def batch(
    folder: Annotated[
        Path,
        typer.Argument(help="Pasta com as fotos. A busca e recursiva.", show_default=False),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Pasta de saida: artefatos, manifesto e log."),
    ] = Path("resultado"),
    store_id: Annotated[
        str | None,
        typer.Option(
            "--store-id",
            help="Identificador do ponto de venda. Sem ele, nada e gravado no historico.",
            show_default=False,
        ),
    ] = None,
    db: Annotated[Path, typer.Option("--db", help="Arquivo SQLite do historico.")] = Path(
        "vitrine.db"
    ),
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help=(
                "Processos em paralelo. 1 roda em processo unico, "
                "que e mais rapido para lote pequeno."
            ),
            min=1,
            max=32,
        ),
    ] = 1,
    detector_name: Annotated[
        str, typer.Option("--detector", help="'contour' ou 'yolo'.")
    ] = "contour",
    weights: Annotated[
        str | None,
        typer.Option("--weights", help="Arquivo .pt, para --detector yolo.", show_default=False),
    ] = None,
    confidence: Annotated[
        float, typer.Option("--conf", help="Confianca minima da deteccao.", min=0.0, max=1.0)
    ] = 0.25,
    invert: Annotated[
        bool,
        typer.Option("--invert", help="Para --detector contour: produtos escuros em fundo claro."),
    ] = False,
    cuts: Annotated[
        str | None,
        typer.Option(
            "--cuts",
            help="Fronteiras das regioes, de 0 a 1. Exemplo: --cuts 0,0.5,1",
            show_default=False,
        ),
    ] = None,
    region_names: Annotated[
        str | None,
        typer.Option(
            "--region-names", help="Nomes das regioes, separados por virgula.", show_default=False
        ),
    ] = None,
    max_size: Annotated[
        int, typer.Option("--max-size", help="Maior lado da imagem em pixels apos reducao.", min=1)
    ] = DEFAULT_MAX_SIZE,
    no_resume: Annotated[
        bool,
        typer.Option("--no-resume", help="Reprocessar tudo, ignorando o manifesto de progresso."),
    ] = False,
    no_artifacts: Annotated[
        bool,
        typer.Option("--no-artifacts", help="Nao gravar imagem anotada nem JSON por foto."),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Imprime o resumo do lote em JSON em stdout.")
    ] = False,
) -> None:
    """Processa uma pasta inteira de fotos.

    O lote e [b]resumivel[/b]: pode interromper com Ctrl+C e rodar o mesmo
    comando de novo que ele continua de onde parou, lendo o manifesto em
    [b]<out>/manifest.jsonl[/b]. Uma foto corrompida nao derruba o lote --
    vira uma linha de erro e a execucao segue.

    [b]Exemplos[/b]

      Lote com quatro processos, gravando no historico do PDV:

        vitrine batch ./fotos --store-id LOJA_12 --workers 4

      Retomar um lote interrompido: exatamente o mesmo comando.

      Reprocessar tudo do zero:

        vitrine batch ./fotos --store-id LOJA_12 --no-resume

    O log estruturado fica em [b]<out>/vitrine.jsonl[/b], uma linha JSON por
    etapa. Para ver so as falhas:

      jq -r 'select(.level=="error") | .image' resultado/vitrine.jsonl
    """
    log_path = out / "vitrine.jsonl"
    logger = logs.setup(log_path)

    try:
        options = _batch_options(
            detector_name,
            weights,
            confidence,
            invert,
            cuts,
            region_names,
            max_size,
            store_id,
            no_artifacts,
        )
        if store_id is not None:
            with Repository(db) as repo:
                resumo = _executar_lote(
                    folder, out, options, workers, not no_resume, logger, repo, store_id
                )
                gravados: int | None = repo.count()
        else:
            resumo = _executar_lote(
                folder, out, options, workers, not no_resume, logger, None, None
            )
            gravados = None
    except UsageError as erro:
        _report_error(erro, log_path)
        raise typer.Exit(EXIT_USAGE) from erro
    except VitrineError as erro:
        _report_error(erro, log_path)
        raise typer.Exit(EXIT_FAILURE) from erro

    if as_json:
        sys.stdout.write(
            json.dumps(
                {
                    "total": resumo.total,
                    "processed": resumo.processed,
                    "skipped": resumo.skipped,
                    "failed": resumo.failed,
                    "interrupted": resumo.interrupted,
                    "duration_s": round(resumo.duration_s, 2),
                    "failures": [{"image": i, "error": e} for i, e in resumo.failures],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    else:
        _print_batch(resumo, out, gravados)

    if resumo.interrupted:
        err_console.print(
            "[yellow]Lote interrompido.[/yellow] Rode o mesmo comando de novo "
            "para continuar de onde parou."
        )
        raise typer.Exit(EXIT_FAILURE)
    raise typer.Exit(EXIT_FAILURE if resumo.failed and not resumo.processed else EXIT_OK)


def _batch_options(
    detector_name: str,
    weights: str | None,
    confidence: float,
    invert: bool,
    cuts: str | None,
    region_names: str | None,
    max_size: int,
    store_id: str | None,
    no_artifacts: bool,
) -> BatchOptions:
    """Valida as opcoes da linha de comando e monta a configuracao do lote."""
    escolha = detector_name.strip().lower()
    if escolha not in {"contour", "yolo"}:
        raise UsageError(
            f"Detector desconhecido: {detector_name!r}.",
            "Os valores aceitos sao 'contour' e 'yolo'.",
        )
    if invert and escolha == "yolo":
        raise UsageError(
            "--invert nao se aplica ao detector 'yolo'.",
            "A opcao existe apenas para o detector por contorno.",
        )
    if weights is not None and escolha == "contour":
        raise UsageError(
            "--weights nao se aplica ao detector 'contour', que nao usa modelo.",
            "Remova --weights, ou use --detector yolo.",
        )

    return BatchOptions(
        detector=DetectorSpec(
            kind="yolo" if escolha == "yolo" else "contour",
            weights=weights,
            confidence=confidence,
            invert=invert,
        ),
        store_id=store_id,
        regions=_parse_regions(cuts, region_names),
        max_size=max_size,
        write_artifacts=not no_artifacts,
    )


def _executar_lote(
    folder: Path,
    out: Path,
    options: BatchOptions,
    workers: int,
    resume: bool,
    logger: logging.Logger,
    repo: Repository | None,
    store_id: str | None,
) -> BatchSummary:
    """Roda o lote, persistindo cada resultado assim que ele fica pronto.

    A persistencia acontece aqui, no processo pai, e nao dentro do worker: com
    varios processos escrevendo no mesmo SQLite, a contencao de lock seria
    certa e o ganho, nenhum.
    """

    def ao_terminar(resultado: ImageOutcome) -> None:
        if repo is not None and store_id is not None and resultado.report is not None:
            repo.save(store_id, resultado.report)

    # O spinner so faz sentido em terminal. Fora dele -- saida redirecionada,
    # pipe, CI -- ele vira ruido no log e ainda custa uma thread de atualizacao
    # por execucao, o que e desperdicio puro.
    girando = console.status("Processando o lote...") if console.is_terminal else nullcontext()
    with girando:
        return run_batch(
            folder,
            out_dir=out,
            options=options,
            workers=workers,
            resume=resume,
            on_result=ao_terminar,
            logger=logger,
        )


def _print_batch(resumo: BatchSummary, out: Path, gravados: int | None) -> None:
    """Resumo do lote em tabela."""
    tabela = Table(title="Lote", header_style="bold")
    tabela.add_column("Metrica")
    tabela.add_column("Valor", justify="right")
    tabela.add_row("Imagens na pasta", str(resumo.total))
    tabela.add_row("Processadas agora", str(resumo.processed))
    tabela.add_row("Puladas (ja no manifesto)", str(resumo.skipped))
    tabela.add_row("Falhas", str(resumo.failed))
    tabela.add_row("Tempo", f"{resumo.duration_s:.1f} s")
    if gravados is not None:
        tabela.add_row("Analises no historico", str(gravados))
    console.print(tabela)

    if resumo.failures:
        err_console.print("[yellow]Imagens que falharam:[/yellow]")
        for imagem, erro in resumo.failures:
            err_console.print(f"  [red]{imagem}[/red]: {erro}")

    console.print(f"[dim]Artefatos e manifesto em {out}; log em {out / 'vitrine.jsonl'}[/dim]")


@app.command()
def history(
    store_id: Annotated[
        str,
        typer.Option("--store-id", help="Identificador do ponto de venda.", show_default=False),
    ],
    last: Annotated[
        int, typer.Option("--last", help="Quantas visitas mostrar.", min=1, max=1000)
    ] = 30,
    db: Annotated[Path, typer.Option("--db", help="Arquivo SQLite do historico.")] = Path(
        "vitrine.db"
    ),
    as_json: Annotated[
        bool, typer.Option("--json", help="Imprime o historico em JSON em stdout.")
    ] = False,
) -> None:
    """Mostra a evolucao de um ponto de venda ao longo do tempo.

    As visitas sao ordenadas pela [b]data da foto[/b], lida do EXIF, e nao pela
    data em que o lote rodou. Um lote processado com uma semana de atraso nao
    embaralha a serie.

    A tabela traz a variacao em relacao a visita anterior: quantos produtos a
    mais ou a menos, quantos pontos percentuais de ocupacao, quantos vazios.

    [b]Exemplo[/b]

      vitrine history --store-id LOJA_12 --last 30
    """
    if not db.is_file():
        _report_error(
            VitrineError(
                f"Nao encontrei o banco de historico em {db}.",
                "O historico e criado por 'vitrine batch --store-id <PDV>'. "
                "Se o arquivo esta em outro lugar, use --db.",
            ),
            Path("vitrine.log"),
        )
        raise typer.Exit(EXIT_FAILURE)

    with Repository(db) as repo:
        visitas = repo.history(store_id, limit=last)
        conhecidos = repo.stores()

    if not visitas:
        err_console.print(f"[yellow]Sem historico para {store_id!r}.[/yellow]")
        if conhecidos:
            err_console.print(f"[dim]PDVs no banco: {', '.join(conhecidos)}[/dim]")
        raise typer.Exit(EXIT_OK)

    if as_json:
        sys.stdout.write(
            json.dumps(
                [
                    {
                        "store_id": v.store_id,
                        "source": v.source,
                        "captured_at": v.captured_at,
                        "detections": v.detections,
                        "shelves": v.shelves,
                        "gaps": v.gaps,
                        "occupancy": v.occupancy,
                        "detector": v.detector,
                        "regions": [
                            {"region": n, "count_share": c, "linear_share": a}
                            for n, c, a in v.regions
                        ],
                    }
                    for v in visitas
                ],
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        raise typer.Exit(EXIT_OK)

    _print_history(store_id, visitas)
    raise typer.Exit(EXIT_OK)


def _print_history(store_id: str, visitas: Sequence[Visit]) -> None:
    """Tabela do historico, com a variacao em relacao a visita anterior."""
    tabela = Table(title=f"Historico: {store_id}", header_style="bold")
    tabela.add_column("Data da foto")
    tabela.add_column("Foto", overflow="ellipsis", max_width=22)
    tabela.add_column("Produtos", justify="right")
    tabela.add_column("Prat.", justify="right")
    tabela.add_column("Ocupacao", justify="right")
    tabela.add_column("Vazios", justify="right")

    nomes = [n for n, _, _ in visitas[0].regions]
    for nome in nomes:
        tabela.add_column(f"{nome}\ncont. | area", justify="right")

    # As visitas vem da mais recente para a mais antiga, entao a visita
    # imediatamente anterior no tempo e a proxima da lista.
    for indice, visita in enumerate(visitas):
        anterior = visitas[indice + 1] if indice + 1 < len(visitas) else None
        produtos = _delta(visita.detections, anterior.detections if anterior else None)
        ocupacao = _delta_pct(visita.occupancy, anterior.occupancy if anterior else None)
        vazios = _delta(visita.gaps, anterior.gaps if anterior else None, invertido=True)
        linha = [
            visita.captured_at.replace("T", " ")[:16],
            visita.source,
            f"{visita.detections}{produtos}",
            str(visita.shelves),
            f"{visita.occupancy:.0%}{ocupacao}",
            f"{visita.gaps}{vazios}",
        ]
        mapa = {n: (c, a) for n, c, a in visita.regions}
        linha.extend(
            f"{mapa[nome][0]:.0%} | {mapa[nome][1]:.0%}" if nome in mapa else "-" for nome in nomes
        )
        tabela.add_row(*linha)

    console.print(tabela)
    console.print(
        "[dim]A variacao e em relacao a visita anterior. Em 'Vazios', menos e melhor.[/dim]"
    )


def _delta(atual: int, anterior: int | None, *, invertido: bool = False) -> str:
    """Variacao absoluta, colorida por ser melhora ou piora.

    ``invertido`` existe para o numero de vazios: ali, cair e melhorar.
    """
    if anterior is None or atual == anterior:
        return ""
    diferenca = atual - anterior
    melhorou = (diferenca < 0) if invertido else (diferenca > 0)
    cor = "green" if melhorou else "red"
    return f" [{cor}]{diferenca:+d}[/{cor}]"


def _delta_pct(atual: float, anterior: float | None) -> str:
    """Variacao de ocupacao, em pontos percentuais."""
    if anterior is None:
        return ""
    diferenca = (atual - anterior) * 100
    if abs(diferenca) < 0.5:
        return ""
    cor = "green" if diferenca > 0 else "red"
    return f" [{cor}]{diferenca:+.0f}pp[/{cor}]"


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
            f"Regioes invalidas: {_mensagem_limpa(exc)}",
            "As fronteiras precisam comecar em 0, terminar em 1 e ser crescentes. "
            "Exemplo: --cuts 0,0.5,1",
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


def _mensagem_limpa(exc: Exception) -> str:
    """Extrai a frase util de um erro de validacao.

    O ``str()`` de um ``ValidationError`` do Pydantic traz contagem de erros,
    nome do modelo, tipo interno e um link para a documentacao da biblioteca.
    Nada disso ajuda quem digitou uma opcao errada na linha de comando -- e
    despejar isso no terminal e a versao educada de imprimir stack trace.
    """
    if isinstance(exc, ValidationError):
        erros = exc.errors()
        if erros:
            texto = str(erros[0].get("msg", ""))
            return texto.removeprefix("Value error, ").strip() or "valor invalido"
    return str(exc)


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

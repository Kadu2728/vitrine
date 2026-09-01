"""Processamento em lote: paralelo, resumivel e tolerante a falha.

E aqui que o projeto deixa de ser script. Quatro exigencias, e a solucao de
cada uma:

**Paralelismo.** ``ProcessPoolExecutor``, porque inferencia e trabalho de CPU e
o GIL tornaria threads inuteis. Mas um modelo de deteccao nao e serializavel:
nao da para mandar o objeto para o worker. Por isso o que viaja e a
``DetectorSpec`` -- uma descricao -- e cada processo constroi o seu detector uma
vez, guardado num cache de modulo. Sem isso, ou o paralelismo nao existe, ou o
modelo seria recarregado a cada imagem.

**Retomada.** Ver ``batch.manifest``. Antes de processar, o lote le o manifesto
e pula o que ja tem entrada.

**Isolamento de falha.** Um worker nunca levanta excecao para fora: ele devolve
sucesso ou fracasso. Uma foto corrompida no meio de 400 vira uma linha de erro
no manifesto, e o lote segue. O contrario -- derrubar tudo por causa de um
arquivo ruim -- e o comportamento que faz alguem desistir da ferramenta.

**Ctrl+C.** Os workers ignoram o sinal e o pai o trata: cancela o que ainda nao
comecou, deixa terminar o que ja esta em voo, fecha o manifesto e sai com codigo
proprio. Sem isso, no Windows o sinal vai para o grupo inteiro de processos e o
manifesto fica pela metade -- perdendo exatamente o progresso que ele existe
para guardar.

**Por que ``workers=1`` nao usa pool.** Subir processo para tres imagens custa
mais do que economiza, e o caminho sequencial e depuravel: excecao aparece com
a pilha inteira em vez de atravessar a fronteira do processo.
"""

from __future__ import annotations

import logging
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from vitrine.batch import manifest
from vitrine.errors import VitrineError
from vitrine.logs import LOGGER_NAME, StageTimer
from vitrine.pipeline import analyze_image
from vitrine.vision.image import DEFAULT_MAX_SIZE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from vitrine.domain.models import RegionSet, ShareReport
    from vitrine.vision.protocols import Detector

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

_DETECTOR_CACHE: dict[tuple[str, str | None, float, bool], Detector] = {}
"""Detector por processo. Carregar o peso uma vez por worker, nao por imagem."""


@dataclass(frozen=True)
class DetectorSpec:
    """Descricao serializavel de um detector.

    O que atravessa a fronteira do processo. Um ``YoloDetector`` carregado nao
    e serializavel -- e nem deveria ser: mandar centenas de megabytes por pipe a
    cada tarefa seria absurdo.
    """

    kind: Literal["contour", "yolo"] = "contour"
    weights: str | None = None
    confidence: float = 0.25
    invert: bool = False

    def build(self) -> Detector:
        """Constroi o detector, reaproveitando o do processo atual."""
        chave = (self.kind, self.weights, self.confidence, self.invert)
        if chave not in _DETECTOR_CACHE:
            _DETECTOR_CACHE[chave] = self._construct()
        return _DETECTOR_CACHE[chave]

    def _construct(self) -> Detector:
        if self.kind == "contour":
            from vitrine.vision.contour import ContourDetector

            return ContourDetector(invert=self.invert)

        from vitrine.vision.yolo import DEFAULT_WEIGHTS, YoloDetector

        return YoloDetector(self.weights or DEFAULT_WEIGHTS, confidence=self.confidence)


@dataclass(frozen=True)
class BatchOptions:
    """Tudo que o lote precisa saber, num objeto serializavel."""

    detector: DetectorSpec = field(default_factory=DetectorSpec)
    store_id: str | None = None
    regions: RegionSet | None = None
    max_size: int | None = DEFAULT_MAX_SIZE
    write_artifacts: bool = True


@dataclass
class ImageOutcome:
    """O que um worker devolve. Nunca uma excecao."""

    path: Path
    key: str
    duration_ms: float
    report: ShareReport | None = None
    error: str | None = None
    stages: dict[str, float] = field(default_factory=dict)
    """Duracao de cada etapa em milissegundos: carregar, detectar, medir."""

    @property
    def ok(self) -> bool:
        """Se a imagem foi processada com sucesso."""
        return self.report is not None


@dataclass(frozen=True)
class BatchSummary:
    """Resultado consolidado do lote."""

    total: int
    processed: int
    skipped: int
    failed: int
    interrupted: bool
    duration_s: float
    failures: tuple[tuple[str, str], ...] = ()


def process_one(path: Path, out_dir: Path | None, options: BatchOptions, key: str) -> ImageOutcome:
    """Processa uma imagem, convertendo qualquer falha em resultado.

    Esta funcao roda dentro do worker. Ela **nao levanta**: erro previsto vira
    ``error`` com a dica junto; erro imprevisto vira ``error`` com o tipo. Um
    ``raise`` aqui atravessaria a fronteira do processo e derrubaria o lote.
    """
    inicio = time.perf_counter()
    cronometro = StageTimer()
    try:
        with cronometro.stage("detector"):
            detector = options.detector.build()
        with cronometro.stage("analyze"):
            resultado = analyze_image(
                path,
                detector,
                max_size=options.max_size,
                regions=options.regions,
                source=path.name,
            )
        if out_dir is not None and options.write_artifacts:
            with cronometro.stage("render"):
                _write_artifacts(resultado, out_dir, path.stem)
    except VitrineError as erro:
        return ImageOutcome(
            path=path,
            key=key,
            duration_ms=_ms(inicio),
            error=f"{erro.message} {erro.hint}",
            stages=cronometro.durations,
        )
    except Exception as erro:
        # Rede de seguranca deliberada. Um bug nosso numa unica imagem nao pode
        # custar o lote inteiro de quem esta em campo; o erro fica registrado
        # com o tipo, no manifesto e no log.
        return ImageOutcome(
            path=path,
            key=key,
            duration_ms=_ms(inicio),
            error=f"{type(erro).__name__}: {erro}",
            stages=cronometro.durations,
        )

    return ImageOutcome(
        path=path,
        key=key,
        duration_ms=_ms(inicio),
        report=resultado.report,
        stages=cronometro.durations,
    )


def run_batch(
    root: Path,
    *,
    out_dir: Path | None = None,
    options: BatchOptions | None = None,
    workers: int = 1,
    resume: bool = True,
    on_result: Callable[[ImageOutcome], None] | None = None,
    logger: logging.Logger | None = None,
) -> BatchSummary:
    """Processa uma pasta de fotos.

    Args:
        root: pasta com as imagens; a busca e recursiva.
        out_dir: onde gravar artefatos e manifesto. ``None`` nao grava nada e
            desliga a retomada.
        options: detector, regioes e demais parametros.
        workers: processos paralelos. ``1`` roda em processo unico.
        resume: pular o que ja consta no manifesto.
        on_result: chamado a cada imagem concluida, para barra de progresso.
        logger: logger estruturado; o padrao e o do projeto.

    Returns:
        O resumo do lote, incluindo se ele foi interrompido.

    Raises:
        VitrineError: se a pasta nao existir ou nao tiver imagem.
    """
    opcoes = options if options is not None else BatchOptions()
    log = logger if logger is not None else logging.getLogger(LOGGER_NAME)

    if not root.is_dir():
        raise VitrineError(
            f"{root} nao e uma pasta.",
            "Aponte para a pasta com as fotos: vitrine batch ./fotos",
        )

    imagens = list(manifest.iter_images(root, IMAGE_SUFFIXES))
    if not imagens:
        raise VitrineError(
            f"Nenhuma imagem encontrada em {root}.",
            f"Extensoes aceitas: {', '.join(sorted(IMAGE_SUFFIXES))}. A busca e recursiva.",
        )

    caminho_manifesto = (out_dir / manifest.MANIFEST_NAME) if out_dir is not None else None
    ja_feitas = (
        manifest.read(caminho_manifesto) if (resume and caminho_manifesto is not None) else {}
    )

    pendentes: list[tuple[Path, str]] = []
    pulados = 0
    for caminho in imagens:
        chave = manifest.image_key(caminho, root)
        if chave in ja_feitas:
            pulados += 1
            continue
        pendentes.append((caminho, chave))

    log.info(
        "batch_start",
        extra={
            "root": str(root),
            "total": len(imagens),
            "pending": len(pendentes),
            "skipped": pulados,
            "workers": workers,
            "detector": opcoes.detector.kind,
        },
    )

    inicio = time.perf_counter()
    processadas = 0
    falhas: list[tuple[str, str]] = []
    interrompido = False

    escritor = manifest.Writer(caminho_manifesto) if caminho_manifesto is not None else None
    contexto = escritor if escritor is not None else _NullWriter()

    with contexto as saida:
        executar = _run_sequential if workers <= 1 else _run_parallel
        try:
            for resultado in executar(pendentes, out_dir, opcoes, workers):
                _registrar(resultado, saida, log)
                if resultado.ok:
                    processadas += 1
                else:
                    falhas.append((resultado.path.name, resultado.error or "erro desconhecido"))
                if on_result is not None:
                    on_result(resultado)
        except KeyboardInterrupt:
            interrompido = True
            log.warning("batch_interrupted", extra={"processed": processadas})

    duracao = time.perf_counter() - inicio
    log.info(
        "batch_done",
        extra={
            "processed": processadas,
            "failed": len(falhas),
            "skipped": pulados,
            "interrupted": interrompido,
            "duration_s": round(duracao, 2),
        },
    )

    return BatchSummary(
        total=len(imagens),
        processed=processadas,
        skipped=pulados,
        failed=len(falhas),
        interrupted=interrompido,
        duration_s=duracao,
        failures=tuple(falhas),
    )


def _run_sequential(
    pendentes: Sequence[tuple[Path, str]],
    out_dir: Path | None,
    opcoes: BatchOptions,
    workers: int,
) -> Any:
    """Caminho de processo unico: mais rapido para lote pequeno e depuravel."""
    del workers
    for caminho, chave in pendentes:
        yield process_one(caminho, out_dir, opcoes, chave)


def _run_parallel(
    pendentes: Sequence[tuple[Path, str]],
    out_dir: Path | None,
    opcoes: BatchOptions,
    workers: int,
) -> Any:
    """Caminho paralelo, com Ctrl+C tratado no pai.

    Os resultados saem na ordem em que ficam prontos, nao na ordem de entrada --
    e por isso que o manifesto guarda a chave de cada imagem em vez de um
    contador de posicao.
    """
    with ProcessPoolExecutor(max_workers=workers, initializer=_ignorar_sigint) as pool:
        pendentes_futuros = {
            pool.submit(process_one, caminho, out_dir, opcoes, chave)
            for caminho, chave in pendentes
        }
        try:
            while pendentes_futuros:
                prontos, pendentes_futuros = wait(pendentes_futuros, return_when=FIRST_COMPLETED)
                for futuro in prontos:
                    yield futuro.result()
        except KeyboardInterrupt:
            # Cancela o que nao comecou; o que ja esta em voo termina sozinho.
            for futuro in pendentes_futuros:
                futuro.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def _ignorar_sigint() -> None:
    """Faz o worker ignorar Ctrl+C, deixando o pai coordenar a parada.

    No Windows o Ctrl+C vai para o grupo inteiro de processos. Sem isto, os
    workers morrem antes de o pai conseguir fechar o manifesto -- e o progresso
    das ultimas imagens se perde.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _registrar(
    resultado: ImageOutcome,
    escritor: Any,
    log: logging.Logger,
) -> None:
    """Grava a entrada no manifesto e a linha no log estruturado."""
    report = resultado.report
    entrada = manifest.Entry(
        key=resultado.key,
        source=resultado.path.name,
        status="ok" if resultado.ok else "failed",
        duration_ms=round(resultado.duration_ms, 1),
        detections=report.total_detections if report is not None else 0,
        shelves=report.shelf_count if report is not None else 0,
        gaps=sum(len(s.gaps) for s in report.shelves) if report is not None else 0,
        error=resultado.error,
    )
    escritor.append(entrada)

    if resultado.ok:
        log.info(
            "image_done",
            extra={
                "image": resultado.path.name,
                "duration_ms": entrada.duration_ms,
                "detections": entrada.detections,
                "shelves": entrada.shelves,
                "gaps": entrada.gaps,
                "stages_ms": resultado.stages,
            },
        )
    else:
        log.error(
            "image_failed",
            extra={
                "image": resultado.path.name,
                "duration_ms": entrada.duration_ms,
                "error": resultado.error,
                "stages_ms": resultado.stages,
            },
        )


def _write_artifacts(resultado: Any, out_dir: Path, stem: str) -> None:
    """Grava JSON e imagem anotada de uma imagem do lote."""
    import cv2

    from vitrine.render.annotate import annotate

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        resultado.report.model_dump_json(indent=2), encoding="utf-8"
    )
    cv2.imwrite(str(out_dir / f"{stem}.anotada.jpg"), annotate(resultado.pixels, resultado.report))


class _NullWriter:
    """Escritor que descarta tudo, para quando nao ha pasta de saida."""

    def __enter__(self) -> _NullWriter:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def append(self, entry: manifest.Entry) -> None:
        """Descarta a entrada."""
        del entry


def _ms(inicio: float) -> float:
    return (time.perf_counter() - inicio) * 1000.0

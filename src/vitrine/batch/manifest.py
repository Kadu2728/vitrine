"""Manifesto de progresso: o que torna o lote retomavel.

Um lote de 400 fotos leva minutos. Fechar o notebook, perder a energia ou dar
Ctrl+C no meio nao pode significar comecar do zero -- e essa exigencia decide o
formato do arquivo.

**Por que JSONL append-only, e nao SQLite nem JSON.** Um JSON unico precisa ser
reescrito inteiro a cada imagem: se a maquina morrer durante a reescrita, o
arquivo fica corrompido e o progresso todo se perde -- exatamente na hora em que
ele mais importa. SQLite resolveria isso, mas traz lock e transacao para um
problema que e uma lista de linhas. Append-only com ``flush`` a cada linha
sobrevive a queda: no pior caso a ultima linha fica truncada, e o leitor
descarta linha invalida e continua.

**A chave de retomada e caminho + tamanho + mtime**, nao hash de conteudo.
Hashear 400 fotos antes de comecar custa a leitura de todas elas, o que anula
boa parte da economia da retomada. Trocar uma foto por outra de tamanho
identico no mesmo nanossegundo escaparia -- risco aceito e declarado. Editar a
imagem muda o mtime e ela e reprocessada, que e o comportamento desejado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import TracebackType
    from typing import TextIO

MANIFEST_NAME = "manifest.jsonl"
"""Nome padrao do manifesto dentro da pasta de saida."""

Status = Literal["ok", "failed"]


@dataclass(frozen=True)
class Entry:
    """Uma imagem ja processada, com o resultado e a chave de retomada."""

    key: str
    """Identidade da imagem: caminho relativo, tamanho e mtime."""

    source: str
    status: Status
    duration_ms: float
    detections: int = 0
    shelves: int = 0
    gaps: int = 0
    error: str | None = None

    def to_json(self) -> str:
        """Serializa a entrada como uma linha JSON."""
        return json.dumps(self.__dict__, ensure_ascii=False)

    @classmethod
    def from_json(cls, linha: str) -> Self | None:
        """Le uma linha do manifesto, devolvendo ``None`` se ela nao servir.

        Linha truncada por queda de energia ou campo faltando nao derruba a
        retomada: a imagem correspondente simplesmente e reprocessada.
        """
        try:
            dados = json.loads(linha)
        except json.JSONDecodeError:
            return None
        if not isinstance(dados, dict) or "key" not in dados:
            return None
        campos = {c: dados[c] for c in cls.__dataclass_fields__ if c in dados}
        try:
            return cls(**campos)
        except TypeError:
            return None


def image_key(path: Path, root: Path) -> str:
    """Identidade estavel de uma imagem dentro do lote.

    O caminho e relativo a raiz para que mover a pasta inteira nao invalide o
    progresso.
    """
    try:
        relativo = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relativo = path.resolve().as_posix()
    info = path.stat()
    return f"{relativo}:{info.st_size}:{info.st_mtime_ns}"


def read(path: Path) -> dict[str, Entry]:
    """Le o manifesto existente, indexado pela chave.

    Arquivo ausente significa lote novo. Linha invalida e descartada em
    silencio -- o custo e reprocessar uma imagem, nao perder o lote.
    """
    if not path.is_file():
        return {}

    entradas: dict[str, Entry] = {}
    with path.open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            entrada = Entry.from_json(linha)
            if entrada is not None:
                entradas[entrada.key] = entrada
    return entradas


class Writer:
    """Escritor append-only do manifesto, com ``flush`` a cada linha.

    O ``flush`` nao e exagero: sem ele o buffer do sistema operacional guarda as
    ultimas dezenas de linhas, e um Ctrl+C perde justamente o progresso mais
    recente -- o que torna o manifesto inutil para o proposito que ele tem.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._arquivo: TextIO | None = None

    def __enter__(self) -> Self:
        """Abre o arquivo em modo de acrescimo."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._arquivo = self._path.open("a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Fecha o arquivo, inclusive quando o lote foi interrompido."""
        if self._arquivo is not None:
            self._arquivo.close()
            self._arquivo = None

    def append(self, entry: Entry) -> None:
        """Grava uma entrada e forca a descarga para o disco."""
        if self._arquivo is None:
            raise RuntimeError("Writer usado fora do bloco 'with'")
        self._arquivo.write(entry.to_json() + "\n")
        self._arquivo.flush()


def iter_images(root: Path, suffixes: frozenset[str]) -> Iterator[Path]:
    """Percorre a pasta em ordem estavel, filtrando por extensao.

    A ordem alfabetica nao e cosmetica: com ela, duas execucoes do mesmo lote
    processam na mesma sequencia, e a retomada e previsivel.
    """
    yield from sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes),
        key=lambda p: p.as_posix(),
    )

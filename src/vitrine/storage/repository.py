"""Histórico por ponto de venda, em SQLite.

**Por que SQLite e não Postgres.** Isto é uma ferramenta de linha de comando que
roda no notebook de um promotor, muitas vezes sem internet, na loja. Exigir um
servidor de banco para guardar algumas centenas de linhas por mês seria trocar
um arquivo por uma infraestrutura. O `sqlite3` é da stdlib: zero dependência,
zero configuração, o banco é um arquivo que cabe num e-mail.

**Por que `sqlite3` e não SQLAlchemy.** São quatro tabelas e meia dúzia de
consultas. Um ORM aqui adicionaria uma dependência e uma camada de indireção
para não resolver problema nenhum.

Decisões de esquema:

- **O relatório inteiro é guardado como JSON**, e as métricas usadas em consulta
  ficam também em colunas próprias. Duplicação deliberada: o JSON preserva o
  contrato completo para quem quiser reprocessar, e as colunas permitem
  ordenar e agregar sem abrir cada documento.
- **A data é a da captura**, lida do EXIF, não a do processamento. Um lote
  rodado com uma semana de atraso não pode embaralhar a série temporal.
- **`UNIQUE(store_id, source, captured_at)` com substituição.** Reprocessar a
  mesma foto atualiza a linha em vez de duplicar o histórico -- o que importa
  quando o lote é retomado ou rodado de novo com outro limiar.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from vitrine.domain.models import ShareReport

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id       TEXT    NOT NULL,
    source         TEXT    NOT NULL,
    captured_at    TEXT    NOT NULL,
    analyzed_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    schema_version TEXT    NOT NULL,
    detector       TEXT,
    detections     INTEGER NOT NULL,
    shelves        INTEGER NOT NULL,
    gaps           INTEGER NOT NULL,
    occupancy      REAL    NOT NULL,
    report         TEXT    NOT NULL,
    UNIQUE(store_id, source, captured_at) ON CONFLICT REPLACE
);

CREATE INDEX IF NOT EXISTS idx_analyses_store_date
    ON analyses(store_id, captured_at DESC);

CREATE TABLE IF NOT EXISTS region_shares (
    analysis_id  INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    region       TEXT    NOT NULL,
    count_share  REAL    NOT NULL,
    linear_share REAL    NOT NULL,
    occupancy    REAL    NOT NULL,
    PRIMARY KEY (analysis_id, region)
);
"""


@dataclass(frozen=True)
class Visit:
    """Uma análise registrada, como ela sai do histórico."""

    store_id: str
    source: str
    captured_at: str
    detections: int
    shelves: int
    gaps: int
    occupancy: float
    detector: str | None
    regions: tuple[tuple[str, float, float], ...] = ()
    """``(nome, share por contagem, share por área)`` de cada região."""


class Repository:
    """Acesso ao banco de histórico.

    Use como gerenciador de contexto::

        with Repository(Path("vitrine.db")) as repo:
            repo.save("LOJA_12", report)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> Self:
        """Abre a conexão e garante o esquema."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        # WAL deixa leitura e escrita conviverem: dá para consultar o histórico
        # enquanto um lote longo ainda está gravando.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Fecha a conexão."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Repository usado fora do bloco 'with'")
        return self._conn

    def save(self, store_id: str, report: ShareReport) -> int:
        """Grava uma análise no histórico.

        Args:
            store_id: identificador do ponto de venda.
            report: o relatório completo.

        Returns:
            O id da linha gravada.
        """
        captured = report.image.captured_at if report.image is not None else None
        if captured is None:
            captured = "0000-00-00T00:00:00"

        with closing(self._db.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO analyses
                    (store_id, source, captured_at, schema_version, detector,
                     detections, shelves, gaps, occupancy, report)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    store_id,
                    report.source or "desconhecido",
                    captured,
                    report.schema_version,
                    report.detector.name if report.detector is not None else None,
                    report.total_detections,
                    report.shelf_count,
                    sum(len(s.gaps) for s in report.shelves),
                    _occupancy(report),
                    report.model_dump_json(),
                ),
            )
            analysis_id = int(cur.lastrowid or 0)

            cur.execute("DELETE FROM region_shares WHERE analysis_id = ?", (analysis_id,))
            cur.executemany(
                """
                INSERT INTO region_shares
                    (analysis_id, region, count_share, linear_share, occupancy)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (analysis_id, r.region, r.count_share, r.linear_share, r.occupancy)
                    for r in report.regions
                ],
            )
        self._db.commit()
        return analysis_id

    def history(self, store_id: str, *, limit: int = 30) -> tuple[Visit, ...]:
        """Últimas visitas a um ponto de venda, da mais recente para a mais antiga.

        Args:
            store_id: identificador do ponto de venda.
            limit: quantas visitas trazer.

        Returns:
            As visitas, com as regiões de cada uma.
        """
        linhas = self._db.execute(
            """
            SELECT id, store_id, source, captured_at, detections, shelves,
                   gaps, occupancy, detector
              FROM analyses
             WHERE store_id = ?
             ORDER BY captured_at DESC, id DESC
             LIMIT ?
            """,
            (store_id, limit),
        ).fetchall()

        visitas: list[Visit] = []
        for linha in linhas:
            regioes = self._db.execute(
                """
                SELECT region, count_share, linear_share
                  FROM region_shares
                 WHERE analysis_id = ?
                 ORDER BY rowid
                """,
                (linha["id"],),
            ).fetchall()
            visitas.append(
                Visit(
                    store_id=linha["store_id"],
                    source=linha["source"],
                    captured_at=linha["captured_at"],
                    detections=linha["detections"],
                    shelves=linha["shelves"],
                    gaps=linha["gaps"],
                    occupancy=linha["occupancy"],
                    detector=linha["detector"],
                    regions=tuple(
                        (r["region"], r["count_share"], r["linear_share"]) for r in regioes
                    ),
                )
            )
        return tuple(visitas)

    def stores(self) -> tuple[str, ...]:
        """Pontos de venda com histórico, em ordem alfabética."""
        linhas = self._db.execute(
            "SELECT DISTINCT store_id FROM analyses ORDER BY store_id"
        ).fetchall()
        return tuple(linha["store_id"] for linha in linhas)

    def load_report(self, store_id: str, source: str, captured_at: str) -> dict[str, object] | None:
        """Recupera o relatório completo de uma análise, como dicionário."""
        linha = self._db.execute(
            """
            SELECT report FROM analyses
             WHERE store_id = ? AND source = ? AND captured_at = ?
            """,
            (store_id, source, captured_at),
        ).fetchone()
        if linha is None:
            return None
        carregado: dict[str, object] = json.loads(linha["report"])
        return carregado

    def count(self) -> int:
        """Número de análises guardadas."""
        linha = self._db.execute("SELECT COUNT(*) AS total FROM analyses").fetchone()
        return int(linha["total"])


def _occupancy(report: ShareReport) -> float:
    """Ocupação média da gôndola, ponderada pela largura de cada prateleira."""
    if not report.shelves:
        return 0.0
    ocupado = sum(s.occupied_length for s in report.shelves)
    disponivel = sum(s.extent.width for s in report.shelves)
    return ocupado / disponivel if disponivel > 0 else 0.0

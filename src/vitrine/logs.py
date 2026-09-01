r"""Log estruturado em JSONL.

Um lote de 400 fotos que roda por vinte minutos e falha em três precisa dizer
**quais** três, por quê, e onde o tempo foi gasto. Log em prosa não responde
isso sem alguém ler linha por linha; JSONL responde com uma linha de ``jq``::

    jq -r 'select(.level=="error") | "\\(.image): \\(.error)"' vitrine.jsonl
    jq -s 'map(select(.event=="image_done") | .duration_ms) | add/length' vitrine.jsonl

Por isso cada registro é um objeto JSON completo numa linha, com o tempo de
cada etapa. Sem dependência externa: a stdlib basta.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

LOGGER_NAME = "vitrine"

_RESERVADOS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__)
"""Atributos que o ``logging`` já usa; o resto do ``extra`` é campo nosso."""


class JsonFormatter(logging.Formatter):
    """Formata cada registro como um objeto JSON numa linha."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializa o registro, promovendo o ``extra`` a campos de primeiro nível."""
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        payload.update(
            {
                chave: valor
                for chave, valor in record.__dict__.items()
                if chave not in _RESERVADOS and not chave.startswith("_")
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup(path: Path | None, *, level: int = logging.INFO) -> logging.Logger:
    """Configura o logger do projeto para escrever JSONL em ``path``.

    Args:
        path: arquivo de log; ``None`` desliga a escrita.
        level: nível mínimo registrado.

    Returns:
        O logger configurado.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for antigo in list(logger.handlers):
        logger.removeHandler(antigo)
        antigo.close()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


class StageTimer:
    """Cronometra etapas e devolve as duracoes, sem escrever nada.

    Nao loga de proposito. As etapas acontecem dentro do worker, onde o logger
    do processo pai nao existe -- e configurar um handler por processo faria
    varios escritores disputarem o mesmo arquivo. O worker mede e devolve os
    numeros; quem registra e o pai, numa linha so por imagem.
    """

    def __init__(self) -> None:
        self.durations: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Cronometra um bloco, registrando a duracao mesmo se ele falhar.

        O ``finally`` importa: saber que a deteccao levou quatro segundos antes
        de estourar e informacao, e perde-la seria desperdicio.
        """
        inicio = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = round((time.perf_counter() - inicio) * 1000.0, 1)

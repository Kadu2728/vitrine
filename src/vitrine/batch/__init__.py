"""Processamento em lote: paralelo, resumivel e tolerante a falha.

E o que separa script de ferramenta. Ver ``batch.runner`` para o motivo de cada
decisao e ``batch.manifest`` para o formato que sustenta a retomada.
"""

from vitrine.batch.runner import BatchOptions, BatchSummary, DetectorSpec, ImageOutcome, run_batch

__all__ = ["BatchOptions", "BatchSummary", "DetectorSpec", "ImageOutcome", "run_batch"]

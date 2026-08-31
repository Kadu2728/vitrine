"""Avaliacao de detectores: correspondencia, precisao, recall e AP.

Separado da camada de visao porque e matematica pura sobre caixas -- nao abre
imagem, nao carrega modelo, e por isso e testavel com casos conferidos no papel.
"""

from vitrine.eval.metrics import EvaluationResult, average_precision, evaluate, match

__all__ = ["EvaluationResult", "average_precision", "evaluate", "match"]

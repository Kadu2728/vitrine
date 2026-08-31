"""Metricas de deteccao: precisao, recall e AP@50.

Escrito antes de existir qualquer numero para reportar, e de proposito. Se a
metrica for implementada depois de ver o resultado, a tentacao de escolher a
definicao mais favoravel e real -- e AP tem mais de uma definicao em circulacao.

Definicoes fixadas aqui, para que o numero de ``benchmarks/results.md`` signifique
sempre a mesma coisa:

**Correspondencia.** Uma predicao e verdadeiro positivo se tem IoU maior ou igual
ao limiar com alguma anotacao ainda nao usada. A atribuicao e gulosa em ordem
decrescente de confianca -- cada anotacao casa com no maximo uma predicao, e
predicoes duplicadas sobre o mesmo produto viram falso positivo, que e
exatamente o comportamento desejado.

**Precisao e recall** sao reportados a um limiar de confianca fixo, porque e
nesse regime que a ferramenta roda de verdade.

**AP** usa interpolacao em todos os pontos (a definicao do VOC a partir de 2010,
tambem usada pelo COCO como base): area sob a envoltoria monotonamente
decrescente da curva precisao-recall, varrendo todas as confiancas. Nao e a
interpolacao de 11 pontos, que produz numeros sistematicamente diferentes.

Classe unica: por decisao de escopo o sistema nao identifica SKU, entao "mAP" e
o AP da unica classe existente. O nome ``mean`` seria falso e nao e usado.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vitrine.domain.models import BoundingBox, Detection

DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


class EvaluationResult(BaseModel):
    """O resultado de uma avaliacao, com o suficiente para reproduzi-la."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    images: int = Field(ge=0)
    ground_truth: int = Field(ge=0, description="Total de produtos anotados.")
    predictions: int = Field(ge=0, description="Predicoes acima do limiar de confianca.")
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    average_precision: float = Field(ge=0.0, le=1.0, description="AP no limiar de IoU informado.")
    iou_threshold: float = Field(gt=0.0, le=1.0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)


def match(
    predictions: Sequence[Detection],
    ground_truth: Sequence[BoundingBox],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> tuple[bool, ...]:
    """Diz, para cada predicao, se ela e verdadeiro positivo.

    As predicoes sao consideradas em ordem decrescente de confianca; o resultado
    volta na ordem original da entrada.

    Args:
        predictions: predicoes do detector para uma imagem.
        ground_truth: caixas anotadas da mesma imagem.
        iou_threshold: IoU minimo para considerar acerto.

    Returns:
        Uma tupla de booleanos alinhada com ``predictions``.

    Raises:
        ValueError: se ``iou_threshold`` estiver fora de ``(0, 1]``.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold precisa estar em (0, 1]; recebido {iou_threshold!r}")

    acertos = [False] * len(predictions)
    usadas = [False] * len(ground_truth)

    ordem = sorted(
        range(len(predictions)),
        key=lambda i: (-predictions[i].confidence, predictions[i].sort_key),
    )
    for indice in ordem:
        melhor_iou = iou_threshold
        melhor_alvo = -1
        for alvo, anotada in enumerate(ground_truth):
            if usadas[alvo]:
                continue
            valor = predictions[indice].box.iou(anotada)
            if valor >= melhor_iou:
                melhor_iou = valor
                melhor_alvo = alvo
        if melhor_alvo >= 0:
            usadas[melhor_alvo] = True
            acertos[indice] = True

    return tuple(acertos)


def evaluate(
    samples: Sequence[tuple[Sequence[Detection], Sequence[BoundingBox]]],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> EvaluationResult:
    """Avalia um detector sobre um conjunto de imagens.

    Args:
        samples: pares ``(predicoes, anotacoes)``, uma entrada por imagem.
        iou_threshold: IoU minimo para considerar acerto.
        confidence_threshold: limiar em que precisao e recall sao reportados.

    Returns:
        O resultado consolidado.

    Raises:
        ValueError: se algum limiar estiver fora da faixa valida.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            f"confidence_threshold precisa estar em [0, 1]; recebido {confidence_threshold!r}"
        )

    total_anotado = sum(len(anotacoes) for _, anotacoes in samples)
    pontuacoes: list[tuple[float, bool]] = []
    acima_do_limiar = 0
    acertos_acima = 0

    for predicoes, anotacoes in samples:
        acertos = match(predicoes, anotacoes, iou_threshold=iou_threshold)
        for predicao, acertou in zip(predicoes, acertos, strict=True):
            pontuacoes.append((predicao.confidence, acertou))
            if predicao.confidence >= confidence_threshold:
                acima_do_limiar += 1
                acertos_acima += int(acertou)

    falsos_positivos = acima_do_limiar - acertos_acima
    falsos_negativos = total_anotado - acertos_acima
    precisao = acertos_acima / acima_do_limiar if acima_do_limiar else 0.0
    recall = acertos_acima / total_anotado if total_anotado else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0.0

    return EvaluationResult(
        images=len(samples),
        ground_truth=total_anotado,
        predictions=acima_do_limiar,
        true_positives=acertos_acima,
        false_positives=max(0, falsos_positivos),
        false_negatives=max(0, falsos_negativos),
        precision=precisao,
        recall=recall,
        f1=f1,
        average_precision=average_precision(pontuacoes, total_anotado),
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )


def average_precision(scored: Sequence[tuple[float, bool]], ground_truth: int) -> float:
    """AP por interpolacao em todos os pontos da curva precisao-recall.

    Args:
        scored: pares ``(confianca, acertou)`` de todas as predicoes, de todas
            as imagens, sem filtro de confianca.
        ground_truth: total de caixas anotadas; e o denominador do recall.

    Returns:
        A area sob a envoltoria decrescente da curva, em ``[0, 1]``. Vale
        ``0.0`` quando nao ha anotacao ou nao ha predicao -- nunca ``NaN``.
    """
    if ground_truth <= 0 or not scored:
        return 0.0

    ordenadas = sorted(scored, key=lambda par: (-par[0], not par[1]))

    recalls: list[float] = []
    precisoes: list[float] = []
    acumulado_verdadeiro = 0
    for posicao, (_, acertou) in enumerate(ordenadas, start=1):
        acumulado_verdadeiro += int(acertou)
        recalls.append(acumulado_verdadeiro / ground_truth)
        precisoes.append(acumulado_verdadeiro / posicao)

    # Envoltoria: a precisao em cada ponto vira a maior precisao dali para a
    # frente. Sem isso a curva serrilha e a area fica abaixo da definicao.
    for indice in range(len(precisoes) - 2, -1, -1):
        precisoes[indice] = max(precisoes[indice], precisoes[indice + 1])

    area = 0.0
    recall_anterior = 0.0
    for recall, precisao in zip(recalls, precisoes, strict=True):
        area += (recall - recall_anterior) * precisao
        recall_anterior = recall
    return min(1.0, area)

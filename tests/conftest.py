"""Configuracao comum da suite.

O perfil do hypothesis e *derandomizado* de proposito. Por padrao o hypothesis
sorteia exemplos e mantem um banco de casos entre execucoes -- ou seja, duas
rodadas da suite testam coisas diferentes. Isso contradiz a regra de
determinismo do projeto: a suite que deveria provar que o sistema e
reprodutivel nao seria, ela propria, reprodutivel.

``HYPOTHESIS_PROFILE=thorough`` roda 500 exemplos por propriedade, para uso
local ou noturno. O padrao cabe no orcamento de 5 segundos.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, settings

from helpers import detection

if TYPE_CHECKING:
    from vitrine import Detection

settings.register_profile(
    "fast",
    max_examples=20,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "thorough",
    max_examples=500,
    deadline=None,
    derandomize=True,
    database=None,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))


@pytest.fixture
def gondola() -> list[Detection]:
    """A gondola canonica dos testes, com os numeros conferidos a mao.

    Prateleira 0 (y 0..100), envelope 0..310, largura mediana 30::

        [====80====]  [====80====]     [30] [30] [30]
        0         80  90        170    200  240  280
                                  ^gap 30

    Prateleira 1 (y 200..300), envelope 0..310, largura mediana 60::

        [==60==]                                [==60==]
        0      60                              250    310
                 ^------------- gap 190 -------------^

    Com o corte em 0.5 (x = 155) a prateleira 0 tem 2 produtos a esquerda e 3 a
    direita, mas 145 de comprimento a esquerda contra 105 a direita: a contagem
    diz "direita" e a area diz "esquerda". E o exemplo que justifica publicar as
    duas metricas.
    """
    return [
        detection(0, 0, 80, 100),
        detection(90, 0, 170, 100),
        detection(200, 0, 230, 100),
        detection(240, 0, 270, 100),
        detection(280, 0, 310, 100),
        detection(0, 200, 60, 300),
        detection(250, 200, 310, 300),
    ]

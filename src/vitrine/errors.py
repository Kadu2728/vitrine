"""Erros do Vitrine, cada um carregando o que fazer a respeito.

A CLI nunca imprime stack trace para quem esta usando a ferramenta. Ela imprime
``mensagem`` e ``hint``, e manda o traceback completo para o arquivo de log. Por
isso todo erro previsivel deste projeto nasce aqui com uma dica junto: uma
mensagem que so diz o que aconteceu obriga o usuario a adivinhar o proximo
passo.
"""

from __future__ import annotations


class VitrineError(Exception):
    """Erro previsivel, com orientacao de correcao.

    Args:
        message: o que aconteceu, em uma frase.
        hint: o que fazer a respeito, concreto o bastante para copiar e colar.
    """

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        """Mensagem seguida da dica, para uso em log e em testes."""
        return f"{self.message} {self.hint}"


class ImageLoadError(VitrineError):
    """A imagem nao pode ser lida, decodificada ou e grande demais."""


class PerspectiveError(VitrineError):
    """Os quatro pontos informados nao formam um quadrilatero utilizavel."""


class DetectorError(VitrineError):
    """O detector nao pode ser construido ou falhou durante a inferencia."""


class UsageError(VitrineError):
    """Combinacao de argumentos invalida na linha de comando."""

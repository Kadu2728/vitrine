"""Persistencia do historico por ponto de venda, em SQLite.

Ferramenta local nao precisa de servidor de banco: o historico e um arquivo.
Ver ``storage.repository`` para as decisoes de esquema.
"""

from vitrine.storage.repository import Repository, Visit

__all__ = ["Repository", "Visit"]

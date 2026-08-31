"""A regra de dependencia, verificada em vez de prometida.

O dominio nao pode importar nada de ``vision``, ``storage``, ``batch``,
``render``, ``pipeline`` ou ``cli``, nem depender de imagem, modelo ou IO. Uma
regra que so existe no README e uma regra que sera quebrada na primeira
segunda-feira apertada. Esta a suite quebra.

O teste tambem e o guarda da Fase 1: se alguem instalar Ultralytics ou OpenCV e
importar no dominio para "adiantar", a suite acusa.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import vitrine

DOMAIN = Path(vitrine.__file__).parent / "domain"

CAMADAS_PROIBIDAS = frozenset(
    {"vitrine.vision", "vitrine.storage", "vitrine.batch", "vitrine.render", "vitrine.cli"}
)
"""Camadas externas ao dominio. A seta de dependencia aponta sempre para dentro."""

BIBLIOTECAS_PROIBIDAS = frozenset(
    {"cv2", "numpy", "PIL", "ultralytics", "torch", "sqlite3", "sqlalchemy", "typer", "rich"}
)
"""Infraestrutura. O dominio calcula; nao le arquivo, nao carrega peso, nao desenha."""


def modulos_do_dominio() -> list[Path]:
    return sorted(DOMAIN.glob("*.py"))


def imports_de(caminho: Path) -> set[str]:
    """Nomes de modulo importados por um arquivo, no nivel de topo do nome."""
    arvore = ast.parse(caminho.read_text(encoding="utf-8"), filename=str(caminho))
    nomes: set[str] = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            nomes.update(alias.name for alias in no.names)
        elif isinstance(no, ast.ImportFrom) and no.module is not None and no.level == 0:
            nomes.add(no.module)
    return nomes


def test_o_dominio_tem_modulos() -> None:
    """Guarda contra o teste passar por nao encontrar arquivo nenhum."""
    assert len(modulos_do_dominio()) >= 5


@pytest.mark.parametrize("modulo", modulos_do_dominio(), ids=lambda p: p.name)
def test_dominio_nao_importa_camada_externa(modulo: Path) -> None:
    for nome in imports_de(modulo):
        raiz = ".".join(nome.split(".")[:2])
        assert raiz not in CAMADAS_PROIBIDAS, (
            f"{modulo.name} importa {nome}: o dominio nao pode depender de camada externa"
        )


@pytest.mark.parametrize("modulo", modulos_do_dominio(), ids=lambda p: p.name)
def test_dominio_nao_importa_infraestrutura(modulo: Path) -> None:
    for nome in imports_de(modulo):
        raiz = nome.split(".")[0]
        assert raiz not in BIBLIOTECAS_PROIBIDAS, (
            f"{modulo.name} importa {nome}: o dominio precisa ser testavel sem "
            f"carregar modelo, abrir imagem ou tocar em disco"
        )


def test_dominio_nao_faz_io() -> None:
    """Nenhuma chamada a ``open`` nos modulos de dominio."""
    for modulo in modulos_do_dominio():
        arvore = ast.parse(modulo.read_text(encoding="utf-8"), filename=str(modulo))
        for no in ast.walk(arvore):
            if isinstance(no, ast.Call) and isinstance(no.func, ast.Name):
                assert no.func.id != "open", f"{modulo.name} chama open()"


def test_o_modelo_e_extra_opcional_e_nao_dependencia_base() -> None:
    """Ultralytics e torch nao podem entrar nas dependencias base.

    A regra R2 -- o modelo e injetado, nunca importado direto na logica -- so
    vale se o pacote realmente instalar e funcionar sem ele. Se o peso virar
    dependencia obrigatoria, a injecao vira decoracao e ``pipx install`` passa
    a baixar gigabytes.
    """
    raiz = Path(vitrine.__file__).parents[2]
    pyproject = (raiz / "pyproject.toml").read_text(encoding="utf-8")
    base = pyproject.split("dependencies = [")[1].split("]")[0]
    for proibida in ("ultralytics", "torch", "sqlalchemy", "fastapi"):
        assert proibida not in base, f"{proibida} nao pertence as dependencias base"
    assert 'yolo = ["ultralytics' in pyproject, "ultralytics precisa existir como extra"


def test_nada_fora_de_yolo_importa_ultralytics() -> None:
    """Apenas ``vision/yolo.py`` pode conhecer o Ultralytics.

    E o modulo que a R2 isola atras do protocolo. Um import em qualquer outro
    lugar significa que a logica passou a depender do framework.
    """
    pacote = Path(vitrine.__file__).parent
    for modulo in sorted(pacote.rglob("*.py")):
        if modulo.name == "yolo.py":
            continue
        fonte = modulo.read_text(encoding="utf-8")
        assert "ultralytics" not in fonte, f"{modulo.name} menciona ultralytics"

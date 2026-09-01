"""Pagina de demonstracao do Vitrine.

**Isto nao faz parte do produto.** Mora fora de ``src/vitrine/`` de proposito e
consome a biblioteca exatamente como qualquer outro cliente faria -- importando
``analyze_image`` e ``annotate``, sem atalho e sem acesso a nada interno. Se
esta pagina sumir amanha, o Vitrine continua inteiro.

Essa separacao nao e formalidade: e a prova da regra R1. Uma interface que so
consegue existir mexendo nas entranhas do pacote denunciaria que a biblioteca
nao e realmente reutilizavel. Esta aqui nao precisa.

O produto continua sem camada web: a saida visual e a imagem anotada e a
integracao se faz por ``--json``. Esta pagina existe para uma coisa so --
permitir que alguem arraste uma foto e veja o resultado sem instalar Python.

Uso local::

    uv run python app/gradio_app.py

Link publico temporario (tunel do Gradio, expira em 72 h)::

    uv run python app/gradio_app.py --share
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gradio as gr

from vitrine import ContourDetector, RegionSet, analyze_image, annotate
from vitrine.errors import VitrineError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

AVISO = """
### O que esta demonstracao mostra, e o que ela nao mostra

O detector disponivel aqui e o **por contorno**, que encontra retangulos de alto
contraste. Ele funciona bem em imagem sintetica ou em gondola muito organizada,
e **funciona mal em foto de supermercado real** -- iluminacao irregular,
embalagem brilhante e produto encostado em produto derrubam o metodo.

Isso e uma limitacao conhecida e declarada, nao um bug. O detector treinado em
gondola (SKU-110K) e o proximo passo do projeto; ate la, as metricas de deteccao
estao registradas como **nao medidas**.

O que esta pagina demonstra de verdade e o **pipeline completo**: carregamento
com correcao de orientacao EXIF, correcao de perspectiva, agrupamento em
prateleiras, share por contagem e por area, deteccao de vazio e renderizacao.
Troque o detector e todo o resto continua igual -- e exatamente esse o ponto da
arquitetura.
"""


def analisar(
    imagem: str | None,
    inverter: bool,
    usar_regioes: bool,
    corte: float,
    nome_esquerda: str,
    nome_direita: str,
    max_size: int,
) -> tuple[NDArray[np.uint8] | None, str, dict[str, Any] | None]:
    """Roda a analise e devolve imagem anotada, resumo em texto e JSON."""
    if not imagem:
        return None, "Envie uma foto de gondola para comecar.", None

    regioes = (
        RegionSet.from_cuts((0.0, corte, 1.0), (nome_esquerda.strip(), nome_direita.strip()))
        if usar_regioes
        else None
    )

    try:
        resultado = analyze_image(
            Path(imagem),
            ContourDetector(invert=inverter),
            regions=regioes,
            max_size=max_size,
        )
    except VitrineError as erro:
        # A pagina herda a regra da CLI: mensagem e dica, nunca stack trace.
        return None, f"**{erro.message}**\n\n{erro.hint}", None

    report = resultado.report
    if report.status == "no_detections":
        return (
            resultado.pixels,
            "**Nenhum produto detectado.**\n\nSe os produtos sao mais escuros "
            "que o fundo, marque *Inverter polaridade*. Em foto de loja real, "
            "este detector costuma falhar mesmo -- ver o aviso abaixo.",
            report.model_dump(),
        )

    return (
        annotate(resultado.pixels, report),
        _resumo(report),
        report.model_dump(),
    )


def _resumo(report: Any) -> str:
    """Monta o resumo em Markdown, com as duas metricas lado a lado."""
    linhas = [
        f"**{report.total_detections} produtos** em "
        f"**{report.shelf_count} prateleira(s)**, "
        f"**{sum(len(s.gaps) for s in report.shelves)} vazio(s)**",
        "",
        "| Prat. | Produtos | Ocupacao | Vazios |",
        "|---|---|---|---|",
    ]
    linhas.extend(
        f"| {s.index} | {s.detection_count} | {s.occupancy:.0%} | {len(s.gaps)} |"
        for s in report.shelves
    )

    if len(report.regions) > 1:
        linhas += [
            "",
            "| Regiao | Share por contagem | Share por area |",
            "|---|---|---|",
        ]
        linhas.extend(
            f"| {r.region} | {r.count_share:.0%} | {r.linear_share:.0%} |" for r in report.regions
        )
        linhas += [
            "",
            "_As duas colunas respondem perguntas diferentes e costumam discordar._",
        ]

    if report.warnings:
        linhas += ["", "**Avisos:**"] + [f"- {a}" for a in report.warnings]

    return "\n".join(linhas)


def construir() -> Any:
    """Monta a interface.

    Devolve ``Any`` porque o ``gr.Blocks`` do Gradio nao carrega tipagem
    suficiente para o mypy estrito: anotar como ``gr.Blocks`` obrigaria a um
    ``cast`` que nao verifica nada de verdade.
    """
    with gr.Blocks(title="Vitrine -- auditoria de gondola") as pagina:
        gr.Markdown(
            "# Vitrine\n"
            "### Auditoria de execucao em ponto de venda por visao computacional\n"
            "Arraste uma foto de gondola. O sistema conta produtos, agrupa em "
            "prateleiras, mede share e aponta ruptura."
        )

        with gr.Row():
            with gr.Column(scale=1):
                entrada = gr.Image(type="filepath", label="Foto da gondola", sources=["upload"])
                inverter = gr.Checkbox(
                    label="Inverter polaridade",
                    info="Marque se os produtos sao mais escuros que o fundo.",
                )
                max_size = gr.Slider(
                    600,
                    3000,
                    value=2000,
                    step=100,
                    label="Maior lado da imagem (px)",
                    info="Reduz antes de analisar. Menor = mais rapido.",
                )
                with gr.Accordion("Dividir a gondola em regioes", open=False):
                    usar_regioes = gr.Checkbox(label="Calcular share entre duas regioes")
                    corte = gr.Slider(
                        0.05,
                        0.95,
                        value=0.5,
                        step=0.05,
                        label="Fronteira",
                        info="Fracao da largura da prateleira onde a divisao cai.",
                    )
                    nome_esquerda = gr.Textbox(
                        value="minha_marca", label="Nome da regiao a esquerda"
                    )
                    nome_direita = gr.Textbox(
                        value="concorrencia", label="Nome da regiao a direita"
                    )
                botao = gr.Button("Analisar", variant="primary")

            with gr.Column(scale=2):
                saida_imagem = gr.Image(label="Resultado", type="numpy")
                saida_texto = gr.Markdown()
                with gr.Accordion("JSON completo (o contrato de integracao)", open=False):
                    saida_json = gr.JSON()

        gr.Markdown(AVISO)

        botao.click(
            analisar,
            inputs=[entrada, inverter, usar_regioes, corte, nome_esquerda, nome_direita, max_size],
            outputs=[saida_imagem, saida_texto, saida_json],
        )

        exemplo = Path(__file__).resolve().parents[1] / "examples" / "gondola.png"
        if exemplo.is_file():
            gr.Examples(
                examples=[[str(exemplo), True]],
                inputs=[entrada, inverter],
                label="Exemplo sintetico (onde o detector por contorno funciona bem)",
            )

    return pagina


def main() -> None:
    """Sobe a pagina."""
    parser = argparse.ArgumentParser(description="Pagina de demonstracao do Vitrine.")
    parser.add_argument(
        "--share",
        action="store_true",
        help=(
            "Cria um link publico temporario pelo tunel do Gradio (expira em 72 h). "
            "Isso expoe a sua maquina para a internet enquanto o processo estiver no ar."
        ),
    )
    parser.add_argument("--port", type=int, default=7860, help="Porta local.")
    argumentos = parser.parse_args()

    gr.set_static_paths([Path(tempfile.gettempdir())])
    construir().launch(share=argumentos.share, server_port=argumentos.port)


if __name__ == "__main__":
    main()

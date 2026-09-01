---
title: Vitrine - Auditoria de Gondola
emoji: 🛒
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: "5.0.0"
app_file: gradio_app.py
pinned: false
---

# Pagina de demonstracao do Vitrine

Esta pasta **nao faz parte do produto**. Ela existe para que alguem consiga
arrastar uma foto e ver o resultado sem instalar Python.

O Vitrine continua sem camada web: a saida visual e a imagem anotada e a
integracao com outros sistemas se faz por `--json`. Esta pagina e apenas mais um
consumidor da biblioteca -- importa `analyze_image` e `annotate` e nada mais.
Se ela sumir, o projeto continua inteiro.

Manter a demonstracao aqui fora, em vez de dentro de `src/vitrine/`, e a prova
pratica da regra R1: uma interface que so conseguisse existir mexendo nas
entranhas do pacote denunciaria que a biblioteca nao e reutilizavel de verdade.

## Rodar localmente

```bash
uv run python app/gradio_app.py
```

Abre em <http://127.0.0.1:7860>.

## Link publico temporario

```bash
uv run python app/gradio_app.py --share
```

Cria um endereco `*.gradio.live` que dura 72 horas. Enquanto o processo estiver
rodando, a sua maquina fica acessivel pela internet -- use com consciencia disso.

## Deploy permanente no Hugging Face Spaces

1. Crie um Space em <https://huggingface.co/new-space> com SDK **Gradio**.
2. Copie `gradio_app.py`, `requirements.txt` e este `README.md` para a raiz do
   Space (o bloco YAML no topo deste arquivo e o que o Spaces le como
   configuracao).
3. `git push`. O build instala `vitrine-shelf` do PyPI -- o que exige que o
   pacote esteja publicado (Fase 4). Ate la, troque a linha do
   `requirements.txt` por uma instalacao direta do repositorio:

   ```
   git+https://github.com/<usuario>/<repo>.git
   ```

## Limitacao que a pagina declara em primeiro plano

O detector disponivel e o **por contorno**, que nao funciona bem em foto de loja
real. Isso esta escrito na propria pagina, acima da area de resultado, e nao em
letra miuda. O detector treinado em gondola e o proximo passo do projeto.

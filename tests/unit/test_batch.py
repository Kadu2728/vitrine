"""Testes do lote: manifesto, retomada, isolamento de falha e paralelismo.

O que se verifica aqui não é que o lote roda — é que ele **sobrevive**: a
interrupção, à imagem corrompida, à execução repetida. Um lote que só funciona
no caminho feliz é um `for` com barra de progresso.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from helpers import synthetic_shelf, write_image
from vitrine.batch import manifest
from vitrine.batch.runner import BatchOptions, DetectorSpec, ImageOutcome, run_batch
from vitrine.errors import VitrineError
from vitrine.logs import setup

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration
"""Estes testes criam arquivos, bancos e invocam a CLI inteira.

Continuam rodando por padrao -- marcar nao e esconder. A marca existe para
que quem esta mexendo no dominio possa rodar
``pytest -m 'not slow and not integration'`` e ter resposta em menos de dois
segundos.
"""


OPCOES = BatchOptions(detector=DetectorSpec(kind="contour"), write_artifacts=False)


def povoar(pasta: Path, quantidade: int = 3) -> Path:
    """Cria uma pasta com N fotos sintéticas válidas."""
    pasta.mkdir(parents=True, exist_ok=True)
    for indice in range(quantidade):
        pixels, _ = synthetic_shelf(rows=2, columns=3)
        write_image(pasta / f"foto_{indice:02d}.png", pixels)
    return pasta


class TestManifesto:
    def test_pasta_sem_manifesto_e_lote_novo(self, tmp_path: Path) -> None:
        assert manifest.read(tmp_path / "manifest.jsonl") == {}

    def test_ida_e_volta(self, tmp_path: Path) -> None:
        caminho = tmp_path / "manifest.jsonl"
        entrada = manifest.Entry(
            key="a.png:10:20", source="a.png", status="ok", duration_ms=12.5, detections=6
        )
        with manifest.Writer(caminho) as escritor:
            escritor.append(entrada)
        assert manifest.read(caminho) == {entrada.key: entrada}

    def test_linha_truncada_nao_derruba_a_leitura(self, tmp_path: Path) -> None:
        """Queda de energia no meio de uma escrita não pode custar o progresso todo."""
        caminho = tmp_path / "manifest.jsonl"
        boa = manifest.Entry(key="a:1:1", source="a.png", status="ok", duration_ms=1.0)
        caminho.write_text(boa.to_json() + "\n" + '{"key": "b", "sta', encoding="utf-8")
        entradas = manifest.read(caminho)
        assert list(entradas) == ["a:1:1"]

    def test_linha_sem_chave_e_descartada(self, tmp_path: Path) -> None:
        caminho = tmp_path / "manifest.jsonl"
        caminho.write_text('{"source": "a.png"}\n', encoding="utf-8")
        assert manifest.read(caminho) == {}

    def test_campo_desconhecido_nao_quebra(self, tmp_path: Path) -> None:
        caminho = tmp_path / "manifest.jsonl"
        caminho.write_text(
            json.dumps(
                {
                    "key": "a:1:1",
                    "source": "a.png",
                    "status": "ok",
                    "duration_ms": 1.0,
                    "campo_do_futuro": 42,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        assert "a:1:1" in manifest.read(caminho)

    def test_escritor_fora_do_with(self, tmp_path: Path) -> None:
        escritor = manifest.Writer(tmp_path / "m.jsonl")
        with pytest.raises(RuntimeError, match="fora do bloco"):
            escritor.append(manifest.Entry(key="a", source="a", status="ok", duration_ms=0.0))

    def test_chave_muda_quando_a_imagem_muda(self, tmp_path: Path) -> None:
        pasta = povoar(tmp_path / "fotos", 1)
        foto = next(iter(pasta.iterdir()))
        antes = manifest.image_key(foto, pasta)

        pixels, _ = synthetic_shelf(rows=3, columns=4)
        write_image(foto, pixels)
        assert manifest.image_key(foto, pasta) != antes

    def test_chave_e_relativa_a_raiz(self, tmp_path: Path) -> None:
        """Mover a pasta inteira não pode invalidar o progresso."""
        pasta = povoar(tmp_path / "fotos", 1)
        foto = next(iter(pasta.iterdir()))
        assert manifest.image_key(foto, pasta).startswith("foto_00.png:")

    def test_busca_e_recursiva_e_ordenada(self, tmp_path: Path) -> None:
        raiz = povoar(tmp_path / "fotos", 2)
        povoar(raiz / "subpasta", 2)
        encontradas = list(manifest.iter_images(raiz, frozenset({".png"})))
        assert len(encontradas) == 4
        assert [p.as_posix() for p in encontradas] == sorted(p.as_posix() for p in encontradas)


class TestLoteBasico:
    def test_processa_a_pasta_inteira(self, tmp_path: Path) -> None:
        resumo = run_batch(povoar(tmp_path / "fotos", 3), out_dir=tmp_path / "out", options=OPCOES)
        assert resumo.total == 3
        assert resumo.processed == 3
        assert resumo.failed == 0
        assert resumo.skipped == 0
        assert resumo.interrupted is False

    def test_grava_artefatos_quando_pedido(self, tmp_path: Path) -> None:
        saida = tmp_path / "out"
        run_batch(
            povoar(tmp_path / "fotos", 2),
            out_dir=saida,
            options=BatchOptions(write_artifacts=True),
        )
        assert (saida / "foto_00.json").is_file()
        assert (saida / "foto_00.anotada.jpg").is_file()

    def test_sem_artefatos_grava_so_o_manifesto(self, tmp_path: Path) -> None:
        saida = tmp_path / "out"
        run_batch(povoar(tmp_path / "fotos", 2), out_dir=saida, options=OPCOES)
        assert (saida / manifest.MANIFEST_NAME).is_file()
        assert not (saida / "foto_00.json").exists()

    def test_pasta_inexistente(self, tmp_path: Path) -> None:
        with pytest.raises(VitrineError, match="nao e uma pasta") as exc:
            run_batch(tmp_path / "fantasma", options=OPCOES)
        assert "vitrine batch" in exc.value.hint

    def test_pasta_sem_imagem(self, tmp_path: Path) -> None:
        vazia = tmp_path / "vazia"
        vazia.mkdir()
        with pytest.raises(VitrineError, match="Nenhuma imagem") as exc:
            run_batch(vazia, options=OPCOES)
        assert "recursiva" in exc.value.hint


class TestRetomada:
    def test_segunda_execucao_pula_tudo(self, tmp_path: Path) -> None:
        pasta = povoar(tmp_path / "fotos", 3)
        saida = tmp_path / "out"
        run_batch(pasta, out_dir=saida, options=OPCOES)

        segunda = run_batch(pasta, out_dir=saida, options=OPCOES)
        assert segunda.processed == 0
        assert segunda.skipped == 3

    def test_retoma_de_onde_parou(self, tmp_path: Path) -> None:
        """O cenário real: metade do lote foi feita e o resto continua."""
        pasta = povoar(tmp_path / "fotos", 4)
        saida = tmp_path / "out"

        # Simula uma execução que terminou só as duas primeiras.
        feitas = sorted(pasta.iterdir())[:2]
        caminho_manifesto = saida / manifest.MANIFEST_NAME
        with manifest.Writer(caminho_manifesto) as escritor:
            for foto in feitas:
                escritor.append(
                    manifest.Entry(
                        key=manifest.image_key(foto, pasta),
                        source=foto.name,
                        status="ok",
                        duration_ms=1.0,
                    )
                )

        resumo = run_batch(pasta, out_dir=saida, options=OPCOES)
        assert resumo.skipped == 2
        assert resumo.processed == 2

    def test_no_resume_reprocessa_tudo(self, tmp_path: Path) -> None:
        pasta = povoar(tmp_path / "fotos", 3)
        saida = tmp_path / "out"
        run_batch(pasta, out_dir=saida, options=OPCOES)

        segunda = run_batch(pasta, out_dir=saida, options=OPCOES, resume=False)
        assert segunda.processed == 3
        assert segunda.skipped == 0

    def test_imagem_alterada_e_reprocessada(self, tmp_path: Path) -> None:
        pasta = povoar(tmp_path / "fotos", 2)
        saida = tmp_path / "out"
        run_batch(pasta, out_dir=saida, options=OPCOES)

        alvo = sorted(pasta.iterdir())[0]
        pixels, _ = synthetic_shelf(rows=3, columns=5)
        write_image(alvo, pixels)

        segunda = run_batch(pasta, out_dir=saida, options=OPCOES)
        assert segunda.processed == 1
        assert segunda.skipped == 1

    def test_sem_pasta_de_saida_nao_ha_retomada(self, tmp_path: Path) -> None:
        pasta = povoar(tmp_path / "fotos", 2)
        run_batch(pasta, options=OPCOES)
        segunda = run_batch(pasta, options=OPCOES)
        assert segunda.processed == 2


class TestIsolamentoDeFalha:
    def test_imagem_corrompida_nao_derruba_o_lote(self, tmp_path: Path) -> None:
        pasta = povoar(tmp_path / "fotos", 3)
        (pasta / "corrompida.png").write_text("isto nao e uma imagem", encoding="utf-8")

        resumo = run_batch(pasta, out_dir=tmp_path / "out", options=OPCOES)
        assert resumo.total == 4
        assert resumo.processed == 3
        assert resumo.failed == 1
        assert resumo.failures[0][0] == "corrompida.png"

    def test_a_falha_traz_a_dica_junto(self, tmp_path: Path) -> None:
        pasta = povoar(tmp_path / "fotos", 1)
        (pasta / "ruim.png").write_bytes(b"\x89PNG quebrado")
        resumo = run_batch(pasta, out_dir=tmp_path / "out", options=OPCOES)
        assert "corrompido" in resumo.failures[0][1]

    def test_falha_entra_no_manifesto_e_nao_e_retentada(self, tmp_path: Path) -> None:
        """Imagem que falhou fica registrada: o lote não fica batendo nela."""
        pasta = povoar(tmp_path / "fotos", 1)
        (pasta / "ruim.png").write_text("nao e imagem", encoding="utf-8")
        saida = tmp_path / "out"

        run_batch(pasta, out_dir=saida, options=OPCOES)
        entradas = manifest.read(saida / manifest.MANIFEST_NAME)
        assert any(e.status == "failed" for e in entradas.values())

        segunda = run_batch(pasta, out_dir=saida, options=OPCOES)
        assert segunda.skipped == 2
        assert segunda.processed == 0


class TestParalelismo:
    @pytest.mark.slow
    def test_dois_workers_produzem_o_mesmo_resultado(self, tmp_path: Path) -> None:
        """Marcado slow: subir processo no Windows custa segundos."""
        pasta = povoar(tmp_path / "fotos", 4)
        sequencial = run_batch(pasta, out_dir=tmp_path / "a", options=OPCOES, workers=1)
        paralelo = run_batch(pasta, out_dir=tmp_path / "b", options=OPCOES, workers=2)
        assert sequencial.processed == paralelo.processed == 4
        assert sequencial.failed == paralelo.failed == 0

    def test_a_especificacao_do_detector_e_serializavel(self) -> None:
        """O que atravessa a fronteira do processo precisa passar pelo pickle."""
        import pickle

        spec = DetectorSpec(kind="contour", invert=True)
        assert pickle.loads(pickle.dumps(spec)) == spec

    def test_as_opcoes_do_lote_sao_serializaveis(self) -> None:
        import pickle

        from vitrine import RegionSet

        opcoes = BatchOptions(
            detector=DetectorSpec(kind="contour"),
            regions=RegionSet.from_cuts((0.0, 0.5, 1.0), ("a", "b")),
        )
        assert pickle.loads(pickle.dumps(opcoes)).regions == opcoes.regions

    def test_o_detector_e_construido_uma_vez_por_processo(self) -> None:
        spec = DetectorSpec(kind="contour")
        assert spec.build() is spec.build()


class TestLogEstruturado:
    def test_gera_uma_linha_json_por_evento(self, tmp_path: Path) -> None:
        log = tmp_path / "vitrine.jsonl"
        logger = setup(log)
        run_batch(
            povoar(tmp_path / "fotos", 2),
            out_dir=tmp_path / "out",
            options=OPCOES,
            logger=logger,
        )
        registros = [json.loads(linha) for linha in log.read_text(encoding="utf-8").splitlines()]
        eventos = [r["event"] for r in registros]
        assert "batch_start" in eventos
        assert eventos.count("image_done") == 2
        assert "batch_done" in eventos

    def test_registra_duracao_por_imagem(self, tmp_path: Path) -> None:
        log = tmp_path / "vitrine.jsonl"
        logger = setup(log)
        run_batch(
            povoar(tmp_path / "fotos", 1),
            out_dir=tmp_path / "out",
            options=OPCOES,
            logger=logger,
        )
        feitas = [
            json.loads(linha)
            for linha in log.read_text(encoding="utf-8").splitlines()
            if json.loads(linha)["event"] == "image_done"
        ]
        assert feitas[0]["duration_ms"] >= 0.0
        assert feitas[0]["detections"] == 6

    def test_falha_sai_com_nivel_de_erro(self, tmp_path: Path) -> None:
        pasta = povoar(tmp_path / "fotos", 1)
        (pasta / "ruim.png").write_text("nao", encoding="utf-8")
        log = tmp_path / "vitrine.jsonl"
        logger = setup(log)
        run_batch(pasta, out_dir=tmp_path / "out", options=OPCOES, logger=logger)

        erros = [
            json.loads(linha)
            for linha in log.read_text(encoding="utf-8").splitlines()
            if json.loads(linha)["level"] == "error"
        ]
        assert len(erros) == 1
        assert erros[0]["image"] == "ruim.png"

    def test_registra_o_tempo_de_cada_etapa(self, tmp_path: Path) -> None:
        """A especificacao pede tempo por etapa, nao so por imagem."""
        log = tmp_path / "vitrine.jsonl"
        logger = setup(log)
        run_batch(
            povoar(tmp_path / "fotos", 1),
            out_dir=tmp_path / "out",
            options=BatchOptions(write_artifacts=True),
            logger=logger,
        )
        feitas = [
            json.loads(linha)
            for linha in log.read_text(encoding="utf-8").splitlines()
            if json.loads(linha)["event"] == "image_done"
        ]
        etapas = feitas[0]["stages_ms"]
        assert set(etapas) == {"detector", "analyze", "render"}
        assert all(valor >= 0.0 for valor in etapas.values())

    def test_etapa_que_falha_ainda_e_cronometrada(self, tmp_path: Path) -> None:
        """Saber que a analise levou tempo antes de estourar e informacao."""
        pasta = povoar(tmp_path / "fotos", 0)
        (pasta / "ruim.png").write_text("nao e imagem", encoding="utf-8")
        log = tmp_path / "vitrine.jsonl"
        logger = setup(log)
        run_batch(pasta, out_dir=tmp_path / "out", options=OPCOES, logger=logger)
        erros = [
            json.loads(linha)
            for linha in log.read_text(encoding="utf-8").splitlines()
            if json.loads(linha)["level"] == "error"
        ]
        assert "analyze" in erros[0]["stages_ms"]

    def test_sem_arquivo_o_log_nao_quebra(self, tmp_path: Path) -> None:
        logger = setup(None)
        resumo = run_batch(
            povoar(tmp_path / "fotos", 1), out_dir=tmp_path / "out", options=OPCOES, logger=logger
        )
        assert resumo.processed == 1


class TestInterrupcao:
    def test_ctrl_c_preserva_o_progresso_ja_feito(self, tmp_path: Path) -> None:
        """O teste que dá sentido ao manifesto.

        Interrompe de verdade no meio do lote e verifica que o que já tinha
        sido processado sobreviveu -- e que rodar de novo continua dali.
        """
        pasta = povoar(tmp_path / "fotos", 5)
        saida = tmp_path / "out"
        processadas: list[str] = []

        def interromper_na_terceira(resultado: ImageOutcome) -> None:
            processadas.append(resultado.path.name)
            if len(processadas) == 3:
                raise KeyboardInterrupt

        resumo = run_batch(
            pasta,
            out_dir=saida,
            options=OPCOES,
            on_result=interromper_na_terceira,
        )
        assert resumo.interrupted is True

        # O manifesto guardou as três: o registro acontece antes do callback.
        entradas = manifest.read(saida / manifest.MANIFEST_NAME)
        assert len(entradas) == 3

        # E a retomada pega exatamente as duas que faltavam.
        segunda = run_batch(pasta, out_dir=saida, options=OPCOES)
        assert segunda.skipped == 3
        assert segunda.processed == 2
        assert segunda.interrupted is False

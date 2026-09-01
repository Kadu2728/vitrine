"""Testes do histórico em SQLite.

Banco temporário a cada teste, criado do zero pelo próprio esquema. Nenhum
`.db` versionado e nenhum estado carregado entre testes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from helpers import detection
from vitrine import (
    DetectorInfo,
    ImageMeta,
    RegionSet,
    Repository,
    ShareReport,
    analyze_detections,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration
"""Estes testes criam arquivos, bancos e invocam a CLI inteira.

Continuam rodando por padrao -- marcar nao e esconder. A marca existe para
que quem esta mexendo no dominio possa rodar
``pytest -m 'not slow and not integration'`` e ter resposta em menos de dois
segundos.
"""


ESQUERDA_DIREITA = RegionSet.from_cuts((0.0, 0.5, 1.0), ("minha", "concorrencia"))


def relatorio(
    *,
    fonte: str = "foto.jpg",
    captura: str = "2026-08-31T10:00:00",
    produtos: int = 3,
    regioes: RegionSet | None = None,
) -> ShareReport:
    """Monta um relatório com procedência, como o pipeline produziria."""
    deteccoes = [detection(i * 100, 0, i * 100 + 60, 100) for i in range(produtos)]
    return analyze_detections(
        deteccoes,
        regions=regioes,
        source=fonte,
        image=ImageMeta(
            name=fonte,
            width=800,
            height=600,
            exif_rotated=False,
            downscale=1.0,
            rectified=False,
            captured_at=captura,
        ),
        detector=DetectorInfo(
            name="contour",
            version="teste",
            confidence_threshold=0.0,
            iou_threshold=0.0,
        ),
    )


class TestEsquema:
    def test_cria_o_banco_do_zero(self, tmp_path: Path) -> None:
        caminho = tmp_path / "sub" / "vitrine.db"
        with Repository(caminho) as repo:
            assert repo.count() == 0
        assert caminho.is_file()

    def test_abrir_de_novo_preserva_os_dados(self, tmp_path: Path) -> None:
        caminho = tmp_path / "vitrine.db"
        with Repository(caminho) as repo:
            repo.save("LOJA_1", relatorio())
        with Repository(caminho) as repo:
            assert repo.count() == 1

    def test_uso_fora_do_with(self, tmp_path: Path) -> None:
        repo = Repository(tmp_path / "v.db")
        with pytest.raises(RuntimeError, match="fora do bloco"):
            repo.count()


class TestGravacao:
    def test_grava_e_recupera(self, tmp_path: Path) -> None:
        with Repository(tmp_path / "v.db") as repo:
            repo.save("LOJA_12", relatorio(produtos=4))
            (visita,) = repo.history("LOJA_12")
        assert visita.store_id == "LOJA_12"
        assert visita.detections == 4
        assert visita.detector == "contour"

    def test_grava_as_regioes(self, tmp_path: Path) -> None:
        with Repository(tmp_path / "v.db") as repo:
            repo.save("LOJA_12", relatorio(regioes=ESQUERDA_DIREITA))
            (visita,) = repo.history("LOJA_12")
        assert [nome for nome, _, _ in visita.regions] == ["minha", "concorrencia"]

    def test_o_relatorio_completo_e_recuperavel(self, tmp_path: Path) -> None:
        """O JSON inteiro fica guardado: dá para reprocessar sem a foto original."""
        with Repository(tmp_path / "v.db") as repo:
            repo.save("LOJA_12", relatorio())
            payload = repo.load_report("LOJA_12", "foto.jpg", "2026-08-31T10:00:00")
        assert payload is not None
        assert payload["schema_version"] == "1.2"
        assert payload["shelves"]

    def test_relatorio_inexistente(self, tmp_path: Path) -> None:
        with Repository(tmp_path / "v.db") as repo:
            assert repo.load_report("LOJA_X", "nada.jpg", "2026-01-01T00:00:00") is None

    def test_reprocessar_a_mesma_foto_substitui(self, tmp_path: Path) -> None:
        """Retomar um lote ou rodar de novo não pode duplicar o histórico."""
        with Repository(tmp_path / "v.db") as repo:
            repo.save("LOJA_12", relatorio(produtos=3))
            repo.save("LOJA_12", relatorio(produtos=5))
            visitas = repo.history("LOJA_12")
        assert len(visitas) == 1
        assert visitas[0].detections == 5

    def test_fotos_diferentes_no_mesmo_dia_convivem(self, tmp_path: Path) -> None:
        with Repository(tmp_path / "v.db") as repo:
            repo.save("LOJA_12", relatorio(fonte="corredor_a.jpg"))
            repo.save("LOJA_12", relatorio(fonte="corredor_b.jpg"))
            assert len(repo.history("LOJA_12")) == 2

    def test_relatorio_sem_imagem_nao_quebra(self, tmp_path: Path) -> None:
        """Análise vinda direto de detecções não tem ImageMeta; ainda assim grava."""
        sem_imagem = analyze_detections([detection(0, 0, 50, 100)], source="avulso")
        with Repository(tmp_path / "v.db") as repo:
            repo.save("LOJA_12", sem_imagem)
            (visita,) = repo.history("LOJA_12")
        assert visita.captured_at == "0000-00-00T00:00:00"

    def test_ocupacao_e_ponderada_pelas_prateleiras(self, tmp_path: Path) -> None:
        # Três produtos de 60 px com vãos de 40: envelope 260, ocupado 180.
        with Repository(tmp_path / "v.db") as repo:
            repo.save("LOJA_12", relatorio(produtos=3))
            (visita,) = repo.history("LOJA_12")
        assert visita.occupancy == pytest.approx(180 / 260)


class TestHistorico:
    def test_ordenado_da_mais_recente_para_a_mais_antiga(self, tmp_path: Path) -> None:
        with Repository(tmp_path / "v.db") as repo:
            for dia in ("01", "15", "08"):
                repo.save(
                    "LOJA_12",
                    relatorio(fonte=f"dia_{dia}.jpg", captura=f"2026-08-{dia}T09:00:00"),
                )
            visitas = repo.history("LOJA_12")
        assert [v.captured_at[8:10] for v in visitas] == ["15", "08", "01"]

    def test_ordena_pela_data_da_foto_e_nao_pela_de_processamento(self, tmp_path: Path) -> None:
        """O ponto do `captured_at`: um lote atrasado não embaralha a série."""
        with Repository(tmp_path / "v.db") as repo:
            # Gravadas fora de ordem, como num lote de fotos antigas.
            repo.save("LOJA_12", relatorio(fonte="nova.jpg", captura="2026-08-20T09:00:00"))
            repo.save("LOJA_12", relatorio(fonte="antiga.jpg", captura="2026-08-01T09:00:00"))
            visitas = repo.history("LOJA_12")
        assert [v.source for v in visitas] == ["nova.jpg", "antiga.jpg"]

    def test_limite(self, tmp_path: Path) -> None:
        with Repository(tmp_path / "v.db") as repo:
            for dia in range(1, 11):
                repo.save(
                    "LOJA_12",
                    relatorio(fonte=f"d{dia}.jpg", captura=f"2026-08-{dia:02d}T09:00:00"),
                )
            assert len(repo.history("LOJA_12", limit=3)) == 3

    def test_pdv_sem_historico(self, tmp_path: Path) -> None:
        with Repository(tmp_path / "v.db") as repo:
            repo.save("LOJA_12", relatorio())
            assert repo.history("LOJA_99") == ()

    def test_pdvs_nao_se_misturam(self, tmp_path: Path) -> None:
        with Repository(tmp_path / "v.db") as repo:
            repo.save("LOJA_1", relatorio(fonte="a.jpg", produtos=2))
            repo.save("LOJA_2", relatorio(fonte="b.jpg", produtos=5))
            assert repo.history("LOJA_1")[0].detections == 2
            assert repo.history("LOJA_2")[0].detections == 5

    def test_lista_os_pdvs_em_ordem(self, tmp_path: Path) -> None:
        with Repository(tmp_path / "v.db") as repo:
            repo.save("LOJA_9", relatorio(fonte="a.jpg"))
            repo.save("LOJA_1", relatorio(fonte="b.jpg"))
            assert repo.stores() == ("LOJA_1", "LOJA_9")

    def test_banco_vazio_nao_tem_pdv(self, tmp_path: Path) -> None:
        with Repository(tmp_path / "v.db") as repo:
            assert repo.stores() == ()

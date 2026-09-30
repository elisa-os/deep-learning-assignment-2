"""Exportação de métricas de avaliação de tracking para CSV.

Classes
--------
PerSequenceMetricsWriter
    Coleta registros durante a avaliação de sequências e grava um CSV
    com as métricas de tracking: IDF1, ID switches, fragmentações,
    erro de contagem de identidades, número de tracks previstas/verdadeiras,
    e detalhes por tipo de matching.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class PerSequenceMetricsWriter:
    """Acumula registros de métricas por sequência e exporta um CSV.

    Parâmetros
    ----------
    output_path : Path
        Diretório onde o CSV será salvo. O diretório é criado automaticamente
        (mkdir -p) na chamada a :meth:`write`.
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path = Path(output_path)
        self._records: list[dict[str, Any]] = []

    @property
    def fieldnames(self) -> list[str]:
        """Ordem garantida das colunas do CSV."""
        return [
            "sequence",
            "n_gt_ids",          # número de identidades verdadeiras na sequência
            "n_pred_ids",        # número de tracks previstos
            "idf1",              # IDF1 (de 0 a 1)
            "id_switches",       # número de trocas de ID
            "fragmentations",    # número de fragmentações
            "mota",              # MOTA (opcional, pode ser None)
            "count_error",       # |n_pred_ids - n_gt_ids|
            "mostly_tracked",    # fração de GT ids rastreados > 80% do tempo
            "mostly_lost",       # fração de GT ids rastreados < 20% do tempo
            "num_frames",        # número de quadros na sequência
            "notes",             # notas adicionais (ex: detector usado)
        ]

    @property
    def record_count(self) -> int:
        """Quantidade de registros acumulados até o momento."""
        return len(self._records)

    def add(
        self,
        sequence: str,
        n_gt_ids: int,
        n_pred_ids: int,
        idf1: float,
        id_switches: int,
        fragmentations: int,
        mota: float | None = None,
        mostly_tracked: float = 0.0,
        mostly_lost: float = 0.0,
        num_frames: int = 0,
        count_error: int | None = None,
        notes: str = "",
    ) -> None:
        """Adiciona um registro para uma única sequência."""
        if count_error is None:
            count_error = abs(n_pred_ids - n_gt_ids)

        record: dict[str, Any] = {
            "sequence": sequence,
            "n_gt_ids": n_gt_ids,
            "n_pred_ids": n_pred_ids,
            "idf1": round(idf1, 6),
            "id_switches": id_switches,
            "fragmentations": fragmentations,
            "mota": round(mota, 6) if mota is not None else "",
            "count_error": count_error,
            "mostly_tracked": round(mostly_tracked, 6),
            "mostly_lost": round(mostly_lost, 6),
            "num_frames": num_frames,
            "notes": notes,
        }
        self._records.append(record)

    def write(self, filename: str = "per_sequence_tracking_metrics.csv") -> Path:
        """Grava o CSV e retorna o caminho do arquivo criado."""
        self._output_path.mkdir(parents=True, exist_ok=True)
        csv_path = self._output_path / filename
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.fieldnames))
            writer.writeheader()
            writer.writerows(self._records)
        return csv_path

    def summarize(self) -> dict[str, float]:
        """Retorna métricas agregadas (média) sobre todas as sequências registradas.

        Retorna
        -------
        dict
            Chaves: idf1_mean, id_switches_mean, fragmentations_mean,
            count_error_mean, n_sequences.
        """
        if not self._records:
            return {
                "idf1_mean": 0.0,
                "id_switches_mean": 0.0,
                "fragmentations_mean": 0.0,
                "count_error_mean": 0.0,
                "n_sequences": 0,
            }

        n = len(self._records)
        return {
            "idf1_mean": sum(r["idf1"] for r in self._records) / n,
            "id_switches_mean": sum(r["id_switches"] for r in self._records) / n,
            "fragmentations_mean": sum(r["fragmentations"] for r in self._records) / n,
            "count_error_mean": sum(r["count_error"] for r in self._records) / n,
            "n_sequences": n,
        }

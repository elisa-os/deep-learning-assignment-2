"""Parte 1 — Baseline por quadro: detecção congelada + associação ingênua por IoU.

Fluxo (nada aqui é treinado):
  A. Estatísticas dos vídeos (densidade, câmera, visibilidade).
  B. Escolha do detector público (DPM / FRCNN / SDP) usando só os vídeos de TREINO.
  C. Variantes da regra de associação (guloso/Hungarian, limiar de IoU, max_age, min_hits),
     ranqueadas no treino; a validação só é reportada.
  D. Avaliação final por vídeo com a configuração do ``config.yaml`` (IDF1, ID switches,
     fragmentações, erro de contagem, AP/mAP) e o gráfico do descolamento.
  E. (opcional) mesma avaliação com o detector pré-treinado do torchvision, se
     ``use_torchvision_detector: true`` e as imagens do MOT17 estiverem em ``data/MOT17``.

Saídas em ``outputs/``: ``parte1_*.csv|json|png``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pa2.association.tracker import track_sequence
from pa2.config import Config
from pa2.metrics.detection import drop_ignored_detections, evaluate_detections
from pa2.mot17 import DETECTORS, Sequence, resolve_split
from pa2.mot17.evaluate import evaluate_tracks
from pa2.utils.visualize import save_figure

# Associação usada para COMPARAR detectores (a mesma para os três)
DETECTOR_COMPARISON_ASSOC = dict(method="hungarian", iou_threshold=0.3, max_age=5, min_hits=1)

VARIANT_GRID = {
    "method": ["greedy", "hungarian"],
    "iou_threshold": [0.3, 0.5],
    "max_age": [1, 5, 30],
    "min_hits": [1, 3],
}


# ─────────────────────────────────────────────────────────────────────────────
# utilidades
# ─────────────────────────────────────────────────────────────────────────────
class Source:
    """Detecções (já sem as que casam com distratores) de um conjunto de vídeos."""

    def __init__(self, name: str, seqs: dict[str, Sequence], dets: dict[str, np.ndarray]):
        self.name = name
        self.seqs = seqs
        self.dets = dets

    @classmethod
    def public(cls, root: Path, videos: list[str], detector: str) -> "Source":
        seqs = {v: Sequence(root, v, detector) for v in videos}
        dets = {}
        for v, s in seqs.items():
            filtered, _ = drop_ignored_detections(s.det, s.gt_pedestrians(), s.gt_distractors())
            dets[v] = filtered
        return cls(detector, seqs, dets)


def _candidate_thresholds(dets: dict[str, np.ndarray]) -> list[float]:
    scores = np.concatenate([d[:, 5] for d in dets.values()])
    return sorted({float(x) for x in np.round(np.quantile(scores, np.linspace(0, 0.95, 20)), 3)})


def best_f1_threshold(src: Source, videos: list[str]) -> tuple[float, dict]:
    """Limiar de score que maximiza o F1 (IoU 0.5) das detecções somadas nos ``videos``."""
    best = (-1.0, -np.inf, {})
    for thr in _candidate_thresholds({v: src.dets[v] for v in videos}):
        tp = n_det = n_gt = 0
        for v in videos:
            r = evaluate_detections(src.dets[v], src.seqs[v].gt_pedestrians(), None, thr, full=False)
            tp, n_det, n_gt = tp + r["n_tp"], n_det + r["n_dets"], n_gt + r["n_gt"]
        p, rec = tp / max(n_det, 1), tp / max(n_gt, 1)
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        if f1 > best[0]:
            best = (f1, thr, {"precision": p, "recall": rec, "f1": f1})
    return best[1], best[2]


def _run_tracker(src: Source, video: str, assoc: dict, min_conf: float) -> dict:
    seq = src.seqs[video]
    tracks = track_sequence(src.dets[video], seq.info.seq_length, min_conf=min_conf, **assoc)
    return evaluate_tracks(seq, tracks)


IDF1_TIE_MARGIN = 0.005


def pick_variant(variants: pd.DataFrame) -> dict:
    """Melhor variante no TREINO: entre as que ficam a até ``IDF1_TIE_MARGIN`` do maior
    IDF1 (diferenças menores que isso são ruído entre vídeos), a com menos ID switches."""
    near = variants[variants["IDF1_train"] >= variants["IDF1_train"].max() - IDF1_TIE_MARGIN]
    row = near.sort_values(["IDsw_train", "IDF1_train"], ascending=[True, False]).iloc[0]
    return {"method": str(row["method"]), "iou_threshold": float(row["iou_threshold"]),
            "max_age": int(row["max_age"]), "min_hits": int(row["min_hits"])}


def _assoc_from_cfg(cfg: Config) -> dict:
    a = cfg.association
    return dict(method=a.method, iou_threshold=a.iou_threshold, max_age=a.max_age,
                min_hits=a.min_hits)


# ─────────────────────────────────────────────────────────────────────────────
# etapas
# ─────────────────────────────────────────────────────────────────────────────
def _stage_stats(root: Path, train: list[str], val: list[str], out: Path) -> pd.DataFrame:
    print("\n--- A. Os vídeos ---\n")
    rows = []
    for v in train + val:
        s = Sequence(root, v, "FRCNN").stats()
        s["split"] = "train" if v in train else "val"
        rows.append(s)
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))
    df.to_csv(out / "parte1_sequences.csv", index=False)
    return df


def _stage_choose_detector(root: Path, train: list[str], out: Path) -> tuple[str, dict]:
    print("\n--- B. Escolha do detector público (só vídeos de treino) ---\n")
    print(f"  associação fixa para comparar: {DETECTOR_COMPARISON_ASSOC}")
    rows = []
    thresholds: dict[str, float] = {}
    for det in DETECTORS:
        src = Source.public(root, train, det)
        thr, pr = best_f1_threshold(src, train)
        thresholds[det] = thr
        ap50, mapv, idf1, ratio, sw = [], [], [], [], []
        for v in train:
            d = evaluate_detections(src.dets[v], src.seqs[v].gt_pedestrians(), None, full=True)
            ap50.append(d["ap50"])
            mapv.append(d["map"])
            r = _run_tracker(src, v, DETECTOR_COMPARISON_ASSOC, thr)
            idf1.append(r["idf1"])
            ratio.append(r["n_pred_ids"] / max(1, r["n_gt_ids"]))
            sw.append(r["id_switches"] / max(1, r["n_gt_ids"]))
        rows.append({
            "detector": det, "score_thr": thr, "AP50": np.mean(ap50), "mAP": np.mean(mapv),
            "precision": pr["precision"], "recall": pr["recall"], "F1": pr["f1"],
            "IDF1": np.mean(idf1), "ids_pred/gt": np.mean(ratio), "IDsw/id": np.mean(sw),
        })
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    df.to_csv(out / "parte1_detector_comparison.csv", index=False)
    best = df.sort_values("IDF1", ascending=False).iloc[0]["detector"]
    print(f"\n  detector com maior IDF1 médio no treino: {best}")
    return str(best), thresholds


def _stage_variants(src: Source, train: list[str], val: list[str], min_conf: float,
                    out: Path) -> pd.DataFrame:
    print("\n--- C. Variantes da regra de associação ---\n")
    keys = list(VARIANT_GRID)
    rows = []
    for combo in itertools.product(*VARIANT_GRID.values()):
        assoc = dict(zip(keys, combo))
        per = {v: _run_tracker(src, v, assoc, min_conf) for v in train + val}
        rows.append({
            **assoc,
            "IDF1_train": np.mean([per[v]["idf1"] for v in train]),
            "IDsw_train": np.mean([per[v]["id_switches"] for v in train]),
            "ids_ratio_train": np.mean([per[v]["n_pred_ids"] / max(1, per[v]["n_gt_ids"])
                                        for v in train]),
            "IDF1_val": np.mean([per[v]["idf1"] for v in val]),
            "IDsw_val": np.mean([per[v]["id_switches"] for v in val]),
        })
    df = pd.DataFrame(rows).sort_values("IDF1_train", ascending=False).reset_index(drop=True)
    print(df.round(3).to_string(index=False))
    df.to_csv(out / "parte1_association_variants.csv", index=False)
    return df


def _evaluate_source(src: Source, assoc: dict, min_conf: float, meta: pd.DataFrame,
                     score_thr_for_pr: float) -> pd.DataFrame:
    rows = []
    for v in src.seqs:
        seq = src.seqs[v]
        det_full = evaluate_detections(src.dets[v], seq.gt_pedestrians(), None, full=True)
        det_op = evaluate_detections(src.dets[v], seq.gt_pedestrians(), None, score_thr_for_pr,
                                     full=False)
        r = _run_tracker(src, v, assoc, min_conf)
        m = meta[meta.sequence == seq.name].iloc[0]
        rows.append({
            "sequence": seq.name, "split": m["split"], "camera": m["camera"],
            "peds_per_frame": m["peds_per_frame"], "mean_visibility": m["mean_visibility"],
            "AP50": det_full["ap50"], "mAP": det_full["map"],
            "precision": det_op["precision"], "recall": det_op["recall"],
            "det_F1": det_op["f1"], "IDF1": r["idf1"], "ID_switches": r["id_switches"],
            "fragmentations": r["fragmentations"],
            "n_gt_ids": r["n_gt_ids"], "n_pred_ids": r["n_pred_ids"],
            "count_error": r["count_error_signed"],
            "ids_pred/gt": r["n_pred_ids"] / max(1, r["n_gt_ids"]),
            "IDsw/id": r["id_switches"] / max(1, r["n_gt_ids"]),
            "MOTA": r["mota"],
        })
    return pd.DataFrame(rows)


def _print_difficulty_axes(df: pd.DataFrame) -> None:
    """Correlação de Spearman (n = nº de vídeos, é só indicativa) entre candidatos a eixo de
    dificuldade e as métricas de identidade."""
    d = df.assign(camera_movel=(df.camera == "móvel").astype(float))
    print("\n  Spearman entre eixo de dificuldade e métrica (n = %d vídeos, só indicativo):" % len(d))
    print(f"  {'eixo':<18}{'IDF1':>8}{'IDsw/id':>10}{'ids prev/verd':>15}")
    for axis in ("peds_per_frame", "mean_visibility", "camera_movel"):
        rho = {m: d[axis].corr(d[m], method="spearman") for m in ("IDF1", "IDsw/id", "ids_pred/gt")}
        print(f"  {axis:<18}{rho['IDF1']:>8.2f}{rho['IDsw/id']:>10.2f}{rho['ids_pred/gt']:>15.2f}")


def _plot_decoupling(df: pd.DataFrame, title: str, path: Path) -> None:
    """Gráfico obrigatório: detecção boa vs. identidade ruim, vídeos ordenados por densidade."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.sort_values("peds_per_frame").reset_index(drop=True)
    x = np.arange(len(d))
    labels = [f"{s.replace('MOT17-', '')}{'*' if sp == 'val' else ''}\n{p:.0f} ped/q\n{c}"
              for s, sp, p, c in zip(d.sequence, d.split, d.peds_per_frame, d.camera)]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    a1.plot(x, d["mAP"], "o-", label="mAP (IoU 0.5:0.95)", color="tab:blue")
    a1.plot(x, d["AP50"], "o--", label="AP50", color="tab:cyan")
    a1.plot(x, d["det_F1"], ":", color="gray", label="F1 da detecção (teto do IDF1)")
    a1.plot(x, d["IDF1"], "s-", label="IDF1", color="tab:red")
    a1.set_ylim(0, 1)
    a1.set_ylabel("métrica (0–1)")
    a1.set_title("Qualidade do detector (por quadro) vs. qualidade das identidades (trajetórias)")
    a1.legend(loc="best", fontsize=8)
    a1.grid(alpha=0.3)

    a2.plot(x, d["ids_pred/gt"], "o-", color="tab:purple", label="ids previstos / ids verdadeiros")
    a2.axhline(1.0, color="gray", ls="--", lw=1)
    a2.set_ylabel("ids previstos / verdadeiros")
    a2.set_xticks(x)
    a2.set_xticklabels(labels, fontsize=8)
    a2.set_xlabel("vídeos ordenados por densidade (pedestres por quadro); * = validação")
    a2.grid(alpha=0.3)
    b = a2.twinx()
    b.plot(x, d["IDsw/id"], "^-", color="tab:green", label="ID switches / id verdadeiro")
    b.set_ylabel("ID switches / id verdadeiro")
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


def _stage_torchvision(cfg: Config, root: Path, videos: list[str], chosen_det: str,
                       meta: pd.DataFrame, assoc: dict, out: Path, device) -> None:
    print("\n--- E. Detector pré-treinado do torchvision ---\n")
    probe = Sequence(root, videos[0], chosen_det)
    if not probe.has_images():
        print(f"  imagens do MOT17 não encontradas em {probe.image_dir}.\n"
              "  Baixe o pacote completo (~5,5 GB) em https://motchallenge.net/data/MOT17/ e extraia\n"
              "  em data/MOT17/ (de preferência com GPU: em CPU leva horas). Pulando esta etapa.")
        return

    from pa2.detection.torchvision_person import TorchvisionPersonDetector

    detector = TorchvisionPersonDetector(device=device)
    seqs = {v: Sequence(root, v, chosen_det) for v in videos}
    dets = {}
    for v, s in seqs.items():
        cache = out / "detections" / "torchvision" / f"{s.name}.txt"
        print(f"  {s.name}: {'lendo cache' if cache.exists() else 'detectando'} -> {cache}")
        raw = detector.detect_sequence(s, cache, progress=None)
        dets[v], _ = drop_ignored_detections(raw, s.gt_pedestrians(), s.gt_distractors())
    src = Source("torchvision", seqs, dets)
    train = [v for v in videos if meta[meta.sequence == f"MOT17-{v}"].iloc[0]["split"] == "train"]
    thr, pr = best_f1_threshold(src, train)
    print(f"  limiar de score (melhor F1 no treino): {thr:.3f}  (P={pr['precision']:.3f}, "
          f"R={pr['recall']:.3f})")
    min_conf = cfg.association.min_conf if cfg.association.min_conf is not None else thr
    df = _evaluate_source(src, assoc, min_conf, meta, thr)
    print(df.round(3).to_string(index=False))
    df.to_csv(out / "parte1_per_sequence_torchvision.csv", index=False)
    _plot_decoupling(df, f"Descolamento — detector torchvision (Faster R-CNN COCO), "
                         f"associação {assoc}", out / "parte1_descolamento_torchvision.png")


# ─────────────────────────────────────────────────────────────────────────────
def run_parte1(cfg: Config, device: torch.device) -> None:
    print("\n[*] Parte 1 — Baseline por quadro")
    root = Path(cfg.data.data_dir)
    if not (root / "train").exists():
        raise SystemExit(f"MOT17 não encontrado em {root}/train. Veja o README (download).")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split = resolve_split(cfg.data.sequence_split)
    train, val = split["train"], split["val"]
    print(f"  split por vídeo — treino: {train} | validação: {val}")

    meta = _stage_stats(root, train, val, out)

    best_det, thresholds = _stage_choose_detector(root, train, out)
    detector = cfg.detector_source or best_det
    if detector != best_det:
        print(f"  ! config usa {detector}, mas os dados apontam {best_det} "
              "(mantendo o config; atualize `detector_source` se quiser seguir os dados)")
    thr = thresholds[detector]
    min_conf = cfg.association.min_conf if cfg.association.min_conf is not None else thr
    print(f"  fonte de detecções do resto do PA: {detector} (score >= {min_conf:.3f})")

    src = Source.public(root, train + val, detector)
    variants = _stage_variants(src, train, val, min_conf, out)

    assoc = _assoc_from_cfg(cfg)
    picked = pick_variant(variants)
    print(f"\n  regra de escolha: maior IDF1 de treino; empate (±{IDF1_TIE_MARGIN}) -> menos ID switches")
    print(f"  melhor variante pela regra: {picked}")
    if picked != assoc:
        print(f"  ! associação do config {assoc} difere da escolhida pelos dados")

    print("\n--- D. Avaliação final por vídeo ---\n")
    print(f"  detector: {detector} | score >= {min_conf:.3f} | associação: {assoc}")
    df = _evaluate_source(src, assoc, min_conf, meta, thr)
    print(df.round(3).to_string(index=False))
    df.to_csv(out / "parte1_per_sequence_public.csv", index=False)

    tr, va = df[df.split == "train"], df[df.split == "val"]
    for name, part in (("treino", tr), ("validação", va)):
        print(f"  média {name}: mAP={part['mAP'].mean():.3f}  IDF1={part['IDF1'].mean():.3f}  "
              f"ids prev/verd={part['ids_pred/gt'].mean():.2f}  IDsw/id={part['IDsw/id'].mean():.2f}")

    _print_difficulty_axes(df)
    _plot_decoupling(
        df, f"Descolamento — detecções públicas {detector}, associação ingênua por IoU "
            f"({assoc['method']}, IoU≥{assoc['iou_threshold']}, max_age={assoc['max_age']}, "
            f"min_hits={assoc['min_hits']})", out / "parte1_descolamento.png")

    with open(out / "parte1_summary.json", "w") as f:
        json.dump({"detector": detector, "min_conf": min_conf, "association": assoc,
                   "split": split, "best_detector_by_train_idf1": best_det,
                   "thresholds": thresholds}, f, indent=2)

    if cfg.use_torchvision_detector:
        _stage_torchvision(cfg, root, train + val, detector, meta, assoc, out, device)
    else:
        print("\n--- E. Detector torchvision: desligado (use_torchvision_detector: false) ---")

    print(f"\n  Resultados em: {out}")
    print("\n[✓] Parte 1 concluída.")

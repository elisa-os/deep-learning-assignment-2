"""Inferência do rastreador final sobre uma sequência qualquer (formato MOTChallenge).

Recebe a PASTA de uma sequência, roda o modelo final **sem retreinar** (GRU de movimento + a mesma gestão
de tracks das Partes 1 a 5), devolve as tracks com identidades consistentes, a contagem de objetos únicos e
um vídeo com as identidades coloridas (a mesma cor para o mesmo id em todo o vídeo).

Contrato da pasta (formato MOTChallenge): ``seqinfo.ini`` e, para as detecções, ``det/det.txt`` e/ou as
imagens em ``img1/``; ``gt/gt.txt`` é opcional (com GT, também saem as métricas).

Fonte de detecções (``detector="auto"``): ``det/det.txt`` se existir (detecções públicas, score >= o limiar
escolhido na Parte 1 para aquele detector); senão, se houver imagens, o Faster R-CNN pré-treinado do
torchvision (classe ``person``, NMS próprio, em cache). Sem nenhum dos dois, erro claro.

Fundo do vídeo: as imagens, se existirem; senão um fundo vazio (as figuras deste projeto usam só as
anotações, sem as imagens do MOT17).

**A contagem de "objetos únicos" é o número de ids distintos que o rastreador emitiu.** Ela superestima: na
validação o rastreador emite ~1,5 ids por pessoa real (ver ``RELATORY_PART3.md``), porque cada fragmentação
de uma identidade cria um id novo. Por isso o resultado traz também a contagem só de ids vistos em pelo menos
``min_track_len`` quadros (descarta ruído de curta duração), e, havendo GT, o número real de identidades.
"""

from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pa2.association.motion import RNNMotion
from pa2.association.motion_tracker import track_sequence_motion
from pa2.config import load_config
from pa2.models import load_checkpoint
from pa2.mot17.evaluate import evaluate_tracks
from pa2.mot17.loader import Sequence

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / "outputs" / "checkpoints" / "final_motion_rnn.pt"
# limiares de score por detector público (escolhidos na Parte 1 no treino; ver outputs/parte1_summary.json)
FALLBACK_THRESHOLDS = {"DPM": -0.295, "FRCNN": 0.05, "SDP": 0.4}
UNKNOWN_DETECTOR_THRESHOLD = 0.4
TORCHVISION_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
def default_assoc() -> dict:
    """Regra de associação das Partes 1 a 5 (``config.yaml`` → ``parte4``)."""
    a = load_config(path=REPO / "pa2" / "config.yaml", parte=4).association
    return dict(method=a.method, iou_threshold=a.iou_threshold, max_age=a.max_age, min_hits=a.min_hits)


def score_threshold_for(detector: str) -> float:
    path = REPO / "outputs" / "parte1_summary.json"
    thr = dict(FALLBACK_THRESHOLDS)
    if path.exists():
        thr.update(json.load(open(path)).get("thresholds", {}))
    return float(thr.get(detector, UNKNOWN_DETECTOR_THRESHOLD))


def id_color(track_id: int) -> tuple[int, int, int]:
    """Cor RGB determinística por id (razão áurea no matiz): o mesmo id tem sempre a mesma cor."""
    h = (int(track_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.78, 0.97)
    return int(r * 255), int(g * 255), int(b * 255)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InferenceResult:
    seq: Sequence
    tracks: list[dict]
    n_frames: int
    detector_source: str
    min_conf: float
    assoc: dict
    checkpoint: str
    n_detections: int
    min_track_len: int = 10
    metrics: dict | None = None
    _by_frame: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for t in self.tracks:
            self._by_frame.setdefault(t["frame"], {})[t["id"]] = (t["bb_left"], t["bb_top"], t["bb_width"],
                                                                 t["bb_height"])

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.tracks, columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
                                                  "conf"])

    def boxes_at(self, frame: int) -> dict:
        return self._by_frame.get(frame, {})

    def counts(self) -> dict:
        t = self.table
        if t.empty:
            return {"ids_unicos": 0, f"ids_unicos_com_{self.min_track_len}+_quadros": 0, "ids_gt": None,
                    "ativos_por_quadro_media": 0.0}
        life = t.groupby("id").size()
        out = {"ids_unicos": int(life.size),
               f"ids_unicos_com_{self.min_track_len}+_quadros": int((life >= self.min_track_len).sum()),
               "ativos_por_quadro_media": float(t.groupby("frame").size().reindex(range(1, self.n_frames + 1),
                                                                               fill_value=0).mean())}
        out["ids_gt"] = int(len(np.unique(self.seq.gt_pedestrians()[:, 1]))) if self.seq.has_gt() else None
        return out

    def save_mot_txt(self, path: str | Path) -> Path:
        """Grava as tracks no formato de resultado do MOTChallenge (frame,id,l,t,w,h,conf,-1,-1,-1)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for t in self.tracks:
                f.write(f"{t['frame']},{t['id']},{t['bb_left']:.2f},{t['bb_top']:.2f},{t['bb_width']:.2f},"
                        f"{t['bb_height']:.2f},{t['conf']:.4f},-1,-1,-1\n")
        return path


def load_detections_for(seq: Sequence, detector: str = "auto", score_threshold: float | None = None,
                        device: str = "cpu", cache_dir: str | Path | None = None):
    """Devolve ``(dets (M, 6), descrição da fonte, limiar de score)``."""
    if detector not in ("auto", "public", "torchvision"):
        raise ValueError("detector deve ser 'auto', 'public' ou 'torchvision'")
    if detector in ("auto", "public") and seq.has_det():
        thr = score_threshold if score_threshold is not None else score_threshold_for(seq.detector)
        return seq.det, f"detecções públicas ({seq.detector}, det/det.txt)", float(thr)
    if detector == "public":
        raise FileNotFoundError(f"{seq.dir}/det/det.txt não existe")
    if seq.has_images():
        from pa2.detection.torchvision_person import TorchvisionPersonDetector
        cache = Path(cache_dir or REPO / "outputs" / "inference_cache") / f"{seq.dir.name}_torchvision.txt"
        dets = TorchvisionPersonDetector(device=device).detect_sequence(seq, cache)
        thr = score_threshold if score_threshold is not None else TORCHVISION_THRESHOLD
        return dets, "Faster R-CNN do torchvision (pessoa, NMS próprio)", float(thr)
    raise FileNotFoundError(
        f"{seq.dir}: sem det/det.txt e sem imagens em {seq.image_dir}; não há de onde tirar detecções")


def run_inference(path: str | Path, checkpoint: str | Path | None = None, detector: str = "auto",
                  score_threshold: float | None = None, max_frames: int | None = None, device: str = "cpu",
                  min_track_len: int = 10, cache_dir: str | Path | None = None) -> InferenceResult:
    """Rastreia a sequência da pasta ``path``. Sem treino: só carrega o checkpoint e roda."""
    seq = Sequence.from_dir(path)
    ckpt = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    model, _ = load_checkpoint(ckpt)
    dets, source, thr = load_detections_for(seq, detector, score_threshold, device, cache_dir)

    n = seq.info.seq_length if max_frames is None else min(max_frames, seq.info.seq_length)
    dets = dets[dets[:, 0] <= n]
    assoc = default_assoc()
    motion = RNNMotion(model, (seq.info.im_width, seq.info.im_height), dt=1.0)
    tracks = track_sequence_motion(dets, n, motion, min_conf=thr, **assoc)

    metrics = None
    if seq.has_gt() and n == seq.info.seq_length:
        r = evaluate_tracks(seq, tracks)
        metrics = {k: r[k] for k in ("idf1", "id_switches", "fragmentations", "n_gt_ids", "n_pred_ids", "mota")}
    try:                                    # caminho relativo ao repositório (portável; sem /home/usuario nas saídas)
        shown = ckpt.resolve().relative_to(REPO)
    except ValueError:
        shown = ckpt
    return InferenceResult(seq=seq, tracks=tracks, n_frames=n, detector_source=source, min_conf=thr, assoc=assoc,
                           checkpoint=str(shown), n_detections=int((dets[:, 5] >= thr).sum()),
                           min_track_len=min_track_len, metrics=metrics)


# ─────────────────────────────────────────────────────────────────────────────
# vídeo
# ─────────────────────────────────────────────────────────────────────────────
class FrameRenderer:
    """Desenha um quadro anotado (BGR, ``uint8``). Sem estado entre quadros: pode ser chamado em qualquer ordem."""

    def __init__(self, result: InferenceResult, scale: float = 0.5, show_gt: bool = False, trail: int = 15,
                 use_images: bool = True):
        import cv2
        self.cv2 = cv2
        self.r, self.scale, self.show_gt, self.trail = result, scale, show_gt, trail
        info = result.seq.info
        self.size = (max(2, int(info.im_width * scale) // 2 * 2), max(2, int(info.im_height * scale) // 2 * 2))
        self.use_images = use_images and result.seq.has_images()
        self.first_seen = result.table.groupby("id").frame.min() if len(result.tracks) else pd.Series(dtype=int)
        self.gt = result.seq.gt_pedestrians() if (show_gt and result.seq.has_gt()) else None

    def _scaled(self, box):
        l, t, w, h = (v * self.scale for v in box)
        return int(round(l)), int(round(t)), int(round(l + w)), int(round(t + h))

    def render(self, frame: int) -> np.ndarray:
        cv2, r = self.cv2, self.r
        img = None
        if self.use_images:
            img = cv2.imread(str(r.seq.image_path(frame)))
            if img is not None:
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        if img is None:
            img = np.full((self.size[1], self.size[0], 3), 28, np.uint8)

        if self.gt is not None:
            for row in self.gt[self.gt[:, 0] == frame]:
                x1, y1, x2, y2 = self._scaled(row[2:6])
                cv2.rectangle(img, (x1, y1), (x2, y2), (110, 110, 110), 1)

        for tid, box in r.boxes_at(frame).items():
            rgb = id_color(tid)
            bgr = (rgb[2], rgb[1], rgb[0])
            if self.trail:
                pts = []
                for f in range(max(1, frame - self.trail), frame + 1):
                    b = r.boxes_at(f).get(tid)
                    if b is not None:
                        pts.append((int((b[0] + b[2] / 2) * self.scale), int((b[1] + b[3]) * self.scale)))
                if len(pts) > 1:
                    cv2.polylines(img, [np.array(pts, np.int32)], False, bgr, 1, cv2.LINE_AA)
            x1, y1, x2, y2 = self._scaled(box)
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(img, str(tid), (x1 + 2, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

        active = len(r.boxes_at(frame))
        uniq = int((self.first_seen <= frame).sum()) if len(self.first_seen) else 0
        hud = f"{r.seq.name}  quadro {frame}/{r.n_frames}  ativos {active}  ids unicos ate aqui {uniq}"
        cv2.putText(img, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        return img


def render_video(result: InferenceResult, out_path: str | Path, scale: float = 0.5, fps: float | None = None,
                 max_frames: int | None = None, show_gt: bool = False, trail: int = 15,
                 use_images: bool = True, codec: str = "mp4v") -> Path:
    """Grava o vídeo anotado (``mp4``, codec ``mp4v``: abre em VLC/players; navegadores podem não tocar)."""
    import cv2
    rd = FrameRenderer(result, scale, show_gt, trail, use_images)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps or result.seq.info.frame_rate, rd.size)
    if not w.isOpened():
        raise RuntimeError(f"não consegui abrir o codec {codec!r}; tente codec='MJPG' com extensão .avi")
    try:
        for f in range(1, (max_frames or result.n_frames) + 1):
            w.write(rd.render(f))
    finally:
        w.release()
    return out_path


def render_gif(result: InferenceResult, out_path: str | Path, scale: float = 0.35, max_frames: int = 120,
               frame_stride: int | None = None, fps: float = 12, show_gt: bool = False, trail: int = 15,
               use_images: bool = True) -> Path:
    """GIF curto (toca em qualquer navegador/notebook). ``frame_stride`` automático para caber em ``max_frames``."""
    import imageio.v2 as imageio
    rd = FrameRenderer(result, scale, show_gt, trail, use_images)
    stride = frame_stride or max(1, int(np.ceil(result.n_frames / max_frames)))
    frames = [rd.render(f)[:, :, ::-1] for f in range(1, result.n_frames + 1, stride)][:max_frames]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)
    return out_path


def sample_frames(result: InferenceResult, frames: list[int], scale: float = 0.4, show_gt: bool = False,
                  trail: int = 15, use_images: bool = True) -> list[np.ndarray]:
    """Quadros anotados em RGB (para mostrar com matplotlib)."""
    rd = FrameRenderer(result, scale, show_gt, trail, use_images)
    return [rd.render(f)[:, :, ::-1] for f in frames]

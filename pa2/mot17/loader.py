"""Loader do MOT17 / MOTChallenge: ground truth, detecções públicas e seqinfo.

Formatos (todos com quadros 1-based e caixas ``left, top, width, height`` em pixels):

``gt/gt.txt``   : frame, id, left, top, w, h, conf, class, visibility
``det/det.txt`` : frame, -1, left, top, w, h, score, ... (DPM tem 10 colunas, os outros 7)

Convenções de avaliação (as mesmas do benchmark):
- GT avaliado = ``class == 1`` (pedestre) e ``conf == 1``.
- Classes "distratoras" (2 pessoa em veículo, 7 pessoa estática, 8 distrator, 12 reflexo)
  não contam como falso positivo se o algoritmo as detecta; ver ``pa2/mot17/evaluate.py``.
- O GT é **contínuo**: um pedestre ocluído continua presente em todos os quadros da sua
  vida, com ``visibility`` baixa. Não há "lacunas" no gt.txt.

Os três detectores públicos (DPM, FRCNN, SDP) compartilham o mesmo vídeo e o mesmo GT:
o split deve ser feito por **vídeo** (02, 04, ...), nunca por detector.
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PEDESTRIAN_CLASS = 1
DISTRACTOR_CLASSES = (2, 7, 8, 12)   # person on vehicle, static person, distractor, reflection
DETECTORS = ("DPM", "FRCNN", "SDP")

# Split por vídeo. Só os vídeos de ``train/`` têm GT (``test/`` não tem).
#   train: 02 (parada, densa), 04 (parada, a mais densa), 05 (móvel, 640x480, 14 fps),
#          10 e 11 (móveis)
#   val  : 09 (parada, esparsa) e 13 (móvel, muita rotatividade de identidades)
# Critério: val contém uma câmera parada e uma móvel, e densidades/fps diferentes.
DEFAULT_SPLIT: dict[str, list[str]] = {
    "train": ["02", "04", "05", "10", "11"],
    "val": ["09", "13"],
}
# Movimento de câmera dos vídeos (MOT16/17: Milan et al., 2016)
CAMERA_MOVING = {"02": False, "04": False, "05": True, "09": False,
                 "10": True, "11": True, "13": True}


@dataclass(frozen=True)
class SeqInfo:
    name: str
    im_dir: str
    frame_rate: int
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str


class Sequence:
    """Uma sequência do MOT17 com um detector público escolhido."""

    def __init__(self, root: str | Path, video: str, detector: str = "FRCNN", split: str = "train"):
        video = video.replace("MOT17-", "")
        if detector not in DETECTORS:
            raise ValueError(f"detector deve ser um de {DETECTORS}, veio {detector!r}")
        self.video = video
        self.detector = detector
        self.name = f"MOT17-{video}"
        self.dir = Path(root) / split / f"MOT17-{video}-{detector}"
        if not self.dir.exists():
            raise FileNotFoundError(f"Sequência não encontrada: {self.dir}")
        self.info = read_seqinfo(self.dir / "seqinfo.ini")
        self._gt: np.ndarray | None = None
        self._det: np.ndarray | None = None

    @classmethod
    def from_dir(cls, path: str | Path) -> "Sequence":
        """Abre uma sequência em formato MOTChallenge a partir da PASTA (qualquer lugar do disco).

        Contrato: a pasta tem ``seqinfo.ini`` e, para detecções, ``det/det.txt`` e/ou as imagens em
        ``img1/``; ``gt/gt.txt`` é opcional (sem GT não há métricas, só contagem). O nome da pasta
        ``MOT17-09-SDP`` dá ``video = "09"`` e ``detector = "SDP"``; outros nomes são aceitos.
        """
        path = Path(path)
        if not (path / "seqinfo.ini").exists():
            raise FileNotFoundError(
                f"{path} não parece uma sequência MOTChallenge: falta seqinfo.ini "
                "(esperado: seqinfo.ini + det/det.txt e/ou img1/, gt/gt.txt opcional)")
        obj = cls.__new__(cls)
        m = re.match(r"^(?P<ds>[A-Za-z]+\d*)-(?P<vid>\d+)(?:-(?P<det>[A-Za-z]+))?$", path.name)
        obj.video = m.group("vid") if m else path.name
        obj.detector = (m.group("det") or "?") if m else "?"
        obj.name = f"{m.group('ds')}-{m.group('vid')}" if m else path.name
        obj.dir = path
        obj.info = read_seqinfo(path / "seqinfo.ini")
        obj._gt = None
        obj._det = None
        return obj

    def has_gt(self) -> bool:
        return (self.dir / "gt" / "gt.txt").exists()

    def has_det(self) -> bool:
        return (self.dir / "det" / "det.txt").exists()

    # ── dados ────────────────────────────────────────────────────────────────
    @property
    def gt(self) -> np.ndarray:
        """Todas as linhas do gt.txt (N, 9)."""
        if self._gt is None:
            path = self.dir / "gt" / "gt.txt"
            if not path.exists():
                raise FileNotFoundError(f"{self.name} não tem GT ({path})")
            self._gt = np.loadtxt(path, delimiter=",", ndmin=2)
        return self._gt

    @property
    def det(self) -> np.ndarray:
        """Detecções públicas (M, 6): frame, left, top, w, h, score."""
        if self._det is None:
            self._det = load_detections(self.dir / "det" / "det.txt")
        return self._det

    def gt_pedestrians(self) -> np.ndarray:
        """GT avaliado: class == 1 e conf == 1. Colunas: frame, id, l, t, w, h, vis."""
        g = self.gt
        g = g[(g[:, 7] == PEDESTRIAN_CLASS) & (g[:, 6] == 1)]
        return g[:, [0, 1, 2, 3, 4, 5, 8]]

    def gt_distractors(self) -> np.ndarray:
        """GT de classes distratoras. Colunas: frame, id, l, t, w, h."""
        g = self.gt
        return g[np.isin(g[:, 7], DISTRACTOR_CLASSES)][:, :6]

    # ── imagens (opcionais: o pacote de anotações não as traz) ───────────────
    @property
    def image_dir(self) -> Path:
        return self.dir / self.info.im_dir

    def has_images(self) -> bool:
        return self.image_dir.exists() and any(self.image_dir.glob(f"*{self.info.im_ext}"))

    def image_path(self, frame: int) -> Path:
        return self.image_dir / f"{frame:06d}{self.info.im_ext}"

    # ── estatísticas ─────────────────────────────────────────────────────────
    def stats(self) -> dict:
        ped = self.gt_pedestrians()
        n = self.info.seq_length
        ids = np.unique(ped[:, 1])
        return {
            "sequence": self.name,
            "camera": "móvel" if CAMERA_MOVING.get(self.video) else "parada",
            "resolution": f"{self.info.im_width}x{self.info.im_height}",
            "fps": self.info.frame_rate,
            "frames": n,
            "gt_ids": len(ids),
            "peds_per_frame": len(ped) / n,
            "mean_visibility": float(ped[:, 6].mean()),
            "det_per_frame": len(self.det) / n,
        }


def read_seqinfo(path: str | Path) -> SeqInfo:
    cp = configparser.ConfigParser()
    cp.read(path)
    s = cp["Sequence"]
    return SeqInfo(
        name=s["name"], im_dir=s.get("imDir", "img1"), frame_rate=int(s["frameRate"]),
        seq_length=int(s["seqLength"]), im_width=int(s["imWidth"]),
        im_height=int(s["imHeight"]), im_ext=s.get("imExt", ".jpg"),
    )


def load_detections(path: str | Path) -> np.ndarray:
    """Lê um det.txt e devolve (M, 6): frame, left, top, w, h, score."""
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    if raw.size == 0:
        return np.zeros((0, 6))
    return raw[:, [0, 2, 3, 4, 5, 6]].copy()


def write_detections(path: str | Path, dets: np.ndarray) -> None:
    """Grava (M, 6) frame, left, top, w, h, score no formato det.txt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fr, x, y, w, h, s in dets:
            f.write(f"{int(fr)},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.5f},-1,-1,-1\n")


def resolve_split(split: dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Usa o split do config se tiver conteúdo; senão o ``DEFAULT_SPLIT``."""
    if split and any(split.get(k) for k in ("train", "val")):
        out = {k: [v.replace("MOT17-", "") for v in split.get(k, [])] for k in ("train", "val")}
    else:
        out = {k: list(v) for k, v in DEFAULT_SPLIT.items()}
    overlap = set(out["train"]) & set(out["val"])
    if overlap:
        raise ValueError(f"vídeos em train e val ao mesmo tempo: {sorted(overlap)}")
    return out


def to_mot_records(arr: np.ndarray, with_id: bool = True) -> list[dict]:
    """Converte (N, >=6) [frame, id, l, t, w, h, ...] (ou sem id) em dicts no formato de metrics."""
    out = []
    if with_id:
        for r in arr:
            out.append({"frame": int(r[0]), "id": int(r[1]), "bb_left": float(r[2]),
                        "bb_top": float(r[3]), "bb_width": float(r[4]), "bb_height": float(r[5]),
                        "conf": 1.0})
    else:
        for r in arr:
            out.append({"frame": int(r[0]), "id": -1, "bb_left": float(r[1]),
                        "bb_top": float(r[2]), "bb_width": float(r[3]),
                        "bb_height": float(r[4]), "conf": float(r[5])})
    return out

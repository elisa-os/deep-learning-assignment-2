"""Gerador de vídeos sintéticos e simulador de detector para o PA2 — Parte 0.

Implementa:
1. Gerador de vídeos (default 128x128, 30-60 quadros, 5-15 elipses em movimento).
   As elipses são pintadas em ordem de profundidade (algoritmo do pintor), então uma
   elipse mais próxima realmente esconde pixels das que estão atrás. Para cada
   elipse, em cada quadro, mede-se a *visibilidade* (fração dos seus pixels que
   aparece na imagem final).
2. Oclusão controlada: um par (alvo, oclusor) é roteirizado para que o alvo passe
   por trás do oclusor e fique totalmente escondido por ~``occlusion_duration``
   quadros, saindo do outro lado. Outras oclusões parciais acontecem naturalmente
   quando as elipses se cruzam.
3. Simulador de detector: descarta p% das caixas, adiciona ruído nas coordenadas e
   injeta falsos positivos.

Convenções de ground truth (iguais às do MOT17)
------------------------------------------------
``gt_tracks`` tem uma caixa *amodal* (a elipse inteira, mesmo escondida) por
objeto e quadro, com o campo ``visibility``. ``conf = 1`` se
``visibility >= min_visibility`` e ``0`` caso contrário (objeto praticamente
escondido: fica fora da avaliação, como as caixas "ignoradas" do MOT17).
``true_boxes`` são as caixas que um detector perfeito veria: só os objetos com
``conf = 1``. Um detector não vê o que está atrás de outro objeto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pa2.utils import set_seed

MIN_VISIBILITY = 0.3        # abaixo disso o objeto conta como escondido
HIDDEN_VISIBILITY = 0.05    # "some de verdade" (usado para medir a duração da oclusão)


@dataclass
class Ellipse:
    """Uma elipse em movimento no plano da imagem."""
    id: int
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float
    vx: float
    vy: float
    color: tuple[float, float, float]  # RGB em [0, 1]
    z: int                              # maior z = mais perto da câmera (pintado por cima)


@dataclass
class SyntheticSequence:
    """Uma sequência sintética completa: frames + ground truth das elipses."""
    frames: list[np.ndarray]
    gt_tracks: list[dict[str, Any]]
    true_boxes: list[dict[str, Any]]
    params: dict[str, Any]
    num_frames: int
    frame_size: int
    n_objects: int
    occlusion_durations: list[int]
    detections: list[dict[str, Any]] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Geometria / rasterização
# ─────────────────────────────────────────────────────────────────────────────
def _ellipse_mask(
    yy: np.ndarray, xx: np.ndarray, cx: float, cy: float, rx: float, ry: float, angle: float
) -> np.ndarray:
    """Máscara booleana da elipse rotacionada (equação implícita, vetorizada)."""
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = xx - cx, yy - cy
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def _ellipse_bbox(e: Ellipse, frame_size: int) -> tuple[int, int, int, int]:
    """Caixa envolvente [left, top, w, h] (inteira) da elipse rotacionada, cortada ao quadro."""
    c, s = math.cos(e.angle), math.sin(e.angle)
    hw = math.sqrt((e.rx * c) ** 2 + (e.ry * s) ** 2)
    hh = math.sqrt((e.rx * s) ** 2 + (e.ry * c) ** 2)
    x1 = max(0, int(math.floor(e.cx - hw)))
    y1 = max(0, int(math.floor(e.cy - hh)))
    x2 = min(frame_size, int(math.ceil(e.cx + hw)))
    y2 = min(frame_size, int(math.ceil(e.cy + hh)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _reflect(p: float, lo: float, hi: float, v: float) -> tuple[float, float]:
    """Reflete posição/velocidade nas paredes [lo, hi]."""
    if p < lo:
        return lo + (lo - p), abs(v)
    if p > hi:
        return hi - (p - hi), -abs(v)
    return p, v


# ─────────────────────────────────────────────────────────────────────────────
# Gerador
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_sequence(
    rng: np.random.Generator,
    *,
    num_frames: int = 45,
    frame_size: int = 128,
    n_objects: int = 8,
    velocity_scale: float = 1.0,
    occlusion_duration: int = 10,
    noise_level: float = 0.0,
    contrast_scale: float = 1.0,
    min_visibility: float = MIN_VISIBILITY,
    seed_offset: int = 0,  # mantido por compatibilidade; a aleatoriedade vem de ``rng``
) -> SyntheticSequence:
    """Gera uma sequência sintética.

    Parâmetros
    ----------
    n_objects          : número de elipses (>= 2 para haver oclusão roteirizada).
    velocity_scale     : velocidade típica em px/quadro das elipses livres.
    occlusion_duration : quadros em que o alvo fica escondido atrás do oclusor
                         (0 = sem oclusão roteirizada). É limitado a
                         ``num_frames - 4`` para o alvo aparecer antes e depois.
    noise_level        : desvio-padrão do ruído gaussiano na imagem (escala [0, 1]).
    contrast_scale     : escala o contraste dos objetos contra o fundo (1 = normal).
    min_visibility     : visibilidade mínima para o objeto contar como visível.
    """
    H = W = frame_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy += 0.5
    xx += 0.5

    scripted = occlusion_duration > 0 and n_objects >= 2
    N = int(min(occlusion_duration, max(0, num_frames - 4))) if scripted else 0
    scripted = scripted and N > 0

    # ── elipses ─────────────────────────────────────────────────────────────
    speed0 = velocity_scale * 1.5
    ellipses: list[Ellipse] = []
    for i in range(1, n_objects + 1):
        heading = float(rng.uniform(0, 2 * math.pi))
        speed = speed0 * float(rng.uniform(0.5, 1.5))
        rx = float(rng.uniform(5, 14))
        ry = float(rng.uniform(5, 14))
        ellipses.append(Ellipse(
            id=i,
            cx=float(rng.uniform(rx + 2, W - rx - 2)),
            cy=float(rng.uniform(ry + 2, H - ry - 2)),
            rx=rx, ry=ry,
            angle=float(rng.uniform(0, math.pi)),
            vx=speed * math.cos(heading),
            vy=speed * math.sin(heading),
            color=tuple(float(c) for c in rng.uniform(0.35, 1.0, 3)),
            z=0,
        ))
    # profundidade: permutação única (maior z = mais perto)
    for e, z in zip(ellipses, rng.permutation(n_objects)):
        e.z = int(z)

    # ── oclusão roteirizada ─────────────────────────────────────────────────
    target = occluder = None
    rel_u = np.zeros(2)
    t_mid = 0.0
    start = -1
    rel_limit = 0.0
    if scripted:
        target, occluder = (ellipses[i] for i in rng.choice(n_objects, 2, replace=False))
        # oclusor grande e quase circular; alvo pequeno
        occluder.rx = float(rng.uniform(20, 28))
        occluder.ry = occluder.rx * float(rng.uniform(0.9, 1.0))
        target.rx = float(rng.uniform(5, 8))
        target.ry = target.rx
        occluder.cx = float(rng.uniform(occluder.rx + 4, W - occluder.rx - 4))
        occluder.cy = float(rng.uniform(occluder.ry + 4, H - occluder.ry - 4))
        occluder.vx *= 0.4
        occluder.vy *= 0.4
        # o oclusor tem que ficar na frente do alvo
        if occluder.z < target.z:
            occluder.z, target.z = target.z, occluder.z

        # o alvo atravessa o oclusor em linha reta (movimento relativo uniforme u).
        # Totalmente dentro enquanto |rel| <= D (D = rr - rt); a visibilidade só passa
        # de HIDDEN_VISIBILITY quando ele sai ~0.15*rt além disso. Calibramos a
        # velocidade para que o intervalo "escondido" tenha N quadros.
        D = min(occluder.rx, occluder.ry) - max(target.rx, target.ry)
        u_mag = 2.0 * (D + 0.15 * target.rx) / N
        ang = float(rng.uniform(0, 2 * math.pi))
        rel_u = u_mag * np.array([math.cos(ang), math.sin(ang)])
        pre = min(8, (num_frames - N) // 2)
        start = int(rng.integers(pre, num_frames - N - pre + 1))   # 0-based, 1º quadro escondido
        t_mid = start + (N - 1) / 2.0
        rel_limit = D + 2 * max(target.rx, target.ry) + 12  # fora do oclusor, sem se afastar muito

    # ── cores / fundo ───────────────────────────────────────────────────────
    bg = np.zeros((H, W, 3), dtype=np.float32)
    bg[..., 0] = 0.05 + 0.02 * (xx / W)
    bg[..., 1] = 0.05 + 0.02 * (yy / H)
    bg[..., 2] = 0.05
    bg_mean = bg.mean(axis=(0, 1))

    colors = {}
    for e in ellipses:
        c = np.asarray(e.color, dtype=np.float32)
        colors[e.id] = np.clip(bg_mean + (c - bg_mean) * contrast_scale, 0, 1)

    draw_order = sorted(ellipses, key=lambda e: e.z)  # fundo -> frente

    frames: list[np.ndarray] = []
    gt_tracks: list[dict[str, Any]] = []
    true_boxes: list[dict[str, Any]] = []
    target_vis: list[float] = []
    margin = 1.0

    for frame_idx in range(num_frames):
        # ── posições deste quadro ───────────────────────────────────────────
        if scripted:
            rel = rel_u * (frame_idx - t_mid)
            norm = float(np.linalg.norm(rel))
            if norm > rel_limit:
                rel = rel * (rel_limit / norm)
            target.cx = float(np.clip(occluder.cx + rel[0], target.rx, W - target.rx))
            target.cy = float(np.clip(occluder.cy + rel[1], target.ry, H - target.ry))

        # ── pintura em ordem de profundidade + mapa de ids ─────────────────
        frame = bg.copy()
        label = np.zeros((H, W), dtype=np.int32)
        full_area: dict[int, int] = {}
        for e in draw_order:
            m = _ellipse_mask(yy, xx, e.cx, e.cy, e.rx, e.ry, e.angle)
            full_area[e.id] = int(m.sum())
            frame[m] = colors[e.id]
            label[m] = e.id

        vis_count = np.bincount(label.ravel(), minlength=n_objects + 1)

        if noise_level > 0:
            frame = frame + rng.normal(0, noise_level, frame.shape).astype(np.float32)
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

        # ── ground truth ────────────────────────────────────────────────────
        for e in ellipses:
            vis = float(vis_count[e.id] / full_area[e.id]) if full_area[e.id] > 0 else 0.0
            x1, y1, w, h = _ellipse_bbox(e, frame_size)
            visible = vis >= min_visibility
            row = {
                "frame": frame_idx + 1,
                "id": e.id,
                "bb_left": x1, "bb_top": y1, "bb_width": w, "bb_height": h,
                "conf": 1.0 if visible else 0.0,
                "visibility": vis,
            }
            gt_tracks.append(row)
            if visible:
                true_boxes.append(dict(row))
            if scripted and e.id == target.id:
                target_vis.append(vis)

        # ── movimento (usado no próximo quadro) ─────────────────────────────
        for e in ellipses:
            if scripted and e.id == target.id:
                continue  # o alvo segue o roteiro relativo ao oclusor
            e.cx += e.vx
            e.cy += e.vy
            e.cx, e.vx = _reflect(e.cx, e.rx + margin, W - e.rx - margin, e.vx)
            e.cy, e.vy = _reflect(e.cy, e.ry + margin, H - e.ry - margin, e.vy)
            e.angle += float(rng.uniform(-0.05, 0.05))

    params: dict[str, Any] = {
        "num_frames": num_frames,
        "frame_size": frame_size,
        "n_objects": n_objects,
        "velocity_scale": velocity_scale,
        "occlusion_duration": N,
        "noise_level": noise_level,
        "contrast_scale": contrast_scale,
        "min_visibility": min_visibility,
        "occlusion_ellipse_id": target.id if scripted else None,
        "covering_ellipse_id": occluder.id if scripted else None,
        "occlusion_start": start + 1 if scripted else None,   # 1º quadro escondido (1-based)
        "target_visibility": target_vis,
        "hidden_frames": (
            [i + 1 for i, v in enumerate(target_vis) if v < HIDDEN_VISIBILITY] if scripted else []
        ),
    }
    return SyntheticSequence(
        frames=frames,
        gt_tracks=gt_tracks,
        true_boxes=true_boxes,
        params=params,
        num_frames=num_frames,
        frame_size=frame_size,
        n_objects=n_objects,
        occlusion_durations=[N],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulador de detector
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedDetector:
    """Simula um detector imperfeito a partir das caixas verdadeiras.

    - ``drop_rate``: probabilidade de descartar cada caixa (falso negativo).
    - ``noise_std``: desvio-padrão (px) do ruído nas coordenadas; left/top usam
      ``noise_std`` e largura/altura ``0.3 * noise_std``.
    - ``fp_rate``: média de falsos positivos por quadro (Poisson).

    Cada detecção devolvida traz ``id = -1`` (o detector não conhece identidades),
    ``src_id`` (id GT de origem, ``-1`` para falso positivo) e ``is_fp``.
    """

    def __init__(
        self,
        *,
        drop_rate: float = 0.1,
        noise_std: float = 3.0,
        fp_rate: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.drop_rate = drop_rate
        self.noise_std = noise_std
        self.fp_rate = fp_rate
        self.rng = rng or np.random.default_rng()

    def detect(
        self,
        true_boxes: list[dict[str, Any]],
        frame_size: int,
        num_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        """Estraga as caixas. Se ``num_frames`` for dado, falsos positivos também
        aparecem em quadros sem nenhuma caixa verdadeira."""
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for tb in true_boxes:
            by_frame.setdefault(tb["frame"], []).append(tb)
        frames = range(1, num_frames + 1) if num_frames else sorted(by_frame)

        detections: list[dict[str, Any]] = []
        for frame in frames:
            for tb in by_frame.get(frame, []):
                if self.rng.random() < self.drop_rate:
                    continue
                nx, ny = self.rng.normal(0, self.noise_std, 2)
                nw, nh = self.rng.normal(0, self.noise_std * 0.3, 2)

                left = float(np.clip(tb["bb_left"] + nx, 0, frame_size - 2))
                top = float(np.clip(tb["bb_top"] + ny, 0, frame_size - 2))
                w = float(min(max(2.0, tb["bb_width"] + nw), frame_size - left))
                h = float(min(max(2.0, tb["bb_height"] + nh), frame_size - top))
                conf = max(0.1, 1.0 - abs(nx) / frame_size - abs(ny) / frame_size)

                detections.append({
                    "frame": frame, "id": -1, "src_id": tb["id"], "is_fp": False,
                    "bb_left": left, "bb_top": top, "bb_width": w, "bb_height": h,
                    "conf": conf,
                })

            for _ in range(int(self.rng.poisson(self.fp_rate))):
                fp_w = int(self.rng.integers(5, 20))
                fp_h = int(self.rng.integers(5, 20))
                detections.append({
                    "frame": frame, "id": -1, "src_id": -1, "is_fp": True,
                    "bb_left": int(self.rng.integers(0, frame_size - fp_w)),
                    "bb_top": int(self.rng.integers(0, frame_size - fp_h)),
                    "bb_width": fp_w, "bb_height": fp_h,
                    "conf": float(self.rng.uniform(0.1, 0.4)),
                })

        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticVideoDataset(Dataset):
    """Dataset PyTorch de sequências sintéticas para o PA2."""

    def __init__(
        self,
        *,
        n_sequences: int = 20,
        num_frames: int = 45,
        frame_size: int = 128,
        n_objects_min: int = 5,
        n_objects_max: int = 12,
        velocity_scale: float = 1.0,
        occlusion_duration_min: int = 5,
        occlusion_duration_max: int = 15,
        noise_level: float = 0.0,
        contrast_scale: float = 1.0,
        detector_drop_rate: float = 0.0,
        detector_noise: float = 0.0,
        detector_fp_rate: float = 0.0,
        seed: int = 42,
    ):
        self.n_sequences = n_sequences
        self.num_frames = num_frames
        self.frame_size = frame_size

        set_seed(seed)
        rng = np.random.default_rng(seed)

        self.simulator = SimulatedDetector(
            drop_rate=detector_drop_rate,
            noise_std=detector_noise,
            fp_rate=detector_fp_rate,
            rng=np.random.default_rng(seed + 999),
        )

        self.sequences: list[SyntheticSequence] = []
        for i in range(n_sequences):
            seq = generate_synthetic_sequence(
                rng=rng,
                num_frames=num_frames,
                frame_size=frame_size,
                n_objects=int(rng.integers(n_objects_min, n_objects_max + 1)),
                velocity_scale=velocity_scale * float(rng.uniform(0.5, 1.5)),
                occlusion_duration=int(rng.integers(occlusion_duration_min, occlusion_duration_max + 1)),
                noise_level=noise_level,
                contrast_scale=contrast_scale,
            )
            seq.detections = self.simulator.detect(seq.true_boxes, frame_size, num_frames)
            self.sequences.append(seq)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq = self.sequences[idx]
        frames_tensor = torch.from_numpy(
            np.stack([f.astype(np.float32) / 255.0 for f in seq.frames], axis=0)
        ).permute(0, 3, 1, 2)

        return {
            "frames": frames_tensor,
            "gt_tracks": seq.gt_tracks,
            "detections": seq.detections if seq.detections is not None else seq.true_boxes,
            "sequence_id": idx,
            "params": seq.params,
            "num_frames": seq.num_frames,
            "frame_size": seq.frame_size,
            "occlusion_duration": seq.occlusion_durations[0] if seq.occlusion_durations else 0,
        }


def make_synthetic_loader(
    n_sequences: int = 20,
    num_frames: int = 45,
    frame_size: int = 128,
    n_objects_min: int = 5,
    n_objects_max: int = 12,
    velocity_scale: float = 1.0,
    occlusion_duration_min: int = 5,
    occlusion_duration_max: int = 15,
    noise_level: float = 0.0,
    contrast_scale: float = 1.0,
    detector_drop_rate: float = 0.0,
    detector_noise: float = 0.0,
    detector_fp_rate: float = 0.0,
    batch_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    split: str = "train",
) -> DataLoader:
    """Cria um DataLoader para o dataset sintético (uma sequência por item)."""
    ds = SyntheticVideoDataset(
        n_sequences=n_sequences,
        num_frames=num_frames,
        frame_size=frame_size,
        n_objects_min=n_objects_min,
        n_objects_max=n_objects_max,
        velocity_scale=velocity_scale,
        occlusion_duration_min=occlusion_duration_min,
        occlusion_duration_max=occlusion_duration_max,
        noise_level=noise_level,
        contrast_scale=contrast_scale,
        detector_drop_rate=detector_drop_rate,
        detector_noise=detector_noise,
        detector_fp_rate=detector_fp_rate,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )

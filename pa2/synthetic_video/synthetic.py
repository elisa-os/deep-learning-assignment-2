"""Gerador de vídeos sintéticos e simulador de detector para o PA2 — Parte 0.

Implementa:
1. Gerador de vídeos 128×128 com 30-60 quadros, 5-15 elipses em movimento
   com ordem de profundidade para oclusão real.
2. Simulador de detector: descarta p% das detecções, adiciona ruído nas
   coordenadas, injeta falsos positivos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pa2.utils import set_seed


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
    color: tuple[float, float, float]  # RGB
    depth: int


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


def _draw_ellipse(
    image: np.ndarray,
    ellipse: Ellipse,
) -> None:
    """Desenha uma elipse no frame respeitando ordem de profundidade (implicitamente pelo caller)."""
    from skimage.draw import ellipse as sk_ellipse

    H, W = image.shape[:2]
    rr, cc = sk_ellipse(
        ellipse.cy, ellipse.cx,
        ellipse.ry, ellipse.rx,
        rotation=ellipse.angle,
        shape=image.shape[:2],
    )
    for y, x in zip(rr, cc):
        if 0 <= y < H and 0 <= x < W:
            image[y, x, 0] = ellipse.color[0]
            image[y, x, 1] = ellipse.color[1]
            image[y, x, 2] = ellipse.color[2]


def _draw_ellipse_fast(
    image: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle: float,
    color: tuple[float, float, float],
) -> None:
    """Desenha elipse usando equação da elipse rotacionada via máscara pixel a pixel (lento mas certeiro)."""
    from skimage.draw import ellipse as sk_ellipse

    rr, cc = sk_ellipse(
        cy, cx,
        ry, rx,
        rotation=angle,
        shape=image.shape[:2],
    )
    for y, x in zip(rr, cc):
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            image[y, x, 0] = color[0]
            image[y, x, 1] = color[1]
            image[y, x, 2] = color[2]


def generate_synthetic_sequence(
    rng: np.random.Generator,
    *,
    num_frames: int = 45,
    frame_size: int = 128,
    n_objects: int = 8,
    velocity_scale: float = 1.0,
    occlusion_duration: int = 10,
    seed_offset: int = 0,
) -> SyntheticSequence:
    """Gera uma sequência sintética com elipses em movimento e oclusão."""
    H = W = frame_size

    ellipses: list[Ellipse] = []
    colors: list[tuple[float, float, float]] = []

    for i in range(n_objects):
        hue = (i / n_objects) * 0.8 + 0.1
        r = math.sin(2 * math.pi * hue) * 0.5 + 0.5
        g = math.sin(2 * math.pi * (hue + 1 / 3)) * 0.5 + 0.5
        b = math.sin(2 * math.pi * (hue + 2 / 3)) * 0.5 + 0.5
        colors.append((r, g, b))

    for i in range(n_objects):
        margin = 20
        cx = float(rng.uniform(margin, W - margin))
        cy = float(rng.uniform(margin, H - margin))
        rx = float(rng.uniform(6, 16))
        ry = float(rng.uniform(6, 16))
        angle = float(rng.uniform(0, 2 * math.pi))

        target_x = float(rng.uniform(margin, W - margin))
        target_y = float(rng.uniform(margin, H - margin))
        dx = target_x - cx
        dy = target_y - cy
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist > 0:
            vx = (dx / dist) * float(rng.uniform(0.3, 1.2)) * velocity_scale
            vy = (dy / dist) * float(rng.uniform(0.3, 1.2)) * velocity_scale
        else:
            vx = float(rng.uniform(-0.5, 0.5)) * velocity_scale
            vy = float(rng.uniform(-0.5, 0.5)) * velocity_scale

        ellipses.append(Ellipse(
            id=i + 1,
            cx=cx, cy=cy,
            rx=rx, ry=ry,
            angle=angle,
            vx=vx, vy=vy,
            color=colors[i],
            depth=int(rng.integers(1, 100)),
        ))

    ellipses_sorted = sorted(ellipses, key=lambda e: e.depth)

    occlusion_ellipse = ellipses[int(rng.integers(0, len(ellipses)))]
    occlusion_start = int(rng.integers(occlusion_duration, num_frames - occlusion_duration))

    covering_ellipses = [e for e in ellipses if e.depth < occlusion_ellipse.depth]
    if not covering_ellipses:
        covering_ellipses = [e for e in ellipses if e.id != occlusion_ellipse.id]
    covering_ellipse = covering_ellipses[int(rng.integers(0, len(covering_ellipses)))]

    frames: list[np.ndarray] = []
    gt_tracks: list[dict[str, Any]] = []
    true_boxes: list[dict[str, Any]] = []
    occlusion_durations: list[int] = [occlusion_duration]

    for frame_idx in range(num_frames):
        frame = np.zeros((H, W, 3), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                frame[y, x, 0] = 0.05 + 0.02 * (x / W)
                frame[y, x, 1] = 0.05 + 0.02 * (y / H)
                frame[y, x, 2] = 0.05

        in_occlusion = (occlusion_start <= frame_idx < occlusion_start + occlusion_duration)

        for ellipse in ellipses_sorted:
            is_occluded = False
            if in_occlusion and ellipse.id == occlusion_ellipse.id:
                dx = ellipse.cx - covering_ellipse.cx
                dy = ellipse.cy - covering_ellipse.cy
                dist = math.sqrt(dx ** 2 + dy ** 2)
                if dist < (covering_ellipse.rx + covering_ellipse.ry) * 0.7:
                    is_occluded = True
            if not is_occluded:
                _draw_ellipse_fast(frame, ellipse.cx, ellipse.cy,
                                   ellipse.rx, ellipse.ry, ellipse.angle,
                                   ellipse.color)

        for ellipse in ellipses:
            ellipse.cx += ellipse.vx
            ellipse.cy += ellipse.vy
            margin2 = 10
            if ellipse.cx < margin2:
                ellipse.cx = margin2
                ellipse.vx = abs(ellipse.vx) * float(rng.uniform(0.5, 1.5))
            elif ellipse.cx > W - margin2:
                ellipse.cx = W - margin2
                ellipse.vx = -abs(ellipse.vx) * float(rng.uniform(0.5, 1.5))
            if ellipse.cy < margin2:
                ellipse.cy = margin2
                ellipse.vy = abs(ellipse.vy) * float(rng.uniform(0.5, 1.5))
            elif ellipse.cy > H - margin2:
                ellipse.cy = H - margin2
                ellipse.vy = -abs(ellipse.vy) * float(rng.uniform(0.5, 1.5))
            ellipse.angle += float(rng.uniform(-0.05, 0.05))

        frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frames.append(frame_uint8)

        for ellipse in ellipses:
            x1 = max(0, int(ellipse.cx - ellipse.rx))
            y1 = max(0, int(ellipse.cy - ellipse.ry))
            w = min(int(ellipse.rx * 2), W - x1)
            h = min(int(ellipse.ry * 2), H - y1)
            gt_tracks.append({
                "frame": frame_idx + 1,
                "id": ellipse.id,
                "bb_left": x1,
                "bb_top": y1,
                "bb_width": w,
                "bb_height": h,
                "conf": 1.0,
            })
            true_boxes.append({
                "frame": frame_idx + 1,
                "id": ellipse.id,
                "bb_left": x1,
                "bb_top": y1,
                "bb_width": w,
                "bb_height": h,
                "conf": 1.0,
            })

    params = {
        "num_frames": num_frames,
        "frame_size": frame_size,
        "n_objects": n_objects,
        "velocity_scale": velocity_scale,
        "occlusion_duration": occlusion_duration,
        "occlusion_start": occlusion_start,
        "occlusion_ellipse_id": occlusion_ellipse.id,
        "covering_ellipse_id": covering_ellipse.id,
        "seed_offset": seed_offset,
    }

    return SyntheticSequence(
        frames=frames,
        gt_tracks=gt_tracks,
        true_boxes=true_boxes,
        params=params,
        num_frames=num_frames,
        frame_size=frame_size,
        n_objects=n_objects,
        occlusion_durations=occlusion_durations,
        detections=None,
    )


class SimulatedDetector:
    """Simula um detector imperfeito sobre as caixas verdadeiras."""

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
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for tb in true_boxes:
            f = tb["frame"]
            if f not in by_frame:
                by_frame[f] = []
            by_frame[f].append(tb)

        for frame, boxes in by_frame.items():
            for tb in boxes:
                if self.rng.random() < self.drop_rate:
                    continue
                noise_x = self.rng.normal(0, self.noise_std)
                noise_y = self.rng.normal(0, self.noise_std)
                noise_w = self.rng.normal(0, self.noise_std * 0.3)
                noise_h = self.rng.normal(0, self.noise_std * 0.3)

                new_left = max(0, tb["bb_left"] + noise_x)
                new_top = max(0, tb["bb_top"] + noise_y)
                new_w = max(2, tb["bb_width"] + noise_w)
                new_h = max(2, tb["bb_height"] + noise_h)
                new_w = min(new_w, frame_size - new_left)
                new_h = min(new_h, frame_size - new_top)
                conf = max(0.1, 1.0 - abs(noise_x) / frame_size - abs(noise_y) / frame_size)

                detections.append({
                    "frame": frame,
                    "id": -1,
                    "bb_left": new_left,
                    "bb_top": new_top,
                    "bb_width": new_w,
                    "bb_height": new_h,
                    "conf": conf,
                })

            n_fp = int(self.rng.poisson(self.fp_rate))
            for _ in range(n_fp):
                fp_left = int(self.rng.integers(0, frame_size - 10))
                fp_top = int(self.rng.integers(0, frame_size - 10))
                fp_w = int(self.rng.integers(5, 20))
                fp_h = int(self.rng.integers(5, 20))
                detections.append({
                    "frame": frame,
                    "id": -1,
                    "bb_left": fp_left,
                    "bb_top": fp_top,
                    "bb_width": fp_w,
                    "bb_height": fp_h,
                    "conf": float(self.rng.uniform(0.1, 0.4)),
                })

        return detections


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
        self.n_objects_min = n_objects_min
        self.n_objects_max = n_objects_max
        self.velocity_scale = velocity_scale
        self.occlusion_duration_min = occlusion_duration_min
        self.occlusion_duration_max = occlusion_duration_max
        self.noise_level = noise_level
        self.contrast_scale = contrast_scale
        self.detector_drop_rate = detector_drop_rate
        self.detector_noise = detector_noise
        self.detector_fp_rate = detector_fp_rate

        set_seed(seed)
        rng = np.random.default_rng(seed)

        self.sequences: list[SyntheticSequence] = []
        self.simulator = SimulatedDetector(
            drop_rate=detector_drop_rate,
            noise_std=detector_noise,
            fp_rate=detector_fp_rate,
            rng=np.random.default_rng(seed + 999),
        )

        for i in range(n_sequences):
            n_obj = int(rng.integers(n_objects_min, n_objects_max + 1))
            oc_dur = int(rng.integers(occlusion_duration_min, occlusion_duration_max + 1))
            vel_scale = velocity_scale * float(rng.uniform(0.5, 1.5))

            seq = generate_synthetic_sequence(
                rng=rng,
                num_frames=num_frames,
                frame_size=frame_size,
                n_objects=n_obj,
                velocity_scale=vel_scale,
                occlusion_duration=oc_dur,
                seed_offset=i * 1000,
            )

            if noise_level > 0:
                for j in range(len(seq.frames)):
                    noise = rng.normal(0, noise_level, seq.frames[j].shape).astype(np.float32)
                    seq.frames[j] = np.clip(seq.frames[j].astype(np.float32) + noise, 0, 255).astype(np.uint8)

            if contrast_scale != 1.0:
                for j in range(len(seq.frames)):
                    mean = seq.frames[j].mean()
                    seq.frames[j] = np.clip(
                        (seq.frames[j].astype(np.float32) - mean) * contrast_scale + mean,
                        0, 255
                    ).astype(np.uint8)

            seq.detections = self.simulator.detect(seq.true_boxes, frame_size)
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
            "detections": seq.detections or seq.true_boxes,
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
    """Cria um DataLoader para o dataset sintético."""
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

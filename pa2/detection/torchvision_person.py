"""Detector de pessoas pré-treinado do torchvision (Faster R-CNN, COCO) — só inferência.

- Classe ``person`` do COCO (rótulo 1); as demais são descartadas.
- O NMS interno do torchvision é *desligado* (``roi_heads.nms_thresh = 1.0``) e o NMS
  aplicado às caixas é o nosso (``pa2.detection.nms``), como exige o enunciado.
- Sem fine-tune. As detecções de uma sequência são gravadas em cache no formato det.txt
  e reaproveitadas pelas Partes 2-5 (a fonte de detecções fica congelada).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from pa2.detection.nms import nms
from pa2.mot17.loader import Sequence, load_detections, write_detections

COCO_PERSON = 1


class TorchvisionPersonDetector:
    def __init__(
        self,
        device: torch.device | str = "cpu",
        score_thresh: float = 0.05,
        nms_iou: float = 0.5,
        model: torch.nn.Module | None = None,
    ):
        """``model`` permite injetar outro modelo no formato do torchvision (usado em testes)."""
        if model is None:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_Weights,
                fasterrcnn_resnet50_fpn,
            )
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.model = model.to(device).eval()
        self.device = device
        self.nms_iou = nms_iou
        # score baixo e NMS interno desligado; o NMS é o nosso. Sem teto de 100 detecções.
        self.model.roi_heads.score_thresh = score_thresh
        self.model.roi_heads.nms_thresh = 1.0
        self.model.roi_heads.detections_per_img = 1000

    @torch.no_grad()
    def detect_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """``images``: lista de HxWx3 RGB uint8. Devolve, por imagem, (n, 5) [l, t, w, h, score]."""
        tensors = [torch.from_numpy(im).permute(2, 0, 1).float().div(255.0).to(self.device)
                   for im in images]
        outputs = self.model(tensors)
        results = []
        for out in outputs:
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            sel = labels == COCO_PERSON
            boxes, scores = boxes[sel], scores[sel]
            keep = nms(boxes, scores, self.nms_iou)
            boxes, scores = boxes[keep], scores[keep]
            ltwh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0],
                             boxes[:, 3] - boxes[:, 1], scores], axis=1) if len(boxes) else np.zeros((0, 5))
            results.append(ltwh)
        return results

    def detect_sequence(
        self,
        seq: Sequence,
        cache_path: str | Path,
        batch_size: int = 2,
        frames: range | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Roda o detector em todos os quadros de ``seq`` (ou ``frames``) e grava o cache.

        Se o cache já existe, devolve o que está nele (apague-o para refazer).
        Retorna (M, 6): frame, l, t, w, h, score.
        """
        import cv2

        cache_path = Path(cache_path)
        if cache_path.exists():
            return load_detections(cache_path)
        if not seq.has_images():
            raise FileNotFoundError(
                f"{seq.name}: imagens não encontradas em {seq.image_dir}. "
                "Baixe o pacote completo do MOT17 (~5,5 GB) e extraia em data/MOT17/."
            )
        frames = frames or range(1, seq.info.seq_length + 1)
        rows: list[np.ndarray] = []
        todo = list(frames)
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            imgs = []
            for f in chunk:
                bgr = cv2.imread(str(seq.image_path(f)))
                if bgr is None:
                    raise FileNotFoundError(seq.image_path(f))
                imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for f, d in zip(chunk, self.detect_batch(imgs)):
                if len(d):
                    rows.append(np.column_stack([np.full(len(d), f), d]))
            if progress:
                progress(min(start + batch_size, len(todo)), len(todo))
        dets = np.concatenate(rows) if rows else np.zeros((0, 6))
        write_detections(cache_path, dets)
        return dets

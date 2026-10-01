"""Detecção: NMS próprio e detector pré-treinado do torchvision (inferência)."""

from .nms import batched_nms, box_iou_xyxy, nms

__all__ = ["batched_nms", "box_iou_xyxy", "nms"]

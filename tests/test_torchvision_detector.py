"""Pós-processamento do detector torchvision, com um modelo falso (sem pesos nem imagens)."""

import numpy as np
import torch

from pa2.detection.torchvision_person import COCO_PERSON, TorchvisionPersonDetector


class _FakeRoiHeads:
    score_thresh = 0.5
    nms_thresh = 0.5
    detections_per_img = 100


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roi_heads = _FakeRoiHeads()

    def forward(self, images):
        boxes = torch.tensor([[10., 10, 50, 110],      # pessoa
                              [11., 11, 51, 111],      # duplicata da pessoa (NMS remove)
                              [200., 10, 240, 110],    # outra pessoa
                              [10., 10, 50, 110]])     # carro (outra classe: descartado)
        return [{"boxes": boxes, "scores": torch.tensor([0.9, 0.8, 0.7, 0.95]),
                 "labels": torch.tensor([COCO_PERSON, COCO_PERSON, COCO_PERSON, 3])}
                for _ in images]


def test_postprocessing_filters_class_applies_own_nms_and_converts_boxes():
    det = TorchvisionPersonDetector(model=_FakeModel(), score_thresh=0.05, nms_iou=0.5)
    out = det.detect_batch([np.zeros((120, 260, 3), np.uint8)])[0]
    assert out.shape == (2, 5)
    assert np.allclose(out[0], [10, 10, 40, 100, 0.9])
    assert np.allclose(out[1], [200, 10, 40, 100, 0.7])


def test_internal_nms_is_disabled_and_threshold_lowered():
    m = _FakeModel()
    TorchvisionPersonDetector(model=m, score_thresh=0.05)
    assert m.roi_heads.nms_thresh == 1.0 and m.roi_heads.score_thresh == 0.05
    assert m.roi_heads.detections_per_img >= 1000


def test_missing_images_raise_clear_error(tmp_path):
    import pytest
    from pa2.mot17.loader import Sequence
    root = tmp_path / "train" / "MOT17-99-FRCNN"
    (root / "det").mkdir(parents=True)
    (root / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-99-FRCNN\nimDir=img1\nframeRate=30\nseqLength=2\n"
        "imWidth=64\nimHeight=64\nimExt=.jpg\n")
    seq = Sequence(tmp_path, "99", "FRCNN")
    det = TorchvisionPersonDetector(model=_FakeModel())
    with pytest.raises(FileNotFoundError, match="imagens"):
        det.detect_sequence(seq, tmp_path / "cache.txt")

"""Loader e avaliação no MOT17 real (pulados se o pacote de anotações não estiver em data/MOT17)."""

from pathlib import Path

import numpy as np
import pytest

from pa2.mot17 import DEFAULT_SPLIT, Sequence, resolve_split
from pa2.mot17.evaluate import evaluate_sequence_detections, evaluate_tracks

ROOT = Path("data/MOT17")
needs_data = pytest.mark.skipif(not (ROOT / "train/MOT17-09-FRCNN/gt/gt.txt").exists(),
                                reason="MOT17 (anotações) não encontrado em data/MOT17")


def test_default_split_is_by_video_and_disjoint():
    s = resolve_split(None)
    assert set(s["train"]).isdisjoint(s["val"]) and s == DEFAULT_SPLIT
    with pytest.raises(ValueError):
        resolve_split({"train": ["02", "09"], "val": ["09"]})


def test_split_from_config_overrides_default():
    assert resolve_split({"train": ["MOT17-02"], "val": ["05"]}) == {"train": ["02"], "val": ["05"]}


@needs_data
def test_loader_shapes_and_conventions():
    s = Sequence(ROOT, "09", "FRCNN")
    assert s.info.seq_length == 525 and (s.info.im_width, s.info.im_height) == (1920, 1080)
    ped = s.gt_pedestrians()
    assert ped.shape[1] == 7 and ped[:, 0].min() >= 1 and ped[:, 0].max() <= 525
    assert (ped[:, 6] >= 0).all() and (ped[:, 6] <= 1).all()
    assert s.det.shape[1] == 6 and len(s.gt_distractors()) > 0


@needs_data
def test_gt_is_identical_across_public_detectors():
    a, b = (Sequence(ROOT, "09", d).gt for d in ("DPM", "SDP"))
    assert a.shape == b.shape and np.allclose(a, b)


@needs_data
@pytest.mark.parametrize("video", ["09", "05"])
def test_gt_as_prediction_gives_perfect_tracking(video):
    s = Sequence(ROOT, video, "FRCNN")
    gt = s.gt_pedestrians()
    tracks = [{"frame": int(r[0]), "id": int(r[1]), "bb_left": r[2], "bb_top": r[3],
               "bb_width": r[4], "bb_height": r[5], "conf": 1.0} for r in gt]
    r = evaluate_tracks(s, tracks)
    assert r["idf1"] == pytest.approx(1.0) and r["id_switches"] == 0 and r["fragmentations"] == 0
    assert r["n_pred_ids"] == r["n_gt_ids"] and r["mota"] == pytest.approx(1.0)


@needs_data
def test_gt_as_detections_gives_perfect_ap():
    s = Sequence(ROOT, "09", "FRCNN")
    gt = s.gt_pedestrians()
    dets = np.column_stack([gt[:, 0], gt[:, 2:6], np.ones(len(gt))])
    r = evaluate_sequence_detections(s, dets)
    assert r["ap50"] == pytest.approx(1.0) and r["recall"] == pytest.approx(1.0)


@needs_data
def test_distractor_predictions_are_not_penalised():
    s = Sequence(ROOT, "09", "FRCNN")
    gt = s.gt_pedestrians()
    ign = s.gt_distractors()
    tracks = [{"frame": int(r[0]), "id": int(r[1]), "bb_left": r[2], "bb_top": r[3],
               "bb_width": r[4], "bb_height": r[5], "conf": 1.0} for r in gt]
    base = evaluate_tracks(s, tracks)
    # acrescenta uma track que segue exatamente os distratores: não pode mudar nada
    extra = [{"frame": int(r[0]), "id": 10_000 + int(r[1]), "bb_left": r[2], "bb_top": r[3],
              "bb_width": r[4], "bb_height": r[5], "conf": 1.0} for r in ign]
    with_ign = evaluate_tracks(s, tracks + extra)
    assert with_ign["IDFP"] == base["IDFP"] and with_ign["idf1"] == pytest.approx(base["idf1"])

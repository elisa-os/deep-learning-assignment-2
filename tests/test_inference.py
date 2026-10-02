"""Inferência sobre uma sequência qualquer: carregamento da pasta, contagem, vídeo e cores consistentes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pa2.inference import (FrameRenderer, id_color, render_gif, render_video, run_inference, sample_frames,
                           score_threshold_for)
from pa2.models import MotionRNN, save_checkpoint
from pa2.mot17 import Sequence

ROOT = Path("data/MOT17")
needs_data = pytest.mark.skipif(not (ROOT / "train/MOT17-09-SDP/gt/gt.txt").exists(),
                                reason="MOT17 (anotações) não encontrado em data/MOT17")
W, H, T = 320, 240, 20


def _make_seq(root: Path, name="MOT17-77-SDP", det=True, gt=False, images=False, noise=True):
    """2 pessoas 30x70 andando 3 px/quadro, bem separadas; detecções = GT (+ ruído de baixo score)."""
    d = root / name
    (d / "det").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=10\nseqLength={T}\nimWidth={W}\nimHeight={H}\nimExt=.jpg\n")
    rows = [(f, i, 20 + 3 * f + 120 * i, 60, 30, 70) for f in range(1, T + 1) for i in (0, 1)]
    if det:
        lines = [f"{f},-1,{x},{y},{w},{h},0.9,-1,-1,-1" for f, i, x, y, w, h in rows]
        if noise:
            lines += [f"{f},-1,5,5,10,10,0.1,-1,-1,-1" for f in range(1, T + 1)]     # abaixo do limiar
        (d / "det" / "det.txt").write_text("\n".join(lines) + "\n")
    if gt:
        (d / "gt").mkdir()
        (d / "gt" / "gt.txt").write_text(
            "\n".join(f"{f},{i + 1},{x},{y},{w},{h},1,1,1.0" for f, i, x, y, w, h in rows) + "\n")
    if images:
        import cv2
        (d / "img1").mkdir()
        for f in range(1, T + 1):
            cv2.imwrite(str(d / "img1" / f"{f:06d}.jpg"), np.full((H, W, 3), (0, 0, 200), np.uint8))   # BGR vermelho
    return d


@pytest.fixture
def ckpt(tmp_path):
    p = tmp_path / "model.pt"
    save_checkpoint(MotionRNN("GRU", 16), p)
    return p


# ── carregamento ────────────────────────────────────────────────────────────
def test_from_dir_parses_names_and_flags(tmp_path):
    d = _make_seq(tmp_path, gt=True)
    s = Sequence.from_dir(d)
    assert (s.video, s.detector, s.name) == ("77", "SDP", "MOT17-77")
    assert s.has_gt() and s.has_det() and not s.has_images() and s.info.seq_length == T


def test_from_dir_accepts_free_names_and_rejects_non_sequences(tmp_path):
    d = _make_seq(tmp_path, name="minha_camera")
    s = Sequence.from_dir(d)
    assert s.name == "minha_camera" and s.detector == "?"
    with pytest.raises(FileNotFoundError, match="seqinfo.ini"):
        Sequence.from_dir(tmp_path)


def test_thresholds_per_detector():
    assert score_threshold_for("SDP") == 0.4 and score_threshold_for("FRCNN") == 0.05
    assert score_threshold_for("?") == 0.4


# ── inferência ──────────────────────────────────────────────────────────────
def test_two_well_separated_people_get_two_consistent_ids(tmp_path, ckpt):
    r = run_inference(_make_seq(tmp_path), checkpoint=ckpt)
    c = r.counts()
    assert c["ids_unicos"] == 2 and c["ids_gt"] is None and r.metrics is None
    t = r.table
    assert set(t.groupby("id").size()) == {T - 2}          # min_hits = 3 descarta 2 quadros por track
    assert r.n_detections == 2 * T                          # o ruído de score 0.1 ficou de fora
    assert "det/det.txt" in r.detector_source


def test_ground_truth_is_optional_and_gives_metrics_when_present(tmp_path, ckpt):
    r = run_inference(_make_seq(tmp_path, gt=True), checkpoint=ckpt)
    assert r.counts()["ids_gt"] == 2
    assert set(r.metrics) >= {"idf1", "id_switches", "fragmentations", "n_gt_ids", "n_pred_ids"}
    assert 0.5 < r.metrics["idf1"] <= 1.0


def test_max_frames_truncates_and_skips_metrics(tmp_path, ckpt):
    r = run_inference(_make_seq(tmp_path, gt=True), checkpoint=ckpt, max_frames=8)
    assert r.n_frames == 8 and r.table.frame.max() <= 8 and r.metrics is None


def test_min_track_len_changes_the_filtered_count_only(tmp_path, ckpt):
    d = _make_seq(tmp_path)
    a = run_inference(d, checkpoint=ckpt, min_track_len=5).counts()
    b = run_inference(d, checkpoint=ckpt, min_track_len=100).counts()
    assert a["ids_unicos"] == b["ids_unicos"] == 2
    assert a["ids_unicos_com_5+_quadros"] == 2 and b["ids_unicos_com_100+_quadros"] == 0


def test_mot_txt_roundtrip(tmp_path, ckpt):
    r = run_inference(_make_seq(tmp_path), checkpoint=ckpt)
    p = r.save_mot_txt(tmp_path / "out" / "res.txt")
    d = pd.read_csv(p, header=None)
    assert d.shape == (len(r.tracks), 10) and set(d[1]) == set(r.table.id)
    assert d.iloc[:, 7:].eq(-1).all().all()


def test_clear_errors_without_any_detection_source(tmp_path, ckpt):
    d = _make_seq(tmp_path, det=False)
    with pytest.raises(FileNotFoundError, match="de onde tirar detecções"):
        run_inference(d, checkpoint=ckpt)
    with pytest.raises(FileNotFoundError, match="det.txt"):
        run_inference(d, checkpoint=ckpt, detector="public")
    with pytest.raises(ValueError):
        run_inference(_make_seq(tmp_path, name="MOT17-78-SDP"), checkpoint=ckpt, detector="foo")


# ── cores e vídeo ───────────────────────────────────────────────────────────
def test_id_color_is_deterministic_and_distinct():
    cols = [id_color(i) for i in range(1, 60)]
    assert cols == [id_color(i) for i in range(1, 60)] and len(set(cols)) == len(cols)
    assert id_color(7) == id_color(7) and id_color(7) != id_color(8)


def test_same_id_same_color_across_frames_on_the_rendered_image(tmp_path, ckpt):
    r = run_inference(_make_seq(tmp_path), checkpoint=ckpt)
    rd = FrameRenderer(r, scale=1.0, trail=0)
    tid, box = next(iter(r.boxes_at(10).items()))
    for f in (5, 10, 15):
        img = rd.render(f)
        b = r.boxes_at(f)[tid]
        x1, y1 = int(round(b[0])), int(round(b[1]))
        rgb = id_color(tid)
        assert tuple(img[y1 + 10, x1]) == (rgb[2], rgb[1], rgb[0])       # borda esquerda da caixa, em BGR


def test_images_are_used_as_background_when_present_and_blank_otherwise(tmp_path, ckpt):
    with_img = run_inference(_make_seq(tmp_path, name="MOT17-80-SDP", images=True), checkpoint=ckpt)
    blank = run_inference(_make_seq(tmp_path, name="MOT17-81-SDP"), checkpoint=ckpt)
    a = FrameRenderer(with_img, scale=1.0).render(3)
    b = FrameRenderer(blank, scale=1.0).render(3)
    assert tuple(a[H - 5, W - 5]) == (0, 0, 200) and tuple(b[H - 5, W - 5]) == (28, 28, 28)
    assert tuple(FrameRenderer(with_img, scale=1.0, use_images=False).render(3)[H - 5, W - 5]) == (28, 28, 28)


def test_render_video_writes_all_frames(tmp_path, ckpt):
    import cv2
    r = run_inference(_make_seq(tmp_path), checkpoint=ckpt)
    p = render_video(r, tmp_path / "v" / "out.mp4", scale=1.0)
    cap = cv2.VideoCapture(str(p))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == T
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (W, H)
    cap.release()


def test_gif_and_sample_frames(tmp_path, ckpt):
    import imageio.v2 as imageio
    r = run_inference(_make_seq(tmp_path), checkpoint=ckpt)
    g = render_gif(r, tmp_path / "o.gif", scale=1.0, max_frames=7)
    assert len(imageio.mimread(g)) <= 7
    fr = sample_frames(r, [1, 10, 20], scale=0.5)
    assert len(fr) == 3 and fr[0].shape == (H // 2, W // 2, 3)


# ── dados reais ─────────────────────────────────────────────────────────────
@needs_data
def test_real_sequence_matches_the_part2_pipeline():
    ck = Path("outputs/checkpoints/final_motion_rnn.pt")
    if not ck.exists():
        pytest.skip("sem checkpoint final")
    r = run_inference(ROOT / "train/MOT17-09-SDP", checkpoint=ck)
    ref = pd.read_csv("outputs/final/parte2_per_sequence.csv", dtype={"sequence": str})
    row = ref[(ref.method == "rnn") & (ref.sequence == "MOT17-09")].iloc[0]
    # a Parte 2 filtra detecções sobre distratores antes de rastrear; aqui não há esse filtro (nem GT):
    # o resultado praticamente não muda (verificado: IDF1 médio 0,595 com e 0,596 sem)
    assert r.metrics["idf1"] == pytest.approx(row.IDF1, abs=0.01)
    assert r.counts()["ids_gt"] == 26


@needs_data
def test_sequence_without_gt_runs():
    ck = Path("outputs/checkpoints/final_motion_rnn.pt")
    if not ck.exists() or not (ROOT / "test/MOT17-01-SDP").exists():
        pytest.skip("sem checkpoint ou sem a pasta test/")
    r = run_inference(ROOT / "test/MOT17-01-SDP", checkpoint=ck, max_frames=60)
    assert r.metrics is None and r.counts()["ids_gt"] is None and r.counts()["ids_unicos"] > 0

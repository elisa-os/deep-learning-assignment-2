"""Matcher guloso por IoU da Parte 0 (baseline sobre os vídeos sintéticos).

A associação das Partes 1 a 5 está em ``pa2/association/tracker.py`` (``IoUTracker``) e
``pa2/association/motion_tracker.py`` (``MotionTracker``). Este módulo ficou só com o
``GreedyMatcher``, que a Parte 0 usa; o resto do esqueleto herdado do PA1 foi removido.

Proibido usar rastreadores prontos ou torchvision.ops.nms. Esta é uma implementação própria.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pa2.metrics.tracking import compute_iou


class GreedyMatcher:
    """Matching guloso por IoU decrescente entre detections e tracks ativos."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.active_tracks: dict[int, dict[str, Any]] = {}
        self.next_track_id: int = 1
        self.dead_tracks: set[int] = set()

    def match_frame(
        self,
        detections: list[dict[str, Any]],
        frame: int,
    ) -> list[tuple[int, int, float, bool]]:
        """Faz matching entre detections e tracks ativos para um frame.

        Retorna lista de (track_id, det_idx, iou, is_new_track).
        is_new_track=True indica que a detecção criou um novo track.
        """
        if not self.active_tracks:
            matches = []
            for i, det in enumerate(detections):
                track_id = self.next_track_id
                self.next_track_id += 1
                self.active_tracks[track_id] = {
                    "ages_since_update": 0,
                    "total_hits": 1,
                    "frames_seen": [frame],
                    "boxes": [np.array([det["bb_left"], det["bb_top"],
                                        det["bb_width"], det["bb_height"]])],
                }
                matches.append((track_id, i, 1.0, True))
            return matches

        active_ids = list(self.active_tracks.keys())
        track_boxes = [self.active_tracks[tid]["boxes"][-1] for tid in active_ids]
        det_boxes = [np.array([d["bb_left"], d["bb_top"],
                               d["bb_width"], d["bb_height"]]) for d in detections]

        iou_mat = np.zeros((len(detections), len(active_ids)))
        for i in range(len(detections)):
            for j in range(len(active_ids)):
                iou_mat[i, j] = compute_iou(det_boxes[i], track_boxes[j])

        pairs = []
        for i in range(len(detections)):
            for j in range(len(active_ids)):
                iou_val = iou_mat[i, j]
                if iou_val >= self.iou_threshold:
                    pairs.append((iou_val, i, j))

        pairs.sort(reverse=True, key=lambda x: x[0])

        matched_detections = set()
        matched_tracks = set()
        matches = []

        for iou_val, det_idx, active_idx in pairs:
            if det_idx in matched_detections or active_idx in matched_tracks:
                continue
            track_id = active_ids[active_idx]
            matches.append((track_id, det_idx, iou_val, False))
            matched_detections.add(det_idx)
            matched_tracks.add(active_idx)

        # Atualiza tracks casados
        for track_id, det_idx, _, _ in matches:
            det = detections[det_idx]
            box = np.array([det["bb_left"], det["bb_top"],
                           det["bb_width"], det["bb_height"]])
            self.active_tracks[track_id]["ages_since_update"] = 0
            self.active_tracks[track_id]["total_hits"] += 1
            self.active_tracks[track_id]["frames_seen"].append(frame)
            self.active_tracks[track_id]["boxes"].append(box)

        # Detectções não casadas → novos tracks
        new_track_matches = []
        for i, det in enumerate(detections):
            if i not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.active_tracks[track_id] = {
                    "ages_since_update": 0,
                    "total_hits": 1,
                    "frames_seen": [frame],
                    "boxes": [np.array([det["bb_left"], det["bb_top"],
                                        det["bb_width"], det["bb_height"]])],
                }
                new_track_matches.append((track_id, i, 0.0, True))

        # Atualiza idade dos tracks não casados
        for tid in active_ids:
            if tid not in matched_tracks:
                self.active_tracks[tid]["ages_since_update"] += 1

        # Mata tracks com idade > max_age
        dead = [tid for tid, state in self.active_tracks.items()
                if state["ages_since_update"] > self.max_age]
        for tid in dead:
            del self.active_tracks[tid]
            self.dead_tracks.add(tid)

        return matches + new_track_matches

    def run(
        self,
        detections: list[dict[str, Any]],
        num_frames: int,
    ) -> list[dict[str, Any]]:
        """Roda o tracking completo sobre uma lista de detecções."""
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for det in detections:
            f = det["frame"]
            if f not in by_frame:
                by_frame[f] = []
            by_frame[f].append(det)

        all_tracks: list[dict[str, Any]] = []

        for frame in range(1, num_frames + 1):
            frame_dets = by_frame.get(frame, [])
            matches = self.match_frame(frame_dets, frame)

            for track_id, det_idx, _, _ in matches:
                det = frame_dets[det_idx]
                all_tracks.append({
                    "frame": frame,
                    "id": track_id,
                    "bb_left": det["bb_left"],
                    "bb_top": det["bb_top"],
                    "bb_width": det["bb_width"],
                    "bb_height": det["bb_height"],
                    "conf": det.get("conf", 1.0),
                })

        return all_tracks

    def reset(self) -> None:
        """Reseta o estado interno do matcher."""
        self.active_tracks = {}
        self.next_track_id = 1
        self.dead_tracks = set()

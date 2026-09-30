"""Associação e gestão de tracks para o PA2.

Implementa:
1. IoU entre caixas e matching por IoU (guloso e Hungarian)
2. Gestão de tracks: nascimento, morte, atualização de estado

Proibido usar rastreadores prontos ou torchvision.ops.nms.
Esta é uma implementação própria.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Any

from pa2.metrics.tracking import compute_iou


class BoxAssociation:
    """Classe base para associação de caixas (IoU entre detecções e tracks)."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold

    def compute_iou_matrix(
        self,
        detections: list[np.ndarray],
        tracks: list[np.ndarray],
    ) -> np.ndarray:
        """Calcula matriz de IoU entre detections (N x 4) e tracks (M x 4)."""
        n = len(detections)
        m = len(tracks)
        iou = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                iou[i, j] = compute_iou(detections[i], tracks[j])
        return iou


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


class HungarianMatcher:
    """Matching ótimo (Hungarian) entre detections e tracks ativos."""

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
        """Faz matching Hungarian entre detections e tracks ativos."""
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

        n, m = len(detections), len(active_ids)
        cost_mat = np.ones((n, m)) * 1e9

        for i in range(n):
            for j in range(m):
                iou_val = compute_iou(det_boxes[i], track_boxes[j])
                if iou_val >= self.iou_threshold:
                    cost_mat[i, j] = -iou_val

        from scipy.optimize import linear_sum_assignment

        try:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
        except ValueError:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        matches = []
        matched_det_indices = set()
        matched_track_ids = set()

        for i, j in zip(row_ind, col_ind):
            if cost_mat[i, j] < 0:
                track_id = active_ids[j]
                iou_val = -cost_mat[i, j]
                matches.append((track_id, i, iou_val, False))
                matched_det_indices.add(i)
                matched_track_ids.add(track_id)

        # Atualiza tracks casados
        for track_id, det_idx, _, _ in matches:
            det = detections[det_idx]
            box = np.array([det["bb_left"], det["bb_top"],
                           det["bb_width"], det["bb_height"]])
            self.active_tracks[track_id]["ages_since_update"] = 0
            self.active_tracks[track_id]["total_hits"] += 1
            self.active_tracks[track_id]["frames_seen"].append(frame)
            self.active_tracks[track_id]["boxes"].append(box)

        # Novos tracks para detecções não casadas
        new_track_matches = []
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
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

        # Atualiza idade de tracks não casados
        for tid in active_ids:
            if tid not in matched_track_ids:
                self.active_tracks[tid]["ages_since_update"] += 1

        # Mata tracks antigos
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
        """Roda o tracking completo (Hungarian) sobre uma sequência."""
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
        self.active_tracks = {}
        self.next_track_id = 1
        self.dead_tracks = set()


class TrackManager:
    """Gerenciador de tracks com estado interno.

    Combina as funcionalidades de matching e gestão de tracks em uma única classe.
    """

    def __init__(
        self,
        matcher: GreedyMatcher | HungarianMatcher,
    ):
        self.matcher = matcher

    def track(self, detections: list[dict[str, Any]], num_frames: int) -> list[dict[str, Any]]:
        """Executa o rastreamento completo."""
        return self.matcher.run(detections, num_frames)


def track_matches_iou(track_id: int, detection: dict[str, Any], track_state: dict[str, Any], iou_threshold: float = 0.3) -> float:
    """Calcula IoU entre um track (última caixa) e uma detecção."""
    track_box = track_state["boxes"][-1]
    det_box = np.array([
        detection["bb_left"],
        detection["bb_top"],
        detection["bb_width"],
        detection["bb_height"],
    ])
    return compute_iou(track_box, det_box)


def simple_greedy_match(
    detections: list[dict[str, Any]],
    prev_tracks: dict[int, dict[str, Any]],
    iou_threshold: float = 0.3,
    max_age: int = 30,
    next_track_id: int = 1,
    frame: int = 1,
    dead_tracks: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], set[int], int]:
    """Matching guloso frame-a-frame, stateless (sem objeto)."""
    if dead_tracks is None:
        dead_tracks = set()

    active_ids = list(prev_tracks.keys())
    track_boxes = [prev_tracks[tid]["boxes"][-1] for tid in active_ids]
    det_boxes = [np.array([d["bb_left"], d["bb_top"],
                           d["bb_width"], d["bb_height"]]) for d in detections]

    iou_mat = np.zeros((len(detections), len(active_ids)))
    for i in range(len(detections)):
        for j in range(len(active_ids)):
            iou_mat[i, j] = compute_iou(det_boxes[i], track_boxes[j])

    pairs = []
    for i in range(len(detections)):
        for j in range(len(active_ids)):
            if iou_mat[i, j] >= iou_threshold:
                pairs.append((iou_mat[i, j], i, j))

    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_det = set()
    matched_trk = set()
    matches: list[tuple[int, int]] = []

    for iou_val, i, j in pairs:
        if i in matched_det or j in matched_trk:
            continue
        track_id = active_ids[j]
        matches.append((track_id, i))
        matched_det.add(i)
        matched_trk.add(j)

    updated_tracks = dict(prev_tracks)
    for track_id, det_idx in matches:
        det = detections[det_idx]
        box = np.array([det["bb_left"], det["bb_top"],
                       det["bb_width"], det["bb_height"]])
        updated_tracks[track_id]["ages_since_update"] = 0
        updated_tracks[track_id]["total_hits"] += 1
        updated_tracks[track_id]["frames_seen"].append(frame)
        updated_tracks[track_id]["boxes"].append(box)

    for i, det in enumerate(detections):
        if i not in matched_det:
            track_id = next_track_id
            next_track_id += 1
            updated_tracks[track_id] = {
                "ages_since_update": 0,
                "total_hits": 1,
                "frames_seen": [frame],
                "boxes": [np.array([det["bb_left"], det["bb_top"],
                                    det["bb_width"], det["bb_height"]])],
            }

    for tid in active_ids:
        if tid not in matched_trk:
            updated_tracks[tid]["ages_since_update"] += 1

    new_dead = set()
    tracks_output: list[dict[str, Any]] = []
    for tid, state in list(updated_tracks.items()):
        if state["ages_since_update"] > max_age:
            del updated_tracks[tid]
            new_dead.add(tid)

    dead_tracks.update(new_dead)

    return tracks_output, updated_tracks, dead_tracks, next_track_id

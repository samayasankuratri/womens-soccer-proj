import argparse
from collections import Counter, defaultdict, deque
from enum import Enum
from typing import Iterator, List

import os
import cv2
import tkinter as tk
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration



PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/best.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

CONFIG = SoccerPitchConfiguration()


def compute_stride(source_video_path: str, min_samples: int = 8) -> int:
    """Compute a stride that guarantees at least `min_samples` frames are sampled."""
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    total_frames = video_info.total_frames
    stride = max(1, total_frames // min_samples)
    return stride


class TeamVoter:
    def __init__(self, window: int = 50, lock_threshold: int = 40, lock_min_fraction: float = 0.82):
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self.locked: dict[int, int] = {}
        self.lock_threshold = lock_threshold
        self.lock_min_fraction = lock_min_fraction

    def update(self, tracker_ids: np.ndarray, team_ids: np.ndarray) -> np.ndarray:
        smoothed = np.empty_like(team_ids)
        for i, (tid, raw_team) in enumerate(zip(tracker_ids, team_ids)):
            if tid in self.locked:
                smoothed[i] = self.locked[tid]
                continue
            self.history[tid].append(int(raw_team))
            counter = Counter(self.history[tid])
            best, best_count = counter.most_common(1)[0]
            smoothed[i] = best
            if (len(self.history[tid]) >= self.lock_threshold
                    and best_count / len(self.history[tid]) >= self.lock_min_fraction):
                self.locked[tid] = best
        return smoothed


class OutlierVoter:
    def __init__(self, window: int = 20, outlier_fraction: float = 0.7):
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self.outlier_fraction = outlier_fraction

    def update(self, tracker_ids: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
        result = np.zeros(len(tracker_ids), dtype=bool)
        for i, (tid, is_outlier) in enumerate(zip(tracker_ids, outlier_mask)):
            self.history[tid].append(bool(is_outlier))
            fraction = sum(self.history[tid]) / len(self.history[tid])
            result[i] = fraction >= self.outlier_fraction
        return result


class ClassVoter:
    def __init__(self, window: int = 30, lock_threshold: int = 20, lock_min_fraction: float = 0.80):
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self.locked: dict[int, int] = {}
        self.lock_threshold = lock_threshold
        self.lock_min_fraction = lock_min_fraction

    def update(self, tracker_ids: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        smoothed = class_ids.copy()
        for i, (tid, raw_class) in enumerate(zip(tracker_ids, class_ids)):
            if tid in self.locked:
                smoothed[i] = self.locked[tid]
                continue
            self.history[tid].append(int(raw_class))
            counter = Counter(self.history[tid])
            best, best_count = counter.most_common(1)[0]
            smoothed[i] = best
            if (len(self.history[tid]) >= self.lock_threshold
                    and best_count / len(self.history[tid]) >= self.lock_min_fraction):
                self.locked[tid] = best
        return smoothed


STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    AERIAL_DUEL = 'AERIAL_DUEL'
    REFEREE_DIAGNOSTIC = 'REFEREE_DIAGNOSTIC'
    PLAYER_DIAGNOSTIC = 'PLAYER_DIAGNOSTIC'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def get_jersey_crops(crops: List[np.ndarray]) -> List[np.ndarray]:
    jersey_crops = []
    for crop in crops:
        h, w = crop.shape[:2]
        y_start = int(h * 0.15)
        y_end = int(h * 0.55)
        x_start = int(w * 0.15)
        x_end = int(w * 0.85)
        sliced = crop[y_start:y_end, x_start:x_end]
        jersey_crops.append(sliced if sliced.size > 0 else crop)
    return jersey_crops


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_players = players_xy[players_team_id == 0]
    team_1_players = players_xy[players_team_id == 1]

    if len(team_0_players) == 0 or len(team_1_players) == 0:
        return np.zeros(len(goalkeepers_xy), dtype=int)

    team_0_centroid = team_0_players.mean(axis=0)
    team_1_centroid = team_1_players.mean(axis=0)

    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    all_keypoints = keypoints.xy[0]
    all_confidence = keypoints.confidence[0] if keypoints.confidence is not None else np.ones(len(all_keypoints))
    mask = (all_keypoints[:, 0] > 1) & (all_keypoints[:, 1] > 1) & (all_confidence > 0.5)

    if np.sum(mask) < 4:
        return draw_pitch(config=CONFIG)

    source_keypoints = all_keypoints[mask].astype(np.float32)
    target_vertices = np.array(CONFIG.vertices)[mask].astype(np.float32)

    try:
        transformer = ViewTransformer(source=source_keypoints, target=target_vertices)
        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(points=xy)
        transformed_xy[:, 0] = np.clip(transformed_xy[:, 0], 0, CONFIG.length)
        transformed_xy[:, 1] = np.clip(transformed_xy[:, 1], 0, CONFIG.width)
    except (ValueError, cv2.error):
        return draw_pitch(config=CONFIG)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


class AerialDuelDetector:
    """
    Detects an aerial duel when ALL of:
    - the ball is currently airborne
    - at least `min_players` players are within `pitch_proximity_meters` of the ball
      (measured in real-world pitch coordinates via homography)
    - at least one player from each team is involved (no same-team false positives)

    When a ViewTransformer is available (pitch keypoints detected), distances are
    computed in pitch-space meters — physically meaningful regardless of camera zoom
    or angle. Falls back to body-height-relative pixel distances when homography
    is unavailable (too few keypoints visible).

    A latch keeps the annotation alive for `latch_frames` frames after the
    conditions drop out, to handle brief ball detection gaps.
    """

    def __init__(
        self,
        pitch_proximity_meters: float = 3.0,    # real-world distance threshold in pitch units (~meters)
        proximity_body_fraction: float = 0.6,   # fallback: fraction of player height when no homography
        min_players: int = 2,
        latch_frames: int = 4,
    ):
        self.pitch_proximity_meters = pitch_proximity_meters
        self.proximity_body_fraction = proximity_body_fraction
        self.min_players = min_players
        self.latch_frames = latch_frames
        self._latch_count = 0

    @staticmethod
    def get_player_duel_points(detections: sv.Detections) -> np.ndarray:
        """Upper-body point (head/chest) in pixel space — better match for aerial headers."""
        xyxy = detections.xyxy
        x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        y = xyxy[:, 1] + 0.25 * (xyxy[:, 3] - xyxy[:, 1])
        return np.stack([x, y], axis=1)

    def detect(
        self,
        ball_tracker: BallTracker,
        players: sv.Detections,
        players_team_id: np.ndarray | None = None,
        transformer: ViewTransformer | None = None,
    ) -> tuple[bool, sv.Detections, np.ndarray | None]:
        ball_center = ball_tracker.get_ball_center()

        if ball_center is None:
            self._latch_count = max(0, self._latch_count - 1)
            return self._latch_count > 0, players[[]], None

        if len(players) == 0:
            self._latch_count = 0
            return False, players[[]], ball_center

        # Only trigger if ball is actually airborne
        if not ball_tracker.is_airborne():
            self._latch_count = max(0, self._latch_count - 1)
            return self._latch_count > 0, players[[]], ball_center

        # --- Compute proximity mask ---
        # Prefer pitch-space (meters) when homography is available; fall back to pixels
        player_points_px = self.get_player_duel_points(players)

        if transformer is not None:
            try:
                # Project player upper-body points and ball center into pitch coordinates
                player_points_pitch = transformer.transform_points(player_points_px)
                ball_center_pitch = transformer.transform_points(ball_center[np.newaxis])[0]
                distances = np.linalg.norm(player_points_pitch - ball_center_pitch, axis=1)
                mask = distances <= self.pitch_proximity_meters
            except (ValueError, cv2.error):
                # Homography degenerate this frame — fall back to pixel distances
                transformer = None

        if transformer is None:
            # Fallback: scale threshold by each player's bounding box height
            player_heights = players.xyxy[:, 3] - players.xyxy[:, 1]
            thresholds = player_heights * self.proximity_body_fraction
            distances = np.linalg.norm(player_points_px - ball_center, axis=1)
            mask = distances <= thresholds

        involved = players[mask]

        # Require players from both teams — eliminates same-team false positives
        has_both_teams = False
        if players_team_id is not None and len(players_team_id) == len(players):
            involved_team_ids = players_team_id[mask]
            has_both_teams = len(np.unique(involved_team_ids)) >= 2
        else:
            has_both_teams = len(involved) >= self.min_players

        currently_dueling = len(involved) >= self.min_players and has_both_teams
        if currently_dueling:
            self._latch_count = self.latch_frames
        else:
            self._latch_count = max(0, self._latch_count - 1)

        return self._latch_count > 0, involved, ball_center


def annotate_aerial_duel(
    frame: np.ndarray,
    players_involved: sv.Detections,
    ball_center: np.ndarray | None,
) -> np.ndarray:
    if ball_center is None or len(players_involved) == 0:
        return frame

    bx, by = map(int, ball_center)
    cv2.circle(frame, (bx, by), 8, (0, 255, 255), 2)

    for i, xyxy in enumerate(players_involved.xyxy.astype(int)):
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        track_id = None
        if players_involved.tracker_id is not None and len(players_involved.tracker_id) > i:
            track_id = players_involved.tracker_id[i]

        label = f"AERIAL DUEL #{track_id}" if track_id is not None else "AERIAL DUEL"
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        px = int((x1 + x2) / 2)
        py = int(y1 + 0.25 * (y2 - y1))
        cv2.line(frame, (px, py), (bx, by), (0, 255, 255), 2)

    cv2.putText(frame, "AERIAL DUEL", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    return frame


def run_referee_diagnostic(source_video_path: str, device: str) -> None:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    stride = compute_stride(source_video_path, min_samples=30)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=stride)

    scores = []
    for frame in tqdm(frame_generator, desc='scanning for referees'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        referee_mask = detections.class_id == REFEREE_CLASS_ID
        scores.extend(detections.confidence[referee_mask].tolist())

    if not scores:
        print("No referee detections found in sampled frames.")
        return

    scores = np.array(scores)
    print(f"\n--- Referee detection confidence scores ({len(scores)} detections) ---")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  p{p:02d}:  {np.percentile(scores, p):.3f}")
    print(f"  mean: {scores.mean():.3f}")

    bin_edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(scores, bins=bin_edges)
    max_count = max(counts) if max(counts) > 0 else 1
    print("\n  Histogram (0.0 → 1.0):")
    for i, count in enumerate(counts):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bar = '#' * int(count / max_count * 30)
        marker = " <-- current threshold" if lo < 0.7 <= hi else ""
        print(f"  [{lo:.1f}-{hi:.1f}] {bar:<30} {count}{marker}")
    print()


def run_player_diagnostic(source_video_path: str, device: str) -> None:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    stride = compute_stride(source_video_path, min_samples=30)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=stride)

    scores = []
    for frame in tqdm(frame_generator, desc='scanning for players'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        player_mask = detections.class_id == PLAYER_CLASS_ID
        scores.extend(detections.confidence[player_mask].tolist())

    if not scores:
        print("No player detections found in sampled frames.")
        return

    scores = np.array(scores)
    print(f"\n--- Player detection confidence scores ({len(scores)} detections) ---")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  p{p:02d}:  {np.percentile(scores, p):.3f}")
    print(f"  mean: {scores.mean():.3f}")

    bin_edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(scores, bins=bin_edges)
    max_count = max(counts) if max(counts) > 0 else 1
    print("\n  Histogram (0.0 → 1.0):")
    for i, count in enumerate(counts):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bar = '#' * int(count / max_count * 30)
        marker = " <-- current threshold" if lo < 0.8 <= hi else ""
        print(f"  [{lo:.1f}-{hi:.1f}] {bar:<30} {count}{marker}")
    print()


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(callback=callback, slice_wh=(640, 640))

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)
        yield annotated_frame


def _build_transformer(keypoints: sv.KeyPoints) -> ViewTransformer | None:
    """
    Build a ViewTransformer from pitch keypoints for homography-based projection.
    Returns None if too few confident keypoints are visible this frame.
    """
    all_keypoints = keypoints.xy[0]
    all_confidence = (
        keypoints.confidence[0]
        if keypoints.confidence is not None
        else np.ones(len(all_keypoints))
    )
    mask = (all_keypoints[:, 0] > 1) & (all_keypoints[:, 1] > 1) & (all_confidence > 0.5)
    if np.sum(mask) < 4:
        return None
    try:
        return ViewTransformer(
            source=all_keypoints[mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32),
        )
    except (ValueError, cv2.error):
        return None


def run_aerial_duel(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Aerial duel detection with:
    - Team colors (full team classification pipeline)
    - Ball detection and tracking with tighter airborne thresholds
    - Pitch homography (ViewTransformer) for real-world meter-based proximity:
        distances between players and ball are computed in pitch coordinates (~meters)
        so the threshold is physically meaningful regardless of camera zoom/angle
    - Falls back to body-height-relative pixel distances when homography unavailable
    - Cross-team check: both teams must be involved for a duel to trigger
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    # --- Phase 1: collect crops for team classifier ---
    stride = compute_stride(source_video_path, min_samples=8)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=stride)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        high_conf_players = detections[
            (detections.class_id == PLAYER_CLASS_ID) & (detections.confidence >= 0.8)
        ]
        crops += get_crops(frame, high_conf_players)

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(get_jersey_crops(crops))

    # --- Phase 2: main processing loop ---
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    voter = TeamVoter()
    class_voter = ClassVoter()
    outlier_voter = OutlierVoter()

    # Tighter airborne thresholds to reduce false positives on rolling/bouncing balls
    ball_tracker = BallTracker(
        buffer_size=20,
        min_airborne_frames=5,           # was 3 — needs more consecutive frames to confirm airborne
        vertical_threshold=4.0,          # was 2.0 — needs stronger upward motion
        vertical_ratio_threshold=0.70,   # was 0.55 — majority of buffer must be rising
        consistency_ratio=0.65,          # was 0.5 — stricter consistency requirement
    )
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    aerial_duel_detector = AerialDuelDetector(
        pitch_proximity_meters=3.0,   # ~3 metres in pitch space — tight, physically meaningful
        proximity_body_fraction=0.6,  # fallback if homography unavailable
        min_players=2,
        latch_frames=4,
    )

    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(callback=ball_callback, slice_wh=(640, 640))

    for frame in frame_generator:
        # --- pitch keypoints → homography transformer ---
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        transformer = _build_transformer(keypoints)  # None if too few keypoints visible

        # --- detect + track all players ---
        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = tracker.update_with_detections(detections)
        detections.class_id = class_voter.update(detections.tracker_id, detections.class_id)

        # --- team classification ---
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        jersey_crops = get_jersey_crops(crops)
        players_team_id = team_classifier.predict(jersey_crops)
        raw_outlier_mask = team_classifier.get_outlier_mask(jersey_crops)
        smoothed_outlier_mask = outlier_voter.update(players.tracker_id, raw_outlier_mask)
        players_team_id = voter.update(players.tracker_id, players_team_id)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        all_detections = sv.Detections.merge([players, goalkeepers, referees])
        player_colors = [
            int(REFEREE_CLASS_ID) if smoothed_outlier_mask[i] else int(players_team_id[i])
            for i in range(len(players_team_id))
        ]
        color_lookup = np.array(
            player_colors +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )

        # --- ball ---
        ball_detections = slicer(frame).with_nms(threshold=0.1)
        ball_detections = ball_tracker.update(ball_detections)

        # --- aerial duel detection ---
        # Primary: pitch-space meter distances via homography (transformer)
        # Fallback: body-height-relative pixel distances if homography unavailable
        # Also requires: ball airborne + players from both teams within range
        is_duel, involved_players, ball_center = aerial_duel_detector.detect(
            ball_tracker=ball_tracker,
            players=players,
            players_team_id=players_team_id,
            transformer=transformer,
        )

        # --- annotate ---
        annotated_frame = frame.copy()

        # Draw all players with team colors
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, all_detections, custom_color_lookup=color_lookup)
        labels = [str(tid) for tid in all_detections.tracker_id]
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, all_detections, labels, custom_color_lookup=color_lookup)

        # Draw ball
        if len(ball_detections) > 0:
            annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)

        # Airborne status indicator
        airborne_text = "BALL IN AIR" if ball_tracker.is_airborne() else "BALL ON GROUND"
        cv2.putText(
            annotated_frame, airborne_text, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 255) if ball_tracker.is_airborne() else (255, 255, 255),
            2, cv2.LINE_AA,
        )

        # Duel annotation
        if is_duel:
            annotated_frame = annotate_aerial_duel(annotated_frame, involved_players, ball_center)

        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    stride = compute_stride(source_video_path, min_samples=8)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=stride)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        high_conf_players = detections[
            (detections.class_id == PLAYER_CLASS_ID) & (detections.confidence >= 0.8)
        ]
        crops += get_crops(frame, high_conf_players)

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(get_jersey_crops(crops))

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    voter = TeamVoter()
    class_voter = ClassVoter()
    outlier_voter = OutlierVoter()
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        detections.class_id = class_voter.update(detections.tracker_id, detections.class_id)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        jersey_crops = get_jersey_crops(crops)
        players_team_id = team_classifier.predict(jersey_crops)
        raw_outlier_mask = team_classifier.get_outlier_mask(jersey_crops)
        smoothed_outlier_mask = outlier_voter.update(players.tracker_id, raw_outlier_mask)
        players_team_id = voter.update(players.tracker_id, players_team_id)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        player_colors = [
            int(REFEREE_CLASS_ID) if smoothed_outlier_mask[i] else int(players_team_id[i])
            for i in range(len(players_team_id))
        ]
        color_lookup = np.array(
            player_colors + goalkeepers_team_id.tolist() + [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    stride = compute_stride(source_video_path, min_samples=8)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=stride)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        high_conf_players = detections[
            (detections.class_id == PLAYER_CLASS_ID) & (detections.confidence >= 0.8)
        ]
        crops += get_crops(frame, high_conf_players)

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(get_jersey_crops(crops))

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    voter = TeamVoter()
    class_voter = ClassVoter()
    outlier_voter = OutlierVoter()
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        detections.class_id = class_voter.update(detections.tracker_id, detections.class_id)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        jersey_crops = get_jersey_crops(crops)
        players_team_id = team_classifier.predict(jersey_crops)
        raw_outlier_mask = team_classifier.get_outlier_mask(jersey_crops)
        smoothed_outlier_mask = outlier_voter.update(players.tracker_id, raw_outlier_mask)
        players_team_id = voter.update(players.tracker_id, players_team_id)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        player_colors = [
            int(REFEREE_CLASS_ID) if smoothed_outlier_mask[i] else int(players_team_id[i])
            for i in range(len(players_team_id))
        ]
        color_lookup = np.array(
            player_colors + goalkeepers_team_id.tolist() + [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(x=w // 2 - radar_w // 2, y=h - radar_h, width=radar_w, height=radar_h)
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    if mode == Mode.REFEREE_DIAGNOSTIC:
        run_referee_diagnostic(source_video_path=source_video_path, device=device)
        return
    elif mode == Mode.PLAYER_DIAGNOSTIC:
        run_player_diagnostic(source_video_path=source_video_path, device=device)
        return

    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(source_video_path=source_video_path, device=device)
    elif mode == Mode.AERIAL_DUEL:
        frame_generator = run_aerial_duel(source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    max_display_height = int(screen_height * 0.9)
    max_display_width = int(screen_width * 0.9)

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            h, w = frame.shape[:2]
            scale = min(max_display_width / w, max_display_height / h)
            if scale < 1:
                display_frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_AREA)
            else:
                display_frame = frame

            cv2.imshow("frame", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    )
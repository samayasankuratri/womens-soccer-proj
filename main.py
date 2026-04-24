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

PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/yolov8n.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/yolov8s.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/yolov8n.pt')

BALL_CLASS_ID = 32
GOALKEEPER_CLASS_ID = 0
PLAYER_CLASS_ID = 0
REFEREE_CLASS_ID = 0

CONFIG = SoccerPitchConfiguration()
STRIDE = 60

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


def compute_stride(source_video_path: str, min_samples: int = 8) -> int:
    """Compute a stride that guarantees at least `min_samples` frames are sampled."""
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    total_frames = video_info.total_frames
    stride = max(1, total_frames // min_samples)
    return stride


class TeamVoter:
    """Majority-vote smoother keyed by ByteTrack tracker_id."""

    def __init__(
        self,
        window: int = 50,
        lock_threshold: int = 40,
        lock_min_fraction: float = 0.82,
    ):
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
            if (
                len(self.history[tid]) >= self.lock_threshold
                and best_count / len(self.history[tid]) >= self.lock_min_fraction
            ):
                self.locked[tid] = best
        return smoothed


class OutlierVoter:
    """Per-tracker smoother for outlier (referee-like) detection."""

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
    """Per-tracker majority-vote smoother for player vs. referee class assignment."""

    def __init__(
        self,
        window: int = 30,
        lock_threshold: int = 20,
        lock_min_fraction: float = 0.80,
    ):
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
            if (
                len(self.history[tid]) >= self.lock_threshold
                and best_count / len(self.history[tid]) >= self.lock_min_fraction
            ):
                self.locked[tid] = best
        return smoothed


class Mode(Enum):
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
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
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    if len(players_xy[players_team_id == 0]) == 0 or len(players_xy[players_team_id == 1]) == 0:
        return np.zeros(len(goalkeepers), dtype=int)

    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)

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
    if keypoints is None:
        return draw_pitch(config=CONFIG)

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
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar
    )
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar
    )
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar
    )
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar
    )
    return radar


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
    percentiles = [10, 25, 50, 75, 90, 95]
    print(f"\n--- Referee detection confidence scores ({len(scores)} detections) ---")
    print(f"  min:  {scores.min():.3f}")
    for p in percentiles:
        print(f"  p{p:02d}:  {np.percentile(scores, p):.3f}")
    print(f"  max:  {scores.max():.3f}")
    print(f"  mean: {scores.mean():.3f}")

    print("\n  Histogram (0.0 → 1.0):")
    bin_edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(scores, bins=bin_edges)
    max_count = max(counts) if max(counts) > 0 else 1
    bar_width = 30
    for i, count in enumerate(counts):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bar = '#' * int(count / max_count * bar_width)
        marker = " <-- current threshold" if lo < 0.7 <= hi else ""
        print(f"  [{lo:.1f}-{hi:.1f}] {bar:<{bar_width}} {count}{marker}")
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
    percentiles = [10, 25, 50, 75, 90, 95]
    print(f"\n--- Player detection confidence scores ({len(scores)} detections) ---")
    print(f"  min:  {scores.min():.3f}")
    for p in percentiles:
        print(f"  p{p:02d}:  {np.percentile(scores, p):.3f}")
    print(f"  max:  {scores.max():.3f}")
    print(f"  mean: {scores.mean():.3f}")

    print("\n  Histogram (0.0 → 1.0):")
    bin_edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(scores, bins=bin_edges)
    max_count = max(counts) if max(counts) > 0 else 1
    bar_width = 30
    for i, count in enumerate(counts):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bar = '#' * int(count / max_count * bar_width)
        marker = " <-- current threshold" if lo < 0.8 <= hi else ""
        print(f"  [{lo:.1f}-{hi:.1f}] {bar:<{bar_width}} {count}{marker}")
    print()


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
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

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

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
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
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
            (detections.class_id == PLAYER_CLASS_ID) & (detections.confidence >= 0.3)
        ]
        crops += get_crops(frame, high_conf_players)

    if not crops:
        raise ValueError("No player crops collected for team classification.")

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

        if detections.tracker_id is None or len(detections) == 0:
            yield frame
            continue

        detections.class_id = class_voter.update(detections.tracker_id, detections.class_id)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        if len(players) > 0:
            player_crops = get_crops(frame, players)
            jersey_crops = get_jersey_crops(player_crops)
            players_team_id = team_classifier.predict(jersey_crops)
            raw_outlier_mask = team_classifier.get_outlier_mask(jersey_crops)
            smoothed_outlier_mask = outlier_voter.update(players.tracker_id, raw_outlier_mask)
            players_team_id = voter.update(players.tracker_id, players_team_id)
        else:
            players_team_id = np.array([], dtype=int)
            smoothed_outlier_mask = np.array([], dtype=bool)

        if len(goalkeepers) > 0 and len(players) > 0:
            goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        else:
            goalkeepers_team_id = np.array([], dtype=int)

        detections_merged = sv.Detections.merge([players, goalkeepers, referees])

        player_colors = [
            int(REFEREE_CLASS_ID) if smoothed_outlier_mask[i] else int(players_team_id[i])
            for i in range(len(players_team_id))
        ]

        color_lookup = np.array(
            player_colors +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )

        labels = [str(tracker_id) for tracker_id in detections_merged.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections_merged, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections_merged, labels=labels, custom_color_lookup=color_lookup)
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

    if not crops:
        raise ValueError("No player crops collected for radar/team classification.")

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(get_jersey_crops(crops))

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    voter = TeamVoter()
    class_voter = ClassVoter()
    outlier_voter = OutlierVoter()

    for frame in frame_generator:
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        if pitch_result.keypoints is not None:
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        else:
            keypoints = None

        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is None or len(detections) == 0:
            yield frame
            continue

        detections.class_id = class_voter.update(detections.tracker_id, detections.class_id)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        if len(players) > 0:
            player_crops = get_crops(frame, players)
            jersey_crops = get_jersey_crops(player_crops)
            players_team_id = team_classifier.predict(jersey_crops)
            raw_outlier_mask = team_classifier.get_outlier_mask(jersey_crops)
            smoothed_outlier_mask = outlier_voter.update(players.tracker_id, raw_outlier_mask)
            players_team_id = voter.update(players.tracker_id, players_team_id)
        else:
            players_team_id = np.array([], dtype=int)
            smoothed_outlier_mask = np.array([], dtype=bool)

        if len(goalkeepers) > 0 and len(players) > 0:
            goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        else:
            goalkeepers_team_id = np.array([], dtype=int)

        detections_merged = sv.Detections.merge([players, goalkeepers, referees])

        player_colors = [
            int(REFEREE_CLASS_ID) if smoothed_outlier_mask[i] else int(players_team_id[i])
            for i in range(len(players_team_id))
        ]

        color_lookup = np.array(
            player_colors +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )

        labels = [str(tracker_id) for tracker_id in detections_merged.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections_merged, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections_merged, labels=labels, custom_color_lookup=color_lookup
        )

        h, w, _ = frame.shape
        radar = render_radar(detections_merged, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape

        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )

        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    if mode == Mode.REFEREE_DIAGNOSTIC:
        run_referee_diagnostic(source_video_path=source_video_path, device=device)
        return
    elif mode == Mode.PLAYER_DIAGNOSTIC:
        run_player_diagnostic(source_video_path=source_video_path, device=device)
        return
    elif mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device)
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
                display_frame = cv2.resize(
                    frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
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
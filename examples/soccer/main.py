import argparse
from collections import Counter, defaultdict, deque
from enum import Enum
from typing import Iterator, List

import os
import cv2
#import tkinter as tk
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
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

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
    """Majority-vote smoother keyed by ByteTrack tracker_id.

    Once a tracker accumulates `lock_threshold` votes with at least
    `lock_min_fraction` of them agreeing, the assignment is locked
    permanently to eliminate flickering.
    """

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
            # Lock once we have enough confident votes
            if (
                len(self.history[tid]) >= self.lock_threshold
                and best_count / len(self.history[tid]) >= self.lock_min_fraction
            ):
                self.locked[tid] = best
        return smoothed

class OutlierVoter:
    """Per-tracker smoother for outlier (referee-like) detection.

    Only flags a tracker as an outlier if at least `outlier_fraction` of its
    recent frames were flagged as outliers, preventing a single bad crop from
    misclassifying a real player as the center ref.
    """

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
    """Per-tracker majority-vote smoother for player vs. referee class assignment.

    Prevents individual players from flickering between PLAYER_CLASS_ID and
    REFEREE_CLASS_ID when detection confidence oscillates near the threshold.
    """

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
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    REFEREE_DIAGNOSTIC = 'REFEREE_DIAGNOSTIC'
    PLAYER_DIAGNOSTIC = 'PLAYER_DIAGNOSTIC'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def get_jersey_crops(crops: List[np.ndarray]) -> List[np.ndarray]:
    """
    Slice each crop to the jersey torso region: rows 15-55%, columns 15-85%.
    Skips the head/face at the top and legs/shorts at the bottom.
    Falls back to the full crop if the slice is too small.
    """
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


def collect_crops(
    frame_generator,
    player_detection_model,
    max_per_frame_ratio: float = 2.0,
) -> List[np.ndarray]:
    """
    Collect high-confidence player crops from a strided frame generator.

    Caps each frame's contribution to 2x the per-frame average (2C) so that
    a single crowded frame (e.g., a corner kick) can't dominate the training set.
    """
    all_frame_crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        high_conf_players = detections[
            (detections.class_id == PLAYER_CLASS_ID) & (detections.confidence >= 0.8)
        ]
        all_frame_crops.append(get_crops(frame, high_conf_players))

    if not all_frame_crops:
        return []

    total = sum(len(fc) for fc in all_frame_crops)
    cap = max(1, int(total / len(all_frame_crops) * max_per_frame_ratio))

    crops = []
    for frame_crops in all_frame_crops:
        if len(frame_crops) > cap:
            idx = np.random.choice(len(frame_crops), cap, replace=False)
            crops.extend([frame_crops[i] for i in idx])
        else:
            crops.extend(frame_crops)
    return crops


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
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
    color_lookup: np.ndarray,
    ball_detections: sv.Detections | None = None
) -> np.ndarray:
    """
    Create a radar view by transforming player positions onto a 2D pitch representation.
    Optionally projects the ball onto the radar as well.
    Only uses keypoints that are actually detected (confidence > 0.5 and within frame).
    """
    # Filter keypoints based on confidence and validity
    all_keypoints = keypoints.xy[0]
    all_confidence = (
        keypoints.confidence[0]
        if keypoints.confidence is not None
        else np.ones(len(all_keypoints))
    )

    # Valid keypoints: detected, confident, and not near origin
    mask = (
        (all_keypoints[:, 0] > 1) &
        (all_keypoints[:, 1] > 1) &
        (all_confidence > 0.5)
    )

    # Need at least 4 points for homography
    if np.sum(mask) < 4:
        return draw_pitch(config=CONFIG)

    source_keypoints = all_keypoints[mask].astype(np.float32)
    target_vertices = np.array(CONFIG.vertices)[mask].astype(np.float32)

    try:
        transformer = ViewTransformer(
            source=source_keypoints,
            target=target_vertices
        )

        # Transform players / goalkeepers / referees
        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(points=xy)

        # Clip to pitch bounds
        transformed_xy[:, 0] = np.clip(transformed_xy[:, 0], 0, CONFIG.length)
        transformed_xy[:, 1] = np.clip(transformed_xy[:, 1], 0, CONFIG.width)

        # Transform ball if present
        transformed_ball_xy = None
        if ball_detections is not None and len(ball_detections) > 0:
            ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            transformed_ball_xy = transformer.transform_points(points=ball_xy)
            transformed_ball_xy[:, 0] = np.clip(transformed_ball_xy[:, 0], 0, CONFIG.length)
            transformed_ball_xy[:, 1] = np.clip(transformed_ball_xy[:, 1], 0, CONFIG.width)

    except (ValueError, cv2.error):
        return draw_pitch(config=CONFIG)

    radar = draw_pitch(config=CONFIG)

    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]),
        radius=20,
        pitch=radar
    )
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]),
        radius=20,
        pitch=radar
    )
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]),
        radius=20,
        pitch=radar
    )
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]),
        radius=20,
        pitch=radar
    )

    # Draw ball on radar
    if transformed_ball_xy is not None and len(transformed_ball_xy) > 0:
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=transformed_ball_xy,
            face_color=sv.Color.from_hex("#FFFFFF"),
            radius=10,
            pitch=radar
        )

    return radar


def run_referee_diagnostic(source_video_path: str, device: str) -> None:
    """
    Sample frames and print confidence score statistics for all REFEREE_CLASS_ID
    detections. Use this to find a good confidence threshold for reclassification.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    stride = compute_stride(source_video_path, min_samples=30)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)

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

    # ASCII histogram with 10 bins from 0.0 to 1.0
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
    """
    Sample frames and print confidence score statistics for all PLAYER_CLASS_ID
    detections. Use this to find a good confidence threshold for crop collection.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    stride = compute_stride(source_video_path, min_samples=30)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)

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
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
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
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
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
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=960, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        if detections.confidence is not None:
            detections = detections[detections.confidence > 0.2]
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
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
    """
    Run team classification on a video and yield annotated frames with team colors.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)

    stride = compute_stride(source_video_path, min_samples=50)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride
    )

    crops = collect_crops(frame_generator, player_detection_model)

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
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers
        )

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])

        player_colors = [
            int(REFEREE_CLASS_ID) if smoothed_outlier_mask[i] else int(players_team_id[i])
            for i in range(len(players_team_id))
        ]

        color_lookup = np.array(
            player_colors +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=960, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    ball_slicer = sv.InferenceSlicer(
        callback=ball_callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    stride = compute_stride(source_video_path, min_samples=50)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride
    )

    crops = collect_crops(frame_generator, player_detection_model)

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

        ball_detections = ball_slicer(frame).with_nms(threshold=0.1)
        if ball_detections.confidence is not None:
            ball_detections = ball_detections[ball_detections.confidence > 0.1]
        ball_detections = ball_tracker.update(ball_detections)

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
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers
        )

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])

        player_colors = [
            int(REFEREE_CLASS_ID) if smoothed_outlier_mask[i] else int(players_team_id[i])
            for i in range(len(players_team_id))
        ]

        color_lookup = np.array(
            player_colors +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup, ball_detections)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape

        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )

        annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
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
    
    # Get screen resolution and calculate appropriate scale
    # root = tk.Tk()
    # screen_width = root.winfo_screenwidth()
    # screen_height = root.winfo_screenheight()
    # root.destroy()
    
    # # Use 90% of screen height to leave room for taskbar
    # max_display_height = int(screen_height * 0.9)
    # max_display_width = int(screen_width * 0.9)
    
    # with sv.VideoSink(target_video_path, video_info) as sink:
    #     for frame in frame_generator:
    #         sink.write_frame(frame)

    #         # Resize frame to fit screen
    #         h, w = frame.shape[:2]
    #         scale = min(max_display_width / w, max_display_height / h)
    #         if scale < 1:
    #             display_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    #         else:
    #             display_frame = frame
            
    #         cv2.imshow("frame", display_frame)
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break
    #     cv2.destroyAllWindows()
    
    with sv.VideoSink(target_video_path, video_info) as sink:
      for frame in frame_generator:
        # Directly write the frame to output, no display
        sink.write_frame(frame)


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

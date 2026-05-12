import argparse
from collections import Counter, defaultdict, deque
from enum import Enum
from typing import Iterator, List

import os
import cv2
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


def collect_crops(
    frame_generator,
    player_detection_model,
    max_per_frame_ratio: float = 2.0,
) -> List[np.ndarray]:
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
    color_lookup: np.ndarray,
    ball_detections: sv.Detections | None = None
) -> np.ndarray:
    all_keypoints = keypoints.xy[0]
    all_confidence = (
        keypoints.confidence[0]
        if keypoints.confidence is not None
        else np.ones(len(all_keypoints))
    )
    mask = (
        (all_keypoints[:, 0] > 1) &
        (all_keypoints[:, 1] > 1) &
        (all_confidence > 0.5)
    )
    if np.sum(mask) < 4:
        return draw_pitch(config=CONFIG)

    source_keypoints = all_keypoints[mask].astype(np.float32)
    target_vertices = np.array(CONFIG.vertices)[mask].astype(np.float32)

    try:
        transformer = ViewTransformer(
            source=source_keypoints,
            target=target_vertices
        )
        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(points=xy)
        transformed_xy[:, 0] = np.clip(transformed_xy[:, 0], 0, CONFIG.length)
        transformed_xy[:, 1] = np.clip(transformed_xy[:, 1], 0, CONFIG.width)

        transformed_ball_xy = None
        if ball_detections is not None and len(ball_detections) > 0:
            ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            transformed_ball_xy = transformer.transform_points(points=ball_xy)
            transformed_ball_xy[:, 0] = np.clip(transformed_ball_xy[:, 0], 0, CONFIG.length)
            transformed_ball_xy[:, 1] = np.clip(transformed_ball_xy[:, 1], 0, CONFIG.width)

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

    if transformed_ball_xy is not None and len(transformed_ball_xy) > 0:
        radar = draw_points_on_pitch(config=CONFIG, xy=transformed_ball_xy,
            face_color=sv.Color.from_hex("#FFFFFF"), radius=10, pitch=radar)

    return radar


def _build_transformer(keypoints: sv.KeyPoints) -> ViewTransformer | None:
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


class AerialDuelDetector:
    """
    Detects an aerial duel when:
    - at least `min_players` players are within `pitch_proximity_meters` of the ball
    - at least one player from each team is involved

    Uses multi-signal airborne detection combining speed, curvature, vertical
    acceleration, and upward motion ratio for robust ball height estimation.

    A latch keeps the annotation alive for `latch_frames` frames after conditions
    drop out, to handle brief ball detection gaps.
    """

    def __init__(
        self,
        pitch_proximity_meters: float = 300.0,
        proximity_body_fraction: float = 0.4,
        min_players: int = 2,
        latch_frames: int = 4,
        airborne_min_frames: int = 5,
        airborne_up_ratio: float = 0.55,
        recent_proximity_frames: int = 30,
        # Airborne detection tuning
        airborne_speed_normaliser: float = 12.0,
        airborne_curve_threshold: float = 15.0,
        airborne_score_threshold: float = 0.45,
    ):
        self.pitch_proximity_meters = pitch_proximity_meters
        self.proximity_body_fraction = proximity_body_fraction
        self.min_players = min_players
        self.latch_frames = latch_frames
        self.airborne_min_frames = airborne_min_frames
        self.airborne_up_ratio = airborne_up_ratio
        self.recent_proximity_frames = recent_proximity_frames
        self.airborne_speed_normaliser = airborne_speed_normaliser
        self.airborne_curve_threshold = airborne_curve_threshold
        self.airborne_score_threshold = airborne_score_threshold
        self._latch_count = 0
        self._near_ball_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=recent_proximity_frames))
        self._frame_index: int = 0

    @staticmethod
    def get_ball_center(ball_tracker: BallTracker) -> np.ndarray | None:
        if len(ball_tracker.buffer) == 0:
            return None
        entry = np.asarray(ball_tracker.buffer[-1], dtype=float).ravel()
        if len(entry) >= 2:
            return entry[:2]
        return entry

    def is_airborne(self, ball_tracker: BallTracker) -> bool:
        """
        Multi-signal airborne detection combining 4 signals:

        1. Upward motion ratio — how often ball moves upward (y decreasing)
        2. Speed — airborne balls move faster than rolling ones
        3. Vertical acceleration — gravity gives airborne balls parabolic
           y-trajectory (linear dy over time)
        4. Trajectory curvature — airborne balls arc; ground balls roll straight

        Tuning (via constructor params):
          airborne_speed_normaliser  — raise if fast ground passes trigger false positives
          airborne_curve_threshold   — raise if slow lofted balls are missed
          airborne_score_threshold   — raise for fewer false positives,
                                       lower for fewer missed detections
        """
        buf = ball_tracker.buffer
        if len(buf) < self.airborne_min_frames:
            return False

        def get_xy(entry) -> tuple[float, float]:
            arr = np.asarray(entry).ravel()
            x = float(arr[0]) if len(arr) >= 1 else 0.0
            y = float(arr[1]) if len(arr) >= 2 else 0.0
            return x, y

        positions = [get_xy(e) for e in buf]
        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])

        # --- Signal 1: upward motion ratio ---
        # y decreasing = ball moving up in image coordinates
        dy = np.diff(ys)
        dx = np.diff(xs)
        up_ratio = float(np.sum(dy < 0)) / max(len(dy), 1)

        # --- Signal 2: speed ---
        # Airborne balls move faster than rolling balls.
        # airborne_speed_normaliser px/frame maps to a score of 1.0.
        speeds = np.sqrt(dx ** 2 + dy ** 2)
        mean_speed = float(np.mean(speeds))
        speed_score = min(1.0, mean_speed / self.airborne_speed_normaliser)

        # --- Signal 3: vertical acceleration (gravity signature) ---
        # Airborne ball: y(t) = at² + bt + c → dy(t) is linear.
        # Fit a line to dy and measure R². High R² = smooth parabolic = airborne.
        if len(dy) >= 4:
            t = np.arange(len(dy), dtype=float)
            coeffs = np.polyfit(t, dy, 1)
            dy_fit = np.polyval(coeffs, t)
            ss_res = float(np.sum((dy - dy_fit) ** 2))
            ss_tot = float(np.sum((dy - dy.mean()) ** 2))
            accel_score = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else 0.0
        else:
            accel_score = 0.0

        # --- Signal 4: trajectory curvature ---
        # Airborne balls arc (deviate from straight line between endpoints).
        # Rolling balls travel roughly straight.
        if len(positions) >= 3:
            start = np.array(positions[0])
            end = np.array(positions[-1])
            line_vec = end - start
            line_len = float(np.linalg.norm(line_vec))
            if line_len > 1e-6:
                deviations = []
                for pos in positions[1:-1]:
                    pt = np.array(pos) - start
                    proj = np.dot(pt, line_vec) / (line_len ** 2)
                    closest = start + proj * line_vec
                    dev = float(np.linalg.norm(np.array(pos) - closest))
                    deviations.append(dev)
                max_deviation = max(deviations) if deviations else 0.0
                curve_score = min(1.0, max_deviation / self.airborne_curve_threshold)
            else:
                curve_score = 0.0
        else:
            curve_score = 0.0

        # --- Weighted combination ---
        # Speed is intentionally near-zero: football is always fast so speed
        # alone is meaningless. Curvature and acceleration are the real signals.
        score = (
            0.15 * up_ratio    +   # upward motion
            0.05 * speed_score +   # speed (near-zero — football always fast)
            0.40 * accel_score +   # parabolic acceleration = strongest airborne signal
            0.40 * curve_score     # curved path = strongest airborne signal
        )

        return score >= self.airborne_score_threshold

    @staticmethod
    def get_player_duel_points(detections: sv.Detections) -> np.ndarray:
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
        ball_center = self.get_ball_center(ball_tracker)

        if ball_center is None:
            self._latch_count = max(0, self._latch_count - 1)
            return self._latch_count > 0, players[[]], None

        if len(players) == 0:
            self._latch_count = 0
            return False, players[[]], ball_center

        player_points_px = self.get_player_duel_points(players)

        if transformer is not None:
            try:
                player_points_pitch = transformer.transform_points(player_points_px)
                ball_center_pitch = transformer.transform_points(ball_center[np.newaxis])[0]
                distances = np.linalg.norm(player_points_pitch - ball_center_pitch, axis=1)
                mask = distances <= self.pitch_proximity_meters
            except (ValueError, cv2.error):
                transformer = None

        if transformer is None:
            player_heights = players.xyxy[:, 3] - players.xyxy[:, 1]
            thresholds = player_heights * self.proximity_body_fraction
            distances = np.linalg.norm(player_points_px - ball_center, axis=1)
            mask = distances <= thresholds

        involved = players[mask]

        if players_team_id is not None and len(players_team_id) == len(players):
            for i, (tid, team) in enumerate(zip(
                players.tracker_id if players.tracker_id is not None else [],
                players_team_id
            )):
                if mask[i]:
                    self._near_ball_history[int(tid)].append((self._frame_index, int(team)))
        self._frame_index += 1

        involved = players[mask]

        current_teams = set()
        if players_team_id is not None and len(players_team_id) == len(players):
            current_teams = set(players_team_id[mask].tolist())

        recently_seen_teams = set(current_teams)
        cutoff = self._frame_index - self.recent_proximity_frames
        for tid, history in self._near_ball_history.items():
            for (frame_idx, team) in history:
                if frame_idx >= cutoff:
                    recently_seen_teams.add(team)
                    break

        has_both_teams = len(recently_seen_teams) >= 2
        currently_dueling = len(involved) >= 2 and has_both_teams

        if currently_dueling:
            self._latch_count = self.latch_frames
        else:
            self._latch_count = max(0, self._latch_count - 1)

        return self._latch_count > 0, involved, ball_center


class DuelOutcomeTracker:
    """
    Tracks the outcome of aerial duels by monitoring which team gains possession
    after the duel ends.

    After a duel ends, watches the next `outcome_window_frames` frames and finds
    which team has a player closest to the ball — that team wins the duel.
    If no clear winner emerges before the window expires, the duel is contested.
    """

    def __init__(
        self,
        outcome_window_frames: int = 30,
        outcome_display_duration: int = 90,
    ):
        self.outcome_window_frames = outcome_window_frames
        self.outcome_display_duration = outcome_display_duration

        self._was_dueling = False
        self._watching = False
        self._watch_countdown = 0
        self._last_duel_teams: set = set()
        self._outcome: str | None = None
        self._outcome_display_frames = 0

    def update(
        self,
        is_duel: bool,
        players: sv.Detections,
        players_team_id: np.ndarray | None,
        ball_center: np.ndarray | None,
        involved_players: sv.Detections,
    ) -> str | None:
        # While duel is active: record involved teams, suppress watch
        if is_duel:
            if (players_team_id is not None and
                    len(involved_players) > 0 and
                    involved_players.tracker_id is not None and
                    players.tracker_id is not None):
                involved_ids = set(involved_players.tracker_id.tolist())
                for i, tid in enumerate(players.tracker_id):
                    if tid in involved_ids and i < len(players_team_id):
                        self._last_duel_teams.add(int(players_team_id[i]))
            self._watching = False
            self._watch_countdown = 0
            self._was_dueling = True

        # Duel just ended: start watch window
        elif self._was_dueling and self._last_duel_teams:
            self._watching = True
            self._watch_countdown = self.outcome_window_frames
            self._was_dueling = False
        else:
            self._was_dueling = False

        # Watch window: check who is closest to ball each frame
        if self._watching and self._watch_countdown > 0:
            self._watch_countdown -= 1

            if (ball_center is not None and
                    players_team_id is not None and
                    len(players) > 0):
                player_points = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                distances = np.linalg.norm(player_points - ball_center, axis=1)
                closest_idx = int(np.argmin(distances))

                if closest_idx < len(players_team_id):
                    winning_team = int(players_team_id[closest_idx])
                    if winning_team in self._last_duel_teams:
                        self._outcome = f"TEAM {winning_team} WINS DUEL"
                        self._outcome_display_frames = self.outcome_display_duration
                        self._watching = False
                        self._watch_countdown = 0
                        self._last_duel_teams = set()

            # Watch window expired with no clear winner
            if self._watch_countdown == 0 and self._watching:
                self._outcome = "DUEL CONTESTED"
                self._outcome_display_frames = self.outcome_display_duration
                self._watching = False
                self._last_duel_teams = set()

        # Count down display timer
        if self._outcome_display_frames > 0:
            self._outcome_display_frames -= 1
            return self._outcome

        return None


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


def annotate_duel_outcome(frame: np.ndarray, outcome: str | None) -> np.ndarray:
    if outcome is None:
        return frame
    color = (0, 255, 0) if "WINS" in outcome else (0, 165, 255)
    cv2.putText(
        frame, outcome, (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA
    )
    return frame


def run_referee_diagnostic(source_video_path: str, device: str) -> None:
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
    ball_annotator = BallAnnotator(radius=10, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=960, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
        slice_wh=(640, 640),
        overlap_wh=(64, 64),
    )
    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.3)
        if detections.confidence is not None:
            detections = detections[detections.confidence > 0.1]
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

    stride = compute_stride(source_video_path, min_samples=50)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)
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
            players, players_team_id, goalkeepers)
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
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_aerial_duel(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Aerial duel detection with:
    - Team colors (full team classification pipeline)
    - Ball detection and tracking
    - Multi-signal airborne detection (speed, curvature, acceleration, up-ratio)
    - Pitch homography for real-world proximity measurement
    - Falls back to body-height-relative pixel distances when homography unavailable
    - Cross-team check: both teams must be involved for a duel to trigger
    - Duel outcome: tracks which team gains possession after the duel ends
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    # --- Phase 1: collect crops for team classifier ---
    stride = compute_stride(source_video_path, min_samples=8)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)
    crops = collect_crops(frame_generator, player_detection_model)

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(get_jersey_crops(crops))

    # --- Phase 2: main processing loop ---
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    voter = TeamVoter()
    class_voter = ClassVoter()
    outlier_voter = OutlierVoter()

    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=10, buffer_size=10)

    aerial_duel_detector = AerialDuelDetector(
        pitch_proximity_meters=300,
        proximity_body_fraction=0.4,
        min_players=2,
        latch_frames=4,
        airborne_min_frames=5,
        # Airborne tuning:
        # - lower airborne_score_threshold if airborne balls show "ON GROUND"
        # - raise airborne_curve_threshold if straight kicks trigger "IN AIR"
        airborne_speed_normaliser=12.0,
        airborne_curve_threshold=25.0,   # raised: football travels straighter than expected
        airborne_score_threshold=0.55,   # raised: require stronger signal to call airborne
    )

    duel_outcome_tracker = DuelOutcomeTracker(
        outcome_window_frames=30,
        outcome_display_duration=90,
    )

    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=960, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=ball_callback,
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
        slice_wh=(640, 640),
        overlap_wh=(64, 64),
    )

    for frame in frame_generator:
        # --- pitch keypoints → homography transformer ---
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        transformer = _build_transformer(keypoints)

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
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)
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
        ball_detections = slicer(frame).with_nms(threshold=0.3)
        if ball_detections.confidence is not None:
            ball_detections = ball_detections[ball_detections.confidence > 0.4]
        ball_detections = ball_tracker.update(ball_detections)

        # --- aerial duel detection ---
        is_duel, involved_players, ball_center = aerial_duel_detector.detect(
            ball_tracker=ball_tracker,
            players=players,
            players_team_id=players_team_id,
            transformer=transformer,
        )

        # --- duel outcome tracking ---
        outcome = duel_outcome_tracker.update(
            is_duel=is_duel,
            players=players,
            players_team_id=players_team_id,
            ball_center=ball_center,
            involved_players=involved_players,
        )

        # --- annotate ---
        annotated_frame = frame.copy()

        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, all_detections, custom_color_lookup=color_lookup)
        labels = (
            [str(tid) for tid in all_detections.tracker_id]
            if all_detections.tracker_id is not None
            else [""] * len(all_detections)
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, all_detections, labels, custom_color_lookup=color_lookup)

        annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)

        # Airborne status indicator
        ball_is_airborne = aerial_duel_detector.is_airborne(ball_tracker)
        airborne_text = "BALL IN AIR" if ball_is_airborne else "BALL ON GROUND"
        cv2.putText(
            annotated_frame, airborne_text, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 255) if ball_is_airborne else (255, 255, 255),
            2, cv2.LINE_AA,
        )

        if is_duel:
            annotated_frame = annotate_aerial_duel(annotated_frame, involved_players, ball_center)

        annotated_frame = annotate_duel_outcome(annotated_frame, outcome)

        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=10, buffer_size=10)

    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=960, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    ball_slicer = sv.InferenceSlicer(
        callback=ball_callback,
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
        slice_wh=(640, 640),
        overlap_wh=(64, 64),
    )

    stride = compute_stride(source_video_path, min_samples=50)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)
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

        ball_detections = ball_slicer(frame).with_nms(threshold=0.3)
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
            players, players_team_id, goalkeepers)
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
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)

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
    elif mode == Mode.AERIAL_DUEL:
        frame_generator = run_aerial_duel(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
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

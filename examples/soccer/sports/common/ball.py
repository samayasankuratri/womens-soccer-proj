from collections import deque

import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    """
    A class to annotate frames with circles of varying radii and colors.

    Attributes:
        radius (int): The maximum radius of the circles to be drawn.
        buffer (deque): A deque buffer to store recent coordinates for annotation.
        color_palette (sv.ColorPalette): A color palette for the circles.
        thickness (int): The thickness of the circle borders.
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):

        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolates the radius between 1 and the maximum radius based on the index.

        Args:
            i (int): The current index in the buffer.
            max_i (int): The maximum index in the buffer.

        Returns:
            int: The interpolated radius.
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotates the frame with circles based on detections.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections containing coordinates.

        Returns:
            np.ndarray: The annotated frame.
        """
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    """
    Tracks the soccer ball and determines if it is airborne using trajectory.
    """

    def __init__(
        self,
        buffer_size: int = 10,
        min_airborne_frames: int = 5,
        vertical_threshold: float = 4.0,
        vertical_ratio_threshold: float = 0.8,
        consistency_ratio: float = 0.6,
    ):
        from collections import deque
        import numpy as np

        self.buffer = deque(maxlen=buffer_size)

        self.min_airborne_frames = min_airborne_frames
        self.vertical_threshold = vertical_threshold
        self.vertical_ratio_threshold = vertical_ratio_threshold
        self.consistency_ratio = consistency_ratio

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)

        if len(self.buffer) == 0:
            index = 0
        else:
            centroid = np.mean(np.concatenate(self.buffer), axis=0)
            distances = np.linalg.norm(xy - centroid, axis=1)
            index = int(np.argmin(distances))

        tracked = detections[[index]]
        tracked_xy = tracked.get_anchors_coordinates(sv.Position.CENTER)

        self.buffer.append(tracked_xy)
        return tracked

    def get_ball_center(self):
        if len(self.buffer) == 0:
            return None

        latest = self.buffer[-1]
        if len(latest) == 0:
            return None

        return latest[0]

    def is_airborne(self) -> bool:
        if len(self.buffer) < self.min_airborne_frames:
            return False

        pts = np.concatenate(self.buffer, axis=0)

        xs = pts[:, 0]
        ys = pts[:, 1]

        dx = np.diff(xs)
        dy = np.diff(ys)

        if len(dx) == 0 or len(dy) == 0:
            return False

        mean_abs_dx = np.mean(np.abs(dx))
        mean_abs_dy = np.mean(np.abs(dy))

        vertical_enough = mean_abs_dy > self.vertical_threshold
        vertical_ratio = mean_abs_dy / (mean_abs_dx + 1e-6)
        vertical_dominant = vertical_ratio > self.vertical_ratio_threshold

        signs = np.sign(dy)
        nonzero_signs = signs[signs != 0]

        if len(nonzero_signs) == 0:
            return False

        dominant_sign = 1 if np.sum(nonzero_signs > 0) >= np.sum(nonzero_signs < 0) else -1
        consistency = np.mean(nonzero_signs == dominant_sign)

        consistent_motion = consistency >= self.consistency_ratio

        return vertical_enough and vertical_dominant and consistent_motion
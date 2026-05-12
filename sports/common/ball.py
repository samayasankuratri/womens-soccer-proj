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
    A class used to track a soccer ball's position across video frames.

    Uses velocity-based prediction: estimates where the ball will be next based on
    recent movement, then picks the detection closest to that predicted position.
    Only confirmed detections are added to the buffer, preventing empty frames from
    polluting the position history.

    Attributes:
        buffer (collections.deque): A deque buffer storing confirmed ball positions.
    """
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the buffer with the best detection and returns it.

        Predicts the next ball position using recent velocity, then selects the
        detection closest to that prediction. Falls back to the last known position
        if velocity cannot be computed. Only appends to the buffer when a detection
        is confirmed, so empty frames never corrupt the position history.

        Args:
            detections (sv.Detections): The current frame's ball detections.

        Returns:
            sv.Detections: The best matching detection, or empty detections if none.
        """
        if len(detections) == 0:
            return detections

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)

        if len(self.buffer) == 0:
            # No history — take highest-confidence detection
            if detections.confidence is not None:
                index = int(np.argmax(detections.confidence))
            else:
                index = 0
            self.buffer.append(xy[index])
            return detections[[index]]

        # Predict next position using velocity from recent frames
        if len(self.buffer) >= 2:
            velocity = self.buffer[-1] - self.buffer[-2]
            predicted = self.buffer[-1] + velocity
        else:
            predicted = self.buffer[-1]

        distances = np.linalg.norm(xy - predicted, axis=1)
        index = int(np.argmin(distances))
        self.buffer.append(xy[index])
        return detections[[index]]

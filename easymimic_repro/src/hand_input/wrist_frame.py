"""MediaPipe 21 hand landmarks -> camera-frame metric-ish points + a wrist rotation.

hand_to_ee needs R_wrist for exactly one thing: its PALM_NORMAL_REFERENCE_AXIS column
(column 2) fixes the sign of the SVD plane normal it computes itself. MediaPipe Hands
reports no wrist rotation, so that frame is rebuilt here from the palm's own geometry --
this module is the adapter between a 21-point detector and the paper-core math layer.

Why cross products and not a second SVD: an SVD plane normal is sign-ambiguous, so a
per-frame flip would turn the relative rotation R_ref^T R_now into a 180 deg jump and
slam the arm. The cross product of the two palm bones (wrist->index MCP, wrist->pinky
MCP) is continuous frame to frame for as long as those bones are visible.

Why the metric conversion: MediaPipe image landmarks are normalized to the image and
carry no scale, while hand_to_ee's gripper calibration (d_min/d_max = 0.02/0.10 m, and
the thenar anchor) needs metres. The scale is the real palm width divided by the same
width measured in the image, so it is a single ratio that cancels perspective: the
palm-width prior is only a fallback when no world landmarks are supplied.

Camera frame: X right, Y down, Z away from the camera (OpenCV convention), origin at the
principal point. hand_to_ee is called with T_base_cam = identity, so this frame is also
the frame the 8-D command is expressed in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_LANDMARKS = 21
WRIST_IDX = 0
INDEX_MCP_IDX = 5
MIDDLE_MCP_IDX = 9
PINKY_MCP_IDX = 17

# Mean adult index-MCP -> pinky-MCP span, metres. Only used when MediaPipe's world
# landmarks are missing; with them the real palm width is measured instead.
PALM_WIDTH_M = 0.085
# |sin| of the angle between the two palm bones below which the palm frame is undefined
# (landmarks collapsed, or the palm seen exactly edge-on).
MIN_PALM_SIN = 1e-3


@dataclass(frozen=True)
class WristFrame:
    """One hand frame, expressed in the camera frame."""

    keypoints: np.ndarray   # (21, 3) metres
    R_wrist: np.ndarray     # (3, 3) proper rotation, columns [X fingers, Y across, Z palm normal]
    scale: float            # metres per image unit (per pixel)
    valid: bool
    reason: str = ""

    @property
    def palm_normal(self):
        return self.R_wrist[:, 2]


def image_points(landmarks, image_size):
    """Normalized MediaPipe landmarks -> camera-frame pixel triple.

    MediaPipe normalizes x by the image width, y by the height, and z by the width with a
    smaller value meaning closer to the camera. Keeping z * forward keeps the triple a
    right-handed (right, down, away) camera frame and a uniform scaling of the hand.
    """
    arr = as_landmarks(landmarks)
    width, height = as_image_size(image_size)
    px = np.empty((NUM_LANDMARKS, 3), dtype=np.float64)
    px[:, 0] = (arr[:, 0] - 0.5) * width
    px[:, 1] = (arr[:, 1] - 0.5) * height
    px[:, 2] = arr[:, 2] * width
    return px


def metric_scale(px, world_landmarks=None):
    """Metres per image unit, from the palm width in 3-D against the same width in the image."""
    px = np.asarray(px, dtype=np.float64)
    span_px = float(np.linalg.norm(px[INDEX_MCP_IDX] - px[PINKY_MCP_IDX]))
    if not np.isfinite(span_px) or span_px <= 0.0:
        raise ValueError("palm span in the image is degenerate (%r px)" % (span_px,))
    if world_landmarks is None or len(world_landmarks) == 0:
        span_m = PALM_WIDTH_M
    else:
        world = as_landmarks(world_landmarks, "world_landmarks")
        span_m = float(np.linalg.norm(world[INDEX_MCP_IDX] - world[PINKY_MCP_IDX]))
    if not np.isfinite(span_m) or span_m <= 0.0:
        raise ValueError("palm span from world landmarks is degenerate (%r m)" % (span_m,))
    return span_m / span_px


def wrist_rotation(points):
    """Palm geometry -> proper rotation whose column 2 is the palm normal.

    X is the finger direction (wrist -> middle MCP, projected off the normal), Z is the
    normal of the two palm bones, Y closes the frame with Y = Z x X so det(R) = +1.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (NUM_LANDMARKS, 3):
        raise ValueError("expected keypoints of shape (%d, 3), got %s" % (NUM_LANDMARKS, p.shape))

    v_index = p[INDEX_MCP_IDX] - p[WRIST_IDX]
    v_pinky = p[PINKY_MCP_IDX] - p[WRIST_IDX]
    n_index = float(np.linalg.norm(v_index))
    n_pinky = float(np.linalg.norm(v_pinky))
    if not (np.isfinite(n_index) and np.isfinite(n_pinky)) or n_index <= 0.0 or n_pinky <= 0.0:
        raise ValueError("wrist -> MCP bones are degenerate")
    normal = np.cross(v_index, v_pinky)
    n_normal = float(np.linalg.norm(normal))
    if n_normal <= MIN_PALM_SIN * n_index * n_pinky:
        raise ValueError("palm bones are collinear; the palm frame is undefined")
    Z = normal / n_normal

    fingers = p[MIDDLE_MCP_IDX] - p[WRIST_IDX]
    X = fingers - float(np.dot(fingers, Z)) * Z
    n_x = float(np.linalg.norm(X))
    if n_x <= 1e-12 * max(n_normal, 1e-12) or n_x <= 0.0:
        raise ValueError("finger direction is parallel to the palm normal")
    X = X / n_x

    Y = np.cross(Z, X)
    Y = Y / float(np.linalg.norm(Y))
    return np.column_stack([X, Y, Z])


def frame_from_points(points, scale):
    """Metric cloud + scale -> WristFrame; degenerate geometry becomes invalid, not an exception."""
    points = np.asarray(points, dtype=np.float64)
    try:
        rotation = wrist_rotation(points)
    except ValueError as exc:
        return invalid_frame(str(exc), keypoints=points, scale=scale)
    return WristFrame(keypoints=points, R_wrist=rotation, scale=float(scale), valid=True)


def build(landmarks, image_size, world_landmarks=None):
    """Full path: normalized landmarks (+ optional world landmarks) -> metric cloud + R_wrist."""
    try:
        arr = as_landmarks(landmarks)
    except (ValueError, TypeError) as exc:
        return invalid_frame(str(exc))
    if not np.all(np.isfinite(arr)):
        return invalid_frame("non-finite landmark")
    try:
        px = image_points(arr, image_size)
        scale = metric_scale(px, world_landmarks)
    except (ValueError, TypeError) as exc:
        return invalid_frame(str(exc))
    if not np.isfinite(scale) or scale <= 0.0:
        return invalid_frame("non-finite scale")
    return frame_from_points(scale * px, scale)


def invalid_frame(reason, keypoints=None, scale=float("nan")):
    """The 'no usable hand frame' value: NaN geometry and valid = False, never zeros."""
    if keypoints is None:
        keypoints = np.full((NUM_LANDMARKS, 3), np.nan)
    return WristFrame(keypoints=np.asarray(keypoints, dtype=np.float64),
                      R_wrist=np.full((3, 3), np.nan), scale=float(scale),
                      valid=False, reason=reason)


class FrameSmoother:
    """Exponential moving average over the metric cloud; R_wrist is re-derived afterwards.

    MediaPipe jitters by a few millimetres per frame, which a position servo reproduces
    as audible buzzing. The state deliberately survives lost frames: when the hand comes
    back the command glides to it over a few ticks instead of teleporting the arm.
    """

    def __init__(self, alpha=0.35):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1], got %r" % (alpha,))
        self.alpha = float(alpha)
        self._points = None

    def reset(self):
        self._points = None

    def update(self, frame):
        if frame is None:
            return invalid_frame("no hand frame")
        if not frame.valid:
            return frame
        if self._points is None:
            self._points = np.array(frame.keypoints, dtype=np.float64)
        else:
            self._points = (1.0 - self.alpha) * self._points + self.alpha * frame.keypoints
        return frame_from_points(self._points, frame.scale)


def as_landmarks(value, name="landmarks"):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (NUM_LANDMARKS, 3):
        raise ValueError("%s must be (%d, 3), got %s" % (name, NUM_LANDMARKS, arr.shape))
    return arr


def as_image_size(image_size):
    size = np.asarray(image_size, dtype=np.float64).reshape(-1)
    if size.size != 2 or not np.all(np.isfinite(size)) or size[0] <= 0.0 or size[1] <= 0.0:
        raise ValueError("image_size must be (width, height) in pixels, got %r" % (image_size,))
    return float(size[0]), float(size[1])

#!/usr/bin/env python3
"""NumPy-only 21-landmark to 8-D end-effector retargeting.

The geometry follows the supplied Action Space Alignment reference:
thenar midpoint for position, a five-point SVD plane frame for orientation,
thumb-index tip distance for the gripper, and a rotation-vector pose command.
``hand_to_ee`` implements the metric camera-frame reference interface.
The separate webcam adapter uses normalized MediaPipe landmarks, so translation is
relative to an armed pose and camera-to-base rotation is an explicit estimate,
not metric HaMeR data or a calibrated hand-eye transform.
"""

from __future__ import annotations

import math

import numpy as np

NUM_KEYPOINTS = 21
THUMB_IP_IDX = 3
THUMB_TIP_IDX = 4
INDEX_MCP_IDX = 5
INDEX_PIP_IDX = 6
INDEX_TIP_IDX = 8
THENAR_ANCHOR_IDS = (THUMB_IP_IDX, INDEX_MCP_IDX)
PLANE_POINT_IDS = (INDEX_MCP_IDX, INDEX_PIP_IDX, 7, INDEX_TIP_IDX, THUMB_IP_IDX)
GRIPPER_TIP_IDS = (THUMB_TIP_IDX, INDEX_TIP_IDX)
D_MIN = 0.02  # metres, thumb/index tip gap for g=0
D_MAX = 0.10  # metres, thumb/index tip gap for g=1

# Camera pseudo-frame: +X image-right, +Y image-down, +Z toward the camera.
# Approximate mapping: screen-right -> base +X, closer -> base +Y,
# screen-down -> base -Z. Replace with measured camera extrinsics for calibrated use.
CAMERA_TO_BASE = np.array(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    dtype=np.float64,
)


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}.")
    return result


def _rotation(value, name: str = "Rotation") -> np.ndarray:
    rotation = _array(value, (3, 3), name)
    if (not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)):
        raise ValueError(f"{name} must be a finite proper rotation (SO(3)).")
    return rotation


def rotmat_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """Stable SO(3) logarithm, including the pi-angle branch."""
    rotation = _rotation(rotation)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    skew = np.array((
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ))
    angle = math.atan2(float(np.linalg.norm(skew)) * 0.5, cosine)
    if angle < 1e-7:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        symmetric = (rotation + rotation.T + 2.0 * np.eye(3)) * 0.25
        axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
        pivot = int(np.argmax(axis))
        if axis[pivot] < 1e-8:
            axis = np.array((1.0, 0.0, 0.0))
        else:
            for index in range(3):
                if index != pivot:
                    axis[index] = symmetric[pivot, index] / axis[pivot]
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        # Below pi, the skew part still determines the axis sign. Ignoring it
        # reverses rotations whose largest axis component is negative.
        if float(axis @ skew) < 0.0:
            axis = -axis
        return angle * axis
    return angle * skew / (2.0 * math.sin(angle))


def rotvec_to_rotmat(vector: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map from a rotation vector to SO(3)."""
    vector = _array(vector, (3,), "Rotation vector")
    if not np.isfinite(vector).all():
        raise ValueError("Rotation vector must be finite.")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def pose_to_ee(position: np.ndarray, rotation: np.ndarray, gripper: float) -> np.ndarray:
    """Pack base-frame metres, rotation vector in radians, g and reserved=0."""
    position = _array(position, (3,), "Position")
    if not np.isfinite(position).all() or not np.isfinite(gripper) or not 0 <= gripper <= 1:
        raise ValueError("Position must be finite and gripper must be in [0, 1].")
    return np.concatenate((position, rotmat_to_rotvec(rotation), (gripper, 0.0)))


def ee_to_pose(ee8: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Unpack a valid EE8 into (base position, base rotation matrix, gripper).

    Invalid commands are rejected before they can reach IK or actuators.
    """
    command = _array(ee8, (8,), "EE8")
    if not np.isfinite(command).all() or not 0 <= command[6] <= 1 or command[7] != 0:
        raise ValueError("EE8 must be finite, with gripper in [0, 1] and reserved=0.")
    return command[:3].copy(), rotvec_to_rotmat(command[3:6]), float(command[6])


def _hand_geometry(
    geometry: np.ndarray, normal_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Shared geometry for metric inputs and the explicitly estimated webcam adapter."""
    geometry = _array(geometry, (NUM_KEYPOINTS, 3), "Keypoints")
    reference = _array(normal_reference, (3,), "Normal reference")
    if not np.isfinite(geometry).all() or not np.isfinite(reference).all():
        raise ValueError("Keypoints and normal reference must be finite.")
    if np.linalg.norm(reference) < 1e-8:
        raise ValueError("Normal reference must be nonzero.")
    index_direction = geometry[INDEX_PIP_IDX] - geometry[INDEX_MCP_IDX]
    if np.linalg.norm(index_direction) < 1e-8:
        raise ValueError("Degenerate hand: index MCP-to-PIP direction is unavailable.")
    plane = geometry[list(PLANE_POINT_IDS)]
    centered = plane - plane.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    if singular_values[1] < 1e-8:
        raise ValueError("Degenerate hand: five landmarks do not define a stable plane.")
    z_axis = vt[-1]
    if float(z_axis @ reference) < 0.0:
        z_axis = -z_axis
    x_axis = index_direction - float(index_direction @ z_axis) * z_axis
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-8:
        raise ValueError("Degenerate hand: index direction is normal to the hand plane.")
    x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    anchor = geometry[list(THENAR_ANCHOR_IDS)].mean(axis=0)
    gap = float(np.linalg.norm(geometry[THUMB_TIP_IDX] - geometry[INDEX_TIP_IDX]))
    residual = float(np.max(np.abs(centered @ z_axis)))
    return anchor, rotation, gap, residual


def hand_to_ee(
    kp: np.ndarray,
    R_wrist: np.ndarray,
    T_base_cam: np.ndarray,
    d_min: float = D_MIN,
    d_max: float = D_MAX,
) -> tuple[np.ndarray, dict]:
    """Convert metric camera-frame keypoints to an absolute base-frame EE8.

    ``kp`` is (21, 3), in metres in the camera frame, not MediaPipe's
    hand-relative world landmarks. ``R_wrist`` has wrist axes as columns in
    that same camera frame. ``T_base_cam`` maps camera coordinates to base:
    p_base = R_bc @ p_camera + t_bc, R_base = R_bc @ R_hand.

    Nonfinite data in ANY input point/matrix yields eight NaNs and valid=False.
    Malformed shapes, non-rigid transforms and degenerate geometry raise
    ValueError. The caller must hold its previous command on either failure.
    """
    points = _array(kp, (NUM_KEYPOINTS, 3), "kp")
    wrist = _array(R_wrist, (3, 3), "R_wrist")
    transform = _array(T_base_cam, (4, 4), "T_base_cam")
    if (not np.isfinite(points).all() or not np.isfinite(wrist).all()
            or not np.isfinite(transform).all()):
        return np.full(8, np.nan), {"valid": False, "reason": "nonfinite_input"}
    _rotation(wrist, "R_wrist")
    rotation_bc = _rotation(transform[:3, :3], "T_base_cam rotation")
    if not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("T_base_cam must have homogeneous last row [0, 0, 0, 1].")
    anchor, rotation_hand, gap, residual = _hand_geometry(points, wrist[:, 2])
    g = normalize_gripper_gap(gap, d_min, d_max)
    ee8 = pose_to_ee(rotation_bc @ anchor + transform[:3, 3],
                     rotation_bc @ rotation_hand, g)
    return ee8, {"valid": True, "p_camera": anchor, "R_hand": rotation_hand,
                 "gap_m": gap, "plane_residual_m": residual}


def blend_rotation(
    previous: np.ndarray | None,
    current: np.ndarray,
    alpha: float = 0.28,
) -> np.ndarray:
    """Low-pass filter rotations and project the blend back onto SO(3)."""
    if previous is None:
        return np.asarray(current, dtype=np.float64).copy()
    u, _, vt = np.linalg.svd(
        (1.0 - alpha) * previous + alpha * np.asarray(current, dtype=np.float64)
    )
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def hand_frame_and_metrics(
    landmarks: np.ndarray,
    image_width: int,
    image_height: int,
    world_landmarks: np.ndarray | None = None,
    normal_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return image thenar, hand rotation, tip gap, image palm size and residual.

    Position and apparent palm size use normalized image landmarks. If
    ``world_landmarks`` is supplied, its 3-D coordinates define the five-point
    plane and thumb/index fingertip distance; MediaPipe describes these as
    metres, but they are relative hand coordinates and do not provide metric
    camera position. Without them, the image landmarks' relative z is used as a
    lower-quality pseudo-3-D fallback. The normal sign follows the previous
    frame when provided; unlike HaMeR, MediaPipe Hands has no wrist rotation.
    """
    raw = np.asarray(landmarks, dtype=np.float64)
    if raw.shape != (NUM_KEYPOINTS, 3) or not np.all(np.isfinite(raw)):
        raise ValueError("Expected 21 finite hand landmarks with x, y, z coordinates.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    image_points = raw.copy()
    image_points[:, 0] *= image_width
    image_points[:, 1] *= image_height
    image_points[:, 2] *= -image_width
    thenar_px = np.mean(image_points[list(THENAR_ANCHOR_IDS), :2], axis=0)
    thenar = np.array((thenar_px[0] / image_width, thenar_px[1] / image_height))

    palm_xy = image_points[[5, 9, 13, 17], :2]
    wrist_xy = image_points[0, :2]
    palm_scale = float(np.median(np.linalg.norm(palm_xy - wrist_xy, axis=1)))
    if palm_scale < 1e-8:
        raise ValueError("Degenerate hand: apparent palm scale is too small.")

    if world_landmarks is None:
        geometry = image_points
    else:
        geometry = np.asarray(world_landmarks, dtype=np.float64)
        if geometry.shape != (NUM_KEYPOINTS, 3) or not np.all(np.isfinite(geometry)):
            raise ValueError("Expected 21 finite 3-D world landmarks.")

    _, hand_rotation, tip_distance, plane_residual = _hand_geometry(
        geometry,
        np.array((0.0, 0.0, 1.0)) if normal_reference is None else normal_reference,
    )
    if world_landmarks is None:
        # Keep the fallback gap independent of image resolution and hand size.
        tip_distance /= palm_scale
    return thenar, hand_rotation, tip_distance, palm_scale, plane_residual


def normalize_gripper_gap(gap: float, closed_gap: float, open_gap: float) -> float:
    """Normalize thumb-index tip separation to [0, 1]."""
    if not np.isfinite((gap, closed_gap, open_gap)).all():
        raise ValueError("Gripper calibration and gap must be finite.")
    if open_gap - closed_gap <= 1e-9:
        raise ValueError("Open-hand fingertip gap must exceed closed-hand gap.")
    return float(np.clip((gap - closed_gap) / (open_gap - closed_gap), 0.0, 1.0))


def relative_hand_to_ee8(
    thenar: np.ndarray,
    hand_rotation: np.ndarray,
    pinch_gap: float,
    palm_scale: float,
    anchor_thenar: np.ndarray,
    anchor_hand_rotation: np.ndarray,
    anchor_palm_scale: float,
    anchor_robot_position: np.ndarray,
    anchor_robot_rotation: np.ndarray,
    closed_gap: float,
    open_gap: float,
    motion_span: float,
    depth_deadzone: float = 0.025,
    depth_full_scale: float = 0.20,
    camera_to_base: np.ndarray = CAMERA_TO_BASE,
) -> np.ndarray:
    """Build ``[base XYZ, base rotvec, normalized gripper, reserved]``.

    Position uses the reference's thenar anchor with relative screen XY and
    apparent palm-size depth. Orientation is the hand's relative spatial
    rotation applied to the robot pose captured at arming. The 8-D structure
    mirrors the reference while remaining usable with monocular MediaPipe.
    """
    values = (palm_scale, anchor_palm_scale, motion_span,
              pinch_gap, closed_gap, open_gap, depth_deadzone, depth_full_scale)
    if not np.isfinite(values).all() or palm_scale <= 0 or anchor_palm_scale <= 0:
        raise ValueError("Palm scales and motion span must be positive and finite.")
    if motion_span <= 0 or depth_deadzone < 0 or depth_full_scale <= depth_deadzone:
        raise ValueError("Workspace span and depth scale range are invalid.")
    change = palm_scale / anchor_palm_scale - 1.0
    magnitude = abs(change)
    if magnitude <= depth_deadzone:
        depth_fraction = 0.0
    else:
        amount = np.clip(
            (magnitude - depth_deadzone) / (depth_full_scale - depth_deadzone), 0.0, 1.0
        )
        depth_fraction = math.copysign(float(amount), change)

    thenar = np.asarray(thenar, dtype=np.float64)
    anchor_thenar = np.asarray(anchor_thenar, dtype=np.float64)
    anchor_position = np.asarray(anchor_robot_position, dtype=np.float64)
    if (thenar.shape != (2,) or anchor_thenar.shape != (2,)
            or anchor_position.shape != (3,)
            or not np.all(np.isfinite(thenar))
            or not np.all(np.isfinite(anchor_thenar))
            or not np.all(np.isfinite(anchor_position))):
        raise ValueError("Hand anchors must be finite 2-vectors and robot pose a finite 3-vector.")
    image_delta = (thenar - anchor_thenar) / 0.25
    R_bc = _rotation(camera_to_base, "Camera-to-base rotation")
    camera_offset = np.array((image_delta[0], image_delta[1], depth_fraction)) * motion_span
    offset = R_bc @ camera_offset
    radius = float(np.linalg.norm(offset))
    if radius > motion_span:
        offset *= motion_span / radius

    R_now = _rotation(hand_rotation, "Hand rotation")
    R_anchor = _rotation(anchor_hand_rotation, "Anchor hand rotation")
    R_robot_anchor = _rotation(anchor_robot_rotation, "Anchor robot rotation")
    camera_delta = R_now @ R_anchor.T
    base_delta = R_bc @ camera_delta @ R_bc.T
    target_rotation = base_delta @ R_robot_anchor
    position = anchor_position + offset
    gripper = normalize_gripper_gap(pinch_gap, closed_gap, open_gap)
    return pose_to_ee(position, target_rotation, gripper)

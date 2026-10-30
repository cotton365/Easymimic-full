#!/usr/bin/env python3
"""Map MediaPipe's 21 hand landmarks to an 8-D JAKA Zu 3 TCP command.

The geometry follows the supplied hand-to-end-effector reference: the thenar
midpoint controls position, a five-point SVD frame controls orientation, and
thumb/index tip separation controls the gripper. Because MediaPipe Hands only
provides relative monocular depth (not metric HaMeR/MANO points or a wrist
rotation), the webcam adapter uses a relative pseudo-3D mapping. This remains a
simulation-only prototype, not calibrated camera-to-robot control.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import numpy as np
import mujoco.viewer
from hand_to_ee import (
    blend_rotation,
    ee_to_pose,
    hand_frame_and_metrics,
    normalize_gripper_gap,
    relative_hand_to_ee8,
    rotmat_to_rotvec,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "jaka_zu3_mujoco.xml"
JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 7))
ACTUATOR_NAMES = tuple(f"actuator_joint_{i}" for i in range(1, 8))
TCP_SITE = "tcp"

# Radius of the allowed TCP target workspace; IK still enforces reachability.
MAX_IMAGE_PLANE_SPAN_M = 0.20
# The palm-size cue is relative only: this reaches full Y travel at a 20% change.
DEPTH_DEADZONE_RATIO = 0.025
DEPTH_FULL_SCALE_RATIO = 0.20
MAX_JOINT_SPEED_RAD_S = 0.45
CAMERA_LOSS_SECONDS = 0.45
GRIPPER_CLOSED_RAD = 0.0
GRIPPER_OPEN_RAD = 0.9
MAX_GRIPPER_SPEED_RAD_S = 1.5
CALIBRATION_MIN_GAP_M = 0.015
CALIBRATION_MIN_GAP_RELATIVE = 0.02
IK_POSITION_TOLERANCE_M = 0.0005
IK_ROTATION_TOLERANCE_RAD = 0.005

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def _joint_addresses(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    qpos = []
    dof = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Model is missing {name}.")
        qpos.append(int(model.jnt_qposadr[joint_id]))
        dof.append(int(model.jnt_dofadr[joint_id]))
    return qpos, dof


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    seed: np.ndarray,
) -> tuple[np.ndarray | None, float, float]:
    """Damped least-squares IK with the model's arm-joint limits enforced."""
    target_position = np.asarray(target_position, dtype=np.float64)
    target_rotation = np.asarray(target_rotation, dtype=np.float64)
    seed = np.asarray(seed, dtype=np.float64)
    if (target_position.shape != (3,) or seed.shape != (6,)
            or not np.isfinite(target_position).all() or not np.isfinite(seed).all()):
        raise ValueError("IK needs a finite target position (3,) and arm seed (6,).")
    rotmat_to_rotvec(target_rotation)  # Reject malformed rotations before modifying scratch data.
    qpos_ids, dof_ids = _joint_addresses(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    if site_id < 0:
        raise RuntimeError(f"Model is missing the {TCP_SITE} site.")

    original_qpos = data.qpos.copy()
    q = np.asarray(seed, dtype=np.float64).copy()
    lower = np.array([model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)] [0] for name in JOINT_NAMES])
    upper = np.array([model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)] [1] for name in JOINT_NAMES])
    q = np.clip(q, lower, upper)
    jac_position = np.zeros((3, model.nv), dtype=np.float64)
    jac_rotation = np.zeros((3, model.nv), dtype=np.float64)
    orientation_scale = 0.15  # metres per radian in the weighted pose error

    try:
        for _ in range(150):
            data.qpos[qpos_ids] = q
            mujoco.mj_forward(model, data)
            current_position = data.site_xpos[site_id].copy()
            current_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
            position_error = np.asarray(target_position) - current_position
            rotation_error = rotmat_to_rotvec(target_rotation @ current_rotation.T)
            position_norm = float(np.linalg.norm(position_error))
            rotation_norm = float(np.linalg.norm(rotation_error))
            if (
                position_norm <= IK_POSITION_TOLERANCE_M
                and rotation_norm <= IK_ROTATION_TOLERANCE_RAD
            ):
                return q, position_norm, rotation_norm

            mujoco.mj_jacSite(model, data, jac_position, jac_rotation, site_id)
            jacobian = np.vstack((
                jac_position[:, dof_ids],
                orientation_scale * jac_rotation[:, dof_ids],
            ))
            error = np.concatenate((position_error, orientation_scale * rotation_error))
            regularized = jacobian @ jacobian.T + (0.025 ** 2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(regularized, error)
            delta = np.clip(delta, -0.10, 0.10)
            q = np.clip(q + delta, lower, upper)

        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        position_norm = float(np.linalg.norm(data.site_xpos[site_id] - target_position))
        rotation_error = rotmat_to_rotvec(
            target_rotation @ data.site_xmat[site_id].reshape(3, 3).T
        )
        rotation_norm = float(np.linalg.norm(rotation_error))
        if (
            position_norm <= IK_POSITION_TOLERANCE_M
            and rotation_norm <= IK_ROTATION_TOLERANCE_RAD
        ):
            return q, position_norm, rotation_norm
        return None, position_norm, rotation_norm
    finally:
        data.qpos[:] = original_qpos
        mujoco.mj_forward(model, data)


def ee8_to_joint_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee8: np.ndarray,
    seed: np.ndarray,
) -> tuple[np.ndarray | None, float, float]:
    """Shared EE8 consumer for camera teleop and metric input sequences.

    Return six arm angles plus the gripper angle, all in radians. Invalid or
    unreachable targets return None; no actuator commands are written here.
    IK residuals describe model kinematics, not physical tracking accuracy.
    """
    try:
        position, rotation, gripper = ee_to_pose(ee8)
    except ValueError:
        return None, float("nan"), float("nan")
    arm, position_error, rotation_error = solve_ik(model, data, position, rotation, seed)
    if arm is None:
        return None, position_error, rotation_error
    gripper_angle = GRIPPER_CLOSED_RAD + gripper * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
    return np.concatenate((arm, (gripper_angle,))), position_error, rotation_error


def make_initial_scene(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Match the project's safe raised pose and tabletop object placement."""
    # A small joint_5 offset avoids the straight-wrist singularity in the
    # previous preview pose and gives XY teleoperation more stable IK.
    home = (0.0, 0.5, 0.8, 0.0, 0.5, 0.0, 0.0)
    for i, name in enumerate(tuple(f"joint_{index}" for index in range(1, 8))):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator_{name}"
        )
        data.qpos[model.jnt_qposadr[joint_id]] = home[i]
        data.ctrl[actuator_id] = home[i]

    for body_name, position in (
        ("body_obj_mug_5", (0.34, 0.10, 0.85)),
        ("body_obj_plate_11", (0.62, -0.12, 0.807)),
    ):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        joint_id = next(
            (
                i for i in range(model.njnt)
                if model.jnt_bodyid[i] == body_id
                and model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
            ),
            None,
        )
        if joint_id is not None:
            address = int(model.jnt_qposadr[joint_id])
            data.qpos[address:address + 3] = position
            data.qpos[address + 3:address + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)


class CameraHandTracker:
    """Read and process the webcam off the physics thread."""

    def __init__(self, camera_index: int):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "Camera tracking needs the optional dependencies. Activate the "
                "dedicated camera environment and run: "
                "python -m pip install -r scripts/requirements-hand-teleop.txt"
            ) from error

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                f"MediaPipe {mp.__version__} does not provide the legacy "
                "solutions.hands API used by this script. Install the pinned "
                "camera requirements in scripts/requirements-hand-teleop.txt."
            )

        self.cv2 = cv2
        self.mp = mp
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                f"Could not open camera {camera_index}. Close other camera apps "
                "or retry with --camera 1."
            )
        self.lock = threading.Lock()
        self.latest = {
            "frame": None, "palm": None, "thenar": None, "hand_rotation": None,
            "pinch_gap": None, "palm_scale": None, "plane_residual": None,
            "gap_unit": "relative", "geometry_source": "none",
            "landmarks": None, "seen": False, "time": 0.0, "error": None,
        }
        self.previous_hand_rotation = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="hand-camera", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            hands_module = self.mp.solutions.hands
            with hands_module.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.60,
                min_tracking_confidence=0.55,
            ) as hands:
                while not self.stop_event.is_set():
                    ok, raw = self.cap.read()
                    if not ok:
                        time.sleep(0.02)
                        continue
                    frame = self.cv2.flip(raw, 1)
                    rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)
                    palm = thenar = hand_rotation = pinch_gap = None
                    palm_scale = plane_residual = landmarks = None
                    gap_unit = "relative"
                    geometry_source = "MediaPipe image 3D proxy"
                    if result.multi_hand_landmarks:
                        points = result.multi_hand_landmarks[0].landmark
                        landmarks = np.asarray(
                            [(point.x, point.y, point.z) for point in points],
                            dtype=np.float64,
                        )
                        world_landmarks = None
                        if result.multi_hand_world_landmarks:
                            world_landmarks = np.asarray(
                                [(point.x, point.y, point.z)
                                 for point in result.multi_hand_world_landmarks[0].landmark],
                                dtype=np.float64,
                            )
                        if world_landmarks is not None:
                            gap_unit = "m"
                            geometry_source = "MediaPipe relative world landmarks"
                        try:
                            (thenar, hand_rotation, pinch_gap, palm_scale,
                             plane_residual) = hand_frame_and_metrics(
                                landmarks,
                                frame.shape[1],
                                frame.shape[0],
                                world_landmarks=world_landmarks,
                                normal_reference=(
                                    None if self.previous_hand_rotation is None
                                    else self.previous_hand_rotation[:, 2]
                                ),
                            )
                            self.previous_hand_rotation = hand_rotation
                            palm = (float(thenar[0]), float(thenar[1]))
                        except (ValueError, np.linalg.LinAlgError):
                            # Render the hand but hold controls on degenerate frames.
                            palm = None
                    with self.lock:
                        self.latest = {
                            "frame": frame,
                            "palm": palm,
                            "thenar": thenar,
                            "hand_rotation": hand_rotation,
                            "pinch_gap": pinch_gap,
                            "gap_unit": gap_unit,
                            "palm_scale": palm_scale,
                            "plane_residual": plane_residual,
                            "geometry_source": geometry_source,
                            "landmarks": landmarks,
                            "seen": thenar is not None,
                            "time": time.monotonic(),
                            "error": None,
                        }
        except Exception as error:
            with self.lock:
                self.latest["error"] = f"{type(error).__name__}: {error}"
                self.latest["time"] = time.monotonic()

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.latest)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.cap.release()


def run_ik_check() -> int:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    make_initial_scene(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    start_q = np.array([
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )]] for name in JOINT_NAMES
    ])
    start_position = data.site_xpos[site_id].copy()
    start_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    cases = (
        (0.0, 0.0, 0.0),
        (0.02, 0.0, 0.0), (-0.02, 0.0, 0.0),
        (0.0, 0.0, 0.02), (0.0, 0.0, -0.02),
        (0.04, 0.0, 0.0), (-0.04, 0.0, 0.0),
        (0.0, 0.0, 0.04), (0.0, 0.0, -0.04),
        (0.0, 0.02, 0.0), (0.0, -0.02, 0.0),
        (0.0, 0.04, 0.0), (0.0, -0.04, 0.0),
    )
    failed = False
    for dx, dy, dz in cases:
        goal = start_position + np.array((dx, dy, dz))
        q, pos_error, rot_error = solve_ik(model, data, goal, start_rotation, start_q)
        ok = q is not None
        print(
            f"target dX={dx:+.3f} dY={dy:+.3f} dZ={dz:+.3f} m: "
            f"{'PASS' if ok else 'FAIL'}, position residual={pos_error:.4f} m, "
            f"orientation residual={rot_error:.4f} rad"
        )
        failed |= not ok
    return 1 if failed else 0


def _draw_camera_overlay(
    frame: np.ndarray,
    cv2,
    text: list[str],
    palm,
    landmarks,
    gripper_opening: float | None,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    cv2.rectangle(output, (int(width * 0.25), int(height * 0.25)),
                  (int(width * 0.75), int(height * 0.75)), (80, 200, 255), 1)
    if palm is not None:
        point = (int(palm[0] * width), int(palm[1] * height))
        cv2.circle(output, point, 9, (0, 255, 80), -1)
    if landmarks is not None:
        pixels = np.column_stack((landmarks[:, 0] * width, landmarks[:, 1] * height))
        pixels = np.rint(pixels).astype(int)
        for start, end in HAND_CONNECTIONS:
            cv2.line(output, tuple(pixels[start]), tuple(pixels[end]), (0, 220, 255), 2, cv2.LINE_AA)
        for index, point in enumerate(pixels):
            color = (0, 130, 255) if index in (4, 8, 12, 16, 20) else (0, 255, 80)
            cv2.circle(output, tuple(point), 4, color, -1, cv2.LINE_AA)
    if gripper_opening is not None:
        bar_left, bar_top = 12, 30 + 25 * len(text)
        bar_right, bar_bottom = min(width - 12, bar_left + 220), bar_top + 15
        cv2.rectangle(output, (bar_left, bar_top), (bar_right, bar_bottom), (60, 60, 60), -1)
        cv2.rectangle(
            output, (bar_left, bar_top),
            (bar_left + int((bar_right - bar_left) * np.clip(gripper_opening, 0.0, 1.0)), bar_bottom),
            (0, 190, 0) if gripper_opening >= 0.65 else (0, 140, 255), -1,
        )
    for index, line in enumerate(text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_width = cv2.getTextSize(line, font, 0.58, 1)[0][0]
        font_scale = 0.58 * min(1.0, (width - 24) / max(text_width, 1))
        cv2.putText(output, line, (12, 25 + 25 * index), font,
                    font_scale, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(output, line, (12, 25 + 25 * index), font,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def run_teleop(camera_index: int, motion_span: float) -> None:
    if not 0.01 <= motion_span <= MAX_IMAGE_PLANE_SPAN_M:
        raise ValueError(
            f"--motion-span must be between 0.01 and {MAX_IMAGE_PLANE_SPAN_M:.2f} m."
        )
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    # IK runs on its own scratch state so it cannot race the passive viewer's
    # physics state while evaluating candidate arm configurations.
    ik_data = mujoco.MjData(model)
    make_initial_scene(model, data)
    qpos_ids, _ = _joint_addresses(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_7")
    if gripper_joint_id < 0:
        raise RuntimeError("Model is missing joint_7.")
    gripper_qpos_id = int(model.jnt_qposadr[gripper_joint_id])
    ctrl_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ACTUATOR_NAMES
    ]
    camera = CameraHandTracker(camera_index)
    target_q = data.qpos[qpos_ids].copy()
    ik_target = target_q.copy()
    armed = False
    anchor_thenar = None
    anchor_hand_rotation = None
    anchor_palm_scale = None
    anchor_robot_position = None
    anchor_robot_rotation = None
    anchor_arm_q = None
    last_hand_seen = 0.0
    last_ik_time = 0.0
    last_snapshot_time = 0.0
    last_metric_frame_time = 0.0
    smoothed_thenar = None
    smoothed_hand_rotation = None
    smoothed_palm_scale = None
    pinch_gap_samples: deque[float] = deque(maxlen=5)
    filtered_pinch_gap = None
    closed_gap = None
    open_gap = None
    calibration_unit = None
    calibration_source = None
    ee8_command = None
    gripper_target = float(data.ctrl[ctrl_ids[6]])
    gripper_command = gripper_target
    status = "CALIBRATE: pinch thumb/index closed + C; widely open + O"
    cv2 = camera.cv2
    window_name = "JAKA Zu 3 hand camera (simulation only)"
    print("Simulation-only hand teleoperation. The camera does not command a real robot.")
    print("Keys: close thumb/index + C; widely open + O; SPACE arm/disarm; R re-centre; Q quit.")
    print("21 landmarks -> thenar XYZ (relative), SVD hand orientation, tip-gap gripper -> EE8 -> IK.")
    print("MediaPipe world landmarks are relative to the hand; webcam depth and camera extrinsics are uncalibrated.")

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=True, show_right_ui=False
        ) as viewer:
            viewer.cam.lookat[:] = [0.48, 0.0, 0.93]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            sim_time = time.monotonic()
            last_sync = sim_time
            last_tick = sim_time
            while viewer.is_running():
                now = time.monotonic()
                snapshot = camera.snapshot()
                if snapshot["error"]:
                    raise RuntimeError(f"Camera tracking thread failed: {snapshot['error']}")
                frame = snapshot["frame"]
                hand_fresh = (
                    snapshot["seen"] and now - snapshot["time"] <= CAMERA_LOSS_SECONDS
                )
                if hand_fresh and snapshot["time"] != last_metric_frame_time:
                    last_hand_seen = snapshot["time"]
                    last_metric_frame_time = snapshot["time"]
                    raw_thenar = np.asarray(snapshot["thenar"], dtype=np.float64)
                    if smoothed_thenar is None:
                        smoothed_thenar = raw_thenar
                    else:
                        smoothed_thenar = 0.72 * smoothed_thenar + 0.28 * raw_thenar

                    raw_hand_rotation = np.asarray(snapshot["hand_rotation"], dtype=np.float64)
                    smoothed_hand_rotation = blend_rotation(
                        smoothed_hand_rotation, raw_hand_rotation, alpha=0.30
                    )
                    raw_scale = snapshot["palm_scale"]
                    if raw_scale is not None:
                        if smoothed_palm_scale is None:
                            smoothed_palm_scale = float(raw_scale)
                        else:
                            smoothed_palm_scale = (
                                0.82 * smoothed_palm_scale + 0.18 * float(raw_scale)
                            )

                    gap = snapshot["pinch_gap"]
                    if gap is not None and np.isfinite(gap):
                        pinch_gap_samples.append(float(gap))
                        filtered_pinch_gap = float(np.median(pinch_gap_samples))
                    if (armed and (
                            snapshot["gap_unit"] != calibration_unit
                            or snapshot["geometry_source"] != calibration_source)):
                        armed = False
                        gripper_target = gripper_command
                        with viewer.lock():
                            target_q = data.qpos[qpos_ids].copy()
                        ik_target = target_q.copy()
                        status = "LANDMARK SOURCE CHANGED: disarmed; re-calibrate before arming"
                    if (armed and filtered_pinch_gap is not None
                            and closed_gap is not None and open_gap is not None
                            and snapshot["gap_unit"] == calibration_unit
                            and snapshot["geometry_source"] == calibration_source):
                        g = normalize_gripper_gap(filtered_pinch_gap, closed_gap, open_gap)
                        gripper_target = GRIPPER_CLOSED_RAD + g * (
                            GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
                        )
                elif not armed and now - last_hand_seen > CAMERA_LOSS_SECONDS:
                    smoothed_thenar = None
                    smoothed_hand_rotation = None
                    smoothed_palm_scale = None
                    filtered_pinch_gap = None
                    pinch_gap_samples.clear()

                # If tracking disappears, immediately stop updating targets;
                # after a short grace period, hold the measured joint pose.
                if armed and now - last_hand_seen > CAMERA_LOSS_SECONDS:
                    armed = False
                    gripper_target = gripper_command
                    with viewer.lock():
                        target_q = data.qpos[qpos_ids].copy()
                    ik_target = target_q.copy()
                    anchor_thenar = anchor_hand_rotation = None
                    anchor_palm_scale = None
                    anchor_robot_position = anchor_robot_rotation = None
                    anchor_arm_q = None
                    smoothed_thenar = None
                    smoothed_hand_rotation = None
                    smoothed_palm_scale = None
                    filtered_pinch_gap = None
                    pinch_gap_samples.clear()
                    ee8_command = None
                    status = "HAND LOST: disarmed and holding current pose"

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if armed:
                        armed = False
                        gripper_target = gripper_command
                        with viewer.lock():
                            target_q = data.qpos[qpos_ids].copy()
                        ik_target = target_q.copy()
                        anchor_thenar = anchor_hand_rotation = None
                        anchor_palm_scale = None
                        anchor_robot_position = anchor_robot_rotation = None
                        anchor_arm_q = None
                        status = "DISARMED: holding current pose"
                    elif hand_fresh and smoothed_thenar is not None:
                        if (closed_gap is None or open_gap is None
                                or snapshot["gap_unit"] != calibration_unit
                                or snapshot["geometry_source"] != calibration_source):
                            status = "CALIBRATE FIRST: closed thumb/index=C, spread=O"
                        else:
                            anchor_thenar = smoothed_thenar.copy()
                            anchor_hand_rotation = smoothed_hand_rotation.copy()
                            anchor_palm_scale = smoothed_palm_scale
                            with viewer.lock():
                                anchor_robot_position = data.site_xpos[site_id].copy()
                                anchor_robot_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
                                target_q = data.qpos[qpos_ids].copy()
                            anchor_arm_q = target_q.copy()
                            ik_target = target_q.copy()
                            armed = True
                            last_hand_seen = now
                            status = "ARMED: move slowly; SPACE to hold"
                    else:
                        status = "Show your hand to the camera before arming"
                elif key in (ord("r"), ord("R")) and armed and hand_fresh and smoothed_thenar is not None:
                    anchor_thenar = smoothed_thenar.copy()
                    anchor_hand_rotation = smoothed_hand_rotation.copy()
                    anchor_palm_scale = smoothed_palm_scale
                    with viewer.lock():
                        anchor_robot_position = data.site_xpos[site_id].copy()
                        anchor_robot_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
                        target_q = data.qpos[qpos_ids].copy()
                    anchor_arm_q = target_q.copy()
                    ik_target = target_q.copy()
                    status = "Re-centred at current pose"
                elif key in (ord("c"), ord("C"), ord("o"), ord("O")):
                    if armed:
                        status = "DISARM before calibrating the hand"
                    elif not hand_fresh or len(pinch_gap_samples) < 3 or filtered_pinch_gap is None:
                        status = "Hold the pose still until the landmarks settle"
                    else:
                        sample = filtered_pinch_gap
                        min_gap_separation = (
                            CALIBRATION_MIN_GAP_M
                            if snapshot.get("gap_unit") == "m"
                            else CALIBRATION_MIN_GAP_RELATIVE
                        )
                        if key in (ord("c"), ord("C")):
                            if calibration_unit is not None and snapshot["gap_unit"] != calibration_unit:
                                open_gap = None
                            if calibration_source is not None and snapshot["geometry_source"] != calibration_source:
                                open_gap = None
                            closed_gap = sample
                            calibration_unit = snapshot["gap_unit"]
                            calibration_source = snapshot["geometry_source"]
                            if open_gap is not None and open_gap - sample < min_gap_separation:
                                open_gap = None
                                status = "Closed pinch saved; open sample too close, press O again"
                            else:
                                status = (
                                    f"Closed tip gap saved ({sample:.3f} {snapshot['gap_unit']}); "
                                    "now spread thumb/index and press O"
                                )
                        else:
                            if closed_gap is None:
                                status = "Record a closed thumb/index pinch first: close and press C"
                            elif (snapshot["gap_unit"] != calibration_unit
                                  or snapshot["geometry_source"] != calibration_source):
                                open_gap = None
                                status = "Landmark distance source changed; record closed pinch again with C"
                            elif sample - closed_gap < min_gap_separation:
                                status = (
                                    f"Open tip gap too close ({sample:.3f} {snapshot['gap_unit']}); "
                                    "spread thumb/index farther and press O"
                                )
                            else:
                                open_gap = sample
                                status = (
                                    f"Calibration ready: closed {closed_gap:.3f}, "
                                    f"open {open_gap:.3f} {snapshot['gap_unit']}"
                                )

                if (armed and hand_fresh and smoothed_thenar is not None
                        and smoothed_hand_rotation is not None
                        and smoothed_palm_scale is not None
                        and filtered_pinch_gap is not None
                        and snapshot["gap_unit"] == calibration_unit
                        and snapshot["geometry_source"] == calibration_source
                        and now - last_ik_time >= 0.04):
                    last_ik_time = now
                    ee8_command = relative_hand_to_ee8(
                        thenar=smoothed_thenar,
                        hand_rotation=smoothed_hand_rotation,
                        pinch_gap=filtered_pinch_gap,
                        palm_scale=smoothed_palm_scale,
                        anchor_thenar=anchor_thenar,
                        anchor_hand_rotation=anchor_hand_rotation,
                        anchor_palm_scale=anchor_palm_scale,
                        anchor_robot_position=anchor_robot_position,
                        anchor_robot_rotation=anchor_robot_rotation,
                        closed_gap=closed_gap,
                        open_gap=open_gap,
                        motion_span=motion_span,
                        depth_deadzone=DEPTH_DEADZONE_RATIO,
                        depth_full_scale=DEPTH_FULL_SCALE_RATIO,
                    )
                    goal = ee8_command[:3]
                    candidate, pos_error, rot_error = ee8_to_joint_targets(
                        model, ik_data, ee8_command, target_q
                    )
                    if candidate is None:
                        # Stop ramping toward the previous solution when the
                        # new hand target is unreachable.
                        ik_target = target_q.copy()
                        status = (
                            f"IK HOLD: target unreachable ({pos_error * 1000:.1f} mm, "
                            f"{math.degrees(rot_error):.1f} deg residual)"
                        )
                    else:
                        ik_target = candidate[:6]
                        gripper_target = float(candidate[6])
                        status = (
                            f"ARMED: TCP target X={goal[0]:.3f} Y={goal[1]:.3f} "
                            f"Z={goal[2]:.3f} m | grip={ee8_command[6]:.2f}"
                        )

                # Rate-limit all arm targets in joint space. The XML's position
                # actuators then track these gently changing references.
                dt = min(max(now - last_tick, 0.0), 0.02)
                last_tick = now
                max_step = MAX_JOINT_SPEED_RAD_S * dt
                target_q += np.clip(ik_target - target_q, -max_step, max_step)
                gripper_delta = gripper_target - gripper_command
                gripper_command += float(np.clip(
                    gripper_delta,
                    -MAX_GRIPPER_SPEED_RAD_S * dt,
                    MAX_GRIPPER_SPEED_RAD_S * dt,
                ))
                with viewer.lock():
                    data.ctrl[ctrl_ids[:6]] = target_q
                    data.ctrl[ctrl_ids[6]] = gripper_command
                    mujoco.mj_step(model, data)

                if now - last_sync >= 1.0 / 60.0:
                    viewer.sync()
                    last_sync = now

                if frame is not None and now - last_snapshot_time >= 1.0 / 30.0:
                    display_openness = ee8_command[6] if armed and hand_fresh and ee8_command is not None else None
                    with viewer.lock():
                        measured_gripper = float(data.qpos[gripper_qpos_id])
                    lines = [
                        "Close thumb/index=C | Spread thumb/index=O | SPACE arm | R centre | Q quit",
                        status,
                        (
                        f"Tip gap: {filtered_pinch_gap:.3f} {snapshot['gap_unit']} | g={display_openness:.2f} | gripper {measured_gripper:.2f}/{gripper_target:.2f} rad"
                            if display_openness is not None
                            else f"Tip gap: {filtered_pinch_gap:.3f} {snapshot['gap_unit']} | gripper {measured_gripper:.2f} rad" if hand_fresh and filtered_pinch_gap is not None
                            else f"Hand: not detected | Gripper: {measured_gripper:.2f} rad"
                        ),
                        f"Cal closed={closed_gap:.3f} open={open_gap:.3f} {snapshot['gap_unit']} | workspace radius {motion_span:.2f} m"
                        if closed_gap is not None and open_gap is not None
                        else f"Calibrate fingertip gap: closed (C), then spread (O) | radius {motion_span:.2f} m",
                    ]
                    if (calibration_unit is not None and (
                            snapshot["gap_unit"] != calibration_unit
                            or snapshot["geometry_source"] != calibration_source)):
                        lines.append("Pinch calibration paused: landmark distance source changed")
                    if armed and anchor_palm_scale and smoothed_palm_scale and hand_fresh:
                        lines.append(
                            f"Relative palm size: {smoothed_palm_scale / anchor_palm_scale:.2f}x"
                            " | closer -> +Y, farther -> -Y (uncalibrated)"
                        )
                    if armed and ee8_command is not None:
                        lines.append(
                            "EE8 p[m]=[" + " ".join(f"{v:+.3f}" for v in ee8_command[:3])
                            + "] rv[rad]=[" + " ".join(f"{v:+.2f}" for v in ee8_command[3:6])
                            + f"] g={ee8_command[6]:.2f} r={ee8_command[7]:.1f}"
                        )
                        lines.append(
                            f"Plane residual={snapshot['plane_residual']:.4f} ({snapshot['geometry_source']})"
                            if snapshot["plane_residual"] is not None
                            else "Hand plane: invalid"
                        )
                    if armed and anchor_arm_q is not None:
                        with viewer.lock():
                            joint_delta_deg = np.degrees(data.qpos[qpos_ids] - anchor_arm_q)
                        lines.extend((
                            "Arm joint change from SPACE (deg): " + " ".join(
                                f"J{i + 1} {joint_delta_deg[i]:+.1f}" for i in range(3)
                            ),
                            "                             " + " ".join(
                                f"J{i + 4} {joint_delta_deg[i + 3]:+.1f}" for i in range(3)
                            ),
                        ))
                    else:
                        lines.append("Show full hand, then calibrate and press SPACE; TCP uses limited IK")
                    shown = _draw_camera_overlay(
                        frame, cv2, lines,
                        tuple(smoothed_thenar) if hand_fresh and smoothed_thenar is not None else None,
                        snapshot["landmarks"] if hand_fresh else None,
                        display_openness,
                    )
                    cv2.imshow(window_name, shown)
                    last_snapshot_time = now
                elapsed = time.monotonic() - now
                time.sleep(max(0.0, model.opt.timestep - elapsed))
    finally:
        camera.close()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: laptop camera 0).")
    parser.add_argument("--motion-span", "--xy-span", dest="motion_span", type=float,
                        default=MAX_IMAGE_PLANE_SPAN_M,
                        help=("Maximum radial XYZ offset from the arm pose at arming, "
                              f"in metres (0.01..{MAX_IMAGE_PLANE_SPAN_M:.2f})."))
    parser.add_argument("--check-ik", action="store_true", help="Run a no-camera local IK reachability check.")
    args = parser.parse_args()
    if args.check_ik:
        return run_ik_check()
    run_teleop(args.camera, args.motion_span)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

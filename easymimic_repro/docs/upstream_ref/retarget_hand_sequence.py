#!/usr/bin/env python3
"""Offline metric hand observations -> EE8 -> JAKA's seven joint targets.

This evaluates kinematics; it does not open the webcam or execute a trajectory.
Input NPZ: kp (N,21,3) camera-frame metres, R_wrist (N,3,3), T_base_cam (4,4).
MediaPipe's hand-relative landmarks alone do not satisfy this input contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from hand_camera_teleop import MODEL_PATH, ee8_to_joint_targets, make_initial_scene
from hand_to_ee import D_MAX, D_MIN, hand_to_ee, normalize_gripper_gap


def retarget_sequence(
    model: mujoco.MjModel,
    kp: np.ndarray,
    R_wrist: np.ndarray,
    T_base_cam: np.ndarray,
    d_min: float = D_MIN,
    d_max: float = D_MAX,
) -> dict[str, np.ndarray]:
    """Map frames in order, using the previous accepted solution as IK seed.

    Every rejected frame holds the previous seven joint targets. ``accepted``
    and ``status`` must be inspected; held values are not successful solutions.
    The first frame starts from the simulator's raised initial pose.
    """
    points = np.asarray(kp, dtype=np.float64)
    wrists = np.asarray(R_wrist, dtype=np.float64)
    transform = np.asarray(T_base_cam, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (21, 3) or len(points) == 0:
        raise ValueError("kp must have shape (N, 21, 3) with N > 0.")
    if wrists.shape != (len(points), 3, 3):
        raise ValueError("R_wrist must have shape (N, 3, 3), matching kp.")
    if transform.shape != (4, 4):
        raise ValueError("T_base_cam must have shape (4, 4).")
    normalize_gripper_gap(d_min, d_min, d_max)
    data = mujoco.MjData(model)
    make_initial_scene(model, data)
    addresses = [int(model.jnt_qposadr[model.joint(f"joint_{i}").id]) for i in range(1, 8)]
    previous = data.qpos[addresses].copy()
    count = len(points)
    output = {
        "ee8": np.full((count, 8), np.nan),
        "joint_targets": np.empty((count, 7)),
        "accepted": np.zeros(count, dtype=bool),
        "position_error_m": np.full(count, np.nan),
        "rotation_error_rad": np.full(count, np.nan),
        "status": np.full(count, "", dtype="U160"),
    }
    for index in range(count):
        try:
            ee8, info = hand_to_ee(points[index], wrists[index], transform, d_min, d_max)
        except ValueError as error:
            output["status"][index] = f"invalid_geometry_or_transform: {error}"
        else:
            output["ee8"][index] = ee8
            if not info["valid"]:
                output["status"][index] = info["reason"]
            else:
                joints, pos_error, rot_error = ee8_to_joint_targets(model, data, ee8, previous[:6])
                output["position_error_m"][index] = pos_error
                output["rotation_error_rad"][index] = rot_error
                if joints is None:
                    output["status"][index] = "ik_unreachable_or_not_converged"
                else:
                    previous = joints
                    output["accepted"][index] = True
                    output["status"][index] = "ok"
        output["joint_targets"][index] = previous
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Metric observations NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="New output NPZ (will not overwrite).")
    parser.add_argument("--closed-gap", type=float, default=D_MIN, help="Closed fingertip gap, metres.")
    parser.add_argument("--open-gap", type=float, default=D_MAX, help="Open fingertip gap, metres.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    with np.load(args.input, allow_pickle=False) as observations:
        missing = {"kp", "R_wrist", "T_base_cam"} - set(observations.files)
        if missing:
            parser.error(f"Missing input arrays: {', '.join(sorted(missing))}")
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        output = retarget_sequence(model, observations["kp"], observations["R_wrist"],
                                   observations["T_base_cam"], args.closed_gap, args.open_gap)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **output)
    accepted = int(output["accepted"].sum())
    print(f"Accepted {accepted}/{len(output['accepted'])} frames; saved {args.output.resolve()}")
    print("Rejected frames hold the previous targets. This file is not a timed or collision-checked trajectory.")
    return 0 if accepted == len(output["accepted"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Headless checks of metric observations -> EE8 -> seven joints -> MuJoCo FK.

Ground truth targets come from known robot joint configurations. These are
synthetic kinematic checks, not tests of live vision or physical tracking.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

import mujoco
import numpy as np

from check_hand_to_ee import synthetic_hand
from hand_camera_teleop import (
    IK_POSITION_TOLERANCE_M, IK_ROTATION_TOLERANCE_RAD, MODEL_PATH,
    ee8_to_joint_targets, make_initial_scene,
)
from retarget_hand_sequence import retarget_sequence


def make_cases(model: mujoco.MjModel):
    data = mujoco.MjData(model)
    make_initial_scene(model, data)
    arm_addresses = [model.jnt_qposadr[model.joint(f"joint_{i}").id] for i in range(1, 7)]
    site = model.site("tcp").id
    start = data.qpos[arm_addresses].copy()
    transform = np.eye(4)
    transform[:3, :3] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    transform[:3, 3] = (.4, -.2, .8)
    _, canonical = synthetic_hand()
    anchor = np.array((0., .025, 0.))
    points, wrists, positions, rotations = [], [], [], []
    offsets = (
        (0, 0, 0, 0, 0, 0),
        (.08, -.06, .05, 0, .02, .03),
        (-.08, .04, -.03, .05, -.06, -.04),
        (.03, .08, -.08, -.03, .05, .07),
        (0, -.05, .08, .08, -.08, -.10),
    )
    for index, offset in enumerate(offsets):
        data.qpos[arm_addresses] = start + offset
        mujoco.mj_forward(model, data)
        position = data.site_xpos[site].copy()
        rotation = data.site_xmat[site].reshape(3, 3).copy()
        position_cam = transform[:3, :3].T @ (position - transform[:3, 3])
        rotation_cam = transform[:3, :3].T @ rotation
        hand = canonical.copy()
        hand[4] = hand[8] + (0, .02 + .08 * index / 4, 0)
        points.append((hand - anchor) @ rotation_cam.T + position_cam)
        wrists.append(rotation_cam)
        positions.append(position)
        rotations.append(rotation)
    return np.array(points), np.array(wrists), transform, positions, rotations


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    kp, wrists, transform, positions, rotations = make_cases(model)
    result = retarget_sequence(model, kp, wrists, transform)
    assert result["accepted"].all(), result["status"]
    np.testing.assert_allclose(result["joint_targets"][:, 6], np.linspace(0, .9, 5), atol=1e-12)
    addresses = [model.jnt_qposadr[model.joint(f"joint_{i}").id] for i in range(1, 8)]
    limits = np.array([model.joint(f"joint_{i}").range for i in range(1, 8)])
    assert np.all(result["joint_targets"] >= limits[:, 0])
    assert np.all(result["joint_targets"] <= limits[:, 1])
    data = mujoco.MjData(model)
    make_initial_scene(model, data)
    site = model.site("tcp").id
    for index, joints in enumerate(result["joint_targets"]):
        data.qpos[addresses] = joints
        mujoco.mj_forward(model, data)
        # Independent FK evaluation, including moving the seventh joint.
        position_error = np.linalg.norm(data.site_xpos[site] - positions[index])
        delta = rotations[index] @ data.site_xmat[site].reshape(3, 3).T
        angle_error = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
        assert position_error <= IK_POSITION_TOLERANCE_M, position_error
        assert angle_error <= IK_ROTATION_TOLERANCE_RAD, angle_error
        print(f"PASS pipeline {index + 1}/5: FK error {position_error * 1000:.4f} mm, "
              f"{np.degrees(angle_error):.4f} deg; joint_7={joints[6]:.3f} rad")

    # Valid -> NaN in unused landmark -> degenerate -> unreachable -> valid.
    # Held targets must recover using the last accepted solution, never NaNs.
    broken = kp.copy()
    broken[1, 12, 0] = np.nan
    broken[2, 6] = broken[2, 5]
    broken[3] += (5., 0, 0)
    held = retarget_sequence(model, broken, wrists, transform)
    np.testing.assert_array_equal(held["accepted"], (True, False, False, False, True))
    np.testing.assert_allclose(held["joint_targets"][1:4],
                               np.repeat(held["joint_targets"][:1], 3, axis=0))
    assert np.isnan(held["ee8"][1]).all()
    assert np.isfinite(held["joint_targets"]).all()
    print("PASS: invalid/degenerate/unreachable frames hold; valid input recovers")

    # The shared controller boundary must reject invalid commands without
    # mutating the simulator state, and restore scratch state after valid IK.
    qpos_before, ctrl_before = data.qpos.copy(), data.ctrl.copy()
    for command in (np.full(8, np.nan), np.zeros(7), np.array((0, 0, 0, 0, 0, 0, .5, 1))):
        candidate, _, _ = ee8_to_joint_targets(model, data, command, result["joint_targets"][0, :6])
        assert candidate is None
    candidate, _, _ = ee8_to_joint_targets(model, data, result["ee8"][0], result["joint_targets"][0, :6])
    assert candidate is not None
    np.testing.assert_array_equal(data.qpos, qpos_before)
    np.testing.assert_array_equal(data.ctrl, ctrl_before)
    print("PASS: EE8 validation and simulator state preservation")

    with tempfile.TemporaryDirectory(prefix="jaka-hand-pipeline-") as directory:
        source, target = Path(directory) / "input.npz", Path(directory) / "output.npz"
        np.savez(source, kp=kp, R_wrist=wrists, T_base_cam=transform)
        subprocess.run([sys.executable, str(Path(__file__).with_name("retarget_hand_sequence.py")),
                        "--input", str(source), "--output", str(target)], check=True)
        with np.load(target, allow_pickle=False) as saved:
            np.testing.assert_allclose(saved["joint_targets"], result["joint_targets"])
            np.testing.assert_array_equal(saved["accepted"], result["accepted"])
    print("PASS: metric sequence CLI and saved output reproduce computed joints")
    print("ALL HAND PIPELINE CHECKS PASSED (synthetic kinematics only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

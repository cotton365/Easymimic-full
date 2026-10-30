#!/usr/bin/env python3
"""Reconstructed reference A-F cases plus webcam and numerical regressions.

These are local tests based on the previously reviewed reference, not a copy
of the external demo or evidence of measured camera/robot accuracy.
"""

from __future__ import annotations

import numpy as np

from hand_to_ee import (
    CAMERA_TO_BASE,
    ee_to_pose,
    hand_frame_and_metrics,
    hand_to_ee,
    normalize_gripper_gap,
    relative_hand_to_ee8,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
)


def synthetic_hand() -> tuple[np.ndarray, np.ndarray]:
    """Return a canonical planar 21-point hand in image and relative 3-D space."""
    world = np.zeros((21, 3), dtype=np.float64)
    world[0] = (-0.06, 0.01, 0.0)
    world[1] = (-0.03, 0.02, 0.0)
    world[2] = (-0.01, 0.035, 0.0)
    world[3] = (0.0, 0.05, 0.0)
    world[4] = (0.10, 0.06, 0.0)
    world[5] = (0.0, 0.0, 0.0)
    world[6] = (0.04, 0.0, 0.0)
    world[7] = (0.075, 0.0, 0.0)
    world[8] = (0.10, 0.0, 0.0)
    world[9] = (0.0, -0.02, 0.0)
    world[10] = (0.045, -0.02, 0.0)
    world[11] = (0.075, -0.02, 0.0)
    world[12] = (0.095, -0.02, 0.0)
    world[13] = (0.0, -0.04, 0.0)
    world[14] = (0.04, -0.04, 0.0)
    world[15] = (0.07, -0.04, 0.0)
    world[16] = (0.088, -0.04, 0.0)
    world[17] = (0.0, -0.055, 0.0)
    world[18] = (0.035, -0.055, 0.0)
    world[19] = (0.06, -0.055, 0.0)
    world[20] = (0.075, -0.055, 0.0)

    # Reconstruct the same projected points through MediaPipe's normalized
    # image coordinates; use a square image to keep the pixel axes isotropic.
    image = np.zeros((21, 3), dtype=np.float64)
    image[:, 0] = 0.5 + world[:, 0] * 1000.0 / 640.0
    image[:, 1] = 0.5 + world[:, 1] * 1000.0 / 640.0
    return image, world


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_value_error(callback, text: str = "") -> None:
    try:
        callback()
    except ValueError as error:
        check(text in str(error), f"unexpected ValueError: {error}")
    else:
        raise AssertionError("expected ValueError")


def check_reference() -> None:
    _, kp = synthetic_hand()
    ee8, info = hand_to_ee(kp, np.eye(3), np.eye(4))
    check(info["valid"], "A: canonical hand must be valid")
    np.testing.assert_allclose(ee8, (0, .025, 0, 0, 0, 0, .5, 0), atol=1e-12)
    p, R, g = ee_to_pose(ee8)
    np.testing.assert_allclose(p, (0, .025, 0), atol=1e-12)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
    check(np.isclose(g, .5), "A: unpack must preserve gripper")
    print("PASS A: metric thenar, identity orientation, 60 mm gap -> g=0.5")

    transform = np.eye(4)
    transform[:3, :3] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    transform[:3, 3] = (.4, 0, .8)
    ee8, _ = hand_to_ee(kp, np.eye(3), transform)
    np.testing.assert_allclose(ee8, (.375, 0, .8, 0, 0, np.pi / 2, .5, 0), atol=1e-12)
    print("PASS B: full camera-to-base rotation AND translation")

    wrist = np.diag((1., -1., -1.))
    flipped = kp @ wrist.T
    ee8, _ = hand_to_ee(flipped, wrist, np.eye(4))
    p, R, _ = ee_to_pose(ee8)
    np.testing.assert_allclose(p, (0, -.025, 0), atol=1e-12)
    np.testing.assert_allclose(R, wrist, atol=1e-12)
    ee8, _ = hand_to_ee(flipped, np.eye(3), np.eye(4))
    np.testing.assert_allclose(ee_to_pose(ee8)[1], np.eye(3), atol=1e-12)
    print("PASS C/C2: wrist normal selects the correct side of the plane")

    degenerate = kp.copy()
    degenerate[6] = degenerate[5]
    expect_value_error(lambda: hand_to_ee(degenerate, np.eye(3), np.eye(4)),
                       "index MCP-to-PIP")
    collinear = np.zeros((21, 3))
    collinear[:, 0] = np.arange(21) * .01
    expect_value_error(lambda: hand_to_ee(collinear, np.eye(3), np.eye(4)),
                       "stable plane")
    print("PASS D: degenerate index direction / collinear hand rejected")

    for bad_value in (np.nan, np.inf, -np.inf):
        for which in range(3):
            inputs = [kp.copy(), np.eye(3), np.eye(4)]
            inputs[which][(12, 0) if which == 0 else (0, 0)] = bad_value
            command, info = hand_to_ee(*inputs)
            check(np.isnan(command).all() and info["valid"] is False,
                  "E: ANY nonfinite input must produce all NaNs and valid=False")
            expect_value_error(lambda: ee_to_pose(command), "finite")
    print("PASS E: unused landmark / wrist / transform nonfinite -> NaN EE8")

    rng = np.random.default_rng(42)
    for _ in range(200):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = rotvec_to_rotmat(axis * np.pi / 2)
        np.testing.assert_allclose(rotvec_to_rotmat(rotmat_to_rotvec(rotation)),
                                   rotation, atol=1e-12)
    # Near pi, compare matrices: opposite axis signs at exactly pi are equivalent.
    axis = np.array((-3., 1., 2.)) / np.sqrt(14)
    for angle in (0., 1e-10, 1e-6, np.pi - 1e-6, np.pi, np.pi + 1e-6):
        rotation = rotvec_to_rotmat(axis * angle)
        np.testing.assert_allclose(rotvec_to_rotmat(rotmat_to_rotvec(rotation)),
                                   rotation, atol=1e-10)
    print("PASS F: 200 random rotation round trips + zero / near-pi cases")

    # Equivariance: changing the camera frame must not change the base result.
    baseline, _ = hand_to_ee(kp, np.eye(3), transform)
    change = np.eye(4)
    change[:3, :3] = rotvec_to_rotmat(np.array((.3, -.7, .2)))
    change[:3, 3] = (.1, -.2, .3)
    changed_kp = kp @ change[:3, :3].T + change[:3, 3]
    changed, _ = hand_to_ee(changed_kp, change[:3, :3], transform @ np.linalg.inv(change))
    np.testing.assert_allclose(changed, baseline, atol=1e-12)
    for gap, expected in ((.0, 0.), (.02, 0.), (.06, .5), (.10, 1.), (.15, 1.)):
        points = kp.copy()
        points[4] = points[8] + (0, gap, 0)
        command, _ = hand_to_ee(points, np.eye(3), np.eye(4))
        check(np.isclose(command[6], expected), "gripper endpoints and clipping")
    expect_value_error(lambda: hand_to_ee(kp[:20], np.eye(3), np.eye(4)), "shape")
    expect_value_error(lambda: hand_to_ee(kp, np.diag((1, 1, -1)), np.eye(4)), "SO(3)")
    bad_transform = np.eye(4)
    bad_transform[3, 0] = 1
    expect_value_error(lambda: hand_to_ee(kp, np.eye(3), bad_transform), "homogeneous")
    expect_value_error(lambda: hand_to_ee(kp, np.eye(3), np.eye(4), .1, .02), "exceed")
    for bad_command in (np.zeros(7), np.array((0, 0, 0, 0, 0, 0, 1.1, 0)),
                        np.array((0, 0, 0, 0, 0, 0, .5, 1))):
        expect_value_error(lambda: ee_to_pose(bad_command))
    print("PASS: coordinate-frame invariance, input contracts and gripper clipping")


def main() -> int:
    check_reference()
    image, world = synthetic_hand()
    thenar, hand_rotation, gap, palm_scale, residual = hand_frame_and_metrics(
        image, 640, 640, world_landmarks=world
    )
    np.testing.assert_allclose(thenar, (.5, .5 + .025 * 1000 / 640), atol=1e-12)
    check(np.allclose(hand_rotation, np.eye(3), atol=1e-10), "canonical frame must be identity")
    check(np.isclose(np.linalg.det(hand_rotation), 1.0), "hand frame must be right-handed")
    check(np.isclose(gap, 0.06), "thumb/index gap must match synthetic 60 mm separation")
    check(palm_scale > 0.0, "apparent palm scale must be positive")
    check(residual < 1e-12, "planar synthetic hand must have zero plane residual")
    print("PASS: thenar anchor, SVD hand frame, gap, and palm scale")

    check(normalize_gripper_gap(0.01, 0.01, 0.09) == 0.0, "closed gap maps to g=0")
    check(normalize_gripper_gap(0.09, 0.01, 0.09) == 1.0, "open gap maps to g=1")
    check(normalize_gripper_gap(0.05, 0.01, 0.09) == 0.5, "midpoint gap maps to g=.5")
    print("PASS: continuous fingertip-gap calibration")

    anchor_position = np.array((0.4, -0.1, 0.8))
    anchor_rotation = np.eye(3)
    common = dict(
        hand_rotation=hand_rotation,
        pinch_gap=0.05,
        palm_scale=palm_scale,
        anchor_thenar=thenar,
        anchor_hand_rotation=hand_rotation,
        anchor_palm_scale=palm_scale,
        anchor_robot_position=anchor_position,
        anchor_robot_rotation=anchor_rotation,
        closed_gap=0.01,
        open_gap=0.09,
        motion_span=0.20,
        camera_to_base=CAMERA_TO_BASE,
    )
    ee8 = relative_hand_to_ee8(thenar=thenar, **common)
    check(ee8.shape == (8,), "command must have eight dimensions")
    check(np.allclose(ee8[:3], anchor_position), "unchanged thenar must hold TCP position")
    check(np.allclose(ee8[3:6], 0.0, atol=1e-10), "unchanged hand must hold EE orientation")
    check(np.isclose(ee8[6], 0.5) and ee8[7] == 0.0, "g and reserved dims must be correct")

    shifted = thenar + np.array((0.125, 0.0))
    moved = relative_hand_to_ee8(thenar=shifted, **common)
    check(np.allclose(moved[:3] - anchor_position, (0.10, 0.0, 0.0)), "thenar image shift maps to X")
    depth = relative_hand_to_ee8(thenar=thenar, **{**common, "palm_scale": palm_scale * 1.20})
    check(np.allclose(depth[:3] - anchor_position, (0.0, 0.20, 0.0)), "palm scale maps relative depth to Y")
    print("PASS: relative position, depth estimate, and 8-D packing")

    angle = 0.3
    rotated = np.array(((np.cos(angle), -np.sin(angle), 0.0),
                        (np.sin(angle), np.cos(angle), 0.0),
                        (0.0, 0.0, 1.0)))
    pose = relative_hand_to_ee8(
        thenar=thenar,
        **{**common, "hand_rotation": rotated @ hand_rotation},
    )
    expected_base_rotation = CAMERA_TO_BASE @ rotated @ CAMERA_TO_BASE.T
    check(np.allclose(rotvec_to_rotmat(pose[3:6]), expected_base_rotation, atol=1e-10), "rotation vector must preserve transformed hand rotation")
    round_trip = rotvec_to_rotmat(rotmat_to_rotvec(rotated))
    check(np.allclose(round_trip, rotated, atol=1e-10), "rotvec conversion must round-trip")
    print("PASS: hand orientation to rotation-vector command")

    bad = image.copy()
    bad[12, 0] = np.nan
    try:
        hand_frame_and_metrics(bad, 640, 640, world_landmarks=world)
    except ValueError:
        print("PASS: invalid landmark frame is rejected")
    else:
        raise AssertionError("non-finite landmark frame must be rejected")
    print("ALL HAND-TO-EE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

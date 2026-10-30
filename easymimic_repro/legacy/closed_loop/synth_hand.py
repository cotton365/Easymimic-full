"""Inverse of hand_to_ee(): desired EE pose -> synthetic 21-keypoint hand.

RETIRED — 针对已退役的占位夹爪（joint 'gripper_jaw'），在本仓库当前资产上必然 KeyError；
保留仅为历史参照，代码不再维护、不参与任何验收，也不随 quick start 出现。
现役入口是 experiments/pose_replay.py + configs/waypoints_cube_lift.json。

hand_to_ee() maps (21, 3) landmarks -> ee8.  Closed-loop tests need the opposite
direction: a hand whose hand_to_ee() output is a pose we *chose*.  This module
inverts every step of that construction, so the command fed to the robot is
known ground truth and any downstream error belongs to IK / physics rather than
to the retargeting math.  The synthetic camera frame *is* the robot base frame
(T_base_cam = identity): no calibration error exists by construction.

What is inverted
----------------
* plane_frame(): the five PLANE_POINT_IDS landmarks sit at z = 0 of the hand
  frame, so the SVD plane normal is +-Z and the fitted frame equals R_hand;
  INDEX_MCP -> INDEX_PIP runs along +X, so plane_frame()'s X axis is exactly
  hand X.
* thenar_anchor(): the landmark pair (THUMB_IP, INDEX_MCP) is placed
  symmetrically about the hand-frame origin, so the commanded position comes
  out as exactly the requested pos.
* gripper_norm(): |THUMB_TIP - INDEX_TIP| = d_min + g * (d_max - d_min), so the
  recovered g is exactly the requested one (no clipping while g stays in
  [0, 1]) and d_t is the true landmark distance.

Gripper-frame convention
------------------------
    R_tool = R_hand @ GRIPPER_FROM_HAND
    GRIPPER_FROM_HAND = [hand Y, -hand X, hand Z]   (columns, det = +1)

Rationale (one sentence): the jaws close along tool x and the fingers extrude
along tool -y, so mapping tool x onto the hand's opposition axis Y = Z x X and
tool -y onto the index-finger direction X makes the gripper close along exactly
the direction a thumb-index pinch closes, with no extra wrist roll needed.

Simplifications worth knowing: the thumb chain is only constrained where
hand_to_ee() reads it (IP lies in the palm plane, TIP fixes d_t), so the
synthetic thumb is effectively a straight IP -> TIP segment and no landmark is
anatomically validated beyond those constraints.  That is enough here because
the landmarks are never rendered; they only feed hand_to_ee().
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EE_DIR = os.path.join(os.path.dirname(_HERE), "ee_retarget")
if _EE_DIR not in sys.path:
    sys.path.insert(0, _EE_DIR)

from hand_to_ee import (  # noqa: E402  (import must follow the sys.path fix-up)
    DEFAULT_D_MAX,
    DEFAULT_D_MIN,
    GRIPPER_TIP_IDS,
    INDEX_DIP_IDX,
    INDEX_MCP_IDX,
    INDEX_PIP_IDX,
    INDEX_TIP_IDX,
    MIDDLE_MCP_IDX,
    NUM_KEYPOINTS,
    PINKY_MCP_IDX,
    RESERVED_VALUE,
    RING_MCP_IDX,
    THUMB_CMC_IDX,
    THUMB_IP_IDX,
    THUMB_MCP_IDX,
    THUMB_TIP_IDX,
    WRIST_IDX,
    rotation_angle_deg,
    rotmat_to_rotvec,
)

# --------------------------------------------------------------------------- #
# Gripper frame <-> hand frame: the module-level configurable knob.
# --------------------------------------------------------------------------- #
_HAND_X = np.array([1.0, 0.0, 0.0])
_HAND_Y = np.array([0.0, 1.0, 0.0])
_HAND_Z = np.array([0.0, 0.0, 1.0])

# Columns are the tool axes expressed in the hand frame: x_tool = hand Y,
# y_tool = -hand X, z_tool = hand Z.
GRIPPER_FROM_HAND = np.column_stack([_HAND_Y, -_HAND_X, _HAND_Z])


def verify_convention(C=GRIPPER_FROM_HAND, tol=1e-12):
    """Return (orthonormality_error, det); raise if C is not a proper rotation."""
    C = np.asarray(C, dtype=np.float64)
    if C.shape != (3, 3):
        raise ValueError("convention matrix must be 3x3, got %s" % (C.shape,))
    err = float(np.max(np.abs(C.T @ C - np.eye(3))))
    det = float(np.linalg.det(C))
    if err > tol or abs(det - 1.0) > tol:
        raise AssertionError(
            "GRIPPER_FROM_HAND is not a proper rotation "
            "(orthonormality err %.3e, det %.15f)" % (err, det))
    return err, det


CONVENTION_ORTHONORMALITY_ERR, CONVENTION_DET = verify_convention()

# --------------------------------------------------------------------------- #
# Hand template in the hand frame (metres): X = index MCP->PIP, Y = Z x X,
# Z = palm normal.  Only the constraints hand_to_ee() reads are load-bearing.
# --------------------------------------------------------------------------- #
ANCHOR_HALF_SEP = 0.025   # THUMB_IP at +Y, INDEX_MCP at -Y -> midpoint = origin

# direction from the index fingertip towards the thumb tip (unit length)
THUMB_TIP_DIR = np.array([-0.55, 0.80, 0.25])
THUMB_TIP_DIR /= np.linalg.norm(THUMB_TIP_DIR)

# finger chains: (MCP index, knuckle y, length scale)
_FINGER_CHAINS = (
    (MIDDLE_MCP_IDX, -0.032, 1.02),
    (RING_MCP_IDX, -0.038, 0.95),
    (PINKY_MCP_IDX, -0.044, 0.82),
)


def _template():
    """21 landmark positions in the hand frame with THUMB_TIP left at zero."""
    t = np.zeros((NUM_KEYPOINTS, 3))
    t[WRIST_IDX] = (-0.060, 0.000, 0.000)
    t[THUMB_CMC_IDX] = (-0.005, 0.046, 0.004)
    t[THUMB_MCP_IDX] = (0.016, 0.040, 0.004)
    t[THUMB_IP_IDX] = (0.000, ANCHOR_HALF_SEP, 0.000)      # thenar anchor pair
    t[THUMB_TIP_IDX] = (0.000, 0.000, 0.000)               # set per frame
    t[INDEX_MCP_IDX] = (0.000, -ANCHOR_HALF_SEP, 0.000)    # thenar anchor pair
    t[INDEX_PIP_IDX] = (0.046, -ANCHOR_HALF_SEP, 0.000)
    t[INDEX_DIP_IDX] = (0.070, -ANCHOR_HALF_SEP, 0.000)
    t[INDEX_TIP_IDX] = (0.090, -ANCHOR_HALF_SEP, 0.000)
    for mcp, y, k in _FINGER_CHAINS:
        t[mcp + 0] = (0.000, y, 0.000)
        t[mcp + 1] = (0.046 * k, y, 0.000)
        t[mcp + 2] = (0.070 * k, y, 0.000)
        t[mcp + 3] = (0.090 * k, y, 0.000)
    return t


def hand_template(g, d_min=DEFAULT_D_MIN, d_max=DEFAULT_D_MAX):
    """Return (21 landmarks in the hand frame, d_t) for normalised opening g."""
    if not d_max > d_min:
        raise ValueError("need d_max > d_min, got %r / %r" % (d_max, d_min))
    g = float(np.clip(g, 0.0, 1.0))
    d_t = d_min + g * (d_max - d_min)
    t = _template()
    i_thumb_tip, i_index_tip = GRIPPER_TIP_IDS
    t[i_thumb_tip] = t[i_index_tip] + d_t * THUMB_TIP_DIR
    return t, d_t


def hand_to_gripper(R_hand):
    """Hand frame -> gripper (tool) frame rotation."""
    return _as_rotation(R_hand) @ GRIPPER_FROM_HAND


def gripper_to_hand(R_tool):
    """Gripper (tool) frame -> hand frame rotation (the inverse mapping)."""
    return _as_rotation(R_tool) @ GRIPPER_FROM_HAND.T


def keypoints_for_pose(pos, R_tool, g, d_min=DEFAULT_D_MIN, d_max=DEFAULT_D_MAX):
    """Landmarks + R_wrist whose hand_to_ee() output is (pos, R_tool, g).

    pos is the commanded position, which for this task is the gripper TCP,
    i.e. tool0 + R_tool @ (0, -0.0325, 0) -- the pad-centre midpoint -- and not
    the bare tool0 site origin.
    """
    R_hand = gripper_to_hand(R_tool)
    local, d_t = hand_template(g, d_min, d_max)
    kp = local @ R_hand.T + np.asarray(pos, dtype=np.float64).reshape(1, 3)
    return kp, R_hand, d_t


def ee8_from_pose(pos, R_tool, g):
    """Ground-truth 8-D command for the same pose, built without keypoints."""
    ee8 = np.zeros(8)
    ee8[0:3] = np.asarray(pos, dtype=np.float64).reshape(3)
    ee8[3:6] = rotmat_to_rotvec(gripper_to_hand(R_tool))
    ee8[6] = float(g)
    ee8[7] = RESERVED_VALUE
    return ee8


def _as_rotation(R):
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("expected a (3, 3) rotation, got %s" % (R.shape,))
    return R

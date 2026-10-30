"""Paper-core math layer: MANO 21 hand keypoints -> 8-D end-effector command.

Robot-agnostic and numpy-only (no scipy, no mujoco). This module reproduces the
math described in the EasyMimic paper for turning a reconstructed hand into a
lower-dimensional end-effector command that any 6-DoF arm can execute:

    ee8 = [pos_xyz(3), rotvec(3), gripper_norm(1), reserved(1)]

Frames
------
All keypoints arrive in the *camera* frame (metres). T_base_cam maps camera ->
robot base, so the 8-D command is expressed in the robot base frame. Rotation
matrices are column-major in the numpy sense: R[:, k] is the k-th basis axis.

Hand frame construction (per the paper)
---------------------------------------
* Z axis      : normal of the best-fit plane through 5 landmarks, via SVD.
* X axis      : index-MCP -> index-PIP direction, projected into that plane.
* Y axis      : Z x X, which closes the frame to a proper rotation (det = +1).
* thenar      : midpoint(thumb IP, index MCP); this is the commanded position.

The plane normal from SVD is sign-ambiguous (+/- n). The sign is resolved
against PALM_NORMAL_REFERENCE_AXIS of the supplied wrist rotation, so a frame
that is flipped 180 degrees about X is never produced: the normal always points
to the same side of the palm as the wrist frame's reference axis. When that
reference axis is (numerically) perpendicular to the plane normal the dot
product carries no information and the SVD sign is kept as-is.

Note on the paper wording "拇指 PIP" (thumb PIP): the MANO/HaMeR 21-joint
convention has no thumb PIP joint (the thumb is CMC/MCP/IP/TIP), so THUMB_IP_IDX
is used as the thumb-side anchor point. Every landmark index is a module-level
constant below so this assumption is a one-line change.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Landmark indices -- HaMeR / MANO 21-keypoint convention.
# --------------------------------------------------------------------------- #
WRIST_IDX = 0          # wrist root
THUMB_CMC_IDX = 1      # thumb carpometacarpal
THUMB_MCP_IDX = 2      # thumb metacarpophalangeal
THUMB_IP_IDX = 3       # thumb interphalangeal -- stands in for the paper's "thumb PIP"
THUMB_TIP_IDX = 4      # thumb fingertip
INDEX_MCP_IDX = 5      # index metacarpophalangeal (knuckle)
INDEX_PIP_IDX = 6      # index proximal interphalangeal
INDEX_DIP_IDX = 7      # index distal interphalangeal
INDEX_TIP_IDX = 8      # index fingertip
MIDDLE_MCP_IDX = 9     # unused by the pose math, listed for completeness
RING_MCP_IDX = 13      # unused
PINKY_MCP_IDX = 17     # unused

# thenar anchor = midpoint of these two landmarks (thumb IP, index MCP)
THENAR_ANCHOR_IDS = (THUMB_IP_IDX, INDEX_MCP_IDX)
# the 5 coplanar landmarks the palm/finger plane is fitted through
PLANE_POINT_IDS = (INDEX_MCP_IDX, INDEX_PIP_IDX, INDEX_DIP_IDX, INDEX_TIP_IDX, THUMB_IP_IDX)
# X axis = unit(p[X_AXIS_TO_IDX] - p[X_AXIS_FROM_IDX]) == index MCP -> index PIP
X_AXIS_FROM_IDX = INDEX_MCP_IDX
X_AXIS_TO_IDX = INDEX_PIP_IDX
# gripper opening = distance between these two landmarks
GRIPPER_TIP_IDS = (THUMB_TIP_IDX, INDEX_TIP_IDX)

NUM_KEYPOINTS = 21
EE_DIM = 8
RESERVED_VALUE = 0.0   # index 7 of the command, reserved for future use (e.g. wrist roll rate)

# Which column of R_wrist orients the palm; used only to de-ambiguate the SVD
# plane-normal sign. +Z of a MANO wrist frame is the palm-normal-ish direction.
PALM_NORMAL_REFERENCE_AXIS = 2
# Below this |dot| the wrist reference carries no sign information.
_SIGN_REFERENCE_EPS = 1e-8

# Default gripper calibration in metres; override per-robot.
DEFAULT_D_MIN = 0.02   # landmark tip distance with the jaws fully closed
DEFAULT_D_MAX = 0.10   # landmark tip distance with the jaws fully open


# --------------------------------------------------------------------------- #
# Rotation helpers (scipy is unavailable in this environment)
# --------------------------------------------------------------------------- #
def rotvec_to_rotmat(rotvec):
    """Rodrigues: axis-angle vector (3,) -> rotation matrix (3, 3)."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        # first-order expansion around the identity
        K = _skew(rotvec)
        return np.eye(3) + K + 0.5 * (K @ K)
    axis = rotvec / theta
    K = _skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotmat_to_rotvec(R):
    """Rotation matrix (3, 3) -> axis-angle vector (3,) with |v| in [0, pi]."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("rotmat_to_rotvec expects a (3, 3) matrix, got %s" % (R.shape,))
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if theta < 1e-7:
        # sin(theta) * axis ~ theta * axis; the vee map is already the answer
        return 0.5 * vee
    if np.pi - theta > 1e-6:
        return vee * (theta / (2.0 * np.sin(theta)))

    # theta ~ pi: recover the axis from (R + I) / 2 == axis * axis^T
    A = 0.5 * (R + np.eye(3))
    k = int(np.argmax(np.diag(A)))
    axis = np.zeros(3)
    axis[k] = np.sqrt(max(A[k, k], 0.0))
    for j in range(3):
        if j != k and axis[k] > 1e-12:
            axis[j] = A[k, j] / axis[k]
    norm = np.linalg.norm(axis)
    if norm < 1e-12:                      # numerically hopeless; identity
        return np.zeros(3)
    return axis / norm * theta


def _skew(v):
    """Skew-symmetric cross-product matrix of a 3-vector."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def rotation_angle_deg(R_a, R_b):
    """Geodesic angle between two rotations, in degrees."""
    M = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


# --------------------------------------------------------------------------- #
# Individual pieces of the math, exposed for unit testing / diagnostics
# --------------------------------------------------------------------------- #
def thenar_anchor(keypoints):
    """Midpoint of thumb IP and index MCP -- the commanded EE position."""
    keypoints = _as_keypoints(keypoints)
    i_a, i_b = THENAR_ANCHOR_IDS
    return 0.5 * (keypoints[i_a] + keypoints[i_b])


def plane_frame(keypoints, R_wrist):
    """Best-fit hand frame in the camera frame.

    Returns (R, z_flipped, plane_residual) where R is a proper rotation whose
    columns are [X, Y, Z] as described in the module docstring and
    plane_residual is the max distance of the 5 landmarks from the fitted plane.
    """
    keypoints = _as_keypoints(keypoints)
    pts = keypoints[list(PLANE_POINT_IDS)]
    centroid = pts.mean(axis=0)

    # normal = singular vector of the smallest singular value
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    Z = Vt[2]

    # sign de-ambiguation against the wrist reference axis
    ref = _as_rotation(R_wrist)[:, PALM_NORMAL_REFERENCE_AXIS]
    dot = float(np.dot(Z, ref))
    z_flipped = False
    if abs(dot) > _SIGN_REFERENCE_EPS and dot < 0.0:
        Z = -Z
        z_flipped = True

    # X from the index finger proximal phalanx, projected off the normal
    X = keypoints[X_AXIS_TO_IDX] - keypoints[X_AXIS_FROM_IDX]
    X = X - np.dot(X, Z) * Z
    X_norm = np.linalg.norm(X)
    if X_norm < 1e-12:
        raise ValueError("degenerate hand: index MCP->PIP direction is parallel to the plane normal")
    X = X / X_norm

    Y = np.cross(Z, X)
    Y = Y / np.linalg.norm(Y)
    R = np.column_stack([X, Y, Z])

    plane_residual = float(np.max(np.abs((pts - centroid) @ Z)))
    return R, z_flipped, plane_residual


def gripper_norm(keypoints, d_min=DEFAULT_D_MIN, d_max=DEFAULT_D_MAX):
    """Normalised gripper opening in [0, 1] plus the raw landmark distance."""
    keypoints = _as_keypoints(keypoints)
    if not d_max > d_min:
        raise ValueError("gripper calibration needs d_max > d_min, got d_min=%r d_max=%r" % (d_min, d_max))
    i_a, i_b = GRIPPER_TIP_IDS
    d_t = float(np.linalg.norm(keypoints[i_a] - keypoints[i_b]))
    g = (d_t - d_min) / (d_max - d_min)
    return float(np.clip(g, 0.0, 1.0)), d_t


def hand_to_ee(keypoints, R_wrist, T_base_cam=None,
               d_min=DEFAULT_D_MIN, d_max=DEFAULT_D_MAX):
    """MANO keypoints -> 8-D EE command in the robot base frame.

    Parameters
    ----------
    keypoints : (21, 3) array-like, metres, camera frame (HaMeR/MANO ordering).
    R_wrist   : (3, 3) wrist rotation in the camera frame. Only its
                PALM_NORMAL_REFERENCE_AXIS column is used, to fix the sign of
                the plane normal.
    T_base_cam: optional (4, 4) homogeneous camera -> base transform. Defaults
                to the identity (command then stays in the camera frame).
    d_min, d_max : gripper calibration in metres.

    Returns
    -------
    ee8 : (8,) float64 array -- [pos(3), rotvec(3), gripper_norm, reserved].
    info : dict of intermediates (R, d_t, g_norm, anchors, z_flipped, ...).

    NaN semantics: if any keypoint (or the wrist/transform) is not finite the
    whole command is NaN and info["valid"] is False. Nothing is silently
    zero-filled -- a NaN frame must stay recognisable downstream as missing.
    """
    keypoints_arr = np.asarray(keypoints, dtype=np.float64)
    if keypoints_arr.shape != (NUM_KEYPOINTS, 3):
        raise ValueError("expected keypoints of shape (%d, 3), got %s"
                         % (NUM_KEYPOINTS, keypoints_arr.shape))
    R_wrist_arr = np.asarray(R_wrist, dtype=np.float64)
    if R_wrist_arr.shape != (3, 3):
        raise ValueError("expected R_wrist of shape (3, 3), got %s" % (R_wrist_arr.shape,))
    T_base_cam_arr = _as_transform(T_base_cam)

    info = {
        "valid": False,
        "R": np.full((3, 3), np.nan),
        "d_t": float("nan"),
        "g_norm": float("nan"),
        "anchor_cam": np.full(3, np.nan),
        "z_flipped": False,
        "plane_residual_m": float("nan"),
    }

    if not (np.all(np.isfinite(keypoints_arr))
            and np.all(np.isfinite(R_wrist_arr))
            and np.all(np.isfinite(T_base_cam_arr))):
        info["ee8"] = np.full(EE_DIM, np.nan)
        return np.full(EE_DIM, np.nan), info

    R_plane_cam, z_flipped, plane_residual = plane_frame(keypoints_arr, R_wrist_arr)
    anchor_cam = thenar_anchor(keypoints_arr)
    g, d_t = gripper_norm(keypoints_arr, d_min, d_max)

    # camera frame -> robot base frame
    R_base_cam = T_base_cam_arr[:3, :3]
    t_base_cam = T_base_cam_arr[:3, 3]
    R_base = R_base_cam @ R_plane_cam
    anchor_base = R_base_cam @ anchor_cam + t_base_cam

    ee8 = np.zeros(EE_DIM, dtype=np.float64)
    ee8[0:3] = anchor_base
    ee8[3:6] = rotmat_to_rotvec(R_base)
    ee8[6] = g
    ee8[7] = RESERVED_VALUE

    info.update({
        "valid": True,
        "ee8": ee8,
        "R": R_base,
        "R_cam": R_plane_cam,
        "anchor_cam": anchor_cam,
        "anchor_base": anchor_base,
        "d_t": d_t,
        "g_norm": g,
        "z_flipped": z_flipped,
        "plane_residual_m": plane_residual,
        "T_base_cam": T_base_cam_arr,
    })
    return ee8, info


def ee_to_pose(ee8):
    """Split an 8-D command into (position (3,), rotation (3, 3), gripper, reserved)."""
    ee8 = np.asarray(ee8, dtype=np.float64).reshape(EE_DIM)
    return ee8[0:3].copy(), rotvec_to_rotmat(ee8[3:6]), float(ee8[6]), float(ee8[7])


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _as_keypoints(keypoints):
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if keypoints.shape != (NUM_KEYPOINTS, 3):
        raise ValueError("expected keypoints of shape (%d, 3), got %s"
                         % (NUM_KEYPOINTS, keypoints.shape))
    return keypoints


def _as_rotation(R):
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("expected a (3, 3) rotation, got %s" % (R.shape,))
    return R


def _as_transform(T):
    if T is None:
        return np.eye(4)
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("expected T_base_cam of shape (4, 4), got %s" % (T.shape,))
    return T

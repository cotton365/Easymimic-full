"""Execution layer: 8-D end-effector pose -> 6 joint angles for the JAKA Zu3.

Damped least squares (Levenberg-Marquardt) inverse kinematics on top of the
MuJoCo numpy bindings. Only numpy + mujoco are used (no scipy, no ikpy).

Why DLS and not a plain pseudo-inverse: the JAKA Zu3 is 6-DoF, so its 6x6 site
Jacobian passes through singular configurations (measured on a target sampled
from the joint limits: singular values [1.83, 1.44, 1.00, 0.33, 0.18, 1e-4]).
There the Moore-Penrose inverse explodes; damping (J J^T + lambda^2 I)^-1 keeps
the step bounded and just yields a slower, still-descending step.

Why the damping is *relative*: with an absolute lambda = 1e-3 the near-singular
Jacobians above produced steps of 60-128 rad and only ~50% of the 100-target
benchmark converged. Damping is therefore scaled to the Jacobian itself,

    lambda_abs = lambda_rel * sqrt(trace(J J^T) / 6)

which is dimensionless in the singular values of J and makes the solver behave
identically in the stretched and folded parts of the workspace.

Solver, per iteration:
  1. forward kinematics for the current q (mj_kinematics + mj_comPos)
  2. 6-D error e = [target_pos - pos ; ROT_WEIGHT * rotvec(R_target R^T)];
     ROT_WEIGHT converts radians into metres so both blocks are commensurate
     in one least-squares problem
  3. dq = J^T (J J^T + lambda_abs^2 I)^-1 e with J = [Jp ; ROT_WEIGHT * Jr]
     from mj_jacSite, restricted to the arm's 6 DoF columns
  4. clamp |dq|_inf to MAX_STEP_RAD, integrate, clip to the joint limits, then
     accept only if the weighted residual drops; otherwise damp harder and
     recompute the same step (up to MAX_STEP_TRIES). No trial step is ever
     applied blindly, so a large MAX_STEP_RAD is safe.

Convergence: position error < POS_TOL_M and orientation error < ANG_TOL_DEG.
An attempt that is only creeping (see CREEP_WINDOW/CREEP_RATIO) is abandoned
early instead of burning all MAX_ITERS iterations.

solve() is one DLS attempt (warm start). solve_robust() adds the global step
that arbitrary targets actually need: when the warm start stalls it retries
from deterministic pseudo-random seeds inside the joint limits and returns the
first solution that converges. Both are reproducible: same inputs + same
n_restarts/restart_seed -> same result.

Pitfall (verified): the MJCF must be loaded from an ABSOLUTE path. A relative
path makes MuJoCo resolve the top-level meshdir inside the includes and fail
with "Error opening file 'assets/jaka_zu3_meshes/assets/objaverse/...'". This
module therefore raises on a relative path instead of silently resolving it
against the cwd. DEFAULT_XML_PATH is already absolute (repo-relative: the repo's own
assets/jaka_zu3/robot.xml) and can be overridden with the EASYMIMIC_JAKA_XML
environment variable, which is read when the solver is constructed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import mujoco

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The arm + jaw model that ships with this repo (nq = nu = 7, no free joints, loads in
# 0.24 s). It is deliberately NOT third_party/jaka-zu3-mujoco/jaka_zu3_mujoco.xml any
# more: upstream gained a <flexcomp name="rag"> (line 117 at f88a837) and MuJoCo 3.16
# refuses that file with "flex vertex/node body 'rag_0' cannot have joint damping".
DEFAULT_XML_PATH = os.path.join(_REPO_ROOT, "assets", "jaka_zu3", "robot.xml")


def default_xml_path():
    """DEFAULT_XML_PATH unless EASYMIMIC_JAKA_XML names another (absolute) model."""
    return os.environ.get("EASYMIMIC_JAKA_XML") or DEFAULT_XML_PATH


DEFAULT_SITE_NAME = "tool0"
# Address the arm DoF by joint name, never by "the first 6 indices": the default model
# has nq = nu = 7 (6 arm joints + jaw), while the scenes used by experiments/ and
# src/closed_loop/ add free-floating props (scene_cube_bowl.xml: nq=14, nv=13).
ARM_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")

POS_TOL_M = 1e-3
ANG_TOL_DEG = 0.5
MAX_ITERS = 100             # per DLS attempt
MAX_STEP_RAD = 0.6          # ~34 deg cap on |dq|_inf; only accepted if it improves
LAMBDA_REL_INIT = 0.1
LAMBDA_REL_MIN = 1e-4
LAMBDA_REL_MAX = 3.0
ROT_WEIGHT = 0.2            # metres per radian, balances the two error blocks
MAX_STEP_TRIES = 8          # damping re-tries before a DLS attempt is called stalled
# Creep detector: a hopeless attempt keeps taking ever-smaller accepted steps and
# would otherwise burn all MAX_ITERS (measured on the previous upstream model: 97.5
# iterations per target against 45.2 with this rule, at identical success; the repo
# asset below averages 40.0). If the weighted residual has not
# dropped by CREEP_RATIO over CREEP_WINDOW iterations while still above
# CREEP_FLOOR (5x the position tolerance), the attempt is abandoned.
CREEP_WINDOW = 10
CREEP_RATIO = 0.9
CREEP_FLOOR_M = 5e-3
DEFAULT_N_RESTARTS = 16     # extra seeds tried by solve_robust()
DEFAULT_RESTART_SEED = 20240501


@dataclass
class IKResult:
    """Outcome of an IK solve."""
    qpos: np.ndarray           # (6,) joint angles in radians
    converged: bool
    pos_err: float             # final position error in metres
    ang_err_deg: float         # final orientation error in degrees
    iters: int                 # total DLS iterations consumed (all restarts)
    restarts: int = 0          # extra seeds consumed beyond the warm start

    def __repr__(self):
        return ("IKResult(converged=%s, pos_err=%.3e m, ang_err=%.4f deg, "
                "iters=%d, restarts=%d)"
                % (self.converged, self.pos_err, self.ang_err_deg, self.iters, self.restarts))


def rotvec_to_rotmat(rotvec):
    """Rodrigues axis-angle -> rotation matrix (local copy keeps this module standalone)."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotmat_to_rotvec(R):
    """Rotation matrix -> axis-angle vector, |v| in [0, pi] (keeps this module standalone).

    Built on the quaternion (Shepperd) extraction instead of arccos(trace) + vee/sin,
    because both of those are ill-conditioned at theta = pi: the trace is -1 +/- 1e-16
    there and arccos amplifies that into a ~1e-8 rad angle error, while vee/sin(theta)
    is 0/0. Measured on the previous body: diag(-1, 1, -1) came back as [0, 0, 0], i.e.
    a full half-turn read as "no error". Measured on the pi branch of
    hand_to_ee.rotmat_to_rotvec (the in-repo reference): 1.5e-3 rad round-trip error at
    theta = pi - 1e-6, 2.5e-8 for an exact pi built as 2 a a^T - I. The route below
    stays under 1.2e-15 over 5000 Haar-random rotations and at theta = pi +/- 1e-12.
    """
    R = np.asarray(R, dtype=np.float64)
    # 4 q_i^2 for q = [w, x, y, z]: the largest component is the safe divisor, which is
    # what makes theta = pi (w = 0) well conditioned instead of a 0/0.
    q4 = np.array([1.0 + R[0, 0] + R[1, 1] + R[2, 2],
                   1.0 + R[0, 0] - R[1, 1] - R[2, 2],
                   1.0 - R[0, 0] + R[1, 1] - R[2, 2],
                   1.0 - R[0, 0] - R[1, 1] + R[2, 2]])
    k = int(np.argmax(q4))
    q = np.zeros(4)
    q[k] = 0.5 * np.sqrt(max(q4[k], 0.0))
    if q[k] < 1e-12:                    # not a rotation to working precision -> identity
        return np.zeros(3)
    s = 0.25 / q[k]
    if k == 0:
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    elif k == 1:
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 1] + R[1, 0]) * s
        q[3] = (R[0, 2] + R[2, 0]) * s
    elif k == 2:
        q[0] = (R[0, 2] - R[2, 0]) * s
        q[1] = (R[0, 1] + R[1, 0]) * s
        q[3] = (R[1, 2] + R[2, 1]) * s
    else:
        q[0] = (R[1, 0] - R[0, 1]) * s
        q[1] = (R[0, 2] + R[2, 0]) * s
        q[2] = (R[1, 2] + R[2, 1]) * s
    q /= np.linalg.norm(q)
    if q[0] < 0.0:                      # same rotation, but keep the arc |v| = theta <= pi
        q = -q
    sin_half = float(np.linalg.norm(q[1:]))
    if sin_half < 1e-300:               # exactly identity
        return np.zeros(3)
    theta = 2.0 * float(np.arctan2(sin_half, q[0]))
    return q[1:] * (theta / sin_half)


def _skew(v):
    """Skew-symmetric cross-product matrix of a 3-vector."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _as_offset(tcp_offset):
    """Normalise the TCP offset argument to a (4, 4) homogeneous transform."""
    if tcp_offset is None:
        return np.eye(4)
    arr = np.asarray(tcp_offset, dtype=np.float64)
    if arr.shape == (3,):
        out = np.eye(4)
        out[:3, 3] = arr
        return out
    if arr.shape == (4, 4):
        return arr.copy()
    raise ValueError("tcp_offset must be (3,) or (4, 4), got %s" % (arr.shape,))


def pose_error(target_pos, target_rot, pos, rot):
    """Return (position error in metres, orientation error in degrees)."""
    pos_err = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float64) - pos))
    M = np.asarray(target_rot, dtype=np.float64).T @ rot
    cos_theta = float(np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0))
    return pos_err, float(np.degrees(np.arccos(cos_theta)))


class JakaZu3IK:
    """Position/velocity kinematics plus DLS IK for the JAKA Zu3 in MuJoCo."""

    def __init__(self, xml_path=None, site_name=DEFAULT_SITE_NAME,
                 tcp_offset=None):
        if xml_path is None:
            xml_path = default_xml_path()
        if not os.path.isabs(xml_path):
            raise ValueError(
                "xml_path must be absolute: loading the JAKA MJCF from a relative path "
                "makes MuJoCo resolve the top-level meshdir inside the includes and fail; "
                "got %r" % (xml_path,))
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(xml_path)
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError("site %r not found in %s" % (site_name, xml_path))

        joint_ids, qpos_adr, dof_adr = [], [], []
        for name in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError("joint %r not found in %s" % (name, xml_path))
            joint_ids.append(jid)
            qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            dof_adr.append(int(self.model.jnt_dofadr[jid]))
        self.joint_ids = np.asarray(joint_ids, dtype=int)
        self.qpos_adr = np.asarray(qpos_adr, dtype=int)
        self.dof_adr = np.asarray(dof_adr, dtype=int)

        if not np.all(self.model.jnt_limited[self.joint_ids]):
            raise ValueError("expected all 6 arm joints to be limited")
        self.joint_limits = np.asarray(self.model.jnt_range[self.joint_ids], dtype=np.float64)
        self.qpos_home = np.asarray(self.model.qpos0[self.qpos_adr], dtype=np.float64).copy()

        # TCP offset from the site frame: (3,) translation or a full (4, 4)
        # homogeneous transform, expressed in the site frame. IK then controls
        # the tool point, not the site origin. Identity by default, which makes
        # "TCP == site" and leaves the measured tool0 numbers unchanged.
        self.tcp_offset = _as_offset(tcp_offset)

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0

    # ------------------------------------------------------------------ #
    # kinematics
    # ------------------------------------------------------------------ #
    def _sync(self, q6):
        self.data.qpos[self.qpos_adr] = q6
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def fk_site(self, q6):
        """Raw site kinematics: joint angles (6,) -> (site position, site rotation)."""
        self._sync(np.asarray(q6, dtype=np.float64).reshape(6))
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def fk(self, q6):
        """TCP kinematics: joint angles (6,) -> (TCP position (3,), TCP rotation (3, 3)).

        With the default identity tcp_offset this is exactly the site pose.
        """
        pos, rot = self.fk_site(q6)
        return self._apply_offset(pos, rot)

    def _apply_offset(self, pos, rot):
        t = self.tcp_offset[:3, 3]
        return pos + rot @ t, rot @ self.tcp_offset[:3, :3]

    def jacobian(self, q6):
        """6 x 6 TCP Jacobian, rows [position; rotation], arm DoF columns only.

        The offset lever arm turns the site's angular velocity into linear
        velocity of the tool point: Jp_tcp = Jp - skew(R_site t) Jr.
        """
        self._sync(np.asarray(q6, dtype=np.float64).reshape(6))
        self._jacp[:] = 0.0
        self._jacr[:] = 0.0
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        Jp = self._jacp[:, self.dof_adr]
        Jr = self._jacr[:, self.dof_adr]
        lever = self.data.site_xmat[self.site_id].reshape(3, 3) @ self.tcp_offset[:3, 3]
        if np.any(lever):
            Jp = Jp - _skew(lever) @ Jr
        return np.vstack([Jp, Jr])

    # ------------------------------------------------------------------ #
    # IK
    # ------------------------------------------------------------------ #
    def _dls_attempt(self, target_pos, target_rot, q_seed, pos_tol, ang_tol_deg,
                     max_iters, max_step, lam_rel, rot_weight, max_step_tries):
        """One damped-least-squares descent from q_seed.

        Returns (q, converged, pos_err, ang_err_deg, iters, pos, rot).
        """
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        q = np.clip(np.asarray(q_seed, dtype=np.float64).reshape(6).copy(), lo, hi)
        it = 0
        hist = []
        for it in range(1, max_iters + 1):
            pos, rot = self.fk(q)
            pos_err, ang_err_deg = pose_error(target_pos, target_rot, pos, rot)
            if pos_err < pos_tol and ang_err_deg < ang_tol_deg:
                return q, True, pos_err, ang_err_deg, it, pos, rot

            score_cur = pos_err + rot_weight * np.radians(ang_err_deg)
            hist.append(score_cur)
            if (len(hist) > CREEP_WINDOW and score_cur > CREEP_FLOOR_M
                    and score_cur > CREEP_RATIO * hist[-CREEP_WINDOW - 1]):
                return q, False, pos_err, ang_err_deg, it, pos, rot

            e = np.concatenate([target_pos - pos,
                                rot_weight * rotmat_to_rotvec(target_rot @ rot.T)])
            J = self.jacobian(q)
            J[3:6, :] *= rot_weight
            JJt = J @ J.T
            lam = lam_rel * np.sqrt(max(np.trace(JJt) / 6.0, 1e-12))

            accepted = False
            for _ in range(max_step_tries):
                try:
                    dq = J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(6), e)
                except np.linalg.LinAlgError:
                    lam = min(lam * 10.0, LAMBDA_REL_MAX)
                    continue
                inf_norm = float(np.max(np.abs(dq)))
                if inf_norm > max_step:
                    dq *= max_step / inf_norm
                q_try = np.clip(q + dq, lo, hi)
                pos_try, rot_try = self.fk(q_try)
                pos_err_try, ang_err_try = pose_error(target_pos, target_rot, pos_try, rot_try)
                if pos_err_try + rot_weight * np.radians(ang_err_try) < score_cur:
                    q = q_try
                    accepted = True
                    break
                lam = min(lam * 4.0, LAMBDA_REL_MAX)
            if not accepted:
                pos, rot = self.fk(q)
                pos_err, ang_err_deg = pose_error(target_pos, target_rot, pos, rot)
                return q, False, pos_err, ang_err_deg, it, pos, rot

        pos, rot = self.fk(q)
        pos_err, ang_err_deg = pose_error(target_pos, target_rot, pos, rot)
        return q, False, pos_err, ang_err_deg, it, pos, rot

    def solve(self, target_pos, target_rot, q_seed=None, *,
              pos_tol=POS_TOL_M, ang_tol_deg=ANG_TOL_DEG, max_iters=MAX_ITERS,
              max_step=MAX_STEP_RAD, lam_rel=LAMBDA_REL_INIT, rot_weight=ROT_WEIGHT,
              max_step_tries=MAX_STEP_TRIES):
        """Single DLS attempt from q_seed (default: the home pose). Never raises on failure."""
        q, conv, pos_err, ang_err_deg, it, _, _ = self._dls_attempt(
            np.asarray(target_pos, dtype=np.float64).reshape(3),
            np.asarray(target_rot, dtype=np.float64),
            self.qpos_home if q_seed is None else q_seed,
            pos_tol, ang_tol_deg, max_iters, max_step, lam_rel, rot_weight, max_step_tries)
        return IKResult(q, conv, pos_err, ang_err_deg, it, 0)

    def solve_robust(self, target_pos, target_rot, q_seed=None, *,
                     n_restarts=DEFAULT_N_RESTARTS, restart_seed=DEFAULT_RESTART_SEED,
                     pos_tol=POS_TOL_M, ang_tol_deg=ANG_TOL_DEG, max_iters=MAX_ITERS,
                     max_step=MAX_STEP_RAD, lam_rel=LAMBDA_REL_INIT, rot_weight=ROT_WEIGHT,
                     max_step_tries=MAX_STEP_TRIES):
        """Warm start, then up to n_restarts deterministic random seeds until convergence.

        Returns the first converged solution; on total failure the best (lowest
        weighted residual) attempt is returned with converged=False. iters is the
        sum over all attempts, restarts the number of extra seeds consumed.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_rot = np.asarray(target_rot, dtype=np.float64)
        if target_rot.shape != (3, 3):
            raise ValueError("target_rot must be (3, 3), got %s" % (target_rot.shape,))

        best = None
        best_score = np.inf
        total_iters = 0

        def _score(pos_err, ang_err_deg):
            return pos_err + rot_weight * np.radians(ang_err_deg)

        q, conv, pos_err, ang_err_deg, it, _, _ = self._dls_attempt(
            target_pos, target_rot, self.qpos_home if q_seed is None else q_seed,
            pos_tol, ang_tol_deg, max_iters, max_step, lam_rel, rot_weight, max_step_tries)
        total_iters += it
        if conv:
            return IKResult(q, True, pos_err, ang_err_deg, total_iters, 0)
        best, best_score = q, _score(pos_err, ang_err_deg)

        rng = np.random.default_rng(restart_seed)
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        for k in range(1, n_restarts + 1):
            seed = lo + (hi - lo) * rng.random(6)
            q, conv, pos_err, ang_err_deg, it, _, _ = self._dls_attempt(
                target_pos, target_rot, seed, pos_tol, ang_tol_deg, max_iters,
                max_step, lam_rel, rot_weight, max_step_tries)
            total_iters += it
            if conv:
                return IKResult(q, True, pos_err, ang_err_deg, total_iters, k)
            score = _score(pos_err, ang_err_deg)
            if score < best_score:
                best, best_score = q, score
        pos_err, ang_err_deg = pose_error(target_pos, target_rot, *self.fk(best))
        return IKResult(best, False, pos_err, ang_err_deg, total_iters, n_restarts)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def within_limits(self, q6, tol=1e-9):
        """True when q6 respects the joint limits within tol."""
        q6 = np.asarray(q6, dtype=np.float64)
        return bool(np.all(q6 >= self.joint_limits[:, 0] - tol)
                    and np.all(q6 <= self.joint_limits[:, 1] + tol))

    def random_qpos(self, rng, margin=1e-3):
        """Uniformly sample joint angles inside the limits (margin keeps off the boundary)."""
        lo = self.joint_limits[:, 0] + margin
        hi = self.joint_limits[:, 1] - margin
        return lo + (hi - lo) * rng.random(6)

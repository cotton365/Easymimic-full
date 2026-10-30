"""Acceptance test for the 8-D EE action layer (easymimic_repro/src/ee_retarget).

Pure asserts, no pytest, numpy + mujoco only:

    ~/miniconda3/envs/mujoco/bin/python easymimic_repro/tests_ee/test_ee_pipeline.py

Sections
  1. analytic synthetic hand     -> anchor / R / d_t / g against closed-form truth
  2. orthonormality + chirality  -> R^T R = I, det = +1, X . Z = 0
  3. NaN propagation             -> any NaN keypoint makes the whole 8-D frame NaN
  4. IK convergence              -> 100 FK-sampled reachable targets, error stats
  5. joint limits                -> every returned solution inside the limits
  6. end-to-end                  -> hand -> EE(8) -> IK -> qpos -> FK residual
  7. rotvec at 180 deg           -> |v| = pi and an exact round-trip
  8. rotvec boundary             -> theta = pi +/- 1e-12/1e-9 and theta ~ 0
  9. rotvec random SO(3)         -> 5000 rotations, round-trip
 10. rotvec vs reference         -> equivalent to hand_to_ee.rotmat_to_rotvec
 11. IK sees a 180 deg error     -> position identical, orientation a half-turn off
 12. default model + env         -> no-arg JakaZu3IK(), EASYMIMIC_JAKA_XML
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "ee_retarget")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import hand_to_ee as H          # noqa: E402
import ik_jaka_zu3 as K         # noqa: E402

D_MIN = 0.02                    # gripper calibration, metres
D_MAX = 0.10
D_MID = 0.5 * (D_MIN + D_MAX)

# index-finger phalanx lengths / offsets used by the synthetic hand (metres)
L_PIP, L_DIP, L_TIP = 0.035, 0.065, 0.090
Y_MCP = -0.04                   # index knuckle offset from the thenar anchor along Y

TOL = 1e-9
failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("  [%s] %s%s" % (status, name, ("" if not detail else "   " + detail)))
    if not condition:
        failures.append(name)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return (float(np.percentile(values, 50)),
            float(np.percentile(values, 95)),
            float(np.max(values)))


# --------------------------------------------------------------------------- #
# synthetic hand: every quantity is known in closed form
# --------------------------------------------------------------------------- #
def make_synthetic_hand(anchor, R, d_t):
    """Build a 21-keypoint hand whose frame is exactly R and grip distance exactly d_t.

    Local coordinates (x, y, z) with basis R:
        index MCP   = (0,      Y_MCP, 0)
        index PIP   = (L_PIP,  Y_MCP, 0)
        index DIP   = (L_DIP,  Y_MCP, 0)
        index TIP   = (L_TIP,  Y_MCP, 0)          -> thumb TIP at (L_TIP - d_t, Y_MCP, 0)
        thumb IP    = (0,     -Y_MCP, 0)
    so the thenar anchor = midpoint(thumb IP, index MCP) = the frame origin, and
    index MCP -> PIP is exactly R[:, 0].
    """
    def P(x, y):
        return anchor + R @ np.array([x, y, 0.0])

    kp = np.zeros((H.NUM_KEYPOINTS, 3))
    kp[H.WRIST_IDX] = P(-0.06, 0.0)
    kp[H.THUMB_CMC_IDX] = P(0.01, 0.03)
    kp[H.THUMB_MCP_IDX] = P(0.0, 0.02)
    kp[H.THUMB_IP_IDX] = P(0.0, -Y_MCP)
    kp[H.THUMB_TIP_IDX] = P(L_TIP - d_t, Y_MCP)
    kp[H.INDEX_MCP_IDX] = P(0.0, Y_MCP)
    kp[H.INDEX_PIP_IDX] = P(L_PIP, Y_MCP)
    kp[H.INDEX_DIP_IDX] = P(L_DIP, Y_MCP)
    kp[H.INDEX_TIP_IDX] = P(L_TIP, Y_MCP)
    for i in (H.MIDDLE_MCP_IDX, H.RING_MCP_IDX, H.PINKY_MCP_IDX):
        for j in range(i, i + 4):
            kp[j] = P(L_TIP - 0.004 * (j - i), Y_MCP - 0.008)
    return kp


def test_1_analytic():
    print("\n[1] analytic synthetic hand (truth by construction)")
    R_true = H.rotvec_to_rotmat(np.array([0.31, -0.72, 1.13]))
    anchor = np.array([0.12, -0.03, 0.44])

    for label, d_t, g_expected in (("closed (d_min)", D_MIN, 0.0),
                                    ("mid (d_mean)", D_MID, 0.5),
                                    ("open (d_max)", D_MAX, 1.0)):
        kp = make_synthetic_hand(anchor, R_true, d_t)
        ee, info = H.hand_to_ee(kp, R_true, None, D_MIN, D_MAX)

        check("anchor %s" % label,
              np.abs(info["anchor_cam"] - anchor).max() < TOL,
              "err=%.3e" % np.abs(info["anchor_cam"] - anchor).max())
        check("pos3 %s" % label,
              np.abs(ee[0:3] - anchor).max() < TOL,
              "err=%.3e" % np.abs(ee[0:3] - anchor).max())
        check("R three axes %s" % label,
              np.abs(info["R"] - R_true).max() < TOL,
              "err=%.3e" % np.abs(info["R"] - R_true).max())
        check("d_t %s" % label,
              abs(info["d_t"] - d_t) < TOL,
              "d_t=%.15f expected=%.15f" % (info["d_t"], d_t))
        check("g %s == %.1f" % (label, g_expected),
              abs(ee[6] - g_expected) < TOL,
              "g=%.15f" % ee[6])
        # rotvec must round-trip to the same rotation
        R_back = H.rotvec_to_rotmat(ee[3:6])
        check("rotvec round-trip %s" % label,
              np.abs(R_back - R_true).max() < TOL,
              "err=%.3e" % np.abs(R_back - R_true).max())
        check("reserved slot %s" % label, ee[7] == H.RESERVED_VALUE and ee.shape == (8,))

    # gripper normalisation is clamped outside the calibration range
    kp = make_synthetic_hand(anchor, R_true, 0.005)
    check("g clipped below d_min", H.hand_to_ee(kp, R_true)[0][6] == 0.0)
    kp = make_synthetic_hand(anchor, R_true, 0.20)
    check("g clipped above d_max", H.hand_to_ee(kp, R_true)[0][6] == 1.0)

    # camera -> base transform must compose, not be ignored
    R_bc = H.rotvec_to_rotmat(np.array([-0.4, 0.9, 0.2]))
    t_bc = np.array([0.5, -0.2, 0.8])
    T_bc = np.eye(4)
    T_bc[:3, :3] = R_bc
    T_bc[:3, 3] = t_bc
    kp = make_synthetic_hand(anchor, R_true, D_MID)
    ee, info = H.hand_to_ee(kp, R_bc @ R_true, T_bc, D_MIN, D_MAX)
    check("T_base_cam position",
          np.abs(ee[0:3] - (R_bc @ anchor + t_bc)).max() < TOL,
          "err=%.3e" % np.abs(ee[0:3] - (R_bc @ anchor + t_bc)).max())
    check("T_base_cam rotation",
          np.abs(info["R"] - R_bc @ R_true).max() < TOL,
          "err=%.3e" % np.abs(info["R"] - R_bc @ R_true).max())

    # documented palm-normal sign rule: reference axis picks the side of the plane
    kp = make_synthetic_hand(anchor, R_true, D_MID)
    R_wrist_flipped = R_true.copy()
    R_wrist_flipped[:, H.PALM_NORMAL_REFERENCE_AXIS] *= -1.0
    ee, info = H.hand_to_ee(kp, R_wrist_flipped, None, D_MIN, D_MAX)
    expected = R_true.copy()
    expected[:, 1] *= -1.0
    expected[:, 2] *= -1.0
    ref_axis = R_wrist_flipped[:, H.PALM_NORMAL_REFERENCE_AXIS]
    # the invariant is the *result*: Z must side with the wrist reference axis.
    # z_flipped itself is only a diagnostic -- it depends on which sign the SVD
    # happened to return, so it is not asserted.
    check("palm-normal sign rule",
          np.abs(info["R"] - expected).max() < TOL
          and float(np.dot(info["R"][:, 2], ref_axis)) > 0.99,
          "err=%.3e  Z.ref=%.6f  z_flipped=%s"
          % (np.abs(info["R"] - expected).max(), float(np.dot(info["R"][:, 2], ref_axis)),
             info["z_flipped"]))


def test_2_orthonormal():
    print("\n[2] orthonormality / chirality")
    rng = np.random.default_rng(7)
    worst_ortho = 0.0
    worst_det = 0.0
    worst_xz = 0.0
    for _ in range(50):
        R_true = H.rotvec_to_rotmat(rng.standard_normal(3))
        kp = make_synthetic_hand(rng.standard_normal(3), R_true, rng.uniform(D_MIN, D_MAX))
        _, info = H.hand_to_ee(kp, R_true)
        R = info["R"]
        worst_ortho = max(worst_ortho, np.abs(R.T @ R - np.eye(3)).max())
        worst_det = max(worst_det, abs(np.linalg.det(R) - 1.0))
        X, Z = R[:, 0], R[:, 2]
        worst_xz = max(worst_xz, abs(float(np.dot(X, Z))))
        check_ok = (np.abs(R.T @ R - np.eye(3)).max() < TOL
                    and abs(np.linalg.det(R) - 1.0) < TOL
                    and abs(float(np.dot(X, Z))) < TOL)
        if not check_ok:
            check("frame %d" % _, False, "R=%s" % np.array2string(R, precision=6))
            break
    else:
        check("50 random hand frames: |R^T R - I|, |det R - 1|, |X.Z| all < %g" % TOL, True,
              "max |R^T R-I|=%.2e, max |det-1|=%.2e, max |X.Z|=%.2e"
              % (worst_ortho, worst_det, worst_xz))


def test_3_nan():
    print("\n[3] NaN propagation (no silent zero-fill)")
    R_true = H.rotvec_to_rotmat(np.array([0.2, 0.1, -0.3]))
    kp = make_synthetic_hand(np.array([0.1, 0.2, 0.3]), R_true, D_MID)
    all_nan = True
    for i in range(H.NUM_KEYPOINTS):
        kp_bad = kp.copy()
        kp_bad[i, 1] = np.nan
        ee, info = H.hand_to_ee(kp_bad, R_true, None, D_MIN, D_MAX)
        if not (np.all(np.isnan(ee)) and not info["valid"]):
            all_nan = False
            check("keypoint %d NaN -> all-NaN output" % i, False,
                  "ee=%s" % np.array2string(ee, precision=4))
            break
    check("each of the 21 keypoints NaN -> all 8 outputs NaN", all_nan)

    # non-finite wrist / transform must be treated the same way
    kp_bad_R = kp.copy()
    R_nan = R_true.copy()
    R_nan[0, 0] = np.nan
    ee, info = H.hand_to_ee(kp_bad_R, R_nan)
    check("NaN R_wrist -> all-NaN output", np.all(np.isnan(ee)) and not info["valid"])
    T_nan = np.eye(4)
    T_nan[1, 3] = np.nan
    ee, info = H.hand_to_ee(kp, R_true, T_nan)
    check("NaN T_base_cam -> all-NaN output", np.all(np.isnan(ee)) and not info["valid"])

    # a valid frame in the same batch must still be finite
    ee, info = H.hand_to_ee(kp, R_true, None, D_MIN, D_MAX)
    check("finite input -> finite output", np.all(np.isfinite(ee)) and info["valid"])


def test_4_ik(ik):
    print("\n[4] IK convergence on 100 FK-sampled reachable targets")
    rng = np.random.default_rng(0)
    targets = []
    for _ in range(100):
        q = ik.random_qpos(rng)
        pos, rot = ik.fk(q)
        targets.append((pos, rot, q))

    warm_ok = 0
    pos_errs, ang_errs, iters_used, restarts_used = [], [], [], []
    fails = []
    for i, (pos, rot, q_true) in enumerate(targets):
        res = ik.solve_robust(pos, rot)
        pos_errs.append(res.pos_err)
        ang_errs.append(res.ang_err_deg)
        iters_used.append(res.iters)
        restarts_used.append(res.restarts)
        warm = ik.solve(pos, rot)
        warm_ok += int(warm.converged)
        if not res.converged:
            fails.append((i, pos, rot, q_true, res))

    pos_errs = np.asarray(pos_errs)
    ang_errs = np.asarray(ang_errs)
    n_ok = len(targets) - len(fails)
    rate = 100.0 * n_ok / len(targets)
    p50_p, p95_p, max_p = stats(pos_errs)
    p50_a, p95_a, max_a = stats(ang_errs)

    print("  warm start only (home seed, single DLS attempt): %d/100 converged" % warm_ok)
    print("  position error  [m]    p50=%.3e  p95=%.3e  max=%.3e" % (p50_p, p95_p, max_p))
    print("  orientation err [deg]  p50=%.4f  p95=%.4f  max=%.4f" % (p50_a, p95_a, max_a))
    print("  success rate           %d/100 = %.1f%%" % (n_ok, rate))
    print("  mean DLS iterations    %.1f  (mean extra restarts %.2f, max %d)"
          % (np.mean(iters_used), np.mean(restarts_used), max(restarts_used)))
    if fails:
        print("  FAILED SAMPLES (not hidden):")
        for i, pos, rot, q_true, res in fails[:5]:
            print("    target #%d pos=%s  q_true=%s  ->  pos_err=%.4e m ang_err=%.4f deg"
                  % (i, np.array2string(pos, precision=4), np.array2string(q_true, precision=3),
                     res.pos_err, res.ang_err_deg))

    check("success rate >= 95%", rate >= 95.0, "measured %.1f%%" % rate)
    check("converged solutions respect the tolerances",
          bool(np.all(pos_errs[:n_ok] < K.POS_TOL_M)) and bool(np.all(ang_errs[:n_ok] < K.ANG_TOL_DEG)))
    return targets


def test_5_limits(ik, targets):
    print("\n[5] joint limits respected")
    worst_margin = 0.0
    all_ok = True
    n_solutions = 0
    for pos, rot, _q in targets:
        res = ik.solve_robust(pos, rot)
        q = res.qpos
        n_solutions += 1
        if not ik.within_limits(q):
            all_ok = False
            print("    out of limits: %s" % np.array2string(q, precision=6))
        margin = float(np.min(np.minimum(q - ik.joint_limits[:, 0], ik.joint_limits[:, 1] - q)))
        worst_margin = min(worst_margin, margin) if n_solutions > 1 else margin
    check("all %d solutions inside the model joint limits" % n_solutions, all_ok,
          "smallest distance to a limit = %.3e rad" % worst_margin)
    check("within_limits() rejects an out-of-range vector",
          not ik.within_limits(ik.joint_limits[:, 1] + 0.01))
    print("  joint limits  [rad]    " + ", ".join(
        "[%.2f, %.2f]" % (lo, hi) for lo, hi in ik.joint_limits))


def test_5b_tcp_and_jacobian(ik):
    print("")
    print("[5b] configurable TCP offset + analytic Jacobian vs finite differences")
    tcp = np.eye(4)
    tcp[:3, :3] = H.rotvec_to_rotmat(np.array([0.0, 0.3, 0.0]))
    tcp[:3, 3] = np.array([0.02, -0.01, -0.08])
    ik_tcp = K.JakaZu3IK(tcp_offset=tcp)

    q = np.array([0.3, 0.7, -0.5, 1.1, 0.4, -0.9])
    p_site, R_site = ik_tcp.fk_site(q)
    p_tcp, R_tcp = ik_tcp.fk(q)
    check("TCP offset composition",
          np.abs(p_tcp - (p_site + R_site @ tcp[:3, 3])).max() < 1e-12
          and np.abs(R_tcp - R_site @ tcp[:3, :3]).max() < 1e-12,
          "|dp|=%.3e" % np.abs(p_tcp - (p_site + R_site @ tcp[:3, 3])).max())
    check("default solver is offset-free",
          np.abs(ik.fk(q)[0] - ik.fk_site(q)[0]).max() == 0.0)

    # central differences must reproduce [Jp; Jr] including the lever-arm term
    h = 1e-6
    J = ik_tcp.jacobian(q)
    Jp_num = np.zeros((3, 6))
    Jr_num = np.zeros((3, 6))
    for j in range(6):
        dq = np.zeros(6)
        dq[j] = h
        p_plus, R_plus = ik_tcp.fk(q + dq)
        p_minus, R_minus = ik_tcp.fk(q - dq)
        Jp_num[:, j] = (p_plus - p_minus) / (2.0 * h)
        Jr_num[:, j] = K.rotmat_to_rotvec(R_plus @ R_minus.T) / (2.0 * h)
    err_p = float(np.abs(J[0:3, :] - Jp_num).max())
    err_r = float(np.abs(J[3:6, :] - Jr_num).max())
    check("Jacobian rows vs central differences", err_p < 1e-5 and err_r < 1e-5,
          "max |dJp|=%.2e  max |dJr|=%.2e" % (err_p, err_r))

    # and the offset is actually used by IK: solve a reachable TCP target
    rng = np.random.default_rng(3)
    q_ref = ik_tcp.random_qpos(rng)
    p_ref, R_ref = ik_tcp.fk(q_ref)
    res = ik_tcp.solve_robust(p_ref, R_ref)
    p_fk, R_fk = ik_tcp.fk(res.qpos)
    pos_err, ang_err = K.pose_error(p_ref, R_ref, p_fk, R_fk)
    check("IK converges on a TCP-offset target",
          res.converged and pos_err < K.POS_TOL_M and ang_err < K.ANG_TOL_DEG,
          "pos_err=%.3e m ang_err=%.4f deg iters=%d" % (pos_err, ang_err, res.iters))


def test_6_end_to_end(ik):
    print("\n[6] end-to-end: synthetic hand -> EE(8) -> IK -> qpos -> FK")
    rng = np.random.default_rng(11)
    worst_pos = 0.0
    worst_ang = 0.0
    for trial in range(5):
        q_ref = ik.random_qpos(rng)
        pos_ref, rot_ref = ik.fk(q_ref)

        # a hand whose thenar anchor sits exactly on the site and whose plane
        # frame equals the site rotation; R_wrist = rot_ref fixes the normal side
        kp = make_synthetic_hand(pos_ref, rot_ref, D_MID)
        ee, info = H.hand_to_ee(kp, rot_ref, None, D_MIN, D_MAX)
        ee_pos, ee_rot, g, _ = H.ee_to_pose(ee)
        pre_pos = float(np.linalg.norm(ee_pos - pos_ref))
        pre_ang = H.rotation_angle_deg(ee_rot, rot_ref)

        res = ik.solve_robust(ee_pos, ee_rot)
        pos_fk, rot_fk = ik.fk(res.qpos)
        pos_err, ang_err = K.pose_error(ee_pos, ee_rot, pos_fk, rot_fk)
        worst_pos = max(worst_pos, pos_err, pre_pos)
        worst_ang = max(worst_ang, ang_err, pre_ang)
        print("  trial %d: hand->EE residual %.2e m / %.2e deg | IK residual %.3e m / %.4f deg "
              "| converged=%s iters=%d | g=%.3f"
              % (trial, pre_pos, pre_ang, pos_err, ang_err, res.converged, res.iters, g))
        check("trial %d EE residual < 1e-3 m / 0.5 deg" % trial,
              pre_pos < 1e-3 and pre_ang < 0.5)
        check("trial %d IK residual < 1e-3 m / 0.5 deg" % trial,
              pos_err < 1e-3 and ang_err < 0.5 and res.converged)
    print("  worst residual         %.3e m / %.4f deg" % (worst_pos, worst_ang))


# --------------------------------------------------------------------------- #
# rotvec <-> rotation matrix: the 180 deg branch, its neighbourhood, and the
# IK-layer counterexample that used to be invisible to the solver
# --------------------------------------------------------------------------- #
def _skew3(v):
    v = np.asarray(v, dtype=np.float64)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _axis180(axis):
    """Exact 180 deg rotation about a unit axis: R = 2 a a^T - I."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return 2.0 * np.outer(a, a) - np.eye(3)


def _haar_rotations(rng, n):
    """n Haar-uniform rotations built from quaternions, independently of the module."""
    out = []
    for _ in range(n):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        out.append(np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]))
    return out


def _rotvec_roundtrip(R):
    """max |R - rotvec_to_rotmat(rotmat_to_rotvec(R))| for one rotation matrix."""
    return float(np.abs(R - K.rotvec_to_rotmat(K.rotmat_to_rotvec(R))).max())


def test_7_rotvec_pi():
    print("\n[7] rotmat_to_rotvec at exactly 180 deg (before the fix: [0, 0, 0])")
    v_bug = K.rotmat_to_rotvec(np.diag([-1.0, 1.0, -1.0]))
    print("  diag(-1, 1, -1) -> v = %s   (defective body returned [0. 0. 0.])"
          % np.array2string(v_bug, precision=9))
    check("diag(-1, 1, -1) is no longer read as zero rotation",
          abs(float(np.linalg.norm(v_bug)) - np.pi) < 1e-12,
          "|v| = %.17f  |v|-pi = %+.2e" % (np.linalg.norm(v_bug),
                                           np.linalg.norm(v_bug) - np.pi))

    cases = [("Rx(pi)", np.diag([1.0, -1.0, -1.0])),
             ("Ry(pi)", np.diag([-1.0, 1.0, -1.0])),
             ("Rz(pi)", np.diag([-1.0, -1.0, 1.0])),
             ("180 deg about (1,1,0)", _axis180([1.0, 1.0, 0.0])),
             ("180 deg about (1,2,3)", _axis180([1.0, 2.0, 3.0]))]
    worst_dpi = 0.0
    worst_rt = 0.0
    for name, R in cases:
        v = K.rotmat_to_rotvec(R)
        nv = float(np.linalg.norm(v))
        d_pi = abs(nv - np.pi)
        rt = _rotvec_roundtrip(R)
        worst_dpi = max(worst_dpi, d_pi)
        worst_rt = max(worst_rt, rt)
        check("%s: |v| = pi and rotvec_to_rotmat(v) = R" % name,
              d_pi < 1e-12 and rt < 1e-12 and np.all(np.isfinite(v)),
              "|v|=%.17f  |v|-pi=%+.2e  max|R-R(v)|=%.3e  v=%s"
              % (nv, nv - np.pi, rt, np.array2string(v, precision=6)))

    rng = np.random.default_rng(180)
    worst_dpi = 0.0
    worst_rt = 0.0
    for _ in range(2000):
        R = _axis180(rng.standard_normal(3))
        v = K.rotmat_to_rotvec(R)
        worst_dpi = max(worst_dpi, abs(float(np.linalg.norm(v)) - np.pi))
        worst_rt = max(worst_rt, _rotvec_roundtrip(R))
    check("2000 random-axis exact 180 deg rotations",
          worst_dpi < 1e-12 and worst_rt < 1e-12,
          "max ||v|-pi|=%.3e  max max|R-R(v)|=%.3e" % (worst_dpi, worst_rt))


def test_8_rotvec_boundary():
    print("\n[8] rotvec boundary: theta = pi +/- 1e-9, pi - 1e-12, and theta ~ 0")
    rng = np.random.default_rng(7)
    worst_rt = 0.0
    worst_pi = 0.0
    for theta in (np.pi, np.pi - 1e-12, np.pi + 1e-12,
                  np.pi - 1e-9, np.pi + 1e-9):
        # a rotation of pi + eps is the same as one of 2 pi - (pi + eps) the other way
        expected = min(theta, 2.0 * np.pi - theta)
        w_ang = 0.0
        w_rt = 0.0
        finite = True
        flipped = False
        for _ in range(100):
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            R = K.rotvec_to_rotmat(axis * theta)
            v = K.rotmat_to_rotvec(R)
            nv = float(np.linalg.norm(v))
            finite = finite and bool(np.all(np.isfinite(v)))
            flipped = flipped or nv > np.pi + 1e-12     # |v| > pi == wrong half-turn
            w_ang = max(w_ang, abs(nv - expected))
            w_rt = max(w_rt, _rotvec_roundtrip(R))
        worst_rt = max(worst_rt, w_rt)
        worst_pi = max(worst_pi, w_ang)
        check("theta = %.12f (+/- 1e-12): finite, |v| <= pi, angle and matrix exact"
              % theta,
              finite and not flipped and w_ang < 1e-14 and w_rt < 1e-12,
              "max ||v|-theta|=%.2e  max|R-R(v)|=%.2e  finite=%s  |v|<=pi=%s"
              % (w_ang, w_rt, finite, not flipped))

    # theta ~ 0: build the matrix by hand, because the module's own rotvec_to_rotmat
    # snaps |rotvec| < 1e-12 to the identity (:121-122) and would mask the check here.
    for theta in (1e-12, 1e-13, 1e-16):
        w_ang = 0.0
        finite = True
        for _ in range(100):
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            A = _skew3(axis)
            R = np.eye(3) + np.sin(theta) * A + (1.0 - np.cos(theta)) * (A @ A)
            v = K.rotmat_to_rotvec(R)
            finite = finite and bool(np.all(np.isfinite(v)))
            w_ang = max(w_ang, abs(float(np.linalg.norm(v)) - theta))
        check("theta = %.0e: finite and |v| = theta" % theta,
              finite and w_ang < 1e-18,
              "max ||v|-theta|=%.2e  finite=%s" % (w_ang, finite))
    check("identity -> [0, 0, 0] exactly",
          np.array_equal(K.rotmat_to_rotvec(np.eye(3)), np.zeros(3)),
          "v = %s" % K.rotmat_to_rotvec(np.eye(3)))
    print("  worst over the pi neighbourhood: max ||v|-theta|=%.3e  max|R-R(v)|=%.3e"
          % (worst_pi, worst_rt))


def test_9_rotvec_random():
    print("\n[9] 5000 random SO(3) round-trips")
    rng = np.random.default_rng(0)
    rots = _haar_rotations(rng, 5000)
    worst = 0.0
    flips = 0
    for R in rots:
        v = K.rotmat_to_rotvec(R)
        worst = max(worst, float(np.abs(R - K.rotvec_to_rotmat(v)).max()))
        if float(np.linalg.norm(v)) > np.pi + 1e-12:
            flips += 1
    check("max |R - rotvec_to_rotmat(rotmat_to_rotvec(R))| < 1e-12",
          worst < 1e-12, "measured %.3e over 5000 rotations" % worst)
    check("every |v| <= pi (no flipped half-turn)", flips == 0,
          "%d of 5000 outside" % flips)


def test_10_rotvec_vs_reference():
    print("\n[10] cross-check: ik_jaka_zu3.rotmat_to_rotvec vs hand_to_ee's")
    rng = np.random.default_rng(11)
    rots = _haar_rotations(rng, 5000)
    worst_same = 0.0
    worst_ik = 0.0
    worst_ref = 0.0
    for R in rots:
        v_ik = K.rotmat_to_rotvec(R)
        v_ref = H.rotmat_to_rotvec(R)
        M = K.rotvec_to_rotmat(v_ik).T @ H.rotvec_to_rotmat(v_ref)
        worst_same = max(worst_same, float(np.degrees(np.arccos(
            np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0)))))
        worst_ik = max(worst_ik, float(np.abs(R - K.rotvec_to_rotmat(v_ik)).max()))
        worst_ref = max(worst_ref, float(np.abs(R - H.rotvec_to_rotmat(v_ref)).max()))
    # the two vectors may differ by which end of the half-turn they pick, so compare
    # the rotations they describe, not the vectors
    check("both describe the same rotation (max angle difference)",
          worst_same < 1e-5, "max %.3e deg over 5000 rotations" % worst_same)
    check("ik_jaka_zu3 round-trip on all 5000 is < 1e-12", worst_ik < 1e-12,
          "max %.3e" % worst_ik)
    check("the reference (arccos + vee/sin, pi branch) is the weaker one",
          worst_ref > worst_ik, "reference max %.3e vs %.3e" % (worst_ref, worst_ik))
    for name, R in (("Rx(pi)", np.diag([1.0, -1.0, -1.0])),
                    ("diag(-1,1,-1)", np.diag([-1.0, 1.0, -1.0]))):
        v_ik = K.rotmat_to_rotvec(R)
        v_ref = H.rotmat_to_rotvec(R)
        M = K.rotvec_to_rotmat(v_ik).T @ H.rotvec_to_rotmat(v_ref)
        ang = float(np.degrees(np.arccos(np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0))))
        print("  %-14s v_ik = %s   v_ref = %s"
              % (name, np.array2string(v_ik, precision=6),
                 np.array2string(v_ref, precision=6)))
        check("%s: both implementations agree" % name, ang < 1e-3,
              "angle between them %.3e deg" % ang)


def test_11_ik_180(ik):
    print("\n[11] IK layer: target with the SAME position and a rotation 180 deg away")
    q_seed = np.array([0.3, 0.7, -0.5, 1.1, 0.4, -0.9])
    pos_ref, rot_ref = ik.fk(q_seed)
    print("  seed q = %s   |  position held fixed, orientation turned by pi"
          % np.array2string(q_seed, precision=3))
    for name, R180 in (("Rx(pi)", np.diag([1.0, -1.0, -1.0])),
                       ("diag(-1,1,-1)", np.diag([-1.0, 1.0, -1.0]))):
        target_rot = R180 @ rot_ref
        pos_now, rot_now = ik.fk(q_seed)
        pos_err, ang_err = K.pose_error(pos_ref, target_rot, pos_now, rot_now)
        v = K.rotmat_to_rotvec(target_rot @ rot_now.T)
        nv = float(np.linalg.norm(v))
        term = K.ROT_WEIGHT * nv
        print("  %-14s pose_error = %.3e m / %.6f deg | rotvec = %s |v| = %.6f"
              % (name, pos_err, ang_err, np.array2string(v, precision=6), nv))
        print("  %-14s DLS orientation term = %.9f  (before the fix: 0.000000000)"
              % ("", term))
        check("%s: reported orientation residual is ~180 deg" % name,
              abs(ang_err - 180.0) < 1e-3, "%.6f deg" % ang_err)
        check("%s: DLS orientation term is rot_weight * pi, not 0" % name,
              abs(term - K.ROT_WEIGHT * np.pi) < 1e-9,
              "%.9f vs %.9f" % (term, K.ROT_WEIGHT * np.pi))
        res = ik.solve(pos_ref, target_rot, q_seed=q_seed)
        moved = float(np.abs(res.qpos - q_seed).max())
        print("  %-14s solve() -> %r  |dq|_inf = %.4f" % ("", res, moved))
        check("%s: the attempt leaves the seed instead of stalling" % name,
              res.iters > 1 and moved > 1e-3,
              "iters=%d |dq|_inf=%.4f (before the fix: iters=1 |dq|_inf=0.0)"
              % (res.iters, moved))


def test_12_default_model():
    print("\n[12] default model: no-arg JakaZu3IK() and EASYMIMIC_JAKA_XML")
    repo = os.path.dirname(_HERE)
    asset = os.path.join(repo, "assets", "jaka_zu3", "robot.xml")
    scene = os.path.join(repo, "assets", "jaka_zu3", "scene_cube_bowl.xml")
    check("DEFAULT_XML_PATH is the repo asset",
          os.path.abspath(K.DEFAULT_XML_PATH) == os.path.abspath(asset),
          K.DEFAULT_XML_PATH)
    check("DEFAULT_XML_PATH no longer points into third_party",
          "third_party" not in K.DEFAULT_XML_PATH)

    t0 = time.perf_counter()
    ik = K.JakaZu3IK()
    load_s = time.perf_counter() - t0
    print("  no-arg JakaZu3IK(): %s" % ik.xml_path)
    print("  nq=%d nv=%d nu=%d njnt=%d  site %r  loaded in %.3f s"
          % (ik.model.nq, ik.model.nv, ik.model.nu, ik.model.njnt,
             ik.site_name, load_s))
    check("no-arg JakaZu3IK() loads the repo asset",
          os.path.abspath(ik.xml_path) == os.path.abspath(asset))
    check("default model is arm + jaw only (nq = nu = njnt = 7, no free joint)",
          ik.model.nq == 7 and ik.model.nu == 7 and ik.model.njnt == 7,
          "nq=%d nu=%d njnt=%d" % (ik.model.nq, ik.model.nu, ik.model.njnt))
    pos_home, rot_home = ik.fk(ik.qpos_home)
    res = ik.solve_robust(pos_home, rot_home)
    check("the no-arg instance solves",
          res.converged and res.pos_err < K.POS_TOL_M and res.ang_err_deg < K.ANG_TOL_DEG,
          "%r" % (res,))
    check("a relative xml_path is still rejected",
          _raises_value_error(lambda: K.JakaZu3IK("assets/jaka_zu3/robot.xml"),
                              "must be absolute"))

    old = os.environ.get("EASYMIMIC_JAKA_XML")
    try:
        os.environ["EASYMIMIC_JAKA_XML"] = scene
        check("EASYMIMIC_JAKA_XML overrides default_xml_path()",
              os.path.abspath(K.default_xml_path()) == os.path.abspath(scene),
              K.default_xml_path())
        ik_env = K.JakaZu3IK()
        check("EASYMIMIC_JAKA_XML is used by the no-arg constructor",
              os.path.abspath(ik_env.xml_path) == os.path.abspath(scene)
              and ik_env.model.nq == 14,
              "nq=%d path=%s" % (ik_env.model.nq, ik_env.xml_path))
    finally:
        if old is None:
            os.environ.pop("EASYMIMIC_JAKA_XML", None)
        else:
            os.environ["EASYMIMIC_JAKA_XML"] = old
    check("unsetting EASYMIMIC_JAKA_XML returns to the repo asset",
          os.path.abspath(K.default_xml_path()) == os.path.abspath(asset),
          K.default_xml_path())

    old_model = os.path.join(repo, "third_party", "jaka-zu3-mujoco",
                             "jaka_zu3_mujoco.xml")
    if os.path.isfile(old_model):
        err = None
        try:
            K.JakaZu3IK(old_model)
        except ValueError as exc:       # MuJoCo 3.16 rejects the flexcomp'ed rag
            err = str(exc)
        check("the old third_party default still fails to load under MuJoCo",
              err is not None and "flex" in err,
              "" if err is None else err.splitlines()[0][:64])
    else:
        print("  (third_party checkout absent: skipping the old-default demo)")


def _raises_value_error(fn, needle):
    try:
        fn()
    except ValueError as exc:
        return needle in str(exc)
    return False


def main():
    print("=" * 78)
    print("8-D EE action layer acceptance test (numpy + mujoco only)")
    print("src: %s" % _SRC)
    print("=" * 78)

    test_1_analytic()
    test_2_orthonormal()
    test_3_nan()
    test_7_rotvec_pi()
    test_8_rotvec_boundary()
    test_9_rotvec_random()
    test_10_rotvec_vs_reference()

    ik = K.JakaZu3IK()
    print("\n  model: %s" % ik.xml_path)
    pos0, rot0 = ik.fk(ik.qpos_home)
    print("  site %r at qpos0 = [%.5f, %.5f, %.5f]" % (ik.site_name, pos0[0], pos0[1], pos0[2]))
    print("  arm DoF columns %s (nq=%d, nv=%d; props have free joints, hence by name)"
          % (ik.dof_adr.tolist(), ik.model.nq, ik.model.nv))

    targets = test_4_ik(ik)
    test_5_limits(ik, targets)
    test_5b_tcp_and_jacobian(ik)
    test_6_end_to_end(ik)
    test_11_ik_180(ik)
    test_12_default_model()

    print("\n" + "=" * 78)
    if failures:
        print("FAILED %d check(s):" % len(failures))
        for name in failures:
            print("  - %s" % name)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

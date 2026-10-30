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
"""

from __future__ import annotations

import os
import sys

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


def main():
    print("=" * 78)
    print("8-D EE action layer acceptance test (numpy + mujoco only)")
    print("src: %s" % _SRC)
    print("=" * 78)

    test_1_analytic()
    test_2_orthonormal()
    test_3_nan()

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

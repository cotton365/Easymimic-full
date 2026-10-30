"""T3: offline minimum closed loop -- synthetic hand -> hand_to_ee -> IK -> MuJoCo replay.

Pipeline (one direction only; nothing after the landmarks is hand-written):

    8-D EE command in the base frame (ground truth for this test)
      -> synth_hand.keypoints_for_pose   21 landmarks + R_wrist, camera frame == base
      -> hand_to_ee.hand_to_ee           must recover the same ee8 to 1e-9
      -> JakaZu3IK.solve                 6 joint angles; TCP = tool0 + R @ (0,-0.0325,0)
      -> MuJoCo position servos          ctrl = IK qpos, jaw ctrl = g * 0.04, 1/30 s/frame
      -> physical grasp, 0.10 m lift, PNG frames

Nothing here writes an EE waypoint by hand: the 75 positions come from the
descent/hold/close/lift schedule, the joint targets come from the IK, and the
only robot input is the servo setpoint.

Two control modes run the *same* trajectory so a plant defect cannot hide
behind a trajectory fix:

    direct -- ctrl = IK qpos, exactly as the task brief words it.
    servo  -- same trajectory and same IK, plus a per-frame integral Jacobian
              correction on the 6 joints (the assets have position servos with
              no gravity feedforward, so the raw setpoint sags tens of mm).

Reported per frame for both: IK iterations, restarts, residuals, |dq|, the
commanded vs achieved TCP, the lowest pad coordinate, and the cube height.

Run: /home/zh/miniconda3/envs/mujoco/bin/python easymimic_repro/src/closed_loop/run_closed_loop.py
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_REPRO = os.path.dirname(_SRC)
for _p in (os.path.join(_SRC, "ee_retarget"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hand_to_ee as h2e            # noqa: E402
import ik_jaka_zu3 as ikmod         # noqa: E402
import mujoco                       # noqa: E402
import synth_hand as sh             # noqa: E402

XML_PATH = os.path.join(_REPRO, "assets", "jaka_zu3", "scene_cube_bowl.xml")
OUT_DIR = os.path.join(_REPRO, "out", "t3_frames")

# pad-centre midpoint in the tool frame: TCP = tool0 + R @ TCP_OFFSET
TCP_OFFSET = np.array([0.0, -0.0325, 0.0])
PAD_HALF_SIZE = np.array([0.005, 0.0225, 0.011])   # gripper_tip_* box size in gripper_pg2.xml

FRAME_DT = 1.0 / 30.0
SETTLE_SECONDS = 1.0
DESCEND_HEIGHT = 0.15
LIFT_HEIGHT = 0.10
N_DESCEND = 25
N_HOLD_OPEN = 5
N_CLOSE = 10
N_LIFT = 15
N_HOLD_END = 20
MAX_RESTARTS = 3            # real-time budget: one warm attempt + at most 3 reseeds
PAD_CLEARANCE = 0.002   # minimum pad-bottom clearance above the table top
GRASP_OVERLAP = 0.004       # how far the pad bottom ends up below the cube top
RENDER_EVERY = 5
RENDER_HW = (480, 640)
MIN_PNG_MEAN = 5.0
DELTA_Q_LIMIT = 0.35


# --------------------------------------------------------------------------- #
# zero-dependency PNG writer / reader (zlib + struct only)
# --------------------------------------------------------------------------- #
def write_png(path, rgb):
    """Write an (h, w, 3) uint8 array as a truecolour PNG; returns bytes written."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)
    return len(png)


def png_stats(path):
    """Decode a PNG produced by write_png: (w, h, mean, nbytes)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("%s is not a PNG" % path)
    pos, w, h, idat = 8, 0, 0, b""
    while pos < len(data):
        (ln,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", data[pos + 8:pos + 16])
        elif tag == b"IDAT":
            idat += data[pos + 8:pos + 8 + ln]
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = 3 * w + 1
    if len(raw) != stride * h:
        raise ValueError("%s: unexpected IDAT size %d" % (path, len(raw)))
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(h, stride)
    if np.any(rows[:, 0] != 0):
        raise ValueError("%s: unexpected PNG row filter" % path)
    return w, h, float(rows[:, 1:].mean()), len(data)


# --------------------------------------------------------------------------- #
# trajectory: pure EE-space schedule, no robot knowledge
# --------------------------------------------------------------------------- #
def build_trajectory(grasp_pos, frame_dt=FRAME_DT):
    """Return (pos (N,3), g (N,), phase (N,) str) for the grasp sequence."""
    gx, gy, gz = (float(v) for v in grasp_pos)
    pos, g, phase = [], [], []

    for i in range(N_DESCEND):
        s = i / (N_DESCEND - 1.0)
        pos.append((gx, gy, gz + DESCEND_HEIGHT * (1.0 - s)))
        g.append(1.0)
        phase.append("descend")
    for _ in range(N_HOLD_OPEN):
        pos.append((gx, gy, gz))
        g.append(1.0)
        phase.append("hold_open")
    for i in range(N_CLOSE):
        pos.append((gx, gy, gz))
        g.append(1.0 - (i + 1) / float(N_CLOSE))
        phase.append("close")
    for i in range(N_LIFT):
        s = (i + 1) / float(N_LIFT)
        pos.append((gx, gy, gz + LIFT_HEIGHT * s))
        g.append(0.0)
        phase.append("lift")
    for _ in range(N_HOLD_END):
        pos.append((gx, gy, gz + LIFT_HEIGHT))
        g.append(0.0)
        phase.append("hold_end")

    return np.array(pos), np.array(g), np.array(phase)


# --------------------------------------------------------------------------- #
# stage 1: keypoints -> hand_to_ee  (acceptance A)
# --------------------------------------------------------------------------- #
def stage_retarget(traj_pos, traj_g, R_tool):
    n = len(traj_g)
    kp = np.zeros((n, h2e.NUM_KEYPOINTS, 3))
    ee8 = np.zeros((n, h2e.EE_DIM))
    ref = np.zeros((n, h2e.EE_DIM))
    pos_err = np.zeros(n)
    ang_err = np.zeros(n)
    g_err = np.zeros(n)
    plane_res = np.zeros(n)
    d_t = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    flipped = np.zeros(n, dtype=bool)

    for i in range(n):
        kp[i], R_wrist, d_t[i] = sh.keypoints_for_pose(traj_pos[i], R_tool, traj_g[i])
        ee8[i], info = h2e.hand_to_ee(kp[i], R_wrist, None,
                                      h2e.DEFAULT_D_MIN, h2e.DEFAULT_D_MAX)
        ref[i] = sh.ee8_from_pose(traj_pos[i], R_tool, traj_g[i])
        pos_err[i] = float(np.linalg.norm(ee8[i, 0:3] - ref[i, 0:3]))
        ang_err[i] = h2e.rotation_angle_deg(h2e.rotvec_to_rotmat(ee8[i, 3:6]),
                                            h2e.rotvec_to_rotmat(ref[i, 3:6]))
        g_err[i] = abs(ee8[i, 6] - ref[i, 6])
        plane_res[i] = info["plane_residual_m"]
        valid[i] = bool(info["valid"])
        flipped[i] = bool(info["z_flipped"])

    return dict(kp=kp, ee8=ee8, ref=ref, pos_err=pos_err, ang_err=ang_err,
                g_err=g_err, plane_res=plane_res, d_t=d_t, valid=valid, flipped=flipped)


# --------------------------------------------------------------------------- #
# stage 2: ee8 -> IK  (acceptance B and C)
# --------------------------------------------------------------------------- #
def stage_ik(ik, ee8, max_restarts=MAX_RESTARTS):
    n = len(ee8)
    q = np.zeros((n, 6))
    iters = np.zeros(n, dtype=int)
    restarts = np.zeros(n, dtype=int)
    solve_ms = np.zeros(n)
    pos_res = np.zeros(n)
    ang_res = np.zeros(n)
    single_ok = np.zeros(n, dtype=bool)
    ok = np.zeros(n, dtype=bool)
    cond = np.zeros(n)

    prev = None
    for i in range(n):
        tgt_p = ee8[i, 0:3]
        tgt_R = h2e.rotvec_to_rotmat(ee8[i, 3:6])
        seed = ik.qpos_home if prev is None else prev
        t0 = time.perf_counter()
        res = ik.solve(tgt_p, tgt_R, seed)
        single_ok[i] = bool(res.converged)
        if not res.converged:
            res = ik.solve_robust(tgt_p, tgt_R, seed, n_restarts=max_restarts)
        solve_ms[i] = 1e3 * (time.perf_counter() - t0)
        q[i] = res.qpos
        iters[i] = res.iters
        restarts[i] = res.restarts
        ok[i] = bool(res.converged)
        pos_res[i] = res.pos_err
        ang_res[i] = res.ang_err_deg
        cond[i] = float(np.linalg.svd(ik.jacobian(res.qpos), compute_uv=False).min())
        prev = res.qpos

    dq = np.zeros(n)
    dq[0] = float(np.max(np.abs(q[0] - ik.qpos_home)))
    dq[1:] = np.max(np.abs(np.diff(q, axis=0), axis=1))
    return dict(q=q, iters=iters, restarts=restarts, solve_ms=solve_ms, pos_res=pos_res,
                ang_res=ang_res, single_ok=single_ok, ok=ok, cond=cond, dq=dq)


# --------------------------------------------------------------------------- #
# MuJoCo plant helpers
# --------------------------------------------------------------------------- #
def arm_qpos_addr(model):
    return np.array([int(model.jnt_qposadr[model.joint(n).id]) for n in ikmod.ARM_JOINT_NAMES])


def tcp_from_state(model, data, site_id):
    """Measured TCP = tool0 + R @ TCP_OFFSET, from the simulated state."""
    R = data.site_xmat[site_id].reshape(3, 3)
    return data.site_xpos[site_id] + R @ TCP_OFFSET, R


def pad_geom_ids(model):
    return [i for i in range(model.ngeom)
            if str(model.geom(i).name).startswith("gripper_tip")]


def pad_extent(model, R):
    """Half extent of a pad box along world z for a pad aligned with rotation R."""
    return float(np.sum(np.abs(R[2, :]) * PAD_HALF_SIZE))


def pad_lowest_z(model, data, pad_ids):
    lo = np.inf
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=np.float64)
    for gid in pad_ids:
        corners = (signs * model.geom_size[gid]) @ data.geom_xmat[gid].reshape(3, 3).T \
            + data.geom_xpos[gid]
        lo = min(lo, float(corners[:, 2].min()))
    return lo


def cube_contact_counts(model, data, cube_geoms, pad_ids):
    """(#cube-pad contacts, #cube-table contacts) in the current state."""
    n_pad = n_table = 0
    for k in range(data.ncon):
        c = data.contact[k]
        pair = (int(c.geom1), int(c.geom2))
        if not (pair[0] in cube_geoms or pair[1] in cube_geoms):
            continue
        other = pair[1] if pair[0] in cube_geoms else pair[0]
        if other in pad_ids:
            n_pad += 1
        else:
            n_table += 1
    return n_pad, n_table


def _dls_dq(ik, q_act, err6, lam=0.08):
    """One damped-least-squares joint step that reduces a 6-D TCP error."""
    J = ik.jacobian(q_act)
    return J.T @ np.linalg.solve(J @ J.T + (lam * lam) * np.eye(6), err6)


def settle(model, data, seconds=SETTLE_SECONDS):
    """Reset to the home keyframe, let gravity settle, return the cube position."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
    qadr = int(model.jnt_qposadr[model.joint("cube_free").id])
    return data.qpos[qadr:qadr + 3].copy()


# --------------------------------------------------------------------------- #
# stage 3: joint targets -> MuJoCo (acceptance D)
# --------------------------------------------------------------------------- #
def run_physics(model, data, ik, traj_pos, traj_g, q_cmd, R_tool, mode,
                renderer=None, camera=None, png_dir=None, render_frames=()):
    site_id = model.site("tool0").id
    qadr = arm_qpos_addr(model)
    pad_ids = pad_geom_ids(model)
    cube_body = model.body("cube").id
    cube_qadr = int(model.jnt_qposadr[model.joint("cube_free").id])
    cube_geoms = tuple(i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == cube_body)

    cube0 = settle(model, data)
    t_start = data.time
    ctrl = data.ctrl.copy()
    comp = np.zeros(6)
    n = len(traj_g)

    rec = {k: np.zeros(n) for k in
           ("t", "cmd_z", "act_z", "trk_err", "ang_err_deg", "pad_low", "jaw_qpos",
            "cube_z", "cube_x", "cube_y", "n_steps", "comp_inf", "n_pad_con", "n_tab_con")}
    rec["act_pos"] = np.zeros((n, 3))
    rec["cmd_pos"] = np.zeros((n, 3))
    pngs = []

    for i in range(n):
        if mode == "servo":
            p_act, R_act = tcp_from_state(model, data, site_id)
            q_act = data.qpos[qadr].copy()
            err = np.concatenate([traj_pos[i] - p_act,
                                  0.5 * h2e.rotmat_to_rotvec(R_tool @ R_act.T)])
            step = np.clip(_dls_dq(ik, q_act, err), -0.06, 0.06)
            comp = np.clip(comp + step, -0.5, 0.5)
            ctrl[0:6] = q_cmd[i] + comp
        else:
            ctrl[0:6] = q_cmd[i]
        ctrl[6] = traj_g[i] * 0.04
        data.ctrl[:] = ctrl

        n_steps = 0
        deadline = t_start + (i + 1) * FRAME_DT
        while data.time < deadline - 1e-9:
            mujoco.mj_step(model, data)
            n_steps += 1

        p_act, R_act = tcp_from_state(model, data, site_id)
        n_pad_con, n_tab_con = cube_contact_counts(model, data, cube_geoms, pad_ids)
        rec["t"][i] = data.time - t_start
        rec["cmd_pos"][i] = traj_pos[i]
        rec["cmd_z"][i] = traj_pos[i][2]
        rec["act_pos"][i] = p_act
        rec["act_z"][i] = p_act[2]
        rec["trk_err"][i] = float(np.linalg.norm(p_act - traj_pos[i]))
        rec["ang_err_deg"][i] = h2e.rotation_angle_deg(R_act, R_tool)
        rec["pad_low"][i] = pad_lowest_z(model, data, pad_ids)
        rec["jaw_qpos"][i] = float(data.qpos[int(model.jnt_qposadr[model.joint("gripper_jaw").id])])
        rec["cube_z"][i] = float(data.qpos[cube_qadr + 2])
        rec["cube_x"][i] = float(data.qpos[cube_qadr + 0])
        rec["cube_y"][i] = float(data.qpos[cube_qadr + 1])
        rec["n_steps"][i] = n_steps
        rec["comp_inf"][i] = float(np.max(np.abs(comp)))
        rec["n_pad_con"][i] = n_pad_con
        rec["n_tab_con"][i] = n_tab_con

        if renderer is not None and i in render_frames:
            path = os.path.join(png_dir, "frame_%03d.png" % i)
            renderer.update_scene(data, camera)
            pngs.append((i, path, write_png(path, renderer.render())))

    rec["cube0"] = cube0
    rec["pngs"] = pngs
    return rec



def final_grasp_geometry(model, data, cube_half):
    """Cube centre in the tool frame plus the pad inner-face gap from the live state."""
    site_id = model.site("tool0").id
    p, R = tcp_from_state(model, data, site_id)
    qadr = int(model.jnt_qposadr[model.joint("cube_free").id])
    centre = data.qpos[qadr:qadr + 3].copy()
    u = R.T @ (centre - p)

    spans = []
    for gid in pad_geom_ids(model):
        along = float((data.geom_xpos[gid] - p) @ R[:, 0])
        half = float(model.geom_size[gid][0])
        spans.append((along - half, along + half))
    spans.sort()
    left_face, right_face = spans[0][1], spans[-1][0]
    jaw = float(data.qpos[int(model.jnt_qposadr[model.joint("gripper_jaw").id])])
    return u, left_face, right_face, jaw


class Checker:
    """Collect PASS/FAIL lines and decide the exit status at the very end."""

    def __init__(self):
        self.items = []

    def check(self, cond, label, detail=""):
        ok = bool(cond)
        self.items.append((ok, label))
        print("  [%s] %s%s" % ("PASS" if ok else "FAIL", label, ("   " + detail) if detail else ""))
        return ok

    def summary(self):
        bad = [lab for ok, lab in self.items if not ok]
        print()
        print("=" * 78)
        print("VERDICT: %d/%d checks passed" % (len(self.items) - len(bad), len(self.items)))
        for lab in bad:
            print("  FAILED: %s" % lab)
        return 0 if not bad else 1


def _pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def print_frame_table_joined(rec, ik_rec, phase):
    print()
    print("  frame phase      cmd EE pos (x, y, z)      act_z  |trk| iters rst   |dq|"
          "      g    cube_z  pad_low    jaw  npad")
    print("  " + "-" * 122)
    for i in range(len(phase)):
        print("  %5d %-9s %7.4f %7.4f %7.4f  %7.4f %6.4f %5d %3d %6.4f  %5.3f  %7.4f %7.4f %6.4f  %4d"
              % (i, phase[i], rec["cmd_pos"][i][0], rec["cmd_pos"][i][1], rec["cmd_pos"][i][2],
                 rec["act_z"][i], rec["trk_err"][i],
                 ik_rec["iters"][i], ik_rec["restarts"][i], ik_rec["dq"][i],
                 1.0 - rec["jaw_qpos"][i] / 0.04,  # jaw opening normalised back to g
                 rec["cube_z"][i], rec["pad_low"][i], rec["jaw_qpos"][i],
                 rec["n_pad_con"][i]))


def phase_tracking(rec, phase):
    out = []
    for name in ("descend", "hold_open", "close", "lift", "hold_end"):
        m = phase == name
        if not np.any(m):
            continue
        out.append((name, int(m.sum()),
                    float(np.mean(rec["trk_err"][m])),
                    float(np.max(rec["trk_err"][m])),
                    float(np.mean(rec["pad_low"][m])),
                    float(np.mean(rec["n_tab_con"][m]))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="T3 offline minimum closed loop")
    ap.add_argument("--modes", default="direct,servo",
                    help="control modes to replay the same trajectory through")
    ap.add_argument("--grasp-overlap", type=float, default=GRASP_OVERLAP,
                    help="how far the pad bottom sits below the cube top (m)")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    chk = Checker()

    print("=" * 78)
    print("T3 offline closed loop: synthetic hand -> 8-D EE -> IK -> MuJoCo -> PNG")
    print("=" * 78)
    print("  python      : %s" % sys.executable)
    print("  mujoco      : %s (numpy %s)" % (mujoco.__version__, np.__version__))
    print("  scene       : %s" % XML_PATH)
    print("  script      : %s" % os.path.abspath(__file__))
    print("  out dir     : %s" % OUT_DIR)
    print("  modes       : %s" % ", ".join(modes))
    print("  convention  : R_tool = R_hand @ C, C = [hand Y, -hand X, hand Z],"
          " det=%.15f orth=%.1e" % (sh.CONVENTION_DET, sh.CONVENTION_ORTHONORMALITY_ERR))

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    ik = ikmod.JakaZu3IK(XML_PATH, "tool0", tcp_offset=TCP_OFFSET)
    print("  model       : nq=%d nv=%d nu=%d dt=%.4f s" %
          (model.nq, model.nv, model.nu, model.opt.timestep))

    # zero posture defines the vertical approach orientation
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    R_tool = data.site_xmat[model.site("tool0").id].reshape(3, 3).copy()
    print("  R_tool (q=0): %s" % np.array2string(R_tool, precision=6, suppress_small=True))

    # ---- geometry: where can the pads actually go? ------------------------- --
    table_gid = model.geom("table_top").id
    table_top = float(model.geom_pos[table_gid][2] + model.geom_size[table_gid][2])
    pad_half_z = pad_extent(model, R_tool)
    cube_body = model.body("cube").id
    cube_geoms = tuple(i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == cube_body)
    cube_half = np.asarray(model.geom_size[cube_geoms[0]], dtype=np.float64)
    cube0 = settle(model, data)
    cube_top = float(cube0[2] + cube_half[2])

    print()
    print("geometry (XML wins over the brief where they disagree)")
    print("  table top z                   : %.6f  (table_top geom pos.z + size.z)" % table_top)
    print("  cube settled xyz              : %.6f %.6f %.6f" % tuple(cube0))
    print("  cube XML nominal xyz          : 0.474000 0.043500 0.810000  (brief said x=0.486)")
    print("  cube top z                    : %.6f  (half size %.4f)" % (cube_top, cube_half[2]))
    print("  pad half extent along world z  : %.6f  (pad box %.4f %.4f %.4f)"
          % (pad_half_z, *PAD_HALF_SIZE))
    z_spec = float(cube0[2])
    z_min = table_top + PAD_CLEARANCE + pad_half_z
    z_grasp = max(z_min, cube_top - args.grasp_overlap + pad_half_z)
    print("  brief: TCP z = cube centre     : %.6f -> pad bottom %.6f = %.2f mm INSIDE the table"
          % (z_spec, z_spec - pad_half_z, 1e3 * (table_top - (z_spec - pad_half_z))))
    print("  deepest collision-free TCP z   : %.6f  (table top + %.3f mm + pad half)"
          % (z_min, 1e3 * PAD_CLEARANCE))
    print("  chosen TCP grasp z             : %.6f  -> pad bottom %.6f = cube top - %.2f mm"
          % (z_grasp, z_grasp - pad_half_z, 1e3 * (cube_top - (z_grasp - pad_half_z))))

    grasp_pos = np.array([cube0[0], cube0[1], z_grasp])
    traj_pos, traj_g, phase = build_trajectory(grasp_pos)
    n_frames = len(traj_g)
    print()
    print("trajectory: %d frames @ %.1f Hz = %.3f s   descend %.3f -> %.4f m, lift +%.3f m"
          % (n_frames, 1.0 / FRAME_DT, n_frames * FRAME_DT, traj_pos[0][2], z_grasp, LIFT_HEIGHT))
    print("  phases: %s" % ", ".join(
        "%s x%d" % (p, int(np.sum(phase == p)))
        for p in ("descend", "hold_open", "close", "lift", "hold_end")))
    chk.check(60 <= n_frames <= 90, "trajectory length %d frames in [60, 90]" % n_frames)

    # ---- A: keypoints -> hand_to_ee round trip ---------------------------- --
    print()
    print("A. reconstruction: synth keypoints -> hand_to_ee vs the pose we asked for")
    ret = stage_retarget(traj_pos, traj_g, R_tool)
    print("  frames checked                : %d" % n_frames)
    print("  max |pos| error               : %.3e m" % ret["pos_err"].max())
    print("  max orientation error         : %.3e deg" % ret["ang_err"].max())
    print("  max |g| error                 : %.3e" % ret["g_err"].max())
    print("  max plane residual (SVD)      : %.3e m" % ret["plane_res"].max())
    print("  d_t range                     : %.5f .. %.5f m (d_min %.3f / d_max %.3f)"
          % (ret["d_t"].min(), ret["d_t"].max(), h2e.DEFAULT_D_MIN, h2e.DEFAULT_D_MAX))
    print("  frames with z_flip inside SVD : %d (sign convention only, no pose change)"
          % int(ret["flipped"].sum()))
    chk.check(bool(ret["valid"].all()), "hand_to_ee info[valid] on every frame")
    chk.check(ret["pos_err"].max() < 1e-9,
              "A pos error < 1e-9 m on all %d frames" % n_frames,
              "max %.3e" % ret["pos_err"].max())
    chk.check(ret["ang_err"].max() < 1e-9,
              "A orientation error < 1e-9 deg on all %d frames" % n_frames,
              "max %.3e" % ret["ang_err"].max())
    chk.check(ret["g_err"].max() < 1e-12, "A gripper opening error < 1e-12",
              "max %.3e" % ret["g_err"].max())

    # ---- B / C: IK ------------------------------------------------------- --
    print()
    print("B. IK: warm start from the previous frame (frame 0 seeds from home qpos)")
    ikr = stage_ik(ik, ret["ee8"])
    q = ikr["q"]
    print("  converged single attempt      : %d/%d (%.1f%%)"
          % (ikr["single_ok"].sum(), n_frames, 100.0 * ikr["single_ok"].mean()))
    print("  converged with <=%d restarts   : %d/%d (%.1f%%)"
          % (MAX_RESTARTS, ikr["ok"].sum(), n_frames, 100.0 * ikr["ok"].mean()))
    print("  frames that needed a restart  : %s"
          % (np.flatnonzero(ikr["restarts"] > 0).tolist() or "none"))
    print("  iterations per frame          : p50 %.1f  p95 %.1f  max %d  total %d"
          % (_pct(ikr["iters"], 50), _pct(ikr["iters"], 95), ikr["iters"].max(),
             int(ikr["iters"].sum())))
    print("  solve wall time (ms)          : p50 %.2f  p95 %.2f  max %.2f  (budget %.1f ms/frame)"
          % (_pct(ikr["solve_ms"], 50), _pct(ikr["solve_ms"], 95), ikr["solve_ms"].max(),
             1e3 * FRAME_DT))
    print("  max position residual         : %.3e m" % ikr["pos_res"].max())
    print("  max orientation residual      : %.3e deg" % ikr["ang_res"].max())
    dq = ikr["dq"]
    print("  trajectory continuity |dq|_inf: p50 %.4f  p95 %.4f  max %.4f rad (limit %.2f)"
          % (_pct(dq, 50), _pct(dq, 95), dq.max(), DELTA_Q_LIMIT))
    jmax = int(np.argmax(dq))
    print("  worst jump at frame %d (%s), sigma_min(J) %.5f, restarts %d"
          % (jmax, phase[jmax], ikr["cond"][jmax], ikr["restarts"][jmax]))
    print("  sigma_min(J) over trajectory  : p50 %.5f  min %.5f at frame %d"
          % (_pct(ikr["cond"], 50), ikr["cond"].min(), int(np.argmin(ikr["cond"]))))
    chk.check(bool(ikr["ok"].all()), "B all %d frames reach the IK tolerance" % n_frames)
    chk.check(dq.max() <= DELTA_Q_LIMIT,
              "B max |dq|_inf = %.4f rad <= %.2f" % (dq.max(), DELTA_Q_LIMIT))

    print()
    print("C. joint limits")
    lo, hi = ik.joint_limits[:, 0], ik.joint_limits[:, 1]
    in_lim = np.array([ik.within_limits(q[i], tol=1e-9) for i in range(n_frames)])
    at_bound = (np.abs(q - lo) <= 1e-9) | (np.abs(q - hi) <= 1e-9)
    clipped = int(np.sum(np.any(at_bound, axis=1)))
    print("  frames within limits (1e-9)   : %d/%d" % (int(in_lim.sum()), n_frames))
    print("  frames sitting exactly on a bound: %d" % clipped)
    print("  q range per joint (rad):")
    for j, name in enumerate(ikmod.ARM_JOINT_NAMES):
        print("    %-8s %8.4f .. %8.4f   limits %8.3f .. %7.3f"
              % (name, q[:, j].min(), q[:, j].max(), lo[j], hi[j]))
    chk.check(bool(in_lim.all()), "C every frame inside the joint limits")
    chk.check(clipped == 0, "C no frame was clipped by a joint limit", "clipped=%d" % clipped)

    # ---- renderer -------------------------------------------------------- --
    renderer = camera = None
    render_frames = set(range(0, n_frames, RENDER_EVERY))
    if not args.no_render:
        renderer = mujoco.Renderer(model, height=RENDER_HW[0], width=RENDER_HW[1])
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.05]
        camera.distance = 0.55
        camera.azimuth = 130.0
        camera.elevation = -18.0

    results = {}
    for mode in modes:
        print()
        print("=" * 78)
        print("D. physics replay, ctrl mode = %s" % mode)
        print("=" * 78)
        if mode not in ("direct", "servo"):
            raise SystemExit("unknown ctrl mode %r" % mode)
        png_dir = os.path.join(OUT_DIR, mode)
        os.makedirs(png_dir, exist_ok=True)
        for f in os.listdir(png_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(png_dir, f))

        rec = run_physics(model, data, ik, traj_pos, traj_g, q, R_tool, mode,
                          renderer=renderer, camera=camera, png_dir=png_dir,
                          render_frames=render_frames)
        print_frame_table_joined(rec, ikr, phase)

        print()
        print("  tracking error by phase (commanded vs achieved TCP):")
        print("    phase      frames   mean|err| m   max|err| m   mean pad_low   mean cube contacts")
        for name, cnt, m_mean, m_max, pl, tc in phase_tracking(rec, phase):
            print("    %-10s %6d   %10.4f   %10.4f   %11.4f  %8.2f"
                  % (name, cnt, m_mean, m_max, pl, tc))

        cube0_run = rec["cube0"]
        print("  cube start xyz               : %.6f %.6f %.6f" % tuple(cube0_run))
        print("  cube final xyz               : %.6f %.6f %.6f"
              % (rec["cube_x"][-1], rec["cube_y"][-1], rec["cube_z"][-1]))
        rise = float(rec["cube_z"][-1] - cube0_run[2])
        print("  cube rise                    : %+.5f m  (required >= +0.08000)" % rise)
        print("  physics steps per frame      : %d .. %d (dt %.4f s, frame %.5f s)"
              % (rec["n_steps"].min(), rec["n_steps"].max(), model.opt.timestep, FRAME_DT))
        print("  final pad gap / jaw / contacts: ", end="")
        u, left_face, right_face, jaw = final_grasp_geometry(model, data, cube_half)
        print("gap %.5f m, jaw %.5f m, pad contacts %d"
              % (right_face - left_face, jaw, int(rec["n_pad_con"][-1])))
        print("  cube centre in tool frame    : x %+0.5f  y %+0.5f  z %+0.5f m" % tuple(u))
        inside = (left_face <= u[0] <= right_face
                  and abs(u[1]) <= PAD_HALF_SIZE[1] + 1e-9
                  and abs(u[2]) <= PAD_HALF_SIZE[2] + 1e-9)
        print("  cube still between the pads  : %s  (|u_y| <= %.4f, |u_z| <= %.4f)"
              % (inside, PAD_HALF_SIZE[1], PAD_HALF_SIZE[2]))

        chk.check(rise >= 0.08, "[%s] D cube lifted >= 0.08 m" % mode, "rise %+.5f m" % rise)
        chk.check(inside, "[%s] D cube still between the two pads" % mode,
                  "u=(%+.4f, %+.4f, %+.4f)" % tuple(u))

        # ---- E: PNG frames ---------------------------------------------- --
        if renderer is not None:
            final_path = os.path.join(png_dir, "final.png")
            renderer.update_scene(data, camera)
            rec["pngs"].append(("final", final_path, write_png(final_path, renderer.render())))
            expected = len(render_frames) + 1
            files = sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))
            stats = [png_stats(os.path.join(png_dir, f)) for f in files]
            means = [s[2] for s in stats]
            total = sum(s[3] for s in stats)
            print()
            print("E. rendering %dx%d" % (RENDER_HW[0], RENDER_HW[1]))
            print("  frames written               : %d (expected %d: every %d frames + final)"
                  % (len(files), expected, RENDER_EVERY))
            print("  mean pixel value             : min %.2f  max %.2f  (assert > %.1f)"
                  % (min(means), max(means), MIN_PNG_MEAN))
            print("  total PNG bytes              : %d (%.2f MiB)" % (total, total / 1048576.0))
            print("  dir                          : %s" % png_dir)
            chk.check(len(files) == expected, "[%s] E PNG count == %d" % (mode, expected),
                      "got %d" % len(files))
            chk.check(min(means) > MIN_PNG_MEAN, "[%s] E no black frames" % mode,
                      "min mean %.2f" % min(means))

        results[mode] = dict(rec=rec, rise=rise, u=u, inside=inside,
                             gap=float(right_face - left_face), jaw=jaw)

    return chk.summary()


if __name__ == "__main__":
    sys.exit(main())


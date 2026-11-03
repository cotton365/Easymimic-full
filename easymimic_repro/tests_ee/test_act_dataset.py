#!/usr/bin/env python3
"""Contract test for the scripted ACT dataset written by experiments/gen_act_dataset.py.

Needs numpy + mujoco, and reads the two constants it must not invent (GRIPPER_CONFIG, the
scene path) straight out of the sources the generator uses. Nothing here calls the
generator or imports its private helpers: the point is that the artefact on disk can be
checked on its own.

Checks, per episode:
  * meta.json / data.npz carry exactly the agreed keys, with the agreed shapes and dtypes;
  * num_frames == len(t) == the number of PNGs in each camera directory, PNGs are 8-bit
    RGB 256x256 named 000000.png .. and the frame clock is k / fps;
  * qpos/act are physically sane: qpos[6] and act[6] inside the hinge range, |dq| per tick
    bounded, act - qpos bounded, cube quaternion unit length, spawn z on the table;
  * the 8-D EE interface holds: pose8_state[k] is the tool0 pose of the recorded qpos[k]
    (recomputed here by MuJoCo FK, <= 1e-9 per element), pose8_action[k][0:7] is
    pose8_state[k+1][0:7], both quaternion sequences are unit length and sign-continuous
    (dot(q_k, q_{k-1}) >= 0, state vs action on the same tick > 0), state g is
    qpos[6] through GRIPPER_CONFIG, and action g is the commanded g that act[6] was made
    from;
  * meta and npz agree, rise >= 0.08 m and grasp_ok is true;
and across episodes: index.json is consistent with the directories it describes, plus the
g -> ctrl -> simulated joint_7 -> g round trip (<= 1e-3) on the arm-only model.
and, once per run, the layout hooks in the generator: layout_table(base, CUBE_XY, BOWL_XY) is
shifted_table(base, 0, 0) field for field, the anchored rows follow the bowl xy, place_bowl
translates all 17 bowl geoms by the delta without deforming the ring, sample_layout draws
continuously inside CUBE_BOX/BOWL_BOX with MIN_SEP_M between the two centres, and
experiments/set_layout.py fails on an unreachable target, passes on the layout the files
describe, prints --sample layouts, and refuses --apply together with any measuring flag.

Usage:  python tests_ee/test_act_dataset.py [--dir out/act_demos_ee] [--episodes 20]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    import mujoco  # noqa: E402
except ImportError as exc:  # pragma: no cover - the FK and gripper checks need it
    raise SystemExit("this test needs mujoco (FK check + gripper round trip): %s" % exc)

from src.pose_cmd.replay import GRIPPER_CONFIG  # noqa: E402

META_KEYS = {"episode", "fps", "num_frames", "camera_hw", "cube_pos", "cube_yaw",
             "rise_m", "slip_m", "grasp_ok", "scene", "control_hz"}
NPZ_KEYS = {"pose8_state", "pose8_action", "qpos", "act", "ik_err", "t", "cube_pos",
            "cube_quat", "rise_m", "slip_m"}
INDEX_KEYS = {"num_episodes", "fps", "control_hz", "task", "camera_names", "state_dim",
              "action_dim", "seed", "mujoco_version", "randomization", "generated_by"}
CAMERAS = ("front", "wrist")
HW = (256, 256)
SCENE = "assets/jaka_zu3/scene_cube_bowl.xml"
ARM_XML = "assets/jaka_zu3/robot.xml"       # arm + jaws only: the gripper round trip
EE_SITE = "tool0"
EE_DIM = 8                                  # x, y, z, qw, qx, qy, qz, g
QUAT = slice(3, 7)
POS = slice(0, 3)

MIN_RISE_M = 0.080
MAX_TICK_STEP_RAD = 0.15     # measured max per-tick joint step 0.045 rad at 20 Hz
MAX_ACT_QPOS_RAD = 0.35      # measured max |act - qpos| 0.222 rad (the hinge's own lag)
HINGE_RANGE = (0.0, 0.9)
HINGE_SLACK = 1e-3
CUBE_Z = 0.815               # table top 0.80 + half height 0.015, +-1 mm
FK_TOL_M = 1e-9              # A2: recorded qpos -> FK must reproduce pose8_state
EE_STEP_M = 0.02             # measured max per-tick EE displacement 0.0101 m
QNORM_TOL = 1e-9
G_TOL = 1e-3                 # A4: g -> ctrl -> simulated joint_7 -> g
G_SLACK = 1e-3               # the hinge settles a hair outside [0, 0.9]
CTRL_PER_G = float(GRIPPER_CONFIG["ctrl_per_g"])
OPEN_AT_G = float(GRIPPER_CONFIG["open_at_g"])
G_JOINT = GRIPPER_CONFIG["joint"]

# ---- the place task (out/act_demos_place): one extra scalar per episode plus a bowl test ----
PLACE_TASK = "pick up the red block and put it in the bowl"
PLACE_METRICS = {"bowl_dxy_m", "in_bowl", "hold_err_m"}
INDEX_PLACE_KEYS = {"task_kind", "table", "bowl_xy_m", "in_bowl_radius_m", "bowl_dxy_max_m",
                    "in_bowl_all", "hold_err_max_m"}
BOWL_XY = (0.300, -0.200)
BOWL_INNER_R_M = 0.045       # the bowl's own inner radius, from the scene geometry
IN_BOWL_RADIUS_M = 0.045     # accepted: cube centre within this of the bowl centre ...
IN_BOWL_SEATED_R_M = 0.030   # ... the stricter "sitting on the floor" reading
IN_BOWL_Z = (0.800, 0.840)   # ... and between the bowl floor and its wall top
MAX_BOWL_DXY_M = 0.005       # acceptance gate per episode
MAX_HOLD_ERR_M = 0.002       # cube still between the pads on the last closed-gripper tick
PLACE_FRAMES = 181           # configs/waypoints_cube_place.json at 20 Hz: 8 waypoints, 181 tick
RELEASE_G = 0.9              # the place table must end with the gripper commanded open

# ---- the layout pair: written by the generator since sample_layout landed, absent from the
# ---- frozen datasets in out/, so required-plus-optional rather than an exact key set
CUBE_XY = (0.474, 0.0435)
OPTIONAL_META_KEYS = {"cube_xy", "bowl_xy"}
OPTIONAL_INDEX_KEYS = {"cube_xy", "bowl_xy", "cube_box_m", "bowl_box_m", "min_sep_m"}


class Checks:
    def __init__(self):
        self.count = 0

    def ok(self, cond, msg):
        self.count += 1
        if not cond:
            raise AssertionError(msg)


def read_png_header(path):
    """(width, height, bit_depth, colour_type) from the IHDR chunk; raises if not a PNG."""
    with open(path, "rb") as fh:
        head = fh.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise AssertionError("%s: not a PNG (bad signature or missing IHDR)" % path)
    width, height, depth, colour = struct.unpack(">IIBB", head[16:26])
    return width, height, depth, colour


_FK = {}


def fk_setup():
    """One model/data pair for the whole run: site poses for an arbitrary qpos."""
    if not _FK:
        model = mujoco.MjModel.from_xml_path(os.path.join(REPO, SCENE))
        _FK["model"] = model
        _FK["data"] = mujoco.MjData(model)
        _FK["site"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    return _FK["model"], _FK["data"], _FK["site"]


def fk_tool(qpos7):
    """tool0 (pos (3,), quat (4,) scalar-first) for one recorded qpos, via MuJoCo itself.
    The cube's free-joint coordinates do not enter the arm chain, so they stay 0."""
    model, data, site = fk_setup()
    data.qpos[:] = 0.0
    data.qpos[:len(qpos7)] = qpos7
    mujoco.mj_forward(model, data)
    pos = np.array(data.site_xpos[site], dtype=float)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(data.site_xmat[site], dtype=float).ravel())
    return pos, quat


def check_gripper_roundtrip(ck):
    """A4: g -> ctrl (the same GRIPPER_CONFIG the replay uses) -> position servo -> the
    simulated joint_7 -> g back. Runs on the arm-only model so no cube can block the jaw
    and no table is in the way; the residual is the servo's sag under gravity."""
    model = mujoco.MjModel.from_xml_path(os.path.join(REPO, ARM_XML))
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, G_JOINT)
    adr = int(model.jnt_qposadr[jid])
    acts = [i for i in range(model.nu)
            if model.actuator_trnid[i][0] == jid]
    ck.ok(len(acts) == 1, "%s: expected exactly one actuator, found %d" % (G_JOINT, len(acts)))
    worst = 0.0
    rows = []
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        data = mujoco.MjData(model)
        data.ctrl[acts[0]] = g / OPEN_AT_G * CTRL_PER_G
        for _ in range(4000):                      # 8 s at dt=0.002: fully settled
            mujoco.mj_step(model, data)
        q7 = float(data.qpos[adr])
        g_back = q7 * OPEN_AT_G / CTRL_PER_G
        ck.ok(abs(g_back - g) <= G_TOL,
              "g -> ctrl -> %s -> g round trip off by %.3e (bound %.1e): g=%.2f q7=%.7f"
              % (G_JOINT, abs(g_back - g), G_TOL, g, q7))
        worst = max(worst, abs(g_back - g))
        rows.append((g, q7, g_back))
    return worst, rows


def check_episode(ck, ep_dir, index, place=False):
    meta_keys = META_KEYS | PLACE_METRICS if place else META_KEYS
    npz_keys = NPZ_KEYS | PLACE_METRICS if place else NPZ_KEYS
    ck.ok(os.path.isdir(ep_dir), "%s: missing episode directory" % ep_dir)
    meta_path = os.path.join(ep_dir, "meta.json")
    npz_path = os.path.join(ep_dir, "data.npz")
    ck.ok(os.path.isfile(meta_path), "%s: missing meta.json" % ep_dir)
    ck.ok(os.path.isfile(npz_path), "%s: missing data.npz" % ep_dir)
    with open(meta_path) as fh:
        meta = json.load(fh)

    ck.ok(set(meta) >= meta_keys and set(meta) - meta_keys <= OPTIONAL_META_KEYS,
          "%s: meta.json keys %s, required %s plus at most %s"
          % (ep_dir, sorted(set(meta)), sorted(meta_keys), sorted(OPTIONAL_META_KEYS)))
    if "cube_xy" in meta:
        ck.ok([float(v) for v in meta["cube_xy"]] == [float(v) for v in meta["cube_pos"][:2]],
              "%s: meta.cube_xy %r != the first two of meta.cube_pos %r"
              % (ep_dir, meta["cube_xy"], meta["cube_pos"]))
        if "bowl_box_m" in index:
            bx0, bx1, by0, by1 = [float(v) for v in index["bowl_box_m"]]
            ck.ok(bx0 <= float(meta["bowl_xy"][0]) <= bx1
                  and by0 <= float(meta["bowl_xy"][1]) <= by1,
                  "%s: meta.bowl_xy %r is outside index.bowl_box_m %r"
                  % (ep_dir, meta["bowl_xy"], index["bowl_box_m"]))
        else:   # a dataset from before the bowl could move: one bowl, the scene's
            ck.ok([float(v) for v in meta["bowl_xy"]] == [float(v) for v in index["bowl_xy_m"]],
                  "%s: meta.bowl_xy %r != index.bowl_xy_m %r"
                  % (ep_dir, meta["bowl_xy"], index["bowl_xy_m"]))
    ck.ok(isinstance(meta["episode"], int), "%s: meta.episode is not an int" % ep_dir)
    ck.ok(meta["scene"] == SCENE, "%s: meta.scene %r" % (ep_dir, meta["scene"]))
    ck.ok(os.path.isfile(os.path.join(REPO, meta["scene"])),
          "%s: meta.scene does not exist under the repo" % ep_dir)
    ck.ok(meta["camera_hw"] == list(HW), "%s: meta.camera_hw %r" % (ep_dir, meta["camera_hw"]))
    ck.ok(float(meta["fps"]) == float(meta["control_hz"]),
          "%s: one frame per control tick, so fps (%r) must equal control_hz (%r)"
          % (ep_dir, meta["fps"], meta["control_hz"]))
    ck.ok(float(meta["control_hz"]) == float(index["control_hz"]),
          "%s: control_hz disagrees with index.json" % ep_dir)
    ck.ok(meta["grasp_ok"] is True, "%s: grasp_ok is not true" % ep_dir)
    ck.ok(float(meta["cube_yaw"]) == 0.0, "%s: this round does not randomize yaw" % ep_dir)
    if place:
        # the cube ends in the bowl, not lifted: rise is ~0 by construction, so the lift gate
        # below would reject every place episode - and would be testing the wrong task.
        ck.ok(int(meta["num_frames"]) == PLACE_FRAMES,
              "%s: %d frames, the place table is %d ticks at %r Hz"
              % (ep_dir, int(meta["num_frames"]), PLACE_FRAMES, meta["fps"]))
        # Two-sided band: the bowl moves with the random layout, so the cube ends on a floor
        # that at the sampled xy can sit a fraction of a millimetre below the cube's start z
        # (measured -0.00004 m .. +0.00196 m over 60 random episodes). The place invariant is
        # in_bowl / bowl_dxy_m below; this check only says "the cube did not travel up".
        ck.ok(-1e-3 <= float(meta["rise_m"]) < 1e-2,
              "%s: rise %.5f m - a place episode drops the cube into the bowl, it must not rise"
              % (ep_dir, float(meta["rise_m"])))
        # slip_m keeps its lift definition (cube drift after the jaws close) and the cube
        # travels ~0.4 m to the bowl, so it is a distance here, not a grasp-quality bound
        ck.ok(0.05 <= float(meta["slip_m"]) < 0.6,
              "%s: slip %.6f m is not the ~0.4 m trip to the bowl" % (ep_dir, float(meta["slip_m"])))
        ck.ok(float(meta["bowl_dxy_m"]) <= MAX_BOWL_DXY_M,
              "%s: bowl_dxy %.5f m is above the %.3f m gate"
              % (ep_dir, float(meta["bowl_dxy_m"]), MAX_BOWL_DXY_M))
        ck.ok(meta["in_bowl"] is True, "%s: in_bowl is %r, expected a bool true"
              % (ep_dir, meta["in_bowl"]))
        ck.ok(float(meta["hold_err_m"]) <= MAX_HOLD_ERR_M,
              "%s: hold_err %.5f m is above the %.3f m gate - the cube was leaving the pads "
              "before the release" % (ep_dir, float(meta["hold_err_m"]), MAX_HOLD_ERR_M))
    else:
        ck.ok(float(meta["rise_m"]) >= MIN_RISE_M,
              "%s: rise %.5f m is below %.3f m" % (ep_dir, float(meta["rise_m"]), MIN_RISE_M))
        ck.ok(0.0 <= float(meta["slip_m"]) < 1e-2,
              "%s: slip %.6f m" % (ep_dir, float(meta["slip_m"])))

    data = np.load(npz_path)
    ck.ok(set(data.files) == npz_keys,
          "%s: data.npz keys %s != %s" % (ep_dir, sorted(set(data.files)), sorted(npz_keys)))
    qpos, act, t = data["qpos"], data["act"], data["t"]
    state8, action8, ik_err = data["pose8_state"], data["pose8_action"], data["ik_err"]
    n = int(meta["num_frames"])
    ck.ok(qpos.shape == (n, 7), "%s: qpos shape %s != (%d, 7)" % (ep_dir, qpos.shape, n))
    ck.ok(act.shape == (n, 7), "%s: act shape %s != (%d, 7)" % (ep_dir, act.shape, n))
    ck.ok(t.shape == (n,), "%s: t shape %s != (%d,)" % (ep_dir, t.shape, n))
    ck.ok(state8.shape == (n, EE_DIM),
          "%s: pose8_state shape %s != (%d, %d)" % (ep_dir, state8.shape, n, EE_DIM))
    ck.ok(action8.shape == (n, EE_DIM),
          "%s: pose8_action shape %s != (%d, %d)" % (ep_dir, action8.shape, n, EE_DIM))
    ck.ok(ik_err.shape == (n,), "%s: ik_err shape %s != (%d,)" % (ep_dir, ik_err.shape, n))
    for key in ("pose8_state", "pose8_action", "qpos", "act", "ik_err", "t",
                "cube_pos", "cube_quat"):
        ck.ok(data[key].dtype == np.float64, "%s: %s dtype %s" % (ep_dir, key, data[key].dtype))
    for key in ("rise_m", "slip_m"):
        ck.ok(data[key].shape == () and data[key].dtype == np.float64,
              "%s: %s is %s %s, expected a float64 scalar"
              % (ep_dir, key, data[key].shape, data[key].dtype))
        ck.ok(abs(float(data[key]) - float(meta[key])) < 1e-12,
              "%s: %s disagrees between data.npz and meta.json" % (ep_dir, key))
    if place:
        # the three place scalars: right dtype/shape, agreeing with meta.json, and the bowl
        # test is not just bowl_dxy renamed
        for key in ("bowl_dxy_m", "hold_err_m"):
            ck.ok(data[key].shape == () and data[key].dtype == np.float64,
                  "%s: %s is %s %s, expected a float64 scalar"
                  % (ep_dir, key, data[key].shape, data[key].dtype))
            ck.ok(abs(float(data[key]) - float(meta[key])) < 1e-12,
                  "%s: %s disagrees between data.npz and meta.json" % (ep_dir, key))
        ck.ok(data["in_bowl"].shape == () and data["in_bowl"].dtype == np.bool_,
              "%s: in_bowl is %s %s, expected a bool scalar"
              % (ep_dir, data["in_bowl"].shape, data["in_bowl"].dtype))
        ck.ok(bool(data["in_bowl"]) is bool(meta["in_bowl"]),
              "%s: in_bowl disagrees between data.npz and meta.json" % ep_dir)
        ck.ok(float(data["hold_err_m"]) >= 0.0, "%s: hold_err_m is negative" % ep_dir)
        ck.ok(IN_BOWL_RADIUS_M <= BOWL_INNER_R_M,
              "the accepted radius %r is outside the bowl's own %r m inner radius"
              % (IN_BOWL_RADIUS_M, BOWL_INNER_R_M))
        ck.ok(IN_BOWL_SEATED_R_M < BOWL_INNER_R_M,
              "the seated radius %r is not inside the bowl's own %r m inner radius"
              % (IN_BOWL_SEATED_R_M, BOWL_INNER_R_M))
        # the end-of-episode cube pose is not recorded, so the z half of in_bowl is checked on
        # the end pose by tests_ee/test_act_rollout.py (bowl_metrics); here the on-disk record
        # only has to agree with itself
        ck.ok((float(data["bowl_dxy_m"]) < IN_BOWL_RADIUS_M) == bool(data["in_bowl"]),
              "%s: in_bowl (%r) disagrees with bowl_dxy %.5f m against the %.3f m radius"
              % (ep_dir, bool(data["in_bowl"]), float(data["bowl_dxy_m"]), IN_BOWL_RADIUS_M))
    ck.ok(data["cube_quat"].shape == (4,), "%s: cube_quat shape %s"
          % (ep_dir, data["cube_quat"].shape))
    ck.ok(data["cube_pos"].shape == (3,), "%s: cube_pos shape %s"
          % (ep_dir, data["cube_pos"].shape))
    ck.ok(np.allclose(data["cube_pos"], np.asarray(meta["cube_pos"], dtype=float), atol=1e-12),
          "%s: cube_pos disagrees between data.npz and meta.json" % ep_dir)
    ck.ok(abs(float(data["cube_pos"][2]) - CUBE_Z) < 1e-3,
          "%s: cube spawn z %.4f is not on the table" % (ep_dir, float(data["cube_pos"][2])))
    ck.ok(abs(float(np.linalg.norm(data["cube_quat"])) - 1.0) < 1e-9,
          "%s: cube_quat is not unit length" % ep_dir)
    ck.ok(abs(abs(float(data["cube_quat"][0])) - 1.0) < 1e-6,
          "%s: cube_quat is not the identity (rotation exists but yaw is not randomized)" % ep_dir)

    # frame clock: t[k] = k / fps, so num_frames ticks cover num_frames / fps seconds
    ck.ok(abs(float(t[0])) < 1e-12, "%s: t[0] = %r, expected 0" % (ep_dir, float(t[0])))
    ck.ok(np.allclose(t, np.arange(n) / float(meta["fps"]), atol=1e-9),
          "%s: t is not k / fps" % ep_dir)
    ck.ok(abs(t[-1] - (n - 1) / float(meta["fps"])) < 1e-9,
          "%s: t[-1] does not match (num_frames - 1) / fps" % ep_dir)

    # physics sanity
    ck.ok(np.all(np.isfinite(qpos)) and np.all(np.isfinite(act)),
          "%s: non-finite values in qpos/act" % ep_dir)
    # joint_7 declares range 0..0.9 but is not `limited`, so the measured angle settles a
    # hair outside it (0.900006 was the worst case); 1 mrad of slack, not "anything goes".
    ck.ok(np.all(qpos[:, 6] >= HINGE_RANGE[0] - HINGE_SLACK)
          and np.all(qpos[:, 6] <= HINGE_RANGE[1] + HINGE_SLACK),
          "%s: gripper hinge %.6f..%.6f leaves %s"
          % (ep_dir, qpos[:, 6].min(), qpos[:, 6].max(), HINGE_RANGE))
    ck.ok(np.all(act[:, 6] >= HINGE_RANGE[0] - HINGE_SLACK)
          and np.all(act[:, 6] <= HINGE_RANGE[1] + HINGE_SLACK),
          "%s: gripper target %.6f..%.6f leaves %s"
          % (ep_dir, act[:, 6].min(), act[:, 6].max(), HINGE_RANGE))
    step = float(np.abs(np.diff(qpos, axis=0)).max()) if n > 1 else 0.0
    ck.ok(step <= MAX_TICK_STEP_RAD,
          "%s: a joint jumps %.4f rad in one tick (bound %.3f)" % (ep_dir, step, MAX_TICK_STEP_RAD))
    gap = float(np.abs(act - qpos).max())
    ck.ok(gap <= MAX_ACT_QPOS_RAD,
          "%s: |act - qpos| reaches %.4f rad (bound %.3f)" % (ep_dir, gap, MAX_ACT_QPOS_RAD))
    if place:
        # a place episode starts open, closes on the cube, carries, and lets go at the end
        ck.ok(qpos[0, 6] > 0.8, "%s: the gripper does not start open (hinge %.3f)"
              % (ep_dir, qpos[0, 6]))
        ck.ok(float(qpos[:, 6].min()) < 0.5,
              "%s: the gripper never closed on the cube (hinge min %.3f; with the 15 mm cube "
              "between the pads it stops near 0.222, not 0)" % (ep_dir, qpos[:, 6].min()))
        ck.ok(float(action8[-1, 7]) >= RELEASE_G,
              "%s: the episode ends with the gripper commanded g=%.4f, i.e. still holding"
              % (ep_dir, float(action8[-1, 7])))
        ck.ok(float(action8[:, 7].min()) == 0.0,
              "%s: no tick commands a fully closed gripper (min g %.4f)"
              % (ep_dir, float(action8[:, 7].min())))
        ck.ok(float(action8[-1, 7]) > float(action8[:, 7].min()),
              "%s: the commanded g never reopens after the grasp" % ep_dir)
    else:
        ck.ok(qpos[:, 6][0] > qpos[:, 6][-1],
              "%s: the gripper never closed (hinge %.3f -> %.3f)"
              % (ep_dir, qpos[0, 6], qpos[-1, 6]))

    # ---- 8-D EE interface -------------------------------------------------------------
    ck.ok(np.all(np.isfinite(state8)) and np.all(np.isfinite(action8)),
          "%s: non-finite values in pose8_state/pose8_action" % ep_dir)
    for label, block in (("pose8_state", state8), ("pose8_action", action8)):
        norm_err = float(np.abs(np.linalg.norm(block[:, QUAT], axis=1) - 1.0).max())
        ck.ok(norm_err <= QNORM_TOL,
              "%s: %s quaternion norm off by %.3e (bound %.1e)" % (ep_dir, label, norm_err,
                                                                   QNORM_TOL))
    # A2: the stored state must be the FK of the stored qpos - not a hand-rolled chain
    pos_err = 0.0
    quat_err = 0.0
    ee_step = 0.0
    for k in range(n):
        pos, quat = fk_tool(qpos[k])
        pos_err = max(pos_err, float(np.abs(pos - state8[k, POS]).max()))
        want = state8[k, QUAT]
        quat_err = max(quat_err, float(min(np.abs(quat - want).max(),
                                           np.abs(quat + want).max())))
        ee_step = max(ee_step, float(np.abs(action8[k, POS] - state8[k, POS]).max()))
    ck.ok(pos_err <= FK_TOL_M,
          "%s: FK(position) of qpos disagrees with pose8_state by %.3e m (bound %.0e)"
          % (ep_dir, pos_err, FK_TOL_M))
    ck.ok(quat_err <= FK_TOL_M,
          "%s: FK(quaternion) of qpos disagrees with pose8_state by %.3e (bound %.0e)"
          % (ep_dir, quat_err, FK_TOL_M))
    # A3: sign continuity inside the episode, and state/action agreeing on every tick
    if n > 1:
        consec = np.einsum("ij,ij->i", state8[1:, QUAT], state8[:-1, QUAT])
        ck.ok(float(consec.min()) >= 0.0,
              "%s: quaternion sign flips between frames (min dot %.6f)"
              % (ep_dir, float(consec.min())))
        pose_step = float(np.abs(action8[:-1, :7] - state8[1:, :7]).max())
        ck.ok(pose_step <= FK_TOL_M,
              "%s: pose8_action[k] is not the pose of pose8_state[k+1] (off by %.3e)"
              % (ep_dir, pose_step))
    same_tick = np.einsum("ij,ij->i", state8[:, QUAT], action8[:, QUAT])
    ck.ok(float(same_tick.min()) > 0.0,
          "%s: pose8_state and pose8_action have opposite quaternion signs on a tick "
          "(min dot %.6f)" % (ep_dir, float(same_tick.min())))
    # the 8-D rows and the joint rows must describe the same gripper state
    state_g = qpos[:, 6] * OPEN_AT_G / CTRL_PER_G
    ck.ok(float(np.abs(state8[:, 7] - state_g).max()) <= 1e-12,
          "%s: pose8_state[:,7] is not qpos[6] through GRIPPER_CONFIG (off by %.3e)"
          % (ep_dir, float(np.abs(state8[:, 7] - state_g).max())))
    ck.ok(float(np.abs(action8[:, 7] - act[:, 6] * OPEN_AT_G / CTRL_PER_G).max()) <= 1e-12,
          "%s: pose8_action[:,7] is not the g that act[6] was built from (off by %.3e)"
          % (ep_dir, float(np.abs(action8[:, 7] - act[:, 6] * OPEN_AT_G / CTRL_PER_G).max())))
    ck.ok(np.all(action8[:, 7] >= -1e-9) and np.all(action8[:, 7] <= 1.0 + 1e-9),
          "%s: commanded g leaves [0, 1]: %.4f..%.4f"
          % (ep_dir, action8[:, 7].min(), action8[:, 7].max()))
    ck.ok(np.all(state8[:, 7] >= -G_SLACK) and np.all(state8[:, 7] <= 1.0 + G_SLACK),
          "%s: measured g leaves [0, 1]: %.4f..%.4f"
          % (ep_dir, state8[:, 7].min(), state8[:, 7].max()))
    ck.ok(ee_step <= EE_STEP_M,
          "%s: the EE jumps %.4f m within one tick (bound %.3f)"
          % (ep_dir, ee_step, EE_STEP_M))
    ck.ok(float(np.abs(ik_err).max()) <= 2.0e-3,
          "%s: IK residual %.2e m exceeds 2.0e-3" % (ep_dir, float(np.abs(ik_err).max())))

    # frames: one PNG per tick per camera, 8-bit RGB, right size, contiguous numbering
    for cam in CAMERAS:
        cam_dir = os.path.join(ep_dir, cam)
        ck.ok(os.path.isdir(cam_dir), "%s: missing %s/ directory" % (ep_dir, cam))
        names = sorted(os.listdir(cam_dir))
        expect = ["%06d.png" % k for k in range(n)]
        ck.ok(names == expect,
              "%s/%s: %d files, expected %d named 000000.png..%06d.png (first mismatch %r)"
              % (ep_dir, cam, len(names), n, n - 1,
                 next((a for a, b in zip(names, expect) if a != b), None)))
        for name in names:
            path = os.path.join(cam_dir, name)
            ck.ok(os.path.getsize(path) > 200, "%s: suspiciously small PNG" % path)
            width, height, depth, colour = read_png_header(path)
            ck.ok((width, height) == HW, "%s: %dx%d, expected %dx%d" % (path, width, height, *HW))
            ck.ok((depth, colour) == (8, 2), "%s: bit depth %d / colour type %d, expected 8 / 2 (RGB)"
                  % (path, depth, colour))

    return {"pose_err": pos_err, "quat_err": quat_err,
            "consec_dot": float(np.einsum("ij,ij->i", state8[1:, QUAT],
                                          state8[:-1, QUAT]).min()) if n > 1 else 1.0,
            "same_tick_dot": float(same_tick.min())}


def table_equal(a, b):
    """Same table, field for field - exact float equality, not allclose."""
    if (a.source, a.frame, a.units, float(a.control_hz)) != (b.source, b.frame, b.units,
                                                             float(b.control_hz)):
        return False, "%r/%r header differs" % (a, b)
    if len(a.waypoints) != len(b.waypoints):
        return False, "%d vs %d waypoints" % (len(a.waypoints), len(b.waypoints))
    for wa, wb in zip(a.waypoints, b.waypoints):
        if wa.label != wb.label or float(wa.duration_s) != float(wb.duration_s):
            return False, "waypoint %s/%s differs" % (wa.label, wb.label)
        if not np.array_equal(np.asarray(wa.ee8, dtype=float), np.asarray(wb.ee8, dtype=float)):
            return False, "waypoint %s ee8 %s != %s" % (wa.label, wa.ee8, wb.ee8)
    return True, ""


def check_layout(ck):
    """The layout hooks: layout_table, place_bowl, sample_layout, and the set_layout.py CLI."""
    import random
    import subprocess

    start = ck.count

    from experiments import gen_act_dataset as gen
    from src.pose_cmd.replay import Replayer
    from src.pose_cmd.waypoints import load_table

    tables = [load_table(os.path.join(REPO, path))
              for path in (gen.TABLE_PATH, gen.PLACE_TABLE_PATH)]

    # layout_table(base, CUBE_XY, BOWL_XY) is today's table, bit for bit
    for base in tables:
        same, why = table_equal(gen.layout_table(base, gen.CUBE_XY, gen.BOWL_XY),
                               gen.shifted_table(base, 0.0, 0.0))
        ck.ok(same, "%s: layout_table(base, CUBE_XY, BOWL_XY) != shifted_table(base, 0, 0) %s"
              % (os.path.basename(base.source), why))

    # ... and at any other layout the grasp column sits on the cube, the anchored rows on the bowl
    cube_xy, bowl_xy = (0.500, 0.100), (0.320, -0.280)
    for base in tables:
        moved = gen.layout_table(base, cube_xy, bowl_xy)
        ck.ok(len(moved.waypoints) == len(base.waypoints),
              "%s: layout_table changed the waypoint count" % os.path.basename(base.source))
        for old, new in zip(base.waypoints, moved.waypoints):
            anchored = old.label in gen.ANCHORED_LABELS
            want = bowl_xy if anchored else cube_xy
            ck.ok((float(new.ee8[0]), float(new.ee8[1])) == want,
                  "%s: %s at (%r, %r), expected %r"
                  % (os.path.basename(base.source), old.label, float(new.ee8[0]),
                     float(new.ee8[1]), want))
            ck.ok(np.array_equal(np.asarray(new.ee8, dtype=float)[[2, 6, 7]],
                                 np.asarray(old.ee8, dtype=float)[[2, 6, 7]]),
                  "%s: %s: z or the non-xy components moved" % (os.path.basename(base.source),
                                                                old.label))
            ck.ok(float(new.duration_s) == float(old.duration_s),
                  "%s: %s: duration changed" % (os.path.basename(base.source), old.label))

    # place_bowl: all 17 bowl geoms move by the delta, the ring keeps its shape
    rp = Replayer(scene_path=os.path.join(REPO, SCENE), control_hz=20.0, render_every=0)
    try:
        ids = gen.bowl_geom_ids(rp.model)
        bottom = gen.bowl_bottom_id(rp.model)
        ck.ok(len(ids) == 17, "the scene has %d bowl_* geoms, expected 17" % len(ids))
        before = np.array(rp.model.geom_pos[ids], dtype=float)
        gen.place_bowl(rp, gen.BOWL_XY)
        ck.ok(np.array_equal(np.array(rp.model.geom_pos[ids], dtype=float), before),
              "place_bowl at the current centre was not a no-op")
        layout = (float(gen.BOWL_XY[0]) + 0.020, float(gen.BOWL_XY[1]) - 0.080)
        dx = layout[0] - before[ids.index(bottom), 0]
        dy = layout[1] - before[ids.index(bottom), 1]
        expect = before.copy()
        expect[:, 0] = expect[:, 0] + dx
        expect[:, 1] = expect[:, 1] + dy
        gen.place_bowl(rp, layout)
        after = np.array(rp.model.geom_pos[ids], dtype=float)
        ck.ok(np.array_equal(after, expect),
              "place_bowl: geom_pos is not pos + delta on all 17 geoms (worst %.1e m)"
              % float(np.abs(after - expect).max()))
        ring0 = np.hypot(before[:, 0] - before[ids.index(bottom), 0],
                         before[:, 1] - before[ids.index(bottom), 1])
        ring1 = np.hypot(after[:, 0] - after[ids.index(bottom), 0],
                         after[:, 1] - after[ids.index(bottom), 1])
        ck.ok(np.abs(ring1 - ring0).max() <= 1e-12,
              "place_bowl changed the wall ring radii by %.1e m" % float(np.abs(ring1 - ring0).max()))
        walls = [k for k, gid in enumerate(ids) if gid != bottom]
        ck.ok(np.abs(ring0[walls] - gen.BOWL_OUTER_R_M).max() <= 1e-6,
              "wall ring radius %.6f..%.6f, expected %.3f (the scene writes 6 decimals)"
              % (float(ring0[walls].min()), float(ring0[walls].max()), gen.BOWL_OUTER_R_M))
        ck.ok(float(np.abs(after[:, 2] - before[:, 2]).max()) == 0.0,
              "place_bowl moved the bowl off the table")
    finally:
        rp.close()

    # sample_layout: continuous uniform over the two measured-safe boxes, MIN_SEP_M apart
    draws = 2000
    got = [gen.sample_layout(random.Random(0)) for _ in range(1)]     # same seed, twice
    got.append(gen.sample_layout(random.Random(0)))
    ck.ok(got[0] == got[1], "sample_layout: the same seed gave %r then %r" % (got[0], got[1]))
    rng = random.Random(0)
    drawn = [gen.sample_layout(rng) for _ in range(draws)]
    outside = [(cube, bowl) for cube, bowl in drawn
               if not (gen.CUBE_BOX[0] <= cube[0] <= gen.CUBE_BOX[1]
                       and gen.CUBE_BOX[2] <= cube[1] <= gen.CUBE_BOX[3]
                       and gen.BOWL_BOX[0] <= bowl[0] <= gen.BOWL_BOX[1]
                       and gen.BOWL_BOX[2] <= bowl[1] <= gen.BOWL_BOX[3])]
    ck.ok(not outside, "sample_layout left the boxes at %d/%d draws (first %r)"
          % (len(outside), draws, outside[0] if outside else None))
    seps = [float(np.hypot(cube[0] - bowl[0], cube[1] - bowl[1])) for cube, bowl in drawn]
    ck.ok(min(seps) >= gen.MIN_SEP_M, "closest cube-bowl pair %.4f m < MIN_SEP_M %.3f m"
          % (min(seps), gen.MIN_SEP_M))
    unique = len({(round(cube[0], 6), round(cube[1], 6)) for cube, _bowl in drawn})
    ck.ok(unique >= 1990,
          "%d distinct cube positions in %d draws - that is a grid, not a continuous sampler"
          % (unique, draws))
    off = [gen.layout_offsets(cube) for cube, _bowl in drawn]
    ck.ok(all(isinstance(ox, float) and isinstance(oy, float) for ox, oy in off),
          "layout_offsets stopped returning plain floats")

    # the CLI: an unreachable target fails, says which height, and writes nothing
    script = os.path.join(REPO, "experiments", "set_layout.py")
    def run(*args):
        return subprocess.run([sys.executable, script] + list(args), cwd=REPO,
                              capture_output=True, text=True)
    here = run("--cube", repr(float(gen.CUBE_XY[0])), repr(float(gen.CUBE_XY[1])),
               "--bowl", repr(float(gen.BOWL_XY[0])), repr(float(gen.BOWL_XY[1])))
    ck.ok(here.returncode == 0 and "PASS" in here.stdout,
          "set_layout.py on the current layout exited %d (stderr %r)"
          % (here.returncode, here.stderr[-200:]))
    far = run("--cube", "0.60", "0.00")
    ck.ok(far.returncode != 0, "set_layout.py --cube 0.60 0.00 exited %d, expected non-zero"
          % far.returncode)
    names_height = any("IK FAIL" in line and "0.815" in line
                       for line in far.stdout.splitlines())
    ck.ok(names_height and "FAIL" in far.stdout,
          "set_layout.py --cube 0.60 0.00 did not name the failing height in %r" % far.stdout[-300:])

    # the grasp screen: one real episode at the default layout passes, and it may not write
    gate = run("--grasp-grid", "0.474:0.474:0.02,0.0435:0.0435:0.02")
    ck.ok(gate.returncode == 0 and "1/1" in gate.stdout and "no failing point" in gate.stdout,
          "--grasp-grid on one reachable point exited %d (stdout %r)"
          % (gate.returncode, gate.stdout[-300:]))
    ck.ok("s/point" in gate.stdout,
          "--grasp-grid did not report the per-point runtime: %r" % gate.stdout[-300:])
    both = run("--grasp-grid", "0.474:0.474:0.02,0.0435:0.0435:0.02", "--apply")
    ck.ok(both.returncode != 0 and "refusing" in both.stdout,
          "--apply together with --grasp-grid exited %d (stdout %r)"
          % (both.returncode, both.stdout[-300:]))
    smp = run("--sample", "3")
    printed = [ln for ln in smp.stdout.splitlines() if "cube (" in ln]
    ck.ok(smp.returncode == 0 and len(printed) == 3,
          "set_layout.py --sample 3 exited %d and printed %d layout lines (stdout %r)"
          % (smp.returncode, len(printed), smp.stdout[-300:]))
    print("  layout hooks: %d checks; set_layout.py verdicts %s"
          % (ck.count - start,
             " ".join("%s=%d" % (name, res.returncode)
                      for name, res in (("current", here), ("0.60 0.00", far),
                                        ("grid", gate), ("grid+apply", both),
                                        ("sample", smp)))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=os.path.join(REPO, "out", "act_demos_ee"))
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args(argv)
    root = args.dir if os.path.isabs(args.dir) else os.path.join(REPO, args.dir)

    ck = Checks()
    index_path = os.path.join(root, "index.json")
    assert os.path.isfile(index_path), "%s: missing index.json" % root
    with open(index_path) as fh:
        index = json.load(fh)
    # the directory says which task it holds; --dir alone switches the contract
    place = index.get("task_kind") == "place"
    index_keys = INDEX_KEYS | INDEX_PLACE_KEYS if place else INDEX_KEYS
    ck.ok(set(index) >= index_keys and set(index) - index_keys <= OPTIONAL_INDEX_KEYS,
          "index.json keys %s, required %s plus at most %s"
          % (sorted(set(index)), sorted(index_keys), sorted(OPTIONAL_INDEX_KEYS)))
    if "cube_xy" in index:
        ck.ok([float(v) for v in index["cube_xy"]] == list(CUBE_XY),
              "index.cube_xy %r != the CUBE_XY the scene holds %r"
              % (index["cube_xy"], list(CUBE_XY)))
    ck.ok(index["task"] == (PLACE_TASK if place else "pick up the red block and lift it"),
          "index.task %r" % index["task"])
    ck.ok(index["camera_names"] == list(CAMERAS), "index.camera_names %r" % index["camera_names"])
    ck.ok(index["state_dim"] == EE_DIM and index["action_dim"] == EE_DIM,
          "index.state_dim/action_dim must both be %d, got %r/%r"
          % (EE_DIM, index["state_dim"], index["action_dim"]))
    ck.ok(isinstance(index["seed"], int), "index.seed is not an int")
    ck.ok(float(index["fps"]) == float(index["control_hz"]), "index.fps != index.control_hz")
    ck.ok(isinstance(index["randomization"], str) and "cube xy" in index["randomization"],
          "index.randomization %r" % index["randomization"])
    if place:
        ck.ok(index["table"].endswith("waypoints_cube_place.json"),
              "index.table %r is not the place table" % (index["table"],))
        ck.ok(list(index["bowl_xy_m"]) == list(BOWL_XY),
              "index.bowl_xy_m %r != %r" % (index["bowl_xy_m"], list(BOWL_XY)))
        # datasets generated before the 2026-11-01 decision record the stricter 0.030 here
        ck.ok(0.0 < float(index["in_bowl_radius_m"]) <= BOWL_INNER_R_M + 1e-12,
              "index.in_bowl_radius_m %r is outside the bowl" % (index["in_bowl_radius_m"],))
        ck.ok(index["in_bowl_all"] is True, "index.in_bowl_all is %r" % (index["in_bowl_all"],))
        ck.ok(float(index["bowl_dxy_max_m"]) <= MAX_BOWL_DXY_M,
              "index.bowl_dxy_max_m %r above %r m"
              % (index["bowl_dxy_max_m"], MAX_BOWL_DXY_M))
        ck.ok(float(index["hold_err_max_m"]) <= MAX_HOLD_ERR_M,
              "index.hold_err_max_m %r above %r m"
              % (index["hold_err_max_m"], MAX_HOLD_ERR_M))
    ck.ok(index["generated_by"] == "experiments/gen_act_dataset.py",
          "index.generated_by %r" % index["generated_by"])
    ck.ok(index["mujoco_version"].startswith("3.16"), "index.mujoco_version %r"
          % index["mujoco_version"])

    eps = sorted(d for d in os.listdir(root)
                 if d.startswith("ep") and os.path.isdir(os.path.join(root, d)))
    ck.ok(eps == ["ep%03d" % k for k in range(args.episodes)],
          "episode directories %s != ep000..ep%03d" % (eps, args.episodes - 1))
    ck.ok(int(index["num_episodes"]) == len(eps),
          "index.num_episodes %r != %d directories found" % (index["num_episodes"], len(eps)))

    worst = {"pose_err": 0.0, "quat_err": 0.0, "consec_dot": 1.0, "same_tick_dot": 1.0}
    bowl_dxy_max = 0.0
    hold_err_max = 0.0
    for k, name in enumerate(eps):
        ep_dir = os.path.join(root, name)
        got = check_episode(ck, ep_dir, index, place)
        worst["pose_err"] = max(worst["pose_err"], got["pose_err"])
        worst["quat_err"] = max(worst["quat_err"], got["quat_err"])
        worst["consec_dot"] = min(worst["consec_dot"], got["consec_dot"])
        worst["same_tick_dot"] = min(worst["same_tick_dot"], got["same_tick_dot"])
        with open(os.path.join(ep_dir, "meta.json")) as fh:
            meta = json.load(fh)
        ck.ok(meta["episode"] == k, "%s: meta.episode %r != %d" % (ep_dir, meta["episode"], k))
        if place:
            bowl_dxy_max = max(bowl_dxy_max, float(meta["bowl_dxy_m"]))
            hold_err_max = max(hold_err_max, float(meta["hold_err_m"]))
            print("  %s  %3d frames  bowl_dxy %.5f m  in_bowl %s  hold %.5f m  cube xy %+0.3f %+0.3f"
                  % (name, int(meta["num_frames"]), float(meta["bowl_dxy_m"]),
                     bool(meta["in_bowl"]), float(meta["hold_err_m"]),
                     float(meta["cube_pos"][0]), float(meta["cube_pos"][1])))
        else:
            print("  %s  %3d frames  rise %.5f m  slip %.6f m  cube xy %+0.3f %+0.3f"
                  % (name, int(meta["num_frames"]), float(meta["rise_m"]), float(meta["slip_m"]),
                     float(meta["cube_pos"][0]), float(meta["cube_pos"][1])))
    if place:
        # the index must describe what is on disk, not what the generator hoped for
        ck.ok(abs(bowl_dxy_max - float(index["bowl_dxy_max_m"])) < 1e-6,
              "index.bowl_dxy_max_m %r != the worst episode %r"
              % (index["bowl_dxy_max_m"], bowl_dxy_max))
        ck.ok(abs(hold_err_max - float(index["hold_err_max_m"])) < 1e-6,
              "index.hold_err_max_m %r != the worst episode %r"
              % (index["hold_err_max_m"], hold_err_max))
        print("  place: %d episodes, bowl_dxy <= %.5f m, hold_err <= %.5f m, all %d frames"
              % (len(eps), bowl_dxy_max, hold_err_max, PLACE_FRAMES))

    g_worst, g_rows = check_gripper_roundtrip(ck)
    check_layout(ck)
    print("  FK(qpos) vs pose8_state: pos %.2e m, quat %.2e (bound %.0e)"
          % (worst["pose_err"], worst["quat_err"], FK_TOL_M))
    print("  quaternion min dot: consecutive %.9f, state-vs-action %.9f"
          % (worst["consec_dot"], worst["same_tick_dot"]))
    print("  g -> ctrl -> %s -> g: max error %.2e (bound %.0e); %s"
          % (G_JOINT, g_worst, G_TOL,
             " ".join("g=%.2f:q7=%.6f" % (g, q7) for g, q7, _ in g_rows)))
    print("%d checks over %d episodes in %s" % (ck.count, len(eps), root))
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

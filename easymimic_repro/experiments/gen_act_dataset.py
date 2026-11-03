#!/usr/bin/env python3
"""Generate ACT training episodes on scene_cube_bowl: pick-and-lift, or pick-and-place.

One episode = an 8-D waypoint table replayed at CONTROL_HZ against a cube and a bowl that have
been put at one drawn layout; every waypoint that chases the cube is re-seated on the cube and
the four ANCHORED_LABELS waypoints on the bowl, so the grasp column follows the cube and the
release column follows the bowl.  The cube keeps its spawn height (0.815 m = table top 0.80 +
half height 0.015) and its yaw.

--table picks the table and with it the task.  The default stays the lift table.  The table's
own labels decide which task it is, not a flag: a table carrying any of ANCHORED_LABELS is the
place task, because those four waypoints live over the bowl.  The lift table has none of those
labels, so its episodes carry only the cube half of the layout (the bowl is still put at the
sampled centre, it is simply never approached).  npz keys, numbers and meta/index/summary keys
are the same for both tasks, plus the layout pair (cube_xy/bowl_xy) they record.

Where the cube and the bowl sit for one episode is decided by exactly one function:

    sample_layout(rng) -> (cube_xy, bowl_xy)

It draws cube_xy uniformly (continuous, one rng.uniform per axis, x then y) inside CUBE_BOX,
bowl_xy uniformly inside BOWL_BOX, and redraws *both* until the two centres are at least
MIN_SEP_M apart.  The boxes are the measured-safe region, not the whole reachable one - the
constant comments below say what each bound was measured against and docs/ACT_PIPELINE.md
section 11.4 has the numbers behind them.  "Continuous" is the point: every real pair inside
the boxes is legal, there is no grid, no rounding and no fixed point list, so a sampler built
on these bounds can produce any layout the boxes contain.

layout_table() re-seats a table on an arbitrary absolute pair (the grasp column follows the
cube, ANCHORED_LABELS follow the bowl, z and durations untouched) and place_bowl() moves the
bowl's 17 static geoms before the first tick, so the episode loop needs nothing else.
experiments/act_rollout.py calls the same sample_layout, so training and evaluation cannot
sample different distributions, and --replay-demos restores an episode's layout from its
meta.json (cube_xy/bowl_xy; a meta.json predating them falls back to cube_pos and BOWL_XY)
instead of resampling it.  meta.json keeps cube_pos with its old meaning and gains
cube_xy/bowl_xy; summary.json's per-episode rows keep dx/dy (now derived from the sampled
centre) and gain the absolute pair.  The data.npz keys and numbers are unchanged.

The place task adds three measurements, written to data.npz, meta.json, index.json and
summary.json (in index.json they are the worst case over the episodes it describes):

    bowl_dxy_m  last-frame cube-centre xy distance to the bowl centre (m)
    in_bowl     bowl_dxy_m < IN_BOWL_RADIUS_M and the cube centre strictly inside
                IN_BOWL_Z, i.e. the cube is resting in the bowl, not on its rim or beside it
    hold_err_m  on the last tick whose *commanded* g was 0, the horizontal distance
                between the cube centre and tool0 - the cube still being between the pads
                at the moment the release starts.  rise_m/slip_m keep their lift definition.

Per control tick an episode stores the 8-D end-effector interface the ACT policy reads and
writes, plus the joint-space debug rows and one 256x256 frame from each camera:

    pose8_state  (N,8)  [x, y, z, qw, qx, qy, qz, g] of site tool0 in the robot base frame
                        (== world frame here) in metres, measured *before* tick k's action
                        is applied; g is the measured jaw opening, qpos[k][6] inverted
                        through GRIPPER_CONFIG (g = 1 open)
    pose8_action (N,8)  the same 8 numbers *after* tick k's action was applied, i.e. one
                        CONTROL_HZ step later - so [0:7] of row k equals [0:7] of
                        pose8_state[k+1] exactly for k < N-1 - except g, which is the g
                        that this tick commanded rather than the jaw position it reached
    qpos (N,7)  [0:6] arm joints, [6] gripper hinge (joint_7)
    act  (N,7)  data.ctrl[:7] after apply(); the six arm targets already carry the
                gravity feed-forward (src/pose_cmd/replay.py:apply), so act - qpos is a
                real servo offset, not a modelling error.  Kept as a debug row.
    ik_err (N,) IK position residual (m) of every tick's solve - the whole series, not just
                the maximum that gates the attempt
    t    (N,)   seconds, t[k] = k / CONTROL_HZ

The EE pose is the pose of the site the IK actually drives (site "tool0", the pinch centre
at assets/jaka_zu3/robot.xml:142), read back through MuJoCo kinematics from the recorded
qpos - it is never re-assembled from a hand-written FK chain, and the FK self-check below
proves the read-back reproduces what was stored.  Quaternions are scalar-first and unit
length.  Within an episode the state sequence is sign-walked so consecutive frames keep
dot(q_k, q_{k-1}) >= 0, while each action quaternion simply takes the sign of its own tick's
state quaternion - the two sequences therefore never disagree on the same tick.

Frames are rendered *before* the tick's action is applied, so frame k shows exactly
qpos[k] and pairs with pose8_state[k].  write_png is the hand-rolled writer from
src/pose_cmd/replay.py (no PIL in this env); row 0 is the top of the picture.

An attempt is thrown away and resampled when the lifted cube ends lower than MIN_RISE_M,
when any IK residual exceeds MAX_IK_ERR_M, when an IK solve fails outright, or when a place
episode misses the place gate (see PLACE_METRICS below) - so the output directory only ever
holds successful episodes.  Same --seed -> same drawn layouts -> the same physics ->
bit-identical qpos (nothing here reads the clock or a thread pool).

The output directory is rebuilt from scratch on every run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mujoco  # noqa: E402

from src.pose_cmd.replay import (GRIPPER_CONFIG, ReplayError, Replayer,  # noqa: E402
                                 g_to_ctrl, write_png)
from src.pose_cmd.waypoints import Table, Waypoint, load_table, plan_trajectory  # noqa: E402

SCENE = "assets/jaka_zu3/scene_cube_bowl.xml"
TABLE_PATH = "configs/waypoints_cube_lift.json"
DEFAULT_OUT = "out/act_demos_ee"
PLACE_TABLE_PATH = "configs/waypoints_cube_place.json"
PLACE_OUT = "out/act_demos_place"

CONTROL_HZ = 20.0            # 108 ticks for the 5.35 s table; rise unchanged vs 50 Hz
CAMERAS = ("front", "wrist")
CAMERA_HW = (256, 256)

EE_SITE = "tool0"            # the site the IK drives: the pinch centre, robot.xml:142
EE_DIM = 8                   # x, y, z, qw, qx, qy, qz, g
FK_TOL = 1e-9                # recorded qpos -> FK must reproduce pose8 to this, per element

CUBE_XY = (0.474, 0.0435)    # cube body xy in the scene = the table's grasp column
CUBE_Z = 0.815               # table top 0.80 + half height 0.015

# The layout sampler's boxes, (XMIN, XMAX, YMIN, YMAX) in metres.  These are the measured-SAFE
# region, not merely the reachable one; every bound below came from a measurement:
#   CUBE_BOX  grasp map - one real place episode per point over x 0.42..0.52 step 0.01 and
#             y -0.10..0.16 step 0.02; the largest all-pass rectangle was x 0.450..0.490,
#             y -0.100..0.140 (hold_err_m 0.0006..0.0016 m inside it), pulled in by 5 mm in x
#             and 10 mm in y.  x stops at 0.485 because past it the pads push the cube instead
#             of holding it, while IK still converges out to ~0.52 - exactly the trap the grasp
#             map exists to close.
#   BOWL_BOX  IK map at the three bowl heights (0.870/0.935/0.945; pos <= 2e-3 m, ang <=
#             2 deg) - largest all-pass rectangle x 0.200..0.420, y -0.320..0.320, pulled in by
#             5 mm in x and 10 mm in y; the -y side is then clipped by the front camera, which
#             projects a bowl corner at y = -0.320 to v = 258 px, i.e. out of the 256 px frame.
#             x starts at 0.310, not 0.205: the release segment's settling error grows as the
#             bowl moves away from the arm root, so bowls at x < 0.30 failed the place gate on
#             the drop (bowl_dxy_m 0.0052..0.1101 m) while every bowl at x >= 0.30 passed, with
#             a good grasp in both cases (hold_err_m 0.0004..0.0016 m).
#   24 draws from exactly these boxes - all four cube corners, all four bowl corners, 12
#   interior pairs and 4 pairs on the separation bound - were all accepted, and all four corner
#   projections stay inside the front camera frame with >= 12 px to spare.
CUBE_BOX = (0.455, 0.485, -0.090, 0.130)
BOWL_BOX = (0.310, 0.415, -0.235, 0.295)
MIN_SEP_M = 0.15             # smallest legal cube-bowl centre distance (bowl outer radius is
                             # 0.049 m, so 0.15 keeps the grasp column clear of the wall ring)
SAMPLE_MAX_TRIES = 1000      # redraws both centres per try; the cap is a bug detector, not an
                             # expected path - a legal pair is the overwhelmingly likely draw

# NOT used by sample_layout() any more: the sampler draws inside CUBE_BOX/BOWL_BOX, and the
# +-0.020 x +-0.005 box these describe is historical.  Only legacy/act_rollout_joint7.py still
# imports them (for its own summary.json "randomization_m" block); experiments/act_rollout.py
# now reports layout_m = {cube_box_m, bowl_box_m, min_sep_m}.  Never read these as the bounds.
RANDOM_X_M = 0.020
RANDOM_Y_M = 0.005

MIN_RISE_M = 0.080           # same bar as src/pose_cmd/replay.py CRITERIA['min_rise_m']
MAX_IK_ERR_M = 2.0e-3        # the 50 Hz baseline's worst IK residual is 9.2e-4 m
EPISODES = 20
SEED = 0
TASK = "pick up the red block and lift it"
PLACE_TASK = "pick up the red block and put it in the bowl"

# Waypoints that sit over the bowl.  The bowl is part of the scene, so it does not move with
# the cube offset: these four labels keep their table xy while the cube-chasing labels are
# shifted by (dx, dy).  A table with none of these labels is the lift task and every waypoint
# is shifted, which is what the pre-place-task episodes were generated with.
ANCHORED_LABELS = ("carry", "lower", "release", "retreat")

BOWL_XY = (0.3, -0.2)    # bowl centre in the scene
BOWL_INNER_R_M = 0.045       # bowl inner radius; wall top z = 0.840, floor z = 0.802
BOWL_OUTER_R_M = 0.049       # wall ring centreline; the bowl's footprint radius (+ wall)
BOWL_GEOM_PREFIX = "bowl_"   # the bowl's geoms: bottom disc + bowl_wall_00 .. _15
IN_BOWL_RADIUS_M = BOWL_INNER_R_M
# Judge == the bowl's own aperture: a cube that slid down and is leaning on the wall counts as
# in the bowl (decision taken 2026-11-01; it used to be 0.030). The stricter "resting flat on the
# floor" reading is kept next to it so both counts can be reported for the same trial.
IN_BOWL_SEATED_R_M = 0.030   # centre here or closer = the cube sits on the floor, not on the wall
IN_BOWL_Z = (0.800, 0.840)   # cube centre while it rests in the bowl (0.817 on the floor)

MAX_BOWL_DXY_M = 0.005       # place gate: the cube must land this close to the bowl centre
MAX_HOLD_ERR_M = 0.002       # place gate: the cube must still be between the pads at release

PLACE_METRICS = ("bowl_dxy_m", "in_bowl", "hold_err_m")


def table_task(base):
    """"place" when the table has any waypoint over the bowl, else "lift"."""
    return "place" if any(w.label in ANCHORED_LABELS for w in base.waypoints) else "lift"


def shifted_table(base, dx, dy):
    """The template table moved by (dx, dy) - except the waypoints anchored over the bowl.

    Orientation is never touched.  A table with no ANCHORED_LABELS waypoint is shifted
    everywhere, i.e. exactly what this function did before the place task existed.
    """
    waypoints = []
    for w in base.waypoints:
        ee8 = np.array(w.ee8, dtype=float)
        if w.label not in ANCHORED_LABELS:
            ee8[0] += dx
            ee8[1] += dy
        waypoints.append(Waypoint(label=w.label, ee8=ee8, duration_s=w.duration_s))
    return Table(source=base.source, frame=base.frame, units=base.units,
                 control_hz=CONTROL_HZ, waypoints=tuple(waypoints))


def layout_table(base, cube_xy, bowl_xy):
    """The template table re-seated on an absolute (cube_xy, bowl_xy) pair.

    The waypoints that chase the cube take cube_xy, the ANCHORED_LABELS waypoints take
    bowl_xy (that is what "anchored over the bowl" means once the bowl itself can move);
    every other waypoint rides along with the cube column.  z, orientation and durations are
    copied through untouched, so this is shifted_table() for a scene whose objects may have
    moved: layout_table(base, CUBE_XY, BOWL_XY) is shifted_table(base, 0.0, 0.0).
    """
    waypoints = []
    for w in base.waypoints:
        ee8 = np.array(w.ee8, dtype=float)
        xy = bowl_xy if w.label in ANCHORED_LABELS else cube_xy
        ee8[0] = float(xy[0])
        ee8[1] = float(xy[1])
        waypoints.append(Waypoint(label=w.label, ee8=ee8, duration_s=w.duration_s))
    return Table(source=base.source, frame=base.frame, units=base.units,
                 control_hz=CONTROL_HZ, waypoints=tuple(waypoints))


def layout_offsets(cube_xy):
    """(dx, dy) of a sampled cube centre relative to CUBE_XY - the offset the reporting uses.

    The sampler returns absolute centres, so the offset is derived here rather than drawn;
    the subtraction is exact to one ulp of CUBE_XY and is never fed back into the physics
    (place_cube_abs() takes the centre, not the offset), so trajectories do not depend on it.
    """
    return (float(cube_xy[0] - CUBE_XY[0]), float(cube_xy[1] - CUBE_XY[1]))


def sample_layout(rng):
    """One episode's (cube_xy, bowl_xy): THE distribution, for training and for evaluation.

    Continuous uniform over two measured-safe boxes, plus one rejection step:

        cube_xy ~ U(CUBE_BOX[0..1] x CUBE_BOX[2..3])      x drawn first, then y
        bowl_xy ~ U(BOWL_BOX[0..1] x BOWL_BOX[2..3])
        both redrawn until hypot(cube_xy - bowl_xy) >= MIN_SEP_M

    Nothing is quantised: every real pair inside the boxes is a legal output, there is no grid
    and no fixed point list.  "Safe" was measured against a real episode per point - the cube
    box is the all-pass rectangle of the grasp map (hold_err_m: the cube is still between the
    pads) and the bowl box is the all-pass rectangle of the IK map at the three bowl heights
    (0.870/0.935/0.945), clipped in -y so the front camera still sees a bowl corner and at
    x >= 0.310 where the drop stopped failing; see the CUBE_BOX/BOWL_BOX comments above and
    docs/ACT_PIPELINE.md section 11.4.  None of it says anything about self-collision, which
    is not modelled here at all - nor about cube heights other than CUBE_Z.

    Deterministic given the rng: same seed -> same first pair, same stream order for every
    caller.  layout_table(), place_bowl(), meta.json and index.json all take whatever this
    returns, and experiments/act_rollout.py calls it too, so a rollout samples the same
    distribution as the demos it is scored on.
    """
    for _try in range(SAMPLE_MAX_TRIES):
        cube_xy = (float(rng.uniform(CUBE_BOX[0], CUBE_BOX[1])),
                   float(rng.uniform(CUBE_BOX[2], CUBE_BOX[3])))
        bowl_xy = (float(rng.uniform(BOWL_BOX[0], BOWL_BOX[1])),
                   float(rng.uniform(BOWL_BOX[2], BOWL_BOX[3])))
        if math.hypot(cube_xy[0] - bowl_xy[0], cube_xy[1] - bowl_xy[1]) >= MIN_SEP_M:
            return cube_xy, bowl_xy
    raise RuntimeError("sample_layout: no pair with cube-bowl >= %r m in %d tries "
                       "(CUBE_BOX %r, BOWL_BOX %r)"
                       % (MIN_SEP_M, SAMPLE_MAX_TRIES, CUBE_BOX, BOWL_BOX))


def place_cube_abs(rp, cube_xy):
    """place_cube() for an absolute centre instead of an offset from CUBE_XY."""
    rp.data.qpos[7:10] = [float(cube_xy[0]), float(cube_xy[1]), CUBE_Z]
    rp.data.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]
    rp.data.qvel[:] = 0.0
    mujoco.mj_forward(rp.model, rp.data)


def place_cube(rp, dx, dy):
    """Replayer.teleport() rewrites the free joint from the home keyframe, so the cube is
    put back afterwards - once, before the first tick."""
    place_cube_abs(rp, (CUBE_XY[0] + dx, CUBE_XY[1] + dy))


def bowl_geom_ids(model):
    """Ids of the bowl's geoms - bottom disc plus the 16 wall boxes, by name prefix."""
    ids = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name is not None and name.startswith(BOWL_GEOM_PREFIX):
            ids.append(gid)
    return ids


def bowl_bottom_id(model):
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bowl_bottom"))


def place_bowl(rp, bowl_xy):
    """Move the bowl to bowl_xy; a no-op when it already sits there.

    The bowl has no joint - it is 17 worldbody geoms - so "moving" it is translating every
    one of its geoms by the same delta and re-running the kinematics.  mj_forward rebuilds
    geom_xpos and every contact from the new geom_pos, so collision geometry really moves
    (the static geoms carry no dof and no inertia the solver integrates).  One caveat is the
    same one the bowl has always had: ncon scratch and warm-start are rebuilt from scratch on
    the next step, and the caller runs this before the first tick, not mid-episode.
    """
    model = rp.model
    bottom = bowl_bottom_id(model)
    dx = float(bowl_xy[0]) - float(model.geom_pos[bottom][0])
    dy = float(bowl_xy[1]) - float(model.geom_pos[bottom][1])
    if dx == 0.0 and dy == 0.0:
        return
    for gid in bowl_geom_ids(model):
        pos = model.geom_pos[gid]
        pos[0] = pos[0] + dx
        pos[1] = pos[1] + dy
    mujoco.mj_forward(model, rp.data)


def bowl_center(model):
    """The bowl's centre as the model currently holds it (the bottom disc's xy)."""
    pos = model.geom_pos[bowl_bottom_id(model)]
    return (float(pos[0]), float(pos[1]))



def hinge_g(q7):
    """joint_7 position -> jaw opening g (0 closed, 1 open): the exact inverse of
    replay.g_to_ctrl(), same GRIPPER_CONFIG, so g = q * open_at_g / ctrl_per_g."""
    return float(q7) * float(GRIPPER_CONFIG["open_at_g"]) / float(GRIPPER_CONFIG["ctrl_per_g"])


def site_pose(data, site_id):
    """(pos (3,), quat (4,) scalar-first) of the site, out of whatever kinematics the data
    already holds. mju_mat2Quat is MuJoCo's own rotation-matrix -> quaternion, w first."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(data.site_xmat[site_id], dtype=float).ravel())
    return np.array(data.site_xpos[site_id], dtype=float), quat


def fk_tool(fk_data, site_id, qpos7):
    """tool0 pose for a recorded qpos. fk_data is a private MjData: the plant's data is
    never touched, so sampling costs nothing in the trajectory (no extra mj_forward on the
    live state, no contact-warmstart perturbation, same physics as before this change).
    The cube's free-joint coordinates are irrelevant to the arm chain, so they stay 0."""
    fk_data.qpos[:] = 0.0
    fk_data.qpos[:len(qpos7)] = qpos7
    mujoco.mj_kinematics(fk_data.model, fk_data)
    return site_pose(fk_data, site_id)


def sign_walk(quats):
    """Flip whole quaternions so that dot(q_k, q_{k-1}) >= 0 and the first frame has
    qw >= 0. The rotation is unchanged: q and -q are the same orientation."""
    out = np.array(quats, dtype=float, copy=True)
    if out[0, 0] < 0.0:
        out[0] = -out[0]
    for k in range(1, len(out)):
        if float(out[k] @ out[k - 1]) < 0.0:
            out[k] = -out[k]
    return out


def fk_selfcheck(model, site_id, qpos_rows, state8):
    """Recompute tool0 for every recorded qpos through a *different* entry point than the
    sampler used (a full mj_forward on a fresh MjData, instead of mj_kinematics on the
    sampler's scratch data) and compare against the stored state8. A mismatch would mean
    the stored pose is not the pose of the recorded qpos - i.e. the policy is being taught
    a state it can never observe. Returns (max position error, max quaternion error), the
    quaternion compared up to sign."""
    check = mujoco.MjData(model)
    max_pos = 0.0
    max_quat = 0.0
    for k, q in enumerate(qpos_rows):
        check.qpos[:] = 0.0
        check.qpos[:len(q)] = q
        mujoco.mj_forward(model, check)
        pos, quat = site_pose(check, site_id)
        want = state8[k, 3:7]
        max_pos = max(max_pos, float(np.abs(pos - state8[k, 0:3]).max()))
        max_quat = max(max_quat, float(min(np.abs(quat - want).max(),
                                           np.abs(quat + want).max())))
    if max_pos > FK_TOL or max_quat > FK_TOL:
        raise AssertionError("FK self-check failed: pos %.3e m, quat %.3e > %.0e m"
                             % (max_pos, max_quat, FK_TOL))
    return max_pos, max_quat


def run_attempt(base, cube_xy, bowl_xy, ep_dir, render):
    """One episode at the layout (cube_xy, bowl_xy). Returns the measurement dict; writes
    data.npz and meta.json into ep_dir (the caller renames it into place only if the attempt
    is accepted)."""
    traj = plan_trajectory(layout_table(base, cube_xy, bowl_xy))
    task = table_task(base)
    rp = Replayer(scene_path=os.path.join(REPO, SCENE), control_hz=CONTROL_HZ,
                  render_every=0)
    renderer = None
    try:
        model = rp.model
        place_bowl(rp, bowl_xy)
        os.makedirs(ep_dir, exist_ok=True)
        cams = []
        if render:
            renderer = mujoco.Renderer(model, height=CAMERA_HW[0], width=CAMERA_HW[1])
            cams = [(name, model.camera(name).id) for name in CAMERAS]
            for name in CAMERAS:
                os.makedirs(os.path.join(ep_dir, name), exist_ok=True)

        first = np.asarray(traj.ee8[0], dtype=float)
        res0 = rp.solve_frame(traj.label[0], first[0:3], first[3:6], rp.home_qpos())
        rp.teleport(np.array(res0.qpos, dtype=float), float(first[6]))
        place_cube_abs(rp, cube_xy)
        cube_start = rp.cube_pos()
        cube_quat = np.array(rp.data.xquat[rp.cube_body], dtype=float)

        # Private kinematics scratch: the plant's data is never re-forward-ed, so the
        # trajectory stays bit-identical to the one this script produced before the EE
        # rows existed (no contact warmstart perturbation, no extra solver calls).
        fk_data = mujoco.MjData(model)
        site_id = rp.site_id
        qpos, act, ik_errs, stamps, state_rows, action_rows = [], [], [], [], [], []
        cube_grasp = None
        hold_err = float("nan")
        for k in range(len(traj)):
            if renderer is not None:
                for name, cam_id in cams:
                    renderer.update_scene(rp.data, camera=cam_id)
                    write_png(os.path.join(ep_dir, name, "%06d.png" % k), renderer.render())
            ee8 = np.asarray(traj.ee8[k], dtype=float)
            q = rp.data.qpos[:7].copy()
            qpos.append(q)
            # state8[k]: EE pose *before* this tick's action, i.e. of qpos[k] - the very
            # frame that was just rendered - plus the jaw opening that qpos[k] encodes
            pos, quat = fk_tool(fk_data, site_id, q)
            state_rows.append(np.concatenate([pos, quat, [hinge_g(q[6])]]))
            # hold_err_m: the last tick that *commands* a closed gripper, i.e. the last
            # moment the cube is supposed to be between the pads.  Both terms are the tick-k
            # pre-action snapshot, the same frame state8[k] describes.
            if task == "place":
                cube_now = rp.cube_pos()
                if float(ee8[6]) == 0.0:
                    hold_err = float(np.hypot(pos[0] - cube_now[0], pos[1] - cube_now[1]))
            res = rp.solve_frame(traj.label[k], ee8[0:3], ee8[3:6],
                                 rp.data.qpos[:6].copy())
            ik_errs.append(float(res.pos_err))
            rp.apply(np.array(res.qpos, dtype=float), float(ee8[6]))
            act.append(rp.data.ctrl[:7].copy())
            if cube_grasp is None and float(ee8[6]) <= 0.0:
                cube_grasp = rp.cube_pos()
            stamps.append(k / CONTROL_HZ)
            rp.step()
            # action8[k]: EE pose *after* this tick's action (one CONTROL_HZ step of
            # physics later, hence equal to state8[k+1][0:7]) paired with the g this tick
            # commanded, not the g the jaw managed to reach
            pos, quat = fk_tool(fk_data, site_id, rp.data.qpos[:7])
            action_rows.append(np.concatenate([pos, quat, [float(ee8[6])]]))

        cube_end = rp.cube_pos()
        rise = float(cube_end[2] - cube_start[2])
        place = {}
        if task == "place":
            if not np.isfinite(hold_err):
                raise AssertionError("place table never commands a closed gripper: "
                                     "hold_err_m undefined")
            bowl_dxy = float(np.hypot(cube_end[0] - float(bowl_xy[0]),
                                      cube_end[1] - float(bowl_xy[1])))
            place = {"bowl_dxy_m": bowl_dxy,
                     "in_bowl": bool(bowl_dxy < IN_BOWL_RADIUS_M
                                     and IN_BOWL_Z[0] < float(cube_end[2]) < IN_BOWL_Z[1]),
                     "hold_err_m": hold_err}
        # Slip is drift after the jaws closed; the table always closes them, but a table
        # edit that never asks for a closed gripper must not divide by nothing here.
        slip = 0.0 if cube_grasp is None else float(np.linalg.norm(cube_end[:2] - cube_grasp[:2]))
        n = len(qpos)
        ik_err = max(ik_errs)
        act_arr = np.stack(act).astype(np.float64)
        qpos_arr = np.stack(qpos).astype(np.float64)

        state8 = np.stack(state_rows).astype(np.float64)
        action8 = np.stack(action_rows).astype(np.float64)
        state8[:, 3:7] = sign_walk(state8[:, 3:7])
        # action quaternions inherit their own tick's state sign, so the pair can never
        # disagree (an independent walk per sequence would allow exactly that)
        flip = np.einsum("ij,ij->i", action8[:, 3:7], state8[:, 3:7]) < 0.0
        action8[flip, 3:7] *= -1.0

        quat_min_dot = float(np.min(np.einsum("ij,ij->i", state8[1:, 3:7],
                                              state8[:-1, 3:7]))) if n > 1 else 1.0
        same_tick_dot = float(np.min(np.einsum("ij,ij->i", state8[:, 3:7],
                                               action8[:, 3:7])))
        if quat_min_dot < 0.0:
            raise AssertionError("quaternion sign walk broken: min consecutive dot %.6f"
                                 % quat_min_dot)
        if same_tick_dot <= 0.0:
            raise AssertionError("pose8_state/pose8_action disagree in sign on a tick: "
                                 "min dot %.6f" % same_tick_dot)
        for label, block in (("state", state8), ("action", action8)):
            norm_err = float(np.abs(np.linalg.norm(block[:, 3:7], axis=1) - 1.0).max())
            if norm_err > 1e-12:
                raise AssertionError("pose8_%s quaternions not unit: %.3e" % (label, norm_err))
        pose_err = float(np.abs(action8[:-1, 0:7] - state8[1:, 0:7]).max()) if n > 1 else 0.0
        if pose_err > 1e-12:
            raise AssertionError("action8[k] is not state8[k+1]'s pose: max %.3e" % pose_err)
        ctrl_err = float(np.abs(act_arr[:, 6] - np.array([g_to_ctrl(g)
                                                          for g in action8[:, 7]])).max())
        if ctrl_err > 1e-12:
            raise AssertionError("act[:6] g and the commanded g disagree: %.3e" % ctrl_err)
        fk_max_pos, fk_max_quat = fk_selfcheck(model, site_id, qpos, state8)

        arrays = {"pose8_state": state8,
                  "pose8_action": action8,
                  "qpos": qpos_arr,
                  "act": act_arr,
                  "ik_err": np.array(ik_errs, dtype=np.float64),
                  "t": np.array(stamps, dtype=np.float64),
                  "cube_pos": np.asarray(cube_start, dtype=np.float64),
                  "cube_quat": cube_quat.astype(np.float64),
                  "rise_m": np.float64(rise),
                  "slip_m": np.float64(slip)}
        if task == "place":
            # scalars, not per-tick series: the two ends of the transfer and the grasp hold
            arrays["bowl_dxy_m"] = np.float64(place["bowl_dxy_m"])
            arrays["in_bowl"] = np.bool_(place["in_bowl"])
            arrays["hold_err_m"] = np.float64(place["hold_err_m"])
        np.savez(os.path.join(ep_dir, "data.npz"), **arrays)
        result = {"num_frames": n, "rise_m": rise, "slip_m": slip, "ik_err_m": ik_err,
                  "fk_err_m": max(fk_max_pos, fk_max_quat), "quat_min_dot": quat_min_dot,
                  "state_action_min_dot": same_tick_dot,
                  "cube_pos": [float(cube_xy[0]), float(cube_xy[1]), float(CUBE_Z)],
                  "cube_xy": [float(cube_xy[0]), float(cube_xy[1])],
                  "bowl_xy": [float(bowl_xy[0]), float(bowl_xy[1])]}
        result.update(place)
        return result
    finally:
        if renderer is not None:
            renderer.close()
        rp.close()


def write_meta(ep_dir, episode, cube_xy, bowl_xy, result):
    meta = {"episode": int(episode),
            "fps": float(CONTROL_HZ),
            "num_frames": int(result["num_frames"]),
            "camera_hw": [int(CAMERA_HW[0]), int(CAMERA_HW[1])],
            "cube_pos": [float(v) for v in result["cube_pos"]],
            "cube_xy": [float(cube_xy[0]), float(cube_xy[1])],
            "bowl_xy": [float(bowl_xy[0]), float(bowl_xy[1])],
            "cube_yaw": 0.0,
            "rise_m": float(result["rise_m"]),
            "slip_m": float(result["slip_m"]),
            "grasp_ok": True,
            "scene": SCENE,
            "control_hz": float(CONTROL_HZ)}
    for key in PLACE_METRICS:
        if key in result:
            meta[key] = (bool(result[key]) if key == "in_bowl" else float(result[key]))
    with open(os.path.join(ep_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return meta


def randomization_note(anchored=False):
    note = ("cube xy ~ U(%.3f..%.3f) x (%.3f..%.3f), bowl xy ~ U(%.3f..%.3f) x (%.3f..%.3f) m, "
            "cube-bowl centre distance >= %.3f m, continuous (no grid), cube yaw fixed"
            % (CUBE_BOX[0], CUBE_BOX[1], CUBE_BOX[2], CUBE_BOX[3],
               BOWL_BOX[0], BOWL_BOX[1], BOWL_BOX[2], BOWL_BOX[3], MIN_SEP_M))
    if anchored:
        note += "; waypoints %s anchored at the bowl, everything else follows the cube" % (
            "/".join(ANCHORED_LABELS),)
    return note


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED,
                    help="offset stream seed (same seed -> identical qpos)")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--table", default=TABLE_PATH,
                    help="waypoint table to replay; its labels pick the task (default: "
                         "the lift table %s)" % TABLE_PATH)
    ap.add_argument("--out", default=None,
                    help="output directory (rebuilt); default %s for the lift table, %s "
                         "for a table with waypoints over the bowl" % (DEFAULT_OUT, PLACE_OUT))
    ap.add_argument("--no-render", action="store_true",
                    help="skip PNGs (contract-incomplete; only for reproducibility checks)")
    ap.add_argument("--max-attempts", type=int, default=200)
    args = ap.parse_args(argv)

    table_path = args.table if os.path.isabs(args.table) else os.path.join(REPO, args.table)
    base = load_table(table_path)
    if base.control_hz != 50.0:
        raise SystemExit("%s: expected the 50 Hz template table, got %g"
                         % (args.table, base.control_hz))
    task = table_task(base)
    out_arg = args.out if args.out is not None else (
        PLACE_OUT if task == "place" else DEFAULT_OUT)
    out_dir = out_arg if os.path.isabs(out_arg) else os.path.join(REPO, out_arg)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    rng = np.random.default_rng(args.seed)
    attempts = 0
    made = 0
    rows = []
    t_wall = time.perf_counter()
    while made < args.episodes:
        if attempts >= args.max_attempts:
            raise SystemExit("gave up after %d attempts with %d episodes accepted"
                             % (attempts, made))
        cube_xy, bowl_xy = sample_layout(rng)
        dx, dy = layout_offsets(cube_xy)
        attempts += 1
        tmp_dir = os.path.join(out_dir, "_attempt")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            result = run_attempt(base, cube_xy, bowl_xy, tmp_dir, render=not args.no_render)
        except ReplayError as exc:
            print("[ep%03d] attempt %d dx %+0.4f dy %+0.4f  rejected: %s"
                  % (made, attempts, dx, dy, exc))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue
        reason = None
        if task == "place":
            # a place episode is judged by where the cube ends up and by whether it was still
            # held when the release started - lifting it is the means, not the goal, so the
            # lift task's MIN_RISE_M bar does not apply here
            if not result["in_bowl"]:
                reason = "cube not in bowl: bowl_dxy %.5f m" % (result["bowl_dxy_m"],)
            elif result["bowl_dxy_m"] > MAX_BOWL_DXY_M:
                reason = "bowl_dxy %.5f m > %.5f m" % (result["bowl_dxy_m"], MAX_BOWL_DXY_M)
            elif result["hold_err_m"] > MAX_HOLD_ERR_M:
                reason = ("hold error %.5f m > %.5f m"
                          % (result["hold_err_m"], MAX_HOLD_ERR_M))
        elif result["rise_m"] < MIN_RISE_M:
            reason = "rise %.5f m < %.5f m" % (result["rise_m"], MIN_RISE_M)
        if reason is None and result["ik_err_m"] > MAX_IK_ERR_M:
            reason = "ik residual %.2e m > %.2e m" % (result["ik_err_m"], MAX_IK_ERR_M)
        if reason is not None:
            print("[ep%03d] attempt %d dx %+0.4f dy %+0.4f  rejected: %s"
                  % (made, attempts, dx, dy, reason))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        ep_dir = os.path.join(out_dir, "ep%03d" % made)
        shutil.rmtree(ep_dir, ignore_errors=True)
        write_meta(tmp_dir, made, cube_xy, bowl_xy, result)
        os.rename(tmp_dir, ep_dir)
        rows.append({"episode": made, "attempt": attempts, "dx": dx, "dy": dy,
                     "cube_xy": [float(cube_xy[0]), float(cube_xy[1])],
                     "bowl_xy": [float(bowl_xy[0]), float(bowl_xy[1])],
                     "num_frames": result["num_frames"], "rise_m": result["rise_m"],
                     "slip_m": result["slip_m"], "ik_err_m": result["ik_err_m"],
                     "fk_err_m": result["fk_err_m"],
                     "quat_min_dot": result["quat_min_dot"],
                     "state_action_min_dot": result["state_action_min_dot"]})
        if task == "place":
            rows[-1].update(bowl_dxy_m=result["bowl_dxy_m"], in_bowl=result["in_bowl"],
                            hold_err_m=result["hold_err_m"])
        print("[ep%03d] attempt %d dx %+0.4f dy %+0.4f  %3d frames  rise %.5f  slip %.6f  "
              "ik %.1e  fk %.1e  qdot %.4f"
              % (made, attempts, dx, dy, result["num_frames"], result["rise_m"],
                 result["slip_m"], result["ik_err_m"], result["fk_err_m"],
                 min(result["quat_min_dot"], result["state_action_min_dot"])))
        if task == "place":
            print("        bowl_dxy %.5f m  in_bowl %s  hold %.4f m"
                  % (result["bowl_dxy_m"], result["in_bowl"], result["hold_err_m"]))
        made += 1

    wall_s = time.perf_counter() - t_wall
    index = {"num_episodes": int(made),
             "fps": float(CONTROL_HZ),
             "control_hz": float(CONTROL_HZ),
             "task": PLACE_TASK if task == "place" else TASK,
             "camera_names": list(CAMERAS),
             "state_dim": EE_DIM,
             "action_dim": EE_DIM,
             "seed": int(args.seed),
             "mujoco_version": str(mujoco.__version__),
             "randomization": randomization_note(anchored=task == "place"),
             "generated_by": "experiments/gen_act_dataset.py",
             # The scene's authored centres, i.e. what the XML holds - NOT where a given
             # episode put them.  The drawn per-episode pair is in each ep%03d/meta.json and
             # the distribution it was drawn from is cube_box_m/bowl_box_m/min_sep_m below.
             "cube_xy": [float(CUBE_XY[0]), float(CUBE_XY[1])],
             "bowl_xy": [float(BOWL_XY[0]), float(BOWL_XY[1])],
             "cube_box_m": [float(v) for v in CUBE_BOX],
             "bowl_box_m": [float(v) for v in BOWL_BOX],
             "min_sep_m": float(MIN_SEP_M)}
    if task == "place":
        index.update(task_kind="place", table=args.table,
                     bowl_xy_m=[float(BOWL_XY[0]), float(BOWL_XY[1])],
                     in_bowl_radius_m=IN_BOWL_RADIUS_M,
                     bowl_dxy_max_m=max(r["bowl_dxy_m"] for r in rows),
                     in_bowl_all=all(r["in_bowl"] for r in rows),
                     hold_err_max_m=max(r["hold_err_m"] for r in rows))
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
        fh.write("\n")
    rises = [r["rise_m"] for r in rows]
    slips = [r["slip_m"] for r in rows]
    ik_errs = [r["ik_err_m"] for r in rows]
    summary = {"attempts": attempts, "episodes": made, "wall_s": round(wall_s, 2),
               "control_hz": CONTROL_HZ,
               "cube_box_m": [float(v) for v in CUBE_BOX],
               "bowl_box_m": [float(v) for v in BOWL_BOX],
               "min_sep_m": float(MIN_SEP_M),
               "seed": args.seed, "rendered": not args.no_render,
               "state_dim": EE_DIM, "action_dim": EE_DIM,
               "ee_site": EE_SITE, "fk_tol_m": FK_TOL,
               "cube_xy": [float(CUBE_XY[0]), float(CUBE_XY[1])],
               "bowl_xy": [float(BOWL_XY[0]), float(BOWL_XY[1])],
               "rise_min_m": min(rises), "rise_max_m": max(rises),
               "slip_max_m": max(slips), "ik_err_max_m": max(ik_errs),
               "fk_check_max_err_m": max(r["fk_err_m"] for r in rows),
               "quat_min_dot": min(r["quat_min_dot"] for r in rows),
               "state_action_min_dot": min(r["state_action_min_dot"] for r in rows),
               "per_episode": rows}
    if task == "place":
        summary.update(task_kind="place", table=args.table,
                       anchored_labels=list(ANCHORED_LABELS),
                       bowl_xy_m=[float(BOWL_XY[0]), float(BOWL_XY[1])],
                       bowl_dxy_min_m=min(r["bowl_dxy_m"] for r in rows),
                       bowl_dxy_max_m=max(r["bowl_dxy_m"] for r in rows),
                       in_bowl_all=all(r["in_bowl"] for r in rows),
                       hold_err_max_m=max(r["hold_err_m"] for r in rows))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print("\n%d/%d episodes in %.1f s, %d attempts, rise %.5f..%.5f m, slip <= %.6f m"
          % (made, args.episodes, wall_s, attempts, min(rises), max(rises), max(slips)))
    print("ik residual <= %.2e m, FK check <= %.2e m, quat min dot %.4f (state) / %.4f "
          "(state vs action)" % (max(ik_errs), summary["fk_check_max_err_m"],
                                 summary["quat_min_dot"], summary["state_action_min_dot"]))
    if task == "place":
        print("in_bowl %d/%d, bowl_dxy %.5f..%.5f m, hold_err <= %.5f m"
              % (sum(1 for r in rows if r["in_bowl"]), made, summary["bowl_dxy_min_m"],
                 summary["bowl_dxy_max_m"], summary["hold_err_max_m"]))
    print("wrote %s and %s" % (os.path.join(out_dir, "index.json"),
                              os.path.join(out_dir, "summary.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Closed-loop rollout of the trained ACT policy on scene_cube_bowl, through the 8-D EE interface.

The policy sees one 256x256 frame per camera plus an 8-D state - the pose of site tool0
(x, y, z, qw, qx, qy, qz) and the measured jaw opening g - and answers with 8 numbers of the
same shape. Nothing it writes reaches the plant directly: the action is decoded through IK
into six joint targets and the opening is mapped through GRIPPER_CONFIG, so the only thing
that lands in data.ctrl is

    a8 = [x, y, z, qw, qx, qy, qz, g]              the policy's action, base frame, metres
      -> quaternion normalised (|q| < 0.5 is a broken action), g clipped into [0, 1]
      -> R = mju_quat2Mat(q), q6 = ik.solve(pos, R, q_seed = current qpos[:6])
      -> ctrl[:6] = q6 + feedforward(q6),  ctrl[6] = g_to_ctrl(g)

Why the detour: the demos store the pose the arm actually reached and the plant is driven in
joints, so every tick of 8-D command has to be turned back into joints - the same path a real
EE-space policy would need in deployment. Two things in that pipeline are load-bearing:

  * the quaternion is only unit up to the network's arithmetic, and mju_quat2Mat does NOT
    normalise (a 3x-scaled quaternion decodes into a non-orthogonal, 7.6-off matrix), so it is
    normalised here, before it is allowed to become a rotation;
  * g is a regression output and a barely-trained model writes 1.1003 or -0.2 (measured on the
    CPU smoke checkpoint), so it is clipped before g_to_ctrl turns it into a hinge target.

An IK miss is a lost tick, not a crash: the previous tick's ctrl is held, the tick is counted,
and the rollout goes on. One bad tick out of 108 must not throw away the other 107.

One control tick = 1/CONTROL_HZ s of physics with the demo's substep count, so a tick here
means the same thing it meant in training. The state quaternion is sign-walked exactly like
the training data (gen_act_dataset.sign_walk): first tick flipped to qw >= 0, then every tick
negated as a whole while dot(q_k, q_{k-1}) < 0. q and -q are the same orientation, but the
network is handed the numbers, so an unwalked 180 deg jump would read as a different state.

Where the arm starts (--start):
  approach  the demos' own first frame: arm at the approach waypoint of the table shifted by
            the cube offset, cube at that same offset. In-distribution start (default).
  nominal   arm at the approach waypoint of the *unshifted* table, cube still offset, so the
            policy has to cancel the offset from the wrist camera alone.
  keyframe  the scene keyframe "home" (the upward fold), cube offset.

Success = rise >= CRITERIA["min_rise_m"] and final cube-centre-to-tool0 distance <=
CRITERIA["max_tool_offset_m"] and no NaN / runaway tick.

--task place swaps the table, the success criterion and the reported numbers, because "lift it"
and "put it in the bowl" are not the same question. A place trial succeeds when the cube's last
frame is in the bowl (centre closer than gen_act_dataset.IN_BOWL_RADIUS_M to the bowl centre
and resting inside gen_act_dataset.IN_BOWL_Z) and the cube is no longer in the jaws, which is
judged physically: at the last frame the cube is RELEASE_SEP_M or more from tool0 (the retreat
that follows opening). The commanded opening is reported (cmd_g_last) but takes no part in the
verdict - the policy's g head saturates at 0.62-0.80 on trials that did let go. rise is still
measured - it is how a lift is judged - but it takes no part in the place verdict: a huge rise
with the cube lying beside the bowl is a failure, and the place metrics reported are
bowl_dxy_m, in_bowl and hold_err_m (on the last tick that commanded a closed gripper, the
cube-centre to tool0 horizontal distance - the cube still between the pads when the release
starts).

--replay-demos <dir> is the offline validation mode: it needs no policy and no checkpoint.
Every episode directory under <dir> is replayed by feeding its own pose8_action rows into the
decode chain above, one per tick, with the arm reset to that episode's own cube offset (read
back from meta.json, not resampled). The trial is then judged by the same classify() the
policy path uses. It is the one real test of the decode path that does not need an 8-D
checkpoint, and it runs in the mujoco env because torch is imported inside load_policy() and
never at module scope.

A replay will not match the recording tick for tick - IK reaches the same pose through a
different joint solution, so the arm drifts a couple of centimetres from where the demo was -
which is why the bar is the acceptance numbers (rise, slip, tool offset, IK residual) and not
bit equality with the recorded qpos.

--viewer plays the finished rollouts back in an interactive MuJoCo window at the control rate
(the loop below keeps replaying until you close the window). The policy itself runs first,
headless, because one process cannot hold both the viewer's GL context and the offscreen one
the cameras need; the playback stores the full qpos of every tick, so what you see is the
trajectory the policy actually produced, cube and contacts included.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mujoco  # noqa: E402

from experiments.gen_act_dataset import (ANCHORED_LABELS, BOWL_XY, CAMERAS,  # noqa: E402
                                         CAMERA_HW, CONTROL_HZ, CUBE_XY, EE_SITE,
                                         IN_BOWL_RADIUS_M, IN_BOWL_Z, PLACE_TASK,
                                         RANDOM_X_M, RANDOM_Y_M, TASK, fk_tool, hinge_g,
                                         place_cube, shifted_table)
from src.pose_cmd.replay import CRITERIA, Replayer, g_to_ctrl, write_png  # noqa: E402
from src.pose_cmd.waypoints import load_table, plan_trajectory  # noqa: E402

SCENE = os.path.join(REPO, "assets", "jaka_zu3", "scene_cube_bowl.xml")
TABLE_PATH = os.path.join(REPO, "configs", "waypoints_cube_lift.json")
PLACE_TABLE_PATH = os.path.join(REPO, "configs", "waypoints_cube_place.json")
DEFAULT_CKPT = os.path.join(REPO, "out", "act_train_ee", "checkpoints", "last",
                            "pretrained_model")
PLACE_CKPT = os.path.join(REPO, "out", "act_train_place", "checkpoints", "last",
                          "pretrained_model")
DEFAULT_OUT = os.path.join(REPO, "out", "act_rollout")
PLACE_OUT = os.path.join(REPO, "out", "act_rollout_place")

EE_DIM = 8             # [x, y, z, qw, qx, qy, qz, g] - the layout pose8_* uses
QUAT_MIN_NORM = 0.5    # below this a quaternion is not a rotation, it is a broken action

MAX_STEPS = 200
PNG_STRIDE = 25
_WARNED_NO_AV = False   # the "mp4 skipped" notice prints once, not 20 times
GRASP_G = 0.1          # "jaws closed" tick: first command with opening g = ctrl/0.9 below this
CTRL_LIMIT = 10.0      # |ctrl| beyond this is a runaway, not an action
JNT_TOL_RAD = 1e-3     # the gripper starts exactly on its upper limit and the solver lets it
                       # sit ~7e-6 past it, so the limit test needs slack
MOVE_TOL_RAD = 0.05    # arm travel below this over the whole trial = it never moved
RELEASE_SEP_M = 0.05   # place: cube-to-tool0 distance at the last frame = the retreat that
                       # follows opening. A cube still between the pads sits ~0.013-0.02 m
                       # from tool0, one that was let go 0.11-0.32 m.


def _torch():
    """torch is imported where it is used and never at module scope: --replay-demos has to run
    in the mujoco env, and that env (mujoco 3.16, the version the demos were generated with)
    has no torch at all."""
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "[roll] this interpreter has no torch, so it cannot run the policy.\n"
            "       the policy path needs the lerobot env; --replay-demos <dir> exercises the\n"
            "       same decode chain in the mujoco env (mujoco 3.16, the demos' version).")
    return torch


def sample_offsets(rng):
    """One trial's cube offset, drawn in the order the demos were drawn (dx then dy)."""
    return (float(rng.uniform(-RANDOM_X_M, RANDOM_X_M)),
            float(rng.uniform(-RANDOM_Y_M, RANDOM_Y_M)))


def runaway(model, qpos, act):
    """True when this tick is broken: non-finite values, a runaway command, or an arm joint
    past its limit. Free-joint rows have no limit, so jnt_limited gates the check."""
    q = np.asarray(qpos[:7], dtype=float)
    a = np.asarray(act[:7], dtype=float)
    if not (np.isfinite(q).all() and np.isfinite(a).all()):
        return True
    if float(np.abs(a).max()) > CTRL_LIMIT:
        return True
    lim = np.asarray(model.jnt_limited[:7], dtype=bool)
    lo = np.asarray(model.jnt_range[:7, 0], dtype=float)
    hi = np.asarray(model.jnt_range[:7, 1], dtype=float)
    return bool(np.any(lim & ((q < lo - JNT_TOL_RAD) | (q > hi + JNT_TOL_RAD))))


def bowl_metrics(cube_end):
    """(bowl_dxy_m, in_bowl) for a cube centre - the place task's own measurement.

    in_bowl needs both terms: horizontal distance rules out the table beside the bowl, and the
    height window rules out a cube perched on the rim (wall top z = 0.840) or below the floor."""
    cube_end = np.asarray(cube_end, dtype=float)
    dxy = float(np.hypot(cube_end[0] - BOWL_XY[0], cube_end[1] - BOWL_XY[1]))
    inside_z = IN_BOWL_Z[0] < float(cube_end[2]) < IN_BOWL_Z[1]
    return dxy, bool(dxy < IN_BOWL_RADIUS_M and inside_z)


def classify_place(rec):
    """Why a place trial ended the way it did - rise is not consulted.

    Success is exactly what the task asks: at the last frame the cube is in the bowl and it is
    no longer in the jaws. "No longer in the jaws" is judged physically - the cube is
    RELEASE_SEP_M or more from tool0, which the retreat after opening produces - and never from
    the commanded opening: the policy's g head saturates at 0.62-0.80 on trials that really did
    let go, so gating on it failed 80/80 physically correct trials (out/act_train_place_20k
    step 6000, 4 seeds). cmd_g_last stays in the record as a measurement."""
    if rec["diverged"]:
        return "diverged"
    if rec["arm_travel_rad"] < MOVE_TOL_RAD:
        return "no_motion"
    if rec["grasp_tick"] is None:
        return "gripper_never_closed"
    if not rec["in_bowl"]:
        return "not_in_bowl"         # dropped beside it, or on the rim, or never got there
    if float(rec["tool_offset_m"]) < RELEASE_SEP_M:
        return "never_released"      # carried to the bowl and still held at the last frame
    return "success"


def classify(rec):
    """Why the trial ended the way it did. Order matters: a broken run is reported as such."""
    if rec["diverged"]:
        return "diverged"
    if rec["arm_travel_rad"] < MOVE_TOL_RAD:
        return "no_motion"
    if rec["grasp_tick"] is None:
        return "gripper_never_closed"
    if rec["rise_m"] < 0.02:
        if rec["rise_peak_m"] >= 0.02:
            return "slipped_out"      # it did leave the table, then the cube came back down
        return "not_lifted"           # closed on air, or the pads never got the cube
    if rec["rise_m"] < CRITERIA["min_rise_m"]:
        return "lift_insufficient"
    if rec["tool_offset_m"] > CRITERIA["max_tool_offset_m"]:
        return "lost_after_lift"
    return "success"


def read_state8(fk_data, site_id, qpos7, prev_quat):
    """The 8-D observation the policy was trained on: the tool0 pose of *this* tick's qpos
    plus the jaw opening that qpos encodes.

    The pose is read through a private MjData (gen_act_dataset.fk_tool), so the plant is never
    re-forwarded to read a state, and it is computed by the very function the dataset used.
    prev_quat carries the sign walk of the previous tick (None on the first tick, which is
    flipped to qw >= 0). Returns the 8-vector and the quaternion to hand back as prev_quat on
    the next tick."""
    pos, quat = fk_tool(fk_data, site_id, qpos7)
    quat = np.asarray(quat, dtype=np.float64)
    if prev_quat is None:
        if quat[0] < 0.0:
            quat = -quat
    elif float(quat @ prev_quat) < 0.0:
        quat = -quat
    return np.concatenate([pos, quat, [hinge_g(qpos7[6])]]), quat


def quat_to_rotmat(quat):
    """MuJoCo's own quaternion -> rotation matrix. The caller has already normalised: see
    decode_pose8 - mju_quat2Mat assumes a unit quaternion and silently returns a scaled
    matrix otherwise."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64).reshape(4))
    return mat.reshape(3, 3)


def decode_pose8(a8):
    """8-D action -> (position, rotation matrix, gripper opening, info) for the IK.

    Three repairs happen here, in this order, and nowhere else:
      * the quaternion is normalised (a bad action, |q| < QUAT_MIN_NORM, is reported instead
        of being amplified into a rotation);
      * the rotation matrix comes from MuJoCo's mju_quat2Mat, the same convention the
        recorded poses were read back with;
      * g is clipped into [0, 1] - the range GRIPPER_CONFIG maps onto the hinge.
    info records what was repaired, so the caller can count it instead of guessing."""
    a8 = np.asarray(a8, dtype=np.float64).reshape(EE_DIM)
    quat = a8[3:7]
    norm = float(np.linalg.norm(quat))
    g_raw = float(a8[7])
    g = float(np.clip(g_raw, 0.0, 1.0))
    info = {"quat_norm": norm, "g_raw": g_raw, "g": g, "clipped": bool(g != g_raw),
            "bad": not (np.isfinite(norm) and np.isfinite(g_raw)) or norm < QUAT_MIN_NORM}
    if info["bad"]:
        return np.array(a8[0:3], dtype=np.float64), None, g, info
    return np.array(a8[0:3], dtype=np.float64), quat_to_rotmat(quat / norm), g, info


def solve_action_ik(rp, pos, rot, q_seed):
    """Warm-started DLS first, then the global restarts. Returns (IKResult, used_fallback).

    Never raises: the converged flag comes back as it is, and what a miss means is the
    caller's decision (here: hold the last ctrl and count the tick). This is the tolerant
    sibling of Replayer.solve_frame, which stops the run on a miss because a waypoint it
    cannot reach means the table is wrong - an action it cannot reach only means a lost tick."""
    res = rp.ik.solve(pos, rot, q_seed)
    if res.converged:
        return res, 0
    return rp.ik.solve_robust(pos, rot, q_seed), 1


def ctrl_row(rp, q6, g):
    """The 7-vector that goes into data.ctrl: six joint targets plus the gravity feed-forward
    (src/pose_cmd/replay.py:203-205), and the gripper hinge target.

    g is clipped here as well as in decode_pose8 - this is the single place the mapping to the
    hinge lives, so a caller that hands in an out-of-range opening (a test, a held ctrl) must
    not be able to drive joint_7 past its 0..0.9 range."""
    q6 = np.asarray(q6, dtype=np.float64).reshape(6)
    ctrl = np.empty(7)
    ctrl[:6] = q6 + (rp.feedforward(q6) if rp.use_ff else 0.0)
    ctrl[6] = g_to_ctrl(float(np.clip(g, 0.0, 1.0)))
    return ctrl


def load_policy(ckpt, device, seed=0):
    """The policy, plus its pre/post processors. This is the only place torch is imported."""
    torch = _torch()
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.manual_seed(int(seed))
    policy = ACTPolicy.from_pretrained(str(ckpt))
    # A 7-D checkpoint would load fine and then fail on the reshape below, so say what is
    # wrong before the arm ever moves.
    try:
        out_dim = int(policy.config.output_features["action"].shape[0])
    except (AttributeError, KeyError, IndexError, TypeError):
        out_dim = None
    if out_dim is not None and out_dim != EE_DIM:
        raise SystemExit(
            "[roll] %s has a %d-dim action head; this rollout drives the 8-D EE interface and "
            "needs %d (x,y,z,qw,qx,qy,qz,g).\n"
            "       train/export the 8-D checkpoint into out/act_train_ee, or run the 7-D one "
            "with legacy/act_rollout_joint7.py" % (ckpt, out_dim, EE_DIM))
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def reset_arm(rp, start, base_table, dx, dy):
    """Put the arm where --start says, cube not yet placed (place_cube runs afterwards)."""
    if start == "keyframe":
        mujoco.mj_resetDataKeyframe(rp.model, rp.data, rp.key_id)
        rp.data.qvel[:] = 0.0
        mujoco.mj_forward(rp.model, rp.data)
        return
    table = shifted_table(base_table, dx, dy) if start == "approach" else base_table
    first = np.asarray(plan_trajectory(table).ee8[0], dtype=float)
    res = rp.solve_frame("approach", first[0:3], first[3:6], rp.home_qpos())
    rp.teleport(np.array(res.qpos, dtype=float), float(first[6]))


def load_episodes(root):
    """Episode directories under --replay-demos, in name order (ep000, ep001, ...)."""
    if not os.path.isdir(root):
        raise SystemExit("[roll] --replay-demos: no such directory %s" % root)
    eps = [os.path.join(root, name) for name in sorted(os.listdir(root))
           if os.path.exists(os.path.join(root, name, "data.npz"))]
    if not eps:
        raise SystemExit("[roll] --replay-demos: no episode with data.npz under %s" % root)
    return eps


def load_demo(ep_dir):
    """pose8_action/pose8_state plus the cube offset the episode was recorded with.

    The offset is read back from meta.json's cube_pos instead of being resampled: a replay
    that started at a different cube would be validating something else. It is the same
    offset gen_act_dataset matched the shifted table with, so --start approach lands on the
    recording's own first frame."""
    with np.load(os.path.join(ep_dir, "data.npz")) as npz:
        action = np.array(npz["pose8_action"], dtype=np.float64)
        state = np.array(npz["pose8_state"], dtype=np.float64)
    with open(os.path.join(ep_dir, "meta.json")) as fh:
        meta = json.load(fh)
    cube = np.asarray(meta["cube_pos"], dtype=np.float64)
    return {"pose8_action": action, "pose8_state": state,
            "dx": float(cube[0] - CUBE_XY[0]), "dy": float(cube[1] - CUBE_XY[1]),
            "dir": ep_dir}


class Video:
    """H.264 (else mpeg4) writer. A wheel without libx264 still gets a playable mp4."""

    def __init__(self, path, fps, width, height):
        import av

        self.av = av
        self.c = av.open(path, "w")
        codec = "libx264" if "libx264" in av.codec.codecs_available else "mpeg4"
        stream = self.c.add_stream(codec, rate=int(round(fps)))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        self.s = stream

    def add(self, rgb):
        frame = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.s.encode(frame):
            self.c.mux(packet)

    def close(self):
        for packet in self.s.encode():
            self.c.mux(packet)
        self.c.close()


def open_video(path, fps, width, height):
    """The mp4 writer, or None when av is not installed.

    av lives in the lerobot env only, and --replay-demos has to run in the mujoco env (mujoco
    3.16 - the version the demos were generated with) which has no encoder at all. A missing
    av therefore costs the mp4 and nothing else: the PNG keyframes come from write_png, which
    needs zlib alone."""
    global _WARNED_NO_AV

    try:
        return Video(path, fps, width, height)
    except ImportError:
        if not _WARNED_NO_AV:
            _WARNED_NO_AV = True
            print("[roll] no av in this interpreter: mp4 skipped, PNG keyframes still "
                  "written", flush=True)
        return None


def rollout(rp, renderer, cam_ids, args, trial_dir, trial, dx, dy,
            policy=None, pre=None, post=None, demo=None):
    """One trial. Returns the record; writes trial_XX.mp4 and PNG keyframes if asked.

    With demo=None the actions come from the policy (renderer must be live, since the images
    are half the observation); with demo=<load_demo()> they come from the episode's own
    pose8_action rows and no camera, no policy and no checkpoint is touched."""
    model = rp.model
    reset_arm(rp, args.start, args.base_table, dx, dy)
    place_cube(rp, dx, dy)
    cube0 = rp.cube_pos()
    q0 = np.array(rp.data.qpos[:7], dtype=float)
    torch = _torch() if policy is not None else None
    # Private kinematics scratch: the plant's data is never re-forwarded to read a state.
    fk_data = mujoco.MjData(model)
    site_id = rp.site_id

    n_steps = min(args.max_steps, len(demo["pose8_action"])) if demo is not None \
        else int(args.max_steps)
    stem = os.path.join(trial_dir, "trial_%02d" % trial)
    video = None
    if renderer is not None and not args.no_video and not args.json_only:
        video = open_video(stem + ".mp4", CONTROL_HZ, 2 * CAMERA_HW[1], CAMERA_HW[0])
    pngs = []
    grasp_cube = cube0.copy()
    prev_quat = None
    cmd_g_last = None
    hold_err = float("nan")
    # Held when a tick cannot be decoded, so even the very first tick has something to fall
    # back on: the reset pose itself, i.e. the arm keeps its feet under it.
    prev_ctrl = ctrl_row(rp, q0[:6], hinge_g(q0[6]))
    rec = {"trial": int(trial), "dx": dx, "dy": dy, "steps": 0, "diverged": False,
           "diverged_tick": None, "grasp_tick": None, "grasp_g": None,
           "cmd_g_min": None, "arm_travel_rad": 0.0, "rise_curve": [], "infer_s": 0.0,
           "mode": "replay" if demo is not None else "policy",
           "ik_s": 0.0, "ik_pos_max_m": 0.0, "ik_ang_max_deg": 0.0, "ik_restarts": 0,
           "ik_fallback": 0, "ik_fail": 0, "ik_limit": 0, "bad_actions": 0, "g_clip": 0,
           "state8_0": None, "state_dev0": None, "state_dev_max": None,
           "trace": [] if args.viewer else None}
    demo_state = demo["pose8_state"] if demo is not None else None
    try:
        for k in range(n_steps):
            q = np.array(rp.data.qpos[:7], dtype=float)
            state8, prev_quat = read_state8(fk_data, site_id, q, prev_quat)
            if k == 0:
                rec["state8_0"] = [float(v) for v in state8]
            if demo_state is not None:
                dev = float(np.abs(state8 - demo_state[k]).max())
                rec["state_dev_max"] = dev if rec["state_dev_max"] is None \
                    else max(rec["state_dev_max"], dev)
                if k == 0:
                    rec["state_dev0"] = dev
            views = None
            if renderer is not None:
                views = {}
                for cam in CAMERAS:
                    renderer.update_scene(rp.data, camera=cam_ids[cam])
                    views[cam] = renderer.render()
            if video is not None:
                video.add(np.concatenate([views[CAMERAS[0]], views[CAMERAS[1]]], axis=1))
            if renderer is not None and not args.json_only and \
                    (k % PNG_STRIDE == 0 or k == n_steps - 1):
                path = "%s_%04d.png" % (stem, k)
                write_png(path, np.concatenate([views[CAMERAS[0]], views[CAMERAS[1]]],
                                               axis=1))
                pngs.append(os.path.basename(path))

            if demo is not None:
                act8 = np.asarray(demo["pose8_action"][k], dtype=np.float64)
            else:
                obs = {"observation.state":
                       torch.from_numpy(np.asarray(state8, dtype=np.float32)).unsqueeze(0),
                       "task": [TASK]}
                for cam in CAMERAS:
                    obs["observation.images." + cam] = (
                        torch.from_numpy(np.ascontiguousarray(views[cam])).permute(2, 0, 1)
                        .float().div_(255.0).unsqueeze(0))
                t0 = time.perf_counter()
                batch = pre(obs)
                with torch.inference_mode():
                    act = post(policy.select_action(batch))
                rec["infer_s"] += time.perf_counter() - t0
                act8 = act.squeeze(0).detach().cpu().numpy().astype(np.float64).reshape(EE_DIM)

            # 8-D action -> joint ctrl. A tick that cannot be decoded holds the last ctrl.
            pos, rot, g, info = decode_pose8(act8)
            if info["clipped"]:
                rec["g_clip"] += 1
            ctrl = prev_ctrl
            if info["bad"]:
                rec["bad_actions"] += 1
            else:
                t0 = time.perf_counter()
                res, fallback = solve_action_ik(rp, pos, rot,
                                                np.array(rp.data.qpos[:6], dtype=float))
                rec["ik_s"] += time.perf_counter() - t0
                rec["ik_fallback"] += int(fallback)
                rec["ik_restarts"] += int(res.restarts)
                if not res.converged:
                    rec["ik_fail"] += 1
                elif not rp.ik.within_limits(res.qpos):
                    rec["ik_limit"] += 1
                else:
                    ctrl = ctrl_row(rp, res.qpos, g)
                    rec["ik_pos_max_m"] = max(rec["ik_pos_max_m"], float(res.pos_err))
                    rec["ik_ang_max_deg"] = max(rec["ik_ang_max_deg"],
                                                float(res.ang_err_deg))

            if np.isfinite(g):
                rec["cmd_g_min"] = g if rec["cmd_g_min"] is None \
                    else min(rec["cmd_g_min"], g)
                cmd_g_last = float(g)
                if g == 0.0:
                    # the tick-k pre-action snapshot, like gen_act_dataset's hold_err_m: the
                    # cube is supposed to be between the pads right here, and this one is the
                    # last such tick, so it is the moment the release starts
                    cube_now = rp.cube_pos()
                    hold_err = float(np.hypot(state8[0] - cube_now[0],
                                              state8[1] - cube_now[1]))
                if rec["grasp_tick"] is None and abs(g) < GRASP_G:
                    rec["grasp_tick"], rec["grasp_g"] = int(k), g
                    grasp_cube = rp.cube_pos()
            if runaway(model, rp.data.qpos, ctrl):
                rec["diverged"], rec["diverged_tick"], rec["steps"] = True, int(k), k
                break

            rp.data.ctrl[:7] = ctrl
            rp.step()
            prev_ctrl = ctrl
            q = np.array(rp.data.qpos[:7], dtype=float)
            if args.viewer:
                rec["trace"].append(np.array(rp.data.qpos, dtype=float))
            rec["arm_travel_rad"] = max(rec["arm_travel_rad"],
                                        float(np.linalg.norm(q[:6] - q0[:6])))
            rec["rise_curve"].append(float(rp.cube_pos()[2] - cube0[2]))
            rec["steps"] = k + 1
            if runaway(model, q, ctrl):
                rec["diverged"], rec["diverged_tick"] = True, int(k)
                break
    finally:
        if video is not None:
            video.close()

    cube_end = rp.cube_pos()
    tool = np.array(rp.data.site_xpos[rp.site_id], dtype=float)
    rec["pngs"] = pngs
    rec["mp4"] = os.path.basename(stem + ".mp4") if video is not None else None
    rec["cube_pos0"] = [float(v) for v in cube0]
    rec["cube_pos_end"] = [float(v) for v in cube_end]
    rec["rise_m"] = float(cube_end[2] - cube0[2])
    rec["rise_peak_m"] = float(max(rec["rise_curve"])) if rec["rise_curve"] else 0.0
    rec["slip_m"] = float(np.linalg.norm(cube_end[:2] - grasp_cube[:2]))
    rec["tool_offset_m"] = float(np.linalg.norm(cube_end - tool))
    rec["infer_ms"] = 1000.0 * rec["infer_s"] / max(1, rec["steps"])
    rec["ik_ms"] = 1000.0 * rec["ik_s"] / max(1, rec["steps"])
    if getattr(args, "task", "lift") == "place":
        rec["bowl_dxy_m"], rec["in_bowl"] = bowl_metrics(cube_end)
        rec["hold_err_m"] = None if not np.isfinite(hold_err) else float(hold_err)
        rec["cmd_g_last"] = cmd_g_last
        rec["released"] = bool(float(rec["tool_offset_m"]) >= RELEASE_SEP_M)
        rec["outcome"] = classify_place(rec)
    else:
        rec["outcome"] = classify(rec)
    rec["success"] = rec["outcome"] == "success"
    del rec["rise_curve"]
    return rec


def play_in_viewer(scene, traces, control_hz, passes=0):
    """Replay recorded rollouts in an interactive MuJoCo window, one after another.

    passes=0 keeps looping until the window is closed. Each frame writes the whole qpos (arm
    and cube), so this is playback, not a second simulation: nothing here can diverge."""
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(os.path.abspath(scene))
    data = mujoco.MjData(model)
    period = 1.0 / control_hz
    loop = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for i, trace in enumerate(traces):
                next_tick = time.monotonic()
                for q in trace:
                    if not viewer.is_running():
                        break
                    data.qpos[:] = q
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    next_tick += period
                    slack = next_tick - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                print("[viewer] loop %d trial %02d: %d frames (%d Hz)"
                      % (loop, i, len(trace), control_hz), flush=True)
                if not viewer.is_running():
                    break
            loop += 1
            if passes and loop >= passes:
                break
    return loop


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=None,
                    help="8-D ACT checkpoint (ignored by --replay-demos); default %s, or %s "
                         "with --task place" % (DEFAULT_CKPT, PLACE_CKPT))
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--task", default="lift", choices=("lift", "place"),
                    help="which task to run and judge: lift (default) or place - place picks "
                         "configs/waypoints_cube_place.json, its own checkpoint default and "
                         "judges the cube in the bowl at the last frame, gripper released")
    ap.add_argument("--trials", type=int, default=None,
                    help="how many trials (default: 10 with a policy, every episode found "
                         "with --replay-demos)")
    ap.add_argument("--out", default=None,
                    help="output directory; default %s, or %s with --task place"
                         % (DEFAULT_OUT, PLACE_OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="override policy.config.n_action_steps for this rollout only (no "
                         "retraining): the policy re-plans after this many ticks instead of "
                         "playing a whole chunk open-loop. 1..chunk_size")
    ap.add_argument("--start", default="approach", choices=("approach", "nominal", "keyframe"))
    ap.add_argument("--replay-demos", default=None, metavar="DIR",
                    help="offline validation: replay every episode under DIR by feeding its "
                         "pose8_action rows through the 8-D decode chain; no policy, no "
                         "checkpoint, runs in the mujoco env")
    ap.add_argument("--json-only", action="store_true",
                    help="no mp4 and no PNG; print the summary as JSON (summary.json is written)")
    ap.add_argument("--no-video", action="store_true", help="no mp4, keep PNG keyframes")
    ap.add_argument("--viewer", action="store_true",
                    help="after the run, replay it in an interactive MuJoCo window "
                         "(the policy runs headless first; close the window to stop)")
    ap.add_argument("--viewer-passes", type=int, default=0,
                    help="how many replay loops before closing; 0 (default) = until you close it")
    args = ap.parse_args(argv)

    out_arg = args.out if args.out is not None \
        else (PLACE_OUT if args.task == "place" else DEFAULT_OUT)
    out_dir = out_arg if os.path.isabs(out_arg) else os.path.join(REPO, out_arg)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_arg = args.ckpt if args.ckpt is not None \
        else (PLACE_CKPT if args.task == "place" else DEFAULT_CKPT)
    ckpt = ckpt_arg if os.path.isabs(ckpt_arg) else os.path.join(REPO, ckpt_arg)
    root = None
    replay = None
    if args.replay_demos is not None:
        root = args.replay_demos if os.path.isabs(args.replay_demos) \
            else os.path.join(REPO, args.replay_demos)
        replay = [load_demo(ep) for ep in load_episodes(root)]
        trials = len(replay) if args.trials is None else min(int(args.trials), len(replay))
    else:
        trials = 10 if args.trials is None else int(args.trials)

    policy = pre = post = None
    if replay is None:
        if not os.path.isdir(ckpt):
            raise SystemExit(
                "[roll] no checkpoint at %s\n"
                "       the 8-D policy has not been trained yet: --replay-demos <dir>\n"
                "       exercises the same decode chain without one, and the retired 7-D\n"
                "       checkpoint is still served by legacy/act_rollout_joint7.py" % ckpt)
        policy, pre, post = load_policy(ckpt, args.device, args.seed)
        if args.n_action_steps is not None:
            n = int(args.n_action_steps)
            if not 1 <= n <= int(policy.config.chunk_size):
                raise SystemExit("[roll] --n-action-steps %d is outside 1..chunk_size %d"
                                 % (n, policy.config.chunk_size))
            policy.config.n_action_steps = n
            policy.reset()
        print("[roll] ckpt=%s device=%s start=%s chunk=%d n_action=%d"
              % (ckpt, args.device, args.start, policy.config.chunk_size,
                 policy.config.n_action_steps), flush=True)
    else:
        print("[roll] replay-demos=%s episodes=%d start=%s (no policy, no checkpoint)"
              % (root, len(replay), args.start), flush=True)

    args.base_table = load_table(PLACE_TABLE_PATH if args.task == "place" else TABLE_PATH)
    rp = Replayer(scene_path=SCENE, control_hz=CONTROL_HZ, render_every=0)
    tool_site = mujoco.mj_id2name(rp.model, mujoco.mjtObj.mjOBJ_SITE, rp.site_id)
    if tool_site != EE_SITE:
        raise SystemExit("[roll] the replayer drives site %r but the demos were recorded on %r"
                         % (tool_site, EE_SITE))
    renderer = None
    cam_ids = None
    if replay is None or not args.json_only:
        renderer = mujoco.Renderer(rp.model, height=CAMERA_HW[0], width=CAMERA_HW[1])
        cam_ids = {name: rp.model.camera(name).id for name in CAMERAS}
    print("[roll] substeps/tick=%d timestep=%.5f -> %.1f Hz" %
          (rp.substeps, rp.model.opt.timestep, 1.0 / (rp.substeps * rp.model.opt.timestep)),
          flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    traces = []
    t_all = time.perf_counter()
    try:
        for i in range(trials):
            demo = replay[i] if replay is not None else None
            dx, dy = (demo["dx"], demo["dy"]) if demo is not None else sample_offsets(rng)
            if policy is not None:
                policy.reset()
            t0 = time.perf_counter()
            rec = rollout(rp, renderer, cam_ids, args, out_dir, i, dx, dy,
                          policy=policy, pre=pre, post=post, demo=demo)
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            if args.viewer and rec["trace"]:
                traces.append(rec.pop("trace"))
            else:
                rec.pop("trace", None)
            rows.append(rec)
            print("[trial %02d] dx %+0.4f dy %+0.4f  steps %3d  rise %+0.4f  peak %+0.4f  "
                  "slip %.4f  tool %.4f  ik %.2e m %.2f deg  fb %d fail %d clip %d  "
                  "grasp %s  %s  %s"
                  % (i, dx, dy, rec["steps"], rec["rise_m"], rec["rise_peak_m"], rec["slip_m"],
                     rec["tool_offset_m"], rec["ik_pos_max_m"], rec["ik_ang_max_deg"],
                     rec["ik_fallback"], rec["ik_fail"], rec["g_clip"], rec["grasp_tick"],
                     rec["outcome"], "" if rec["mp4"] is None else rec["mp4"]), flush=True)
            if args.task == "place":
                print("            bowl_dxy %.5f m  in_bowl %s  released %s  hold %s  "
                      "last g %s"
                      % (rec["bowl_dxy_m"], rec["in_bowl"], rec["released"],
                         "n/a" if rec["hold_err_m"] is None else "%.5f m" % rec["hold_err_m"],
                         rec["cmd_g_last"]), flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        rp.close()

    ok = sum(1 for r in rows if r["success"])
    total = time.perf_counter() - t_all
    summary = {
        "ckpt": ckpt if replay is None else None, "device": args.device, "start": args.start,
        "seed": args.seed,
        "replay_demos": root if replay is not None else None,
        "mode": "replay" if replay is not None else "policy",
        "ee_layout": ["x", "y", "z", "qw", "qx", "qy", "qz", "g"],
        "ee_site": EE_SITE,
        "control_hz": CONTROL_HZ, "substeps_per_tick": int(rp.substeps),
        "max_steps": args.max_steps,
        "task": PLACE_TASK if args.task == "place" else TASK,
        "criteria": {"min_rise_m": CRITERIA["min_rise_m"],
                     "max_tool_offset_m": CRITERIA["max_tool_offset_m"],
                     "max_slip_m": CRITERIA["max_slip_m"],
                     "max_ik_err_m": 2.0e-3,
                     "grasp_g_below": GRASP_G, "ctrl_limit": CTRL_LIMIT},
        "randomization_m": {"x": RANDOM_X_M, "y": RANDOM_Y_M, "cube_xy0": list(CUBE_XY)},
        "trials": len(rows), "successes": int(ok),
        "success_rate": round(ok / len(rows), 4) if rows else 0.0,
        "outcomes": {k: sum(1 for r in rows if r["outcome"] == k)
                     for k in sorted({r["outcome"] for r in rows})},
        "rise_min_m": round(min((r["rise_m"] for r in rows), default=0.0), 6),
        "rise_max_m": round(max((r["rise_m"] for r in rows), default=0.0), 6),
        "slip_max_m": round(max((r["slip_m"] for r in rows), default=0.0), 6),
        "tool_offset_max_m": round(max((r["tool_offset_m"] for r in rows), default=0.0), 6),
        "ik_pos_err_max_m": max((r["ik_pos_max_m"] for r in rows), default=0.0),
        "ik_ang_err_max_deg": max((r["ik_ang_max_deg"] for r in rows), default=0.0),
        "ik_fallback_ticks": sum(r["ik_fallback"] for r in rows),
        "ik_fail_ticks": sum(r["ik_fail"] for r in rows),
        "ik_limit_ticks": sum(r["ik_limit"] for r in rows),
        "ik_restarts": sum(r["ik_restarts"] for r in rows),
        "bad_action_ticks": sum(r["bad_actions"] for r in rows),
        "g_clip_ticks": sum(r["g_clip"] for r in rows),
        "state_dev0_max": max((r["state_dev0"] for r in rows
                               if r["state_dev0"] is not None), default=None),
        "state_dev_max": max((r["state_dev_max"] for r in rows
                              if r["state_dev_max"] is not None), default=None),
        "infer_ms_mean": round(float(np.mean([r["infer_ms"] for r in rows])), 1) if rows else 0.0,
        "ik_ms_mean": round(float(np.mean([r["ik_ms"] for r in rows])), 1) if rows else 0.0,
        "wall_s_total": round(total, 1), "per_trial": rows,
    }
    if args.task == "place":
        # the place verdict and its own criteria; rise/slip/tool_offset stay in the summary as
        # measurements but the place criteria list does not mention them
        summary["task_kind"] = "place"
        summary["table"] = os.path.relpath(PLACE_TABLE_PATH, REPO)
        summary["anchored_labels"] = list(ANCHORED_LABELS)
        summary["bowl_xy_m"] = [float(BOWL_XY[0]), float(BOWL_XY[1])]
        summary["criteria"] = {"release_sep_m": RELEASE_SEP_M,
                               "in_bowl_radius_m": IN_BOWL_RADIUS_M,
                               "in_bowl_z": [float(IN_BOWL_Z[0]), float(IN_BOWL_Z[1])],
                               "max_ik_err_m": 2.0e-3,
                               "grasp_g_below": GRASP_G, "ctrl_limit": CTRL_LIMIT}
        summary["bowl_dxy_min_m"] = round(min((r["bowl_dxy_m"] for r in rows),
                                              default=0.0), 6)
        summary["bowl_dxy_max_m"] = round(max((r["bowl_dxy_m"] for r in rows),
                                              default=0.0), 6)
        summary["in_bowl_count"] = sum(1 for r in rows if r["in_bowl"])
        summary["in_bowl_all"] = bool(rows) and summary["in_bowl_count"] == len(rows)
        summary["hold_err_max_m"] = max((r["hold_err_m"] for r in rows
                                         if r["hold_err_m"] is not None), default=None)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    if args.json_only:
        print(json.dumps({k: v for k, v in summary.items() if k != "per_trial"}))
    else:
        print("\n%d/%d success (%s)  %s  ik %.2e m / %.2f deg  %.1f ms/tick  %.1f s total"
              "  -> %s"
              % (ok, len(rows), summary["outcomes"], args.start, summary["ik_pos_err_max_m"],
                 summary["ik_ang_err_max_deg"], summary["infer_ms_mean"], total,
                 os.path.join(out_dir, "summary.json")))
        if args.task == "place":
            print("place: in_bowl %d/%d, bowl_dxy %.5f..%.5f m, hold_err <= %s m"
                  % (summary["in_bowl_count"], len(rows), summary["bowl_dxy_min_m"],
                     summary["bowl_dxy_max_m"],
                     "n/a" if summary["hold_err_max_m"] is None
                     else "%.5f" % summary["hold_err_max_m"]))
    if args.viewer:
        if not traces:
            print("[viewer] nothing to replay: no trial produced frames", flush=True)
        else:
            print("[viewer] opening the MuJoCo window: %d trial(s), %d frames total; "
                  "close the window to stop" % (len(traces), sum(len(t) for t in traces)),
                  flush=True)
            sys.stdout.flush()
            loops = play_in_viewer(SCENE, traces, CONTROL_HZ, args.viewer_passes)
            print("[viewer] window closed after %d loop(s)" % loops, flush=True)
            sys.stdout.flush()
            os._exit(0)  # GL teardown crashes on this driver once the viewer is gone
    return 0


if __name__ == "__main__":
    sys.exit(main())

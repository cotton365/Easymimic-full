#!/usr/bin/env python3
"""Generate ACT training episodes for the pick-and-lift task on scene_cube_bowl.

One episode = the 8-D waypoint table configs/waypoints_cube_lift.json replayed at
CONTROL_HZ against a cube whose xy has been shifted by one random offset; every waypoint
is shifted by the same offset, so the grasp column follows the cube.  The cube keeps its
spawn height (0.815 m = table top 0.80 + half height 0.015) and its yaw.

Per control tick an episode stores the 7 measured joint positions, the 7 actuator targets
that were actually sent, and one 256x256 frame from each camera:

    qpos (N,7)  [0:6] arm joints, [6] gripper hinge (joint_7)
    act  (N,7)  data.ctrl[:7] after apply(); the six arm targets already carry the
                gravity feed-forward (src/pose_cmd/replay.py:apply), so act - qpos is a
                real servo offset, not a modelling error
    t    (N,)   seconds, t[k] = k / CONTROL_HZ

Frames are rendered *before* the tick's action is applied, so frame k shows exactly
qpos[k].  write_png is the hand-rolled writer from src/pose_cmd/replay.py (no PIL in this
env); row 0 is the top of the picture.

An attempt is thrown away and resampled when the lifted cube ends lower than MIN_RISE_M,
when any IK residual exceeds MAX_IK_ERR_M, or when an IK solve fails outright - so the
output directory only ever holds successful grasps.  Same --seed -> same offsets -> the
same physics -> bit-identical qpos (nothing here reads the clock or a thread pool).

The output directory is rebuilt from scratch on every run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mujoco  # noqa: E402

from src.pose_cmd.replay import ReplayError, Replayer, write_png  # noqa: E402
from src.pose_cmd.waypoints import Table, Waypoint, load_table, plan_trajectory  # noqa: E402

SCENE = "assets/jaka_zu3/scene_cube_bowl.xml"
TABLE_PATH = "configs/waypoints_cube_lift.json"
DEFAULT_OUT = "out/act_demos"

CONTROL_HZ = 20.0            # 108 ticks for the 5.35 s table; rise unchanged vs 50 Hz
CAMERAS = ("front", "wrist")
CAMERA_HW = (256, 256)

CUBE_XY = (0.474, 0.0435)    # cube body xy in the scene = the table's grasp column
CUBE_Z = 0.815               # table top 0.80 + half height 0.015

# Measured (scratch/sweep_xy.py): every dx in +-0.030 and dy in +-0.006 lifts the cube,
# rise 0.1183..0.1193 m, slip <= 6.0e-4 m.  The published range keeps a margin.
RANDOM_X_M = 0.020
RANDOM_Y_M = 0.005

MIN_RISE_M = 0.080           # same bar as src/pose_cmd/replay.py CRITERIA['min_rise_m']
MAX_IK_ERR_M = 2.0e-3        # the 50 Hz baseline's worst IK residual is 9.2e-4 m
EPISODES = 20
SEED = 0
TASK = "pick up the red block and lift it"


def shifted_table(base, dx, dy):
    """The template table with every waypoint moved by (dx, dy); orientation untouched."""
    waypoints = []
    for w in base.waypoints:
        ee8 = np.array(w.ee8, dtype=float)
        ee8[0] += dx
        ee8[1] += dy
        waypoints.append(Waypoint(label=w.label, ee8=ee8, duration_s=w.duration_s))
    return Table(source=base.source, frame=base.frame, units=base.units,
                 control_hz=CONTROL_HZ, waypoints=tuple(waypoints))


def place_cube(rp, dx, dy):
    """Replayer.teleport() rewrites the free joint from the home keyframe, so the cube is
    put back afterwards - once, before the first tick."""
    rp.data.qpos[7:10] = [CUBE_XY[0] + dx, CUBE_XY[1] + dy, CUBE_Z]
    rp.data.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]
    rp.data.qvel[:] = 0.0
    mujoco.mj_forward(rp.model, rp.data)


def run_attempt(base, dx, dy, ep_dir, render):
    """One episode. Returns the measurement dict; writes data.npz and meta.json into
    ep_dir (the caller renames it into place only if the attempt is accepted)."""
    traj = plan_trajectory(shifted_table(base, dx, dy))
    rp = Replayer(scene_path=os.path.join(REPO, SCENE), control_hz=CONTROL_HZ,
                  render_every=0)
    renderer = None
    try:
        model = rp.model
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
        place_cube(rp, dx, dy)
        cube_start = rp.cube_pos()
        cube_quat = np.array(rp.data.xquat[rp.cube_body], dtype=float)

        qpos, act, stamps = [], [], []
        cube_grasp = None
        ik_err = 0.0
        for k in range(len(traj)):
            if renderer is not None:
                for name, cam_id in cams:
                    renderer.update_scene(rp.data, camera=cam_id)
                    write_png(os.path.join(ep_dir, name, "%06d.png" % k), renderer.render())
            ee8 = np.asarray(traj.ee8[k], dtype=float)
            qpos.append(rp.data.qpos[:7].copy())
            res = rp.solve_frame(traj.label[k], ee8[0:3], ee8[3:6],
                                 rp.data.qpos[:6].copy())
            ik_err = max(ik_err, float(res.pos_err))
            rp.apply(np.array(res.qpos, dtype=float), float(ee8[6]))
            act.append(rp.data.ctrl[:7].copy())
            if cube_grasp is None and float(ee8[6]) <= 0.0:
                cube_grasp = rp.cube_pos()
            stamps.append(k / CONTROL_HZ)
            rp.step()

        cube_end = rp.cube_pos()
        rise = float(cube_end[2] - cube_start[2])
        # Slip is drift after the jaws closed; the table always closes them, but a table
        # edit that never asks for a closed gripper must not divide by nothing here.
        slip = 0.0 if cube_grasp is None else float(np.linalg.norm(cube_end[:2] - cube_grasp[:2]))
        n = len(qpos)

        np.savez(os.path.join(ep_dir, "data.npz"),
                 qpos=np.stack(qpos).astype(np.float64),
                 act=np.stack(act).astype(np.float64),
                 t=np.array(stamps, dtype=np.float64),
                 cube_pos=np.asarray(cube_start, dtype=np.float64),
                 cube_quat=cube_quat.astype(np.float64),
                 rise_m=np.float64(rise),
                 slip_m=np.float64(slip))
        return {"num_frames": n, "rise_m": rise, "slip_m": slip, "ik_err_m": ik_err,
                "cube_pos": [float(CUBE_XY[0] + dx), float(CUBE_XY[1] + dy), float(CUBE_Z)]}
    finally:
        if renderer is not None:
            renderer.close()
        rp.close()


def write_meta(ep_dir, episode, dx, dy, result):
    meta = {"episode": int(episode),
            "fps": float(CONTROL_HZ),
            "num_frames": int(result["num_frames"]),
            "camera_hw": [int(CAMERA_HW[0]), int(CAMERA_HW[1])],
            "cube_pos": [float(v) for v in result["cube_pos"]],
            "cube_yaw": 0.0,
            "rise_m": float(result["rise_m"]),
            "slip_m": float(result["slip_m"]),
            "grasp_ok": True,
            "scene": SCENE,
            "control_hz": float(CONTROL_HZ)}
    with open(os.path.join(ep_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return meta


def randomization_note():
    return "cube xy in [-%.3f, %.3f] x [-%.3f, %.3f] m, no yaw" % (
        RANDOM_X_M, RANDOM_X_M, RANDOM_Y_M, RANDOM_Y_M)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED,
                    help="offset stream seed (same seed -> identical qpos)")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (rebuilt)")
    ap.add_argument("--no-render", action="store_true",
                    help="skip PNGs (contract-incomplete; only for reproducibility checks)")
    ap.add_argument("--max-attempts", type=int, default=200)
    args = ap.parse_args(argv)

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    base = load_table(os.path.join(REPO, TABLE_PATH))
    if base.control_hz != 50.0:
        raise SystemExit("%s: expected the 50 Hz template table, got %g"
                         % (TABLE_PATH, base.control_hz))
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
        dx = float(rng.uniform(-RANDOM_X_M, RANDOM_X_M))
        dy = float(rng.uniform(-RANDOM_Y_M, RANDOM_Y_M))
        attempts += 1
        tmp_dir = os.path.join(out_dir, "_attempt")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            result = run_attempt(base, dx, dy, tmp_dir, render=not args.no_render)
        except ReplayError as exc:
            print("[ep%03d] attempt %d dx %+0.4f dy %+0.4f  rejected: %s"
                  % (made, attempts, dx, dy, exc))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue
        reason = None
        if result["rise_m"] < MIN_RISE_M:
            reason = "rise %.5f m < %.5f m" % (result["rise_m"], MIN_RISE_M)
        elif result["ik_err_m"] > MAX_IK_ERR_M:
            reason = "ik residual %.2e m > %.2e m" % (result["ik_err_m"], MAX_IK_ERR_M)
        if reason is not None:
            print("[ep%03d] attempt %d dx %+0.4f dy %+0.4f  rejected: %s"
                  % (made, attempts, dx, dy, reason))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        ep_dir = os.path.join(out_dir, "ep%03d" % made)
        shutil.rmtree(ep_dir, ignore_errors=True)
        write_meta(tmp_dir, made, dx, dy, result)
        os.rename(tmp_dir, ep_dir)
        rows.append({"episode": made, "attempt": attempts, "dx": dx, "dy": dy,
                     "num_frames": result["num_frames"], "rise_m": result["rise_m"],
                     "slip_m": result["slip_m"], "ik_err_m": result["ik_err_m"]})
        print("[ep%03d] attempt %d dx %+0.4f dy %+0.4f  %3d frames  rise %.5f  slip %.6f  ik %.1e"
              % (made, attempts, dx, dy, result["num_frames"], result["rise_m"],
                 result["slip_m"], result["ik_err_m"]))
        made += 1

    wall_s = time.perf_counter() - t_wall
    index = {"num_episodes": int(made),
             "fps": float(CONTROL_HZ),
             "control_hz": float(CONTROL_HZ),
             "task": TASK,
             "camera_names": list(CAMERAS),
             "state_dim": 7,
             "action_dim": 7,
             "seed": int(args.seed),
             "mujoco_version": str(mujoco.__version__),
             "randomization": randomization_note(),
             "generated_by": "experiments/gen_act_dataset.py"}
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
        fh.write("\n")
    rises = [r["rise_m"] for r in rows]
    slips = [r["slip_m"] for r in rows]
    summary = {"attempts": attempts, "episodes": made, "wall_s": round(wall_s, 2),
               "control_hz": CONTROL_HZ, "random_x_m": RANDOM_X_M,
               "random_y_m": RANDOM_Y_M, "seed": args.seed, "rendered": not args.no_render,
               "rise_min_m": min(rises), "rise_max_m": max(rises),
               "slip_max_m": max(slips), "per_episode": rows}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print("\n%d/%d episodes in %.1f s, %d attempts, rise %.5f..%.5f m, slip <= %.6f m"
          % (made, args.episodes, wall_s, attempts, min(rises), max(rises), max(slips)))
    print("wrote %s and %s" % (os.path.join(out_dir, "index.json"),
                              os.path.join(out_dir, "summary.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

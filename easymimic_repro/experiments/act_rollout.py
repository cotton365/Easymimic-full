#!/usr/bin/env python3
"""Closed-loop rollout of the trained ACT policy on scene_cube_bowl.

The policy sees one 256x256 frame per camera plus data.qpos[:7]; the only thing that reaches
the plant is its action, written straight into data.ctrl[:7]. The demo actions already carry
the gravity feed-forward (src/pose_cmd/replay.py:apply), so nothing is added on top of them.

One control tick = 1/CONTROL_HZ s of physics with the demo's substep count, so a tick here
means the same thing it meant in training.

Where the arm starts (--start):
  approach  the demos' own first frame: arm at the approach waypoint of the table shifted by
            the cube offset, cube at that same offset. In-distribution start (default).
  nominal   arm at the approach waypoint of the *unshifted* table, cube still offset, so the
            policy has to cancel the offset from the wrist camera alone.
  keyframe  the scene keyframe "home" (the upward fold), cube offset.

Success = rise >= CRITERIA["min_rise_m"] and final cube-centre-to-tool0 distance <=
CRITERIA["max_tool_offset_m"] and no NaN / runaway tick.

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
import torch  # noqa: E402

from experiments.gen_act_dataset import (CAMERAS, CAMERA_HW, CONTROL_HZ, CUBE_XY,  # noqa: E402
                                         RANDOM_X_M, RANDOM_Y_M, TASK, place_cube,
                                         shifted_table)
from src.pose_cmd.replay import CRITERIA, Replayer, write_png  # noqa: E402
from src.pose_cmd.waypoints import load_table, plan_trajectory  # noqa: E402

SCENE = os.path.join(REPO, "assets", "jaka_zu3", "scene_cube_bowl.xml")
TABLE_PATH = os.path.join(REPO, "configs", "waypoints_cube_lift.json")
DEFAULT_CKPT = os.path.join(REPO, "out", "act_train", "checkpoints", "last", "pretrained_model")
DEFAULT_OUT = os.path.join(REPO, "out", "act_rollout")

MAX_STEPS = 200
PNG_STRIDE = 25
GRASP_G = 0.1          # "jaws closed" tick: first command with opening g = ctrl/0.9 below this
CTRL_LIMIT = 10.0      # |ctrl| beyond this is a runaway, not an action
JNT_TOL_RAD = 1e-3     # the gripper starts exactly on its upper limit and the solver lets it
                       # sit ~7e-6 past it, so the limit test needs slack
MOVE_TOL_RAD = 0.05    # arm travel below this over the whole trial = it never moved


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


def load_policy(ckpt, device):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(ckpt))
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


def rollout(policy, pre, post, rp, renderer, cam_ids, args, trial_dir, trial, dx, dy):
    """One trial. Returns the record; writes trial_XX.mp4 and PNG keyframes if asked."""
    model = rp.model
    reset_arm(rp, args.start, args.base_table, dx, dy)
    place_cube(rp, dx, dy)
    cube0 = rp.cube_pos()
    q0 = np.array(rp.data.qpos[:7], dtype=float)

    stem = os.path.join(trial_dir, "trial_%02d" % trial)
    video = None
    if not args.no_video and not args.json_only:
        video = Video(stem + ".mp4", CONTROL_HZ, 2 * CAMERA_HW[1], CAMERA_HW[0])
    pngs = []
    grasp_cube = cube0.copy()
    rec = {"trial": int(trial), "dx": dx, "dy": dy, "steps": 0, "diverged": False,
           "diverged_tick": None, "grasp_tick": None, "grasp_g": None,
           "cmd_g_min": None, "arm_travel_rad": 0.0, "rise_curve": [], "infer_s": 0.0,
           "trace": [] if args.viewer else None}
    try:
        for k in range(args.max_steps):
            obs = {"observation.state":
                   torch.from_numpy(np.asarray(rp.data.qpos[:7], dtype=np.float32)).unsqueeze(0),
                   "task": [TASK]}
            views = {}
            for cam in CAMERAS:
                renderer.update_scene(rp.data, camera=cam_ids[cam])
                rgb = renderer.render()
                views[cam] = rgb
                obs["observation.images." + cam] = (
                    torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
                    .float().div_(255.0).unsqueeze(0))
            if video is not None:
                video.add(np.concatenate([views[CAMERAS[0]], views[CAMERAS[1]]], axis=1))
            if not args.json_only and (k % PNG_STRIDE == 0 or k == args.max_steps - 1):
                path = "%s_%04d.png" % (stem, k)
                write_png(path, np.concatenate([views[CAMERAS[0]], views[CAMERAS[1]]], axis=1))
                pngs.append(os.path.basename(path))

            t0 = time.perf_counter()
            batch = pre(obs)
            with torch.inference_mode():
                act = post(policy.select_action(batch))
            rec["infer_s"] += time.perf_counter() - t0
            act = act.squeeze(0).detach().cpu().numpy().astype(np.float64)

            g = float(act[6]) / 0.9
            rec["cmd_g_min"] = g if rec["cmd_g_min"] is None else min(rec["cmd_g_min"], g)
            if rec["grasp_tick"] is None and abs(g) < GRASP_G:
                rec["grasp_tick"], rec["grasp_g"] = int(k), g
                grasp_cube = rp.cube_pos()
            if runaway(model, rp.data.qpos, act):
                rec["diverged"], rec["diverged_tick"], rec["steps"] = True, int(k), k
                break

            rp.data.ctrl[:7] = act
            rp.step()
            q = np.array(rp.data.qpos[:7], dtype=float)
            if args.viewer:
                rec["trace"].append(np.array(rp.data.qpos, dtype=float))
            rec["arm_travel_rad"] = max(rec["arm_travel_rad"],
                                        float(np.linalg.norm(q[:6] - q0[:6])))
            rec["rise_curve"].append(float(rp.cube_pos()[2] - cube0[2]))
            rec["steps"] = k + 1
            if runaway(model, q, act):
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
    rec["outcome"] = classify(rec)
    rec["success"] = rec["outcome"] == "success"
    del rec["rise_curve"]
    return rec


def play_in_viewer(scene, traces, control_hz, passes=0):
    """Replay recorded rollouts in an interactive MuJoCo window, one after another.

    passes=0 keeps looping until the window is closed. Each frame writes the whole qpos (arm
    and cube), so this is playback, not a second simulation: nothing here can diverge.
    """
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
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--start", default="approach", choices=("approach", "nominal", "keyframe"))
    ap.add_argument("--json-only", action="store_true",
                    help="no mp4 and no PNG; print the summary as JSON (summary.json is written)")
    ap.add_argument("--no-video", action="store_true", help="no mp4, keep PNG keyframes")
    ap.add_argument("--viewer", action="store_true",
                    help="after the run, replay it in an interactive MuJoCo window "
                         "(policy runs headless first; close the window to stop)")
    ap.add_argument("--viewer-passes", type=int, default=0,
                    help="how many replay loops before closing; 0 (default) = until you close it")
    args = ap.parse_args(argv)

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(REPO, args.ckpt)
    torch.manual_seed(args.seed)

    policy, pre, post = load_policy(ckpt, args.device)
    print("[roll] ckpt=%s device=%s start=%s chunk=%d n_action=%d"
          % (ckpt, args.device, args.start, policy.config.chunk_size,
             policy.config.n_action_steps), flush=True)

    args.base_table = load_table(TABLE_PATH)
    rp = Replayer(scene_path=SCENE, control_hz=CONTROL_HZ, render_every=0)
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
        for i in range(args.trials):
            dx, dy = sample_offsets(rng)
            policy.reset()
            t0 = time.perf_counter()
            rec = rollout(policy, pre, post, rp, renderer, cam_ids, args, out_dir, i, dx, dy)
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            if args.viewer and rec["trace"]:
                traces.append(rec.pop("trace"))
            else:
                rec.pop("trace", None)
            rows.append(rec)
            print("[trial %02d] dx %+0.4f dy %+0.4f  steps %3d  rise %+0.4f  peak %+0.4f  "
                  "slip %.4f  tool %.4f  grasp %s  infer %.0f ms/tick  %s  %s"
                  % (i, dx, dy, rec["steps"], rec["rise_m"], rec["rise_peak_m"], rec["slip_m"],
                     rec["tool_offset_m"], rec["grasp_tick"], rec["infer_ms"],
                     rec["outcome"], "" if rec["mp4"] is None else rec["mp4"]), flush=True)
    finally:
        renderer.close()
        rp.close()

    ok = sum(1 for r in rows if r["success"])
    total = time.perf_counter() - t_all
    summary = {
        "ckpt": ckpt, "device": args.device, "start": args.start, "seed": args.seed,
        "control_hz": CONTROL_HZ, "substeps_per_tick": int(rp.substeps),
        "max_steps": args.max_steps, "task": TASK,
        "criteria": {"min_rise_m": CRITERIA["min_rise_m"],
                     "max_tool_offset_m": CRITERIA["max_tool_offset_m"],
                     "grasp_g_below": GRASP_G, "ctrl_limit": CTRL_LIMIT},
        "randomization_m": {"x": RANDOM_X_M, "y": RANDOM_Y_M, "cube_xy0": list(CUBE_XY)},
        "trials": len(rows), "successes": int(ok),
        "success_rate": round(ok / len(rows), 4) if rows else 0.0,
        "outcomes": {k: sum(1 for r in rows if r["outcome"] == k)
                     for k in sorted({r["outcome"] for r in rows})},
        "infer_ms_mean": round(float(np.mean([r["infer_ms"] for r in rows])), 1) if rows else 0.0,
        "wall_s_total": round(total, 1), "per_trial": rows,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    if args.json_only:
        print(json.dumps({k: v for k, v in summary.items() if k != "per_trial"}))
    else:
        print("\n%d/%d success (%s)  %s  %.1f ms/tick  %.1f s total  -> %s"
              % (ok, len(rows), summary["outcomes"], args.start, summary["infer_ms_mean"],
                 total, os.path.join(out_dir, "summary.json")))
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

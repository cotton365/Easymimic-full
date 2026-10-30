"""Replay an 8-D waypoint table on the JAKA Zu3 scene.

    python experiments/pose_replay.py                  # configs/waypoints_cube_lift.json
    python experiments/pose_replay.py --table other.json
    python experiments/pose_replay.py --selftest       # schema + IK counterexamples, no replay

Exit codes: 0 every check passed, 1 a criterion failed, 2 the table was rejected,
3 the table could not be executed (IK miss or a solution on a joint limit).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.ee_retarget.ik_jaka_zu3 import rotvec_to_rotmat  # noqa: E402
from src.pose_cmd.replay import (CRITERIA, DEFAULT_FRAME_DIR, DEFAULT_TABLE, FAILS,
                                 RENDER_EVERY, ReplayError, Replayer, check, droop_probe,
                                 g_to_ctrl, replay_table)  # noqa: E402
from src.pose_cmd.waypoints import (WaypointError, interp_rotvec, load_table,  # noqa: E402
                                    parse_table, plan_trajectory)


def rel(path):
    return os.path.relpath(path, REPO_ROOT)


def table_of(*waypoints, hz=50.0):
    return parse_table({"meta": {"frame": "base", "units": "m", "control_hz": hz},
                        "waypoints": list(waypoints)}, source="selftest table")


def wp(label, pos, rotvec, g, duration_s):
    return {"label": label, "pos": list(pos), "rotvec": list(rotvec), "g": float(g),
            "duration_s": float(duration_s)}


def print_report(table, traj, rep, droop):
    print("")
    print("  segment                      frames   iters  |dp| end (m)  |dR| end (deg)"
          "  max |dp| (m)")
    for row in rep.segments:
        print("  %-28s %6d  %6d    %9.2e      %9.4f      %9.2e"
              % (row.label, row.frames, row.iters, row.pos_err, row.ang_err_deg,
                 row.max_pos_err))
    print("")
    print("[cube  ] z %.5f -> %.5f m, rise %.5f m"
          % (rep.cube_start[2], rep.cube_end[2], rep.rise))
    print("[slip  ] horizontal drift since the gripper closed: %.6f m" % rep.slip)
    print("[pads  ] gripper_pad_fixed %d, gripper_pad_moving %d contacts at the last frame"
          % (rep.pad_contacts.get("gripper_pad_fixed", 0),
             rep.pad_contacts.get("gripper_pad_moving", 0)))
    print("[hold  ] cube centre %.5f m from tool0 at the last frame" % rep.tool_offset)
    for use_ff in (True, False):
        probe = droop.get(use_ff)
        if probe is None:
            continue
        print("[droop ] gravity feedforward %-3s: |dp| %.5f m, dz %+.5f m after 1.0 s hold"
              % ("ON" if use_ff else "OFF", probe["pos_err"], probe["dz"]))
    if droop.get(True) and droop.get(False):
        print("[droop ] feedforward removes %+.5f m of vertical sag"
              % (droop[True]["dz"] - droop[False]["dz"]))
    print("[time  ] %d frames, %.2f s sim, %.2f s wall (%.1f ms per frame)"
          % (rep.frames, rep.sim_time, rep.wall_s,
             1000.0 * rep.wall_s / max(1, rep.frames)))
    if rep.frames_png:
        last = rep.frames_png[-1]
        print("[frames] %d PNG in %s (last %s, %d bytes)"
              % (len(rep.frames_png), rel(DEFAULT_FRAME_DIR), rel(last),
                 os.path.getsize(last)))
    else:
        print("[frames] none written")
    print("")


def run_checks(traj, rep):
    print("  criteria (hard-coded in src/pose_cmd/replay.py)")
    check(rep.rise >= CRITERIA["min_rise_m"],
          "cube rise %.5f m >= %.5f m" % (rep.rise, CRITERIA["min_rise_m"]))
    check(rep.slip < CRITERIA["max_slip_m"],
          "cube horizontal slip %.6f m < %.6f m" % (rep.slip, CRITERIA["max_slip_m"]))
    for name in ("gripper_pad_fixed", "gripper_pad_moving"):
        count = rep.pad_contacts.get(name, 0)
        check(count >= CRITERIA["min_pad_contacts"],
              "%s contacts %d >= %d at the last frame"
              % (name, count, CRITERIA["min_pad_contacts"]))
    check(rep.tool_offset <= CRITERIA["max_tool_offset_m"],
          "cube centre %.5f m from tool0 <= %.5f m (still in the jaws)"
          % (rep.tool_offset, CRITERIA["max_tool_offset_m"]))
    expected = len(range(0, len(traj), RENDER_EVERY))
    check(len(rep.frames_png) == expected,
          "frames: %d PNG for %d control frames, one every %d (%d expected)"
          % (len(rep.frames_png), len(traj), RENDER_EVERY, expected))
    return bool(rep.frames_png) and os.path.getsize(rep.frames_png[-1]) > 0


def run_main(args, t0):
    print("== pose replay: %s ==" % rel(args.table))
    try:
        table = load_table(args.table)
        traj = plan_trajectory(table)
    except WaypointError as exc:
        print("waypoint table rejected: %s" % exc, file=sys.stderr)
        return 2
    print("[table ] %d waypoints, meta.frame = %s, meta.units = %s, meta.control_hz = %g"
          % (len(table.waypoints), table.frame, table.units, table.control_hz))
    print("[plan  ] %d control frames, %.3f s of table time, first frame |ee8 - wp[0]| = %.1e"
          % (len(traj), traj.duration_s,
             float(np.abs(traj.ee8[0] - table.waypoints[0].ee8).max())))

    droop = {}
    for use_ff in (True, False):
        try:
            droop[use_ff] = droop_probe(table, use_ff=use_ff, control_hz=table.control_hz)
        except ReplayError as exc:
            print("[droop ] probe with feedforward %s failed: %s"
                  % ("ON" if use_ff else "OFF", exc))

    try:
        traj, rep = replay_table(table, render=not args.no_render,
                                 control_hz=table.control_hz)
    except ReplayError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    print_report(table, traj, rep, droop)
    run_checks(traj, rep)
    print("")
    print("[total ] %.2f s wall" % (time.perf_counter() - t0))
    if FAILS:
        print("RESULT: %d CHECK(S) FAILED" % len(FAILS))
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


def run_selftest(args, t0):
    print("== selftest: table validation, interpolation, IK failure ==")
    print("  schema")
    table = table_of(wp("a", [0.474, 0.0435, 0.91], [0, math.pi, 0], 1.0, 1.0),
                     wp("b", [0.474, 0.0435, 0.81], [0, math.pi, 0], 0.0, 0.5), hz=100.0)
    check(len(table.waypoints) == 2 and table.control_hz == 100.0,
          "valid table accepted: %d waypoints at %g Hz" % (len(table.waypoints),
                                                           table.control_hz))

    def rejects(name, doc):
        try:
            parse_table(doc)
        except WaypointError as exc:
            return check(True, "%s rejected: %s" % (name, exc))
        except Exception as exc:  # a raw traceback here would be a selftest failure too
            return check(False, "%s raised %s instead of WaypointError" % (name, type(exc)))
        return check(False, "%s accepted but should have been rejected" % name)

    meta = {"frame": "base", "units": "m", "control_hz": 50.0}
    base = wp("good", [0.474, 0.0435, 0.91], [0, math.pi, 0], 1.0, 1.0)
    missing = dict(base)
    del missing["g"]
    rejects("waypoint without g", {"meta": meta, "waypoints": [base, missing]})
    rejects("pos with 2 numbers",
            {"meta": meta, "waypoints": [base, wp("short", [0.474, 0.0435], [0, 0, 0], 1, 1)]})
    rejects("rotvec with 4 numbers",
            {"meta": meta, "waypoints": [base, wp("long", [0.4, 0, 0.9], [0, 0, 0, 0], 1, 1)]})
    rejects("non-finite pos",
            {"meta": meta, "waypoints": [base,
                                         wp("nan", [0.4, float("nan"), 0.9], [0, 0, 0], 1, 1)]})
    rejects("g = 1.5", {"meta": meta, "waypoints": [base,
                                                    wp("open", [0.4, 0, 0.9], [0, 0, 0], 1.5, 1)]})
    rejects("g = -0.1", {"meta": meta, "waypoints": [base,
                                                     wp("close", [0.4, 0, 0.9], [0, 0, 0], -0.1, 1)]})
    rejects("duration_s = 0",
            {"meta": meta, "waypoints": [base,
                                         wp("zero", [0.4, 0, 0.9], [0, 0, 0], 1.0, 0.0)]})
    rejects("empty waypoints", {"meta": meta, "waypoints": []})
    rejects("meta.frame = camera",
            {"meta": dict(meta, frame="camera"), "waypoints": [base]})
    missing_path = os.path.join(REPO_ROOT, "configs", "no_such_table.json")
    try:
        load_table(missing_path)
    except WaypointError as exc:
        check(True, "missing file rejected: %s" % exc)
    except Exception as exc:
        check(False, "missing file raised %s" % type(exc))
    else:
        check(False, "missing file was accepted")

    print("  interpolation")
    traj = plan_trajectory(table)
    check(float(np.abs(traj.ee8[0] - table.waypoints[0].ee8).max()) < 1e-12,
          "first frame equals waypoint[0] exactly (|diff| = %.1e)"
          % float(np.abs(traj.ee8[0] - table.waypoints[0].ee8).max()))
    check(abs(float(traj.ee8[-1, 6]) - table.waypoints[-1].g) < 1e-12,
          "last frame g = %.3f matches the last waypoint" % table.waypoints[-1].g)

    same = table_of(wp("a", [0.474, 0.0435, 0.9], [0, math.pi, 0], 1.0, 1.0),
                    wp("b", [0.474, 0.0435, 0.9], [0, -math.pi, 0], 1.0, 0.2), hz=100.0)
    pair = plan_trajectory(same)
    target = rotvec_to_rotmat([0, math.pi, 0])
    worst = max(float(np.abs(rotvec_to_rotmat(row[3:6]) - target).max())
                for row in pair.ee8)
    check(worst < 1e-9,
          "rotvec (0, +pi, 0) -> (0, -pi, 0) is the same pose at every sample "
          "(max |dR| = %.1e)" % worst)
    a, b = [0.0, 0.99 * math.pi, 0.0], [0.0, -0.99 * math.pi, 0.0]
    swept, prev = 0.0, rotvec_to_rotmat(interp_rotvec(a, b, 0.0))
    for k in range(1, 101):
        cur = rotvec_to_rotmat(interp_rotvec(a, b, k / 100.0))
        rel_rot = cur @ prev.T
        swept += math.acos(max(-1.0, min(1.0, (float(np.trace(rel_rot)) - 1.0) / 2.0)))
        prev = cur
    expect = 0.02 * math.pi
    check(abs(swept - expect) < 1e-6,
          "near-pi pair sweeps %.6f rad, shortest arc %.6f rad (a component-wise ramp "
          "would sweep %.4f)" % (swept, expect, 1.98 * math.pi))

    print("  gripper mapping")
    check(abs(g_to_ctrl(1.0) - 0.9) < 1e-12 and abs(g_to_ctrl(0.0)) < 1e-12
          and abs(g_to_ctrl(0.5) - 0.45) < 1e-12,
          "g -> ctrl: g=1 -> %.3f, g=0 -> %.3f, g=0.5 -> %.3f"
          % (g_to_ctrl(1.0), g_to_ctrl(0.0), g_to_ctrl(0.5)))

    print("  outside the workspace")
    far = table_of(wp("x2", [2.0, 0.0, 0.9], [0, math.pi, 0], 1.0, 0.4), hz=50.0)
    rp = None
    try:
        rp = Replayer(render_every=0)
        rp.run(plan_trajectory(far), render=False)
    except ReplayError as exc:
        print("  error: %s" % exc)
        check("IK not converged" in str(exc),
              "pos = [2.0, 0.0, 0.9] m is refused instead of replayed")
    except Exception as exc:
        check(False, "x = 2.0 m raised %s instead of ReplayError" % type(exc))
    else:
        check(False, "pos = [2.0, 0.0, 0.9] m was replayed, expected an IK failure")
    finally:
        if rp is not None:
            rp.close()

    print("")
    print("[total ] %.2f s wall" % (time.perf_counter() - t0))
    if FAILS:
        print("RESULT: %d CHECK(S) FAILED" % len(FAILS))
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Replay an 8-D waypoint table on the JAKA Zu3 scene.")
    ap.add_argument("--table", default=DEFAULT_TABLE,
                    help="waypoint table JSON (default: configs/waypoints_cube_lift.json)")
    ap.add_argument("--no-render", action="store_true", help="skip the PNG frames")
    ap.add_argument("--selftest", action="store_true",
                    help="run the schema/interpolation/IK counterexamples only")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    FAILS.clear()
    t0 = time.perf_counter()
    try:
        return run_selftest(args, t0) if args.selftest else run_main(args, t0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

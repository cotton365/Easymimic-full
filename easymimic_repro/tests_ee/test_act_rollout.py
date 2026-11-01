#!/usr/bin/env python3
"""Contract test for the closed-loop ACT rollout in experiments/act_rollout.py.

Pure logic + one artefact check: no simulation, no policy, no rendering. The three helpers
that decide a trial's fate - the cube randomisation, the runaway detector and the outcome
classifier - are exercised on hand-built inputs so a wrong threshold or a wrong branch order
fails here instead of silently mislabelling ten 10-second rollouts.

Usage:  python tests_ee/test_act_rollout.py [--summary out/act_rollout/summary.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from experiments.act_rollout import (  # noqa: E402
    CTRL_LIMIT, GRASP_G, JNT_TOL_RAD, MOVE_TOL_RAD, classify, runaway, sample_offsets)
from experiments.gen_act_dataset import RANDOM_X_M, RANDOM_Y_M  # noqa: E402
from src.pose_cmd.replay import CRITERIA  # noqa: E402

SUCCESS_RISE = CRITERIA["min_rise_m"]
SUCCESS_TOOL = CRITERIA["max_tool_offset_m"]


class Checks:
    def __init__(self):
        self.count = 0

    def ok(self, cond, msg):
        self.count += 1
        if not cond:
            raise AssertionError(msg)


class FakeModel:
    """Only the joint-limit rows runaway() reads."""

    def __init__(self, limited, lo, hi):
        self.jnt_limited = np.asarray(limited, dtype=bool)
        self.jnt_range = np.stack([np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)], axis=1)


def fake_model():
    lo = np.zeros(7)
    hi = np.zeros(7) + np.pi
    hi[6] = 0.9                                   # gripper hinge: [0, 0.9]
    limited = np.ones(7, dtype=bool)
    limited[6] = True
    return FakeModel(limited, lo, hi)


def rec(**over):
    base = {"diverged": False, "arm_travel_rad": 1.0, "grasp_tick": 50,
            "rise_m": 0.12, "rise_peak_m": 0.12, "tool_offset_m": 0.002}
    base.update(over)
    return base


def check_offsets(ck):
    rng = np.random.default_rng(7)
    dx, dy = sample_offsets(rng)
    ck.ok(isinstance(dx, float) and isinstance(dy, float), "sample_offsets: not plain floats")
    ck.ok(-RANDOM_X_M <= dx <= RANDOM_X_M, "sample_offsets: dx %r outside +-%r" % (dx, RANDOM_X_M))
    ck.ok(-RANDOM_Y_M <= dy <= RANDOM_Y_M, "sample_offsets: dy %r outside +-%r" % (dy, RANDOM_Y_M))
    again = np.random.default_rng(7)
    ck.ok(sample_offsets(again) == (dx, dy), "sample_offsets: same seed gave a different draw")

    xs, ys = [], []
    rng = np.random.default_rng(0)
    for _ in range(4000):
        a, b = sample_offsets(rng)
        xs.append(a)
        ys.append(b)
    ck.ok(max(xs) <= RANDOM_X_M and min(xs) >= -RANDOM_X_M, "sample_offsets: x escaped the box")
    ck.ok(max(ys) <= RANDOM_Y_M and min(ys) >= -RANDOM_Y_M, "sample_offsets: y escaped the box")
    # the cube must be reachable across the whole box, so both signs have to show up
    ck.ok(min(xs) < -0.9 * RANDOM_X_M and max(xs) > 0.9 * RANDOM_X_M,
          "sample_offsets: never used the x extremes")
    ck.ok(min(ys) < -0.9 * RANDOM_Y_M and max(ys) > 0.9 * RANDOM_Y_M,
          "sample_offsets: never used the y extremes")


def check_runaway(ck):
    m = fake_model()
    q = np.zeros(7)
    a = np.zeros(7)
    ck.ok(runaway(m, q, a) is False, "runaway: an all-zero tick is flagged")
    ck.ok(GRASP_G < 0.1 + 1e-12, "GRASP_G moved; the |g|<0.1 grasp-tick contract changed")

    # regression: the hinge starts exactly on its upper limit and the solver rests 7.48e-6 past it
    q_lit = q.copy()
    q_lit[6] = 0.9 + 7.48e-6
    ck.ok(runaway(m, q_lit, a) is False, "runaway: 7.48e-6 past the hinge limit is flagged")
    ck.ok(JNT_TOL_RAD > 7.48e-6, "runaway: JNT_TOL_RAD no longer covers the solver's limit overlap")
    q_bad = q.copy()
    q_bad[2] = np.pi + 0.01
    ck.ok(runaway(m, q_bad, a) is True, "runaway: an arm joint past its limit is not flagged")

    ck.ok(runaway(m, np.array([np.nan] + [0.0] * 6), a) is True, "runaway: NaN qpos not flagged")
    ck.ok(runaway(m, q, np.array([0.0, 0.0, np.inf, 0, 0, 0, 0])) is True,
          "runaway: inf ctrl not flagged")
    big = a.copy()
    big[0] = CTRL_LIMIT + 0.5
    ck.ok(runaway(m, q, big) is True, "runaway: |ctrl| above the limit not flagged")
    at = a.copy()
    at[0] = CTRL_LIMIT
    ck.ok(runaway(m, q, at) is False, "runaway: |ctrl| exactly at the limit is a valid action")

    # free-joint rows have no limits: the cube must never trip this
    m2 = fake_model()
    m2.jnt_limited[2] = False
    q_free = q.copy()
    q_free[2] = 5.0
    ck.ok(runaway(m2, q_free, a) is False, "runaway: unlimited joint flagged (limit gate broken)")


def check_classify(ck):
    ck.ok(classify(rec(diverged=True, rise_m=0.2)) == "diverged",
          "classify: diverged run reported as something else")
    ck.ok(classify(rec(arm_travel_rad=MOVE_TOL_RAD - 1e-6,
                       grasp_tick=None)) == "no_motion",
          "classify: a motionless run is not reported as no_motion (or lost to a later branch)")
    ck.ok(classify(rec(grasp_tick=None, rise_m=0.0)) == "gripper_never_closed",
          "classify: missing grasp tick not reported as gripper_never_closed")
    ck.ok(classify(rec(rise_m=0.01, rise_peak_m=0.01)) == "not_lifted",
          "classify: rise 0.01 m, never higher, not not_lifted")
    ck.ok(classify(rec(rise_m=0.005, rise_peak_m=0.06)) == "slipped_out",
          "classify: lifted-then-dropped not reported as slipped_out")
    ck.ok(classify(rec(rise_m=0.005, rise_peak_m=0.01)) == "not_lifted",
          "classify: a 10 mm wobble counted as slipped_out")
    ck.ok(classify(rec(rise_m=0.05)) == "lift_insufficient",
          "classify: rise 0.05 m not lift_insufficient")
    ck.ok(classify(rec(rise_m=SUCCESS_RISE, tool_offset_m=SUCCESS_TOOL + 1e-6)) == "lost_after_lift",
          "classify: lifted then lost is not lost_after_lift")
    ck.ok(classify(rec(rise_m=SUCCESS_RISE, tool_offset_m=SUCCESS_TOOL)) == "success",
          "classify: exactly on the acceptance edge is not success")
    ck.ok(classify(rec()) == "success", "classify: a good trial is not success")

    # the acceptance criterion itself, spelled out by the brief
    ck.ok(abs(SUCCESS_RISE - 0.08) < 1e-12, "min_rise_m drifted from 0.08")
    ck.ok(abs(SUCCESS_TOOL - 0.04) < 1e-12, "max_tool_offset_m drifted from 0.04")
    ck.ok(classify(rec(rise_m=0.079)) != "success", "classify: a 0.079 m lift counts as success")


def check_summary(ck, path):
    if not os.path.exists(path):
        print("  (no summary at %s - artefact checks skipped)" % path)
        return
    with open(path) as fh:
        s = json.load(fh)
    for key in ("success_rate", "per_trial", "trials", "device", "ckpt"):
        ck.ok(key in s, "summary: missing key %r" % key)
    ck.ok(len(s["per_trial"]) == int(s["trials"]),
          "summary: %d per_trial rows for trials=%r" % (len(s["per_trial"]), s["trials"]))
    wins = sum(1 for r in s["per_trial"] if r["success"])
    ck.ok(abs(float(s["success_rate"]) - wins / len(s["per_trial"])) < 1e-9,
          "summary: success_rate does not match the per_trial flags")
    for r in s["per_trial"]:
        tag = "trial %02d" % r["trial"]
        ck.ok(r["outcome"] in ("success", "no_motion", "gripper_never_closed", "not_lifted",
                               "slipped_out", "lift_insufficient", "lost_after_lift", "diverged"),
              "%s: unknown outcome %r" % (tag, r["outcome"]))
        ck.ok((r["outcome"] == "success") == bool(r["success"]),
              "%s: outcome %r disagrees with success=%r" % (tag, r["outcome"], r["success"]))
        ck.ok(r["rise_m"] >= -0.05, "%s: rise %r means the cube fell through the table"
              % (tag, r["rise_m"]))
    print("  summary: %d trials, %d success, rate %.2f, %.1f ms/tick"
          % (len(s["per_trial"]), wins, float(s["success_rate"]),
             float(s.get("infer_ms_mean", float("nan")))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", default=os.path.join(REPO, "out", "act_rollout", "summary.json"))
    args = ap.parse_args(argv)

    ck = Checks()
    check_offsets(ck)
    print("  cube randomisation: in-box, reproducible, uses the extremes")
    check_runaway(ck)
    print("  runaway detector: limits, NaN/inf, ctrl bound, free-joint gate")
    check_classify(ck)
    print("  outcome classifier: every branch, in order")
    check_summary(ck, args.summary)
    print("%d checks" % ck.count)
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

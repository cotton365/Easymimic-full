#!/usr/bin/env python3
"""Contract test for the closed-loop ACT rollout in experiments/act_rollout.py.

Pure logic + one artefact check: no simulation, no policy, no rendering. The helpers that
decide a trial's fate - the cube randomisation, the runaway detector and the outcome
classifier - are exercised on hand-built inputs so a wrong threshold or a wrong branch order
fails here instead of silently mislabelling ten 10-second rollouts.

The 8-D half of the file guards the two repairs the decode chain must never lose: the action
quaternion is normalised before it becomes a rotation (mju_quat2Mat does not normalise, so a
3x-scaled quaternion would decode into a non-orthogonal matrix), and g is clipped into [0, 1]
before g_to_ctrl turns it into a joint_7 target (a barely-trained model writes 1.1003). Both
are cheap to state and expensive to notice: without them the arm still moves, it just stops
being the arm the demos were recorded on.

Usage:  python tests_ee/test_act_rollout.py [--summary out/act_rollout/summary.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import experiments.act_rollout as ar  # noqa: E402
from experiments.act_rollout import (  # noqa: E402
    CTRL_LIMIT, EE_DIM, GRASP_G, JNT_TOL_RAD, MOVE_TOL_RAD, QUAT_MIN_NORM, RELEASE_SEP_M,
    bowl_metrics, classify, classify_place, ctrl_row, decode_pose8, load_demo, load_episodes,
    quat_to_rotmat, read_state8, runaway, sample_offsets)
from experiments.gen_act_dataset import (  # noqa: E402
    ANCHORED_LABELS, BOWL_XY, CUBE_BOX, CUBE_XY, IN_BOWL_RADIUS_M, IN_BOWL_SEATED_R_M,
    IN_BOWL_Z, hinge_g, shifted_table, table_task)
from src.pose_cmd.replay import CRITERIA, g_to_ctrl  # noqa: E402
from src.pose_cmd.waypoints import load_table  # noqa: E402

SUCCESS_RISE = CRITERIA["min_rise_m"]
SUCCESS_TOOL = CRITERIA["max_tool_offset_m"]
IK_TOL_M = 2.0e-3
MAX_BOWL_DXY_M = 0.005       # acceptance gate for a generated/replayed place episode
POLICY_HOLD_ERR_M = 0.030    # a policy trial's grasp slop: the cube slides in the jaws
                             # while the arm descends (measured 0.011-0.025 m at step 6000).
                             # 0.002 m is what a replay of the demo achieves, not a policy.
POLICY_BOWL_DXY_M = IN_BOWL_RADIUS_M   # what a *policy* trial is held to: inside the bowl.
                               # The 5 mm gate is a property of the deterministic demo run;
                               # a 2000-step policy places the cube up to ~0.028 m off centre
                               # and that cube is in the bowl, not next to it.
MAX_HOLD_ERR_M = 0.002       # cube still between the pads when the release starts
PLACE_OUTCOMES = ("success", "no_motion", "gripper_never_closed", "never_released",
                  "not_in_bowl", "diverged")


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


class FakeReplayer:
    """Only what ctrl_row() reads: the feed-forward switch and the feed-forward itself."""

    def __init__(self, ff=None):
        self.use_ff = ff is not None
        self._ff = ff

    def feedforward(self, q6):
        return self._ff(q6)


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
    """sample_offsets() is the layout sampler's (dx, dy) view: CUBE_XY subtracted from a
    continuous draw inside CUBE_BOX, so its box is that box shifted - not a +-0.02 one."""
    dx_lo, dx_hi = CUBE_BOX[0] - CUBE_XY[0], CUBE_BOX[1] - CUBE_XY[0]
    dy_lo, dy_hi = CUBE_BOX[2] - CUBE_XY[1], CUBE_BOX[3] - CUBE_XY[1]
    tol = 1e-12
    rng = np.random.default_rng(7)
    dx, dy = sample_offsets(rng)
    ck.ok(isinstance(dx, float) and isinstance(dy, float), "sample_offsets: not plain floats")
    ck.ok(dx_lo - tol <= dx <= dx_hi + tol,
          "sample_offsets: dx %r outside %r..%r" % (dx, dx_lo, dx_hi))
    ck.ok(dy_lo - tol <= dy <= dy_hi + tol,
          "sample_offsets: dy %r outside %r..%r" % (dy, dy_lo, dy_hi))
    again = np.random.default_rng(7)
    ck.ok(sample_offsets(again) == (dx, dy), "sample_offsets: same seed gave a different draw")

    xs, ys = [], []
    rng = np.random.default_rng(0)
    for _ in range(4000):
        a, b = sample_offsets(rng)
        xs.append(a)
        ys.append(b)
    ck.ok(max(xs) <= dx_hi + tol and min(xs) >= dx_lo - tol,
          "sample_offsets: x escaped the cube box")
    ck.ok(max(ys) <= dy_hi + tol and min(ys) >= dy_lo - tol,
          "sample_offsets: y escaped the cube box")
    # the box must actually be used, so both ends show up within 10% of the span
    ck.ok(min(xs) < dx_lo + 0.1 * (dx_hi - dx_lo) and max(xs) > dx_hi - 0.1 * (dx_hi - dx_lo),
          "sample_offsets: never used the x extremes")
    ck.ok(min(ys) < dy_lo + 0.1 * (dy_hi - dy_lo) and max(ys) > dy_hi - 0.1 * (dy_hi - dy_lo),
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
    ck.ok(abs(CRITERIA["max_slip_m"] - 0.005) < 1e-12, "max_slip_m drifted from 0.005")
    ck.ok(classify(rec(rise_m=0.079)) != "success", "classify: a 0.079 m lift counts as success")


def check_classify_place(ck):
    """The place verdict: cube in the bowl *and* the gripper let go - rise must not decide."""
    base = {"diverged": False, "arm_travel_rad": 1.0, "grasp_tick": 50, "in_bowl": True,
            "bowl_dxy_m": 0.004, "hold_err_m": 0.001, "cmd_g_last": 1.0,
            "tool_offset_m": 0.12}
    ck.ok(classify_place(base) == "success", "classify_place: a good trial is not success")

    # 1) rise is measured but never consulted: the same lift that wins the lift task loses
    #    here as soon as the cube is not in the bowl. If place ever honours rise, this pair
    #    of assertions is what breaks.
    big_rise = dict(base, rise_m=0.12, rise_peak_m=0.12, tool_offset_m=0.0,
                    in_bowl=False, bowl_dxy_m=0.25)
    ck.ok(classify_place(big_rise) == "not_in_bowl",
          "classify_place: a 0.12 m lift that missed the bowl is not not_in_bowl (rise leaked "
          "into the place criterion)")
    ck.ok(classify(big_rise) == "success",
          "classify: the lift rule should still call a 0.12 m lift a success")
    ck.ok(classify_place(dict(base, rise_m=0.0, rise_peak_m=0.0, tool_offset_m=0.25)) == "success",
          "classify_place: in the bowl with no rise and the tool far away is not success "
          "(place consulted rise or tool_offset)")

    # 2) "let go" is judged physically - the cube is far from tool0 - and never from the
    #    commanded opening. The g head of a real policy saturates at 0.62-0.80 on trials that
    #    did let go, so gating on g >= 0.9 failed 80/80 correct trials (step 6000, 4 seeds).
    ck.ok(classify_place(dict(base, tool_offset_m=RELEASE_SEP_M - 1e-9)) == "never_released",
          "classify_place: a cube still between the pads passed")
    ck.ok(classify_place(dict(base, tool_offset_m=0.0)) == "never_released",
          "classify_place: tool0 sitting on the cube passed")
    ck.ok(classify_place(dict(base, tool_offset_m=RELEASE_SEP_M)) == "success",
          "classify_place: exactly on the separation edge is not success")
    ck.ok(classify_place(dict(base, cmd_g_last=0.62)) == "success",
          "classify_place: a released trial failed because the policy's g head reads 0.62")
    ck.ok(classify_place(dict(base, cmd_g_last=None)) == "success",
          "classify_place: a released trial failed because it never commanded a gripper value")
    ck.ok(classify_place(dict(base, cmd_g_last=1.0, tool_offset_m=0.005)) == "never_released",
          "classify_place: a fully open command with the cube still in the jaws passed")

    # 3) branch order: the earlier reasons still win over the bowl test
    ck.ok(classify_place(dict(base, diverged=True, in_bowl=False)) == "diverged",
          "classify_place: a diverged trial is reported as something else")
    ck.ok(classify_place(dict(base, arm_travel_rad=MOVE_TOL_RAD - 1e-6, grasp_tick=None))
          == "no_motion", "classify_place: a motionless trial is not no_motion")
    ck.ok(classify_place(dict(base, grasp_tick=None)) == "gripper_never_closed",
          "classify_place: a trial that never closed the jaws is not gripper_never_closed")

    # 4) the bowl window itself: on the centre, on the rim of the accepted radius, on the
    #    wall top, below the bowl floor
    def inb(x, y, z):
        return bool(bowl_metrics(np.array([x, y, z], dtype=float))[1])

    ck.ok(inb(BOWL_XY[0], BOWL_XY[1], 0.817) is True, "bowl_metrics: the bowl centre is not in")
    dxy, ok_c = bowl_metrics(np.array([BOWL_XY[0] + IN_BOWL_RADIUS_M - 1e-6, BOWL_XY[1], 0.817]))
    ck.ok(bool(ok_c) is True and abs(dxy - (IN_BOWL_RADIUS_M - 1e-6)) < 1e-12,
          "bowl_metrics: just inside the accepted radius is not in (dxy %r)" % dxy)
    ck.ok(inb(BOWL_XY[0] + IN_BOWL_RADIUS_M + 1e-6, BOWL_XY[1], 0.817) is False,
          "bowl_metrics: just outside the accepted radius counts as in")
    ck.ok(inb(BOWL_XY[0], BOWL_XY[1], IN_BOWL_Z[1] + 1e-6) is False,
          "bowl_metrics: a cube on the wall top counts as in the bowl")
    ck.ok(inb(BOWL_XY[0], BOWL_XY[1], IN_BOWL_Z[0] - 1e-6) is False,
          "bowl_metrics: a cube below the bowl floor counts as in the bowl")
    ck.ok(IN_BOWL_RADIUS_M <= 0.045,
          "the accepted radius %r is outside the bowl's own 0.045 m inner radius"
          % IN_BOWL_RADIUS_M)
    ck.ok(IN_BOWL_SEATED_R_M < IN_BOWL_RADIUS_M,
          "the seated radius %r is not stricter than the accepted one %r"
          % (IN_BOWL_SEATED_R_M, IN_BOWL_RADIUS_M))
    ck.ok(abs(RELEASE_SEP_M - 0.05) < 1e-12, "RELEASE_SEP_M drifted from 0.05")
    ck.ok(abs(MAX_BOWL_DXY_M - 0.005) < 1e-12 and abs(MAX_HOLD_ERR_M - 0.002) < 1e-12,
          "the place gates drifted from 0.005 m / 0.002 m")


def check_anchoring(ck):
    """shifted_table: the cube-side waypoints follow the offset, the bowl-side ones do not.

    The bowl is part of the scene, so a cube offset must not drag the carry/lower/release/
    retreat waypoints off it. Every expectation here is spelled out instead of read back from
    the module under test: the point is to go red when ANCHORED_LABELS stops naming the four
    bowl-side labels, not to agree with it.
    """
    anchored = {"carry", "lower", "release", "retreat"}
    cube_side = {"approach", "descend", "close", "lift"}
    base = load_table(ar.PLACE_TABLE_PATH)
    ck.ok(len(base.waypoints) == 8, "place table: %d waypoints, expected 8"
          % len(base.waypoints))
    ck.ok({w.label for w in base.waypoints} == anchored | cube_side,
          "place table labels %s != the 8 expected ones"
          % sorted(w.label for w in base.waypoints))

    dx, dy = 0.017, 0.004
    moved = shifted_table(base, dx, dy)
    ck.ok(len(moved.waypoints) == len(base.waypoints),
          "shifted_table: waypoint count changed")
    ck.ok(float(base.control_hz) == 50.0,
          "place table: meta.control_hz %r, expected the 50 Hz authoring rate"
          % (base.control_hz,))
    ck.ok(float(moved.control_hz) == float(ar.CONTROL_HZ),
          "shifted_table: replays at %r Hz, expected the generator's %r"
          % (moved.control_hz, ar.CONTROL_HZ))
    for a, b in zip(base.waypoints, moved.waypoints):
        ck.ok(a.label == b.label and float(b.duration_s) == float(a.duration_s),
              "shifted_table: %r lost its label or duration" % a.label)
        want = np.zeros(2) if a.label in anchored else np.array([dx, dy])
        got = np.asarray(b.ee8[0:2], dtype=float) - np.asarray(a.ee8[0:2], dtype=float)
        ck.ok(np.allclose(got, want, atol=1e-12),
              "shifted_table: %r moved by (%+0.4f, %+0.4f), expected (%+0.4f, %+0.4f) - the "
              "bowl does not follow the cube offset" % (a.label, got[0], got[1], want[0],
                                                        want[1]))
        ck.ok(np.array_equal(np.asarray(b.ee8[2:], dtype=float),
                             np.asarray(a.ee8[2:], dtype=float)),
              "shifted_table: %r changed its z, rotvec or g" % a.label)

    ck.ok(tuple(ANCHORED_LABELS) == ("carry", "lower", "release", "retreat"),
          "ANCHORED_LABELS is %r, expected the four bowl-side labels" % (ANCHORED_LABELS,))
    ck.ok(table_task(base) == "place",
          "table_task: a table with the bowl-side labels is not reported as place")
    ck.ok(table_task(load_table(ar.TABLE_PATH)) == "lift",
          "table_task: the lift table is not reported as lift")

    carry = next(w for w in moved.waypoints if w.label == "carry")
    ck.ok(np.allclose(np.asarray(carry.ee8[0:2], dtype=float), np.asarray(BOWL_XY, float),
                      atol=1e-12),
          "shifted_table: carry is not over the bowl centre after shifting: %r"
          % (carry.ee8[0:3],))
    approach = next(w for w in moved.waypoints if w.label == "approach")
    ck.ok(np.allclose(np.asarray(approach.ee8[0:2], dtype=float) - np.array([dx, dy]),
                      np.asarray(CUBE_XY, float), atol=1e-12),
          "shifted_table: approach no longer sits over the shifted cube: %r"
          % (approach.ee8[0:3],))


def check_decode(ck):
    """The 8-D action -> pose/rotation/opening decode: normalisation and clipping."""
    ck.ok(EE_DIM == 8, "EE_DIM is not 8; the pose8_* layout and the policy head must agree")
    ck.ok(abs(g_to_ctrl(1.0) - 0.9) < 1e-12,
          "g_to_ctrl(1.0) is no longer 0.9: the [0, 0.9] hinge range moved")

    pos = np.array([0.5, 0.02, 0.9])
    quat = np.array([0.7071067811865476, 0.0, 0.7071067811865476, 0.0])   # 90 deg about y
    a8 = np.concatenate([pos, quat, [0.4]])
    p0, r0, g0, i0 = decode_pose8(a8)
    ck.ok(np.allclose(p0, pos), "decode: the position is not passed through untouched")
    ck.ok(i0["bad"] is False, "decode: a unit quaternion is reported as a broken action")
    ck.ok(g0 == 0.4 and i0["clipped"] is False, "decode: an in-range g was clipped")
    ck.ok(r0 is not None, "decode: no rotation for a valid action")
    ck.ok(float(np.abs(r0 @ r0.T - np.eye(3)).max()) < 1e-12, "decode: R is not orthonormal")
    ck.ok(abs(float(np.linalg.det(r0)) - 1.0) < 1e-12, "decode: R is not a proper rotation")
    ck.ok(float(np.abs(quat_to_rotmat(quat) - r0).max()) < 1e-15,
          "decode: the rotation is not mju_quat2Mat of the (already unit) quaternion")

    # B3: the network hands over a quaternion that is unit only up to its arithmetic, and
    # mju_quat2Mat does NOT normalise - decoding 3 q must therefore land on the same R as q.
    a3 = a8.copy()
    a3[3:7] = 3.0 * quat
    p3, r3, g3, i3 = decode_pose8(a3)
    ck.ok(abs(i3["quat_norm"] - 3.0) < 1e-12, "decode: quat_norm is not the raw |q|")
    ck.ok(i3["bad"] is False, "decode: a 3x-scaled quaternion is reported as broken")
    ck.ok(r3 is not None, "decode: a 3x-scaled quaternion produced no rotation")
    dev = float(np.abs(r3 - r0).max())
    ck.ok(dev <= 1e-12, "decode: scaling the action quaternion by 3 changed the rotation by %r"
          % dev)
    raw = np.zeros(9)
    import mujoco
    mujoco.mju_quat2Mat(raw, np.array(a3[3:7], dtype=np.float64))
    ck.ok(float(np.abs(raw.reshape(3, 3) - r0).max()) > 1.0,
          "decode: the test itself is wrong - mju_quat2Mat seems to normalise, so B3 would "
          "pass without the normalisation in decode_pose8")

    # a broken action is reported, never amplified into a rotation
    ab = a8.copy()
    ab[3:7] = QUAT_MIN_NORM * 0.2 * quat
    pb, rb, gb, ib = decode_pose8(ab)
    ck.ok(ib["bad"] is True, "decode: |q| %.2f is accepted as a rotation" % ib["quat_norm"])
    ck.ok(rb is None, "decode: a broken quaternion still produced a rotation matrix")
    ck.ok(np.isfinite(gb), "decode: a broken quaternion also lost the gripper opening")
    an = a8.copy()
    an[3] = np.nan
    ck.ok(decode_pose8(an)[3]["bad"] is True, "decode: a NaN quaternion is not flagged")
    an = a8.copy()
    an[7] = np.nan
    ck.ok(decode_pose8(an)[3]["bad"] is True, "decode: a NaN g is not flagged")

    # B2 (decode half): g outside [0, 1] is clipped, on both sides, and counted
    hi = a8.copy()
    hi[7] = 1.1
    ck.ok(decode_pose8(hi)[2] == 1.0, "decode: g=1.1 was not clipped to 1.0")
    ck.ok(decode_pose8(hi)[3]["clipped"] is True, "decode: g=1.1 was not counted as a clip")
    lo = a8.copy()
    lo[7] = -0.2
    ck.ok(decode_pose8(lo)[2] == 0.0, "decode: g=-0.2 was not clipped to 0.0")
    ck.ok(decode_pose8(lo)[3]["clipped"] is True, "decode: g=-0.2 was not counted as a clip")
    for edge in (0.0, 1.0):
        e = a8.copy()
        e[7] = edge
        ck.ok(decode_pose8(e)[2] == edge and decode_pose8(e)[3]["clipped"] is False,
              "decode: g=%r on the boundary was reported as clipped" % edge)


def check_ctrl_row(ck):
    """The 8-D action -> data.ctrl mapping: feed-forward plus the clipped gripper target."""
    q6 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    rp = FakeReplayer(lambda q: np.full(6, 0.01))
    c = ctrl_row(rp, q6, 1.1)
    ck.ok(len(c) == 7, "ctrl_row: the plant takes 7 numbers, got %d" % len(c))
    ck.ok(np.allclose(c[:6], q6 + 0.01),
          "ctrl_row: ctrl[:6] is not q6 + feedforward(q6)")
    ck.ok(c[6] == g_to_ctrl(1.0), "ctrl_row: g=1.1 was not clipped before g_to_ctrl")
    ck.ok(abs(c[6] - 0.9) < 1e-12, "ctrl_row: g=1.1 did not land on the open hinge limit")
    lo_g, hi_g = fake_model().jnt_range[6]
    ck.ok(lo_g - 1e-12 <= c[6] <= hi_g + 1e-12,
          "ctrl_row: g=1.1 drove joint_7 to %r, outside [%r, %r]" % (c[6], lo_g, hi_g))
    cn = ctrl_row(rp, q6, -0.2)
    ck.ok(cn[6] == g_to_ctrl(0.0), "ctrl_row: g=-0.2 was not clipped before g_to_ctrl")
    ck.ok(lo_g - 1e-12 <= cn[6] <= hi_g + 1e-12,
          "ctrl_row: g=-0.2 drove joint_7 to %r, outside [%r, %r]" % (cn[6], lo_g, hi_g))
    mid = ctrl_row(rp, q6, 0.5)
    ck.ok(mid[6] == g_to_ctrl(0.5), "ctrl_row: an in-range g was not mapped 1:1")
    ck.ok(np.allclose(ctrl_row(FakeReplayer(None), q6, 0.5)[:6], q6),
          "ctrl_row: the feed-forward is added even when use_ff is off")


def check_state8(ck):
    """The observation half: tool0 pose, the sign walk, and the opening in g units."""
    real = ar.fk_tool
    mine = np.array([-0.7071067811865476, 0.0, -0.7071067811865476, 0.0])   # qw < 0 on tick 0
    flip = np.array([0.7071067811865476, 0.0, 0.7071067811865476, 0.0])
    same = np.array([0.6, 0.2, 0.7, 0.1]) / np.linalg.norm([0.6, 0.2, 0.7, 0.1])
    quats = [mine, -flip, same]
    pos = np.array([0.5, 0.02, 0.9])
    try:
        ar.fk_tool = lambda fk_data, site_id, qpos7: (pos, quats[min(len(quats), 1) - 1])
        q7 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.45])
        got = []
        prev = None
        for q in quats:
            ar.fk_tool = lambda fk_data, site_id, qpos7, q=q: (pos, np.array(q))
            s8, prev = read_state8(None, 0, q7, prev)
            ck.ok(len(s8) == EE_DIM, "state8: %d numbers, the policy head takes %d"
                  % (len(s8), EE_DIM))
            ck.ok(np.allclose(s8[0:3], pos), "state8: the position is not the tool0 position")
            ck.ok(abs(s8[7] - hinge_g(q7[6])) < 1e-15,
                  "state8: the opening is not hinge_g(qpos[6]) (measured g)")
            got.append(s8[3:7])
    finally:
        ar.fk_tool = real
    # first tick: flipped to qw >= 0; q and -q are the same orientation
    ck.ok(got[0][0] >= 0.0, "state8: the first tick was not flipped to qw >= 0")
    ck.ok(np.allclose(got[0], -mine), "state8: the first-tick flip changed more than the sign")
    # second tick: -flip is 180 deg away in numbers, dot < 0, so it comes back as +flip
    ck.ok(float(got[1] @ got[0]) > 0.0, "state8: the sign walk left a dot<0 tick in place")
    ck.ok(np.allclose(got[1], got[0]), "state8: q and -q were not made continuous")
    # third tick: dot > 0, so it must be passed through untouched
    ck.ok(np.allclose(got[2], same), "state8: a dot>0 quaternion was modified")


def check_replay_plumbing(ck):
    """--replay-demos: episode discovery and the offset read back from meta.json."""
    root = tempfile.mkdtemp(prefix="act_rollout_ep")
    q7 = np.linspace(0.0, 1.0, 8)
    for name, cube in (("ep001", (CUBE_XY[0] + 0.012, CUBE_XY[1] - 0.003, 0.815)),
                       ("ep000", (CUBE_XY[0] - 0.004, CUBE_XY[1] + 0.002, 0.815))):
        d = os.path.join(root, name)
        os.makedirs(d)
        np.savez(os.path.join(d, "data.npz"), pose8_state=np.tile(q7, (3, 1)),
                 pose8_action=np.tile(q7, (3, 1)))
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump({"cube_pos": list(cube), "num_frames": 3}, fh)
    eps = load_episodes(root)
    ck.ok(len(eps) == 2, "load_episodes: found %d of 2 episodes" % len(eps))
    ck.ok([os.path.basename(e) for e in eps] == ["ep000", "ep001"],
          "load_episodes: episodes are not in name order")
    demo = load_demo(eps[0])
    ck.ok(demo["pose8_action"].shape == (3, EE_DIM),
          "load_demo: pose8_action is not (N, %d)" % EE_DIM)
    ck.ok(abs(demo["dx"] - (-0.004)) < 1e-12 and abs(demo["dy"] - 0.002) < 1e-12,
          "load_demo: the cube offset is not cube_pos - CUBE_XY (dx=%r dy=%r)"
          % (demo["dx"], demo["dy"]))
    try:
        load_episodes(os.path.join(root, "nope"))
    except SystemExit:
        pass
    else:
        ck.ok(False, "load_episodes: a missing directory did not stop the run")
    try:
        load_episodes(tempfile.mkdtemp(prefix="act_rollout_empty"))
    except SystemExit:
        pass
    else:
        ck.ok(False, "load_episodes: an empty directory did not stop the run")


class _ViewerCalled(Exception):
    """Raised by the viewer stub: the real call site ends in os._exit(0), so the stub has to
    leave main() *before* that, and an exception is the only exit it can take from inside."""


def check_viewer_stub(ck):
    """--viewer must be reachable without opening a GL window: swap play_in_viewer for a stub.

    Two properties, both from the brief: the window is not opened unless asked for, and when
    it is asked for it receives the scene, the recorded traces, the control rate and the
    requested number of passes - in that order.
    """
    calls = []

    def stub(scene, traces, control_hz, passes=0):
        calls.append({"scene": scene, "traces": traces, "hz": control_hz, "passes": passes})
        raise _ViewerCalled()

    real = ar.play_in_viewer
    ar.play_in_viewer = stub
    out = tempfile.mkdtemp(prefix="act_rollout_viewer")
    try:
        # 1) no --viewer: the stub must not be reached at all
        ar.main(["--replay-demos", os.path.join(REPO, "out", "act_demos_place"),
                 "--task", "place", "--trials", "1", "--json-only", "--no-video",
                 "--out", os.path.join(out, "headless")])
        ck.ok(not calls, "viewer: play_in_viewer was called without --viewer")

        # 2) --viewer: called exactly once, with the place scene and one recorded trace
        try:
            ar.main(["--replay-demos", os.path.join(REPO, "out", "act_demos_place"),
                     "--task", "place", "--trials", "1", "--json-only", "--no-video",
                     "--viewer", "--viewer-passes", "3",
                     "--out", os.path.join(out, "viewer")])
        except _ViewerCalled:
            pass
        ck.ok(len(calls) == 1, "viewer: play_in_viewer called %d times, expected 1"
              % len(calls))
        if calls:
            got = calls[0]
            ck.ok(os.path.abspath(got["scene"]) == os.path.abspath(ar.SCENE),
                  "viewer: scene %r" % (got["scene"],))
            ck.ok(os.path.isfile(got["scene"]), "viewer: scene %r does not exist"
                  % (got["scene"],))
            ck.ok(isinstance(got["traces"], list) and len(got["traces"]) == 1,
                  "viewer: %r traces for a 1-trial run" % (len(got["traces"]),))
            bowl_xy, frames = got["traces"][0]
            trace = np.asarray(frames, dtype=float)
            ck.ok(trace.ndim == 2 and trace.shape[1] == 14,
                  "viewer: trace shape %s, expected (frames, 14) - arm + gripper + cube "
                  "freejoint" % (trace.shape,))
            ck.ok(trace.shape[0] == 181, "viewer: trace has %d frames, the place episode is 181"
                  % (trace.shape[0],))
            # the layout must travel with the frames: the window rebuilds the model from the
            # XML, so without this the bowl sits at the scene default and every random-layout
            # replay looks like the cube missed it (the exact bug this check guards)
            written = json.loads(open(os.path.join(out, "viewer", "summary.json")).read())
            ran = written["per_trial"][0]["bowl_xy"]
            ck.ok(abs(float(bowl_xy[0]) - float(ran[0])) < 1e-12
                  and abs(float(bowl_xy[1]) - float(ran[1])) < 1e-12,
                  "viewer: bowl_xy %r is not the trial's %r" % (bowl_xy, ran))
            ck.ok(abs(float(got["hz"]) - float(ar.CONTROL_HZ)) < 1e-12,
                  "viewer: control_hz %r" % (got["hz"],))
            ck.ok(int(got["passes"]) == 3, "viewer: passes %r, expected the --viewer-passes 3"
                  % (got["passes"],))
    finally:
        ar.play_in_viewer = real


def check_sources(ck):
    """Contracts on the files themselves: the lazy torch import and the retired 7-D copy."""
    src = open(os.path.join(REPO, "experiments", "act_rollout.py")).read()
    top = [ln for ln in src.splitlines()
           if ln.startswith("import torch") or ln.startswith("from torch")]
    ck.ok(not top, "act_rollout.py: torch is imported at module scope (%r)" % (top,))
    lazy = [ln for ln in src.splitlines() if ln.strip() == "import torch"]
    ck.ok(len(lazy) == 1,
          "act_rollout.py: expected exactly one lazy 'import torch' inside _torch(), got %d"
          % len(lazy))
    ck.ok('"--replay-demos"' in src, "act_rollout.py: --replay-demos is not a CLI flag")

    leg = os.path.join(REPO, "legacy", "act_rollout_joint7.py")
    ck.ok(os.path.exists(leg), "legacy/act_rollout_joint7.py is missing")
    head = open(leg).read().splitlines()[:8]
    ck.ok(any("RETIRED" in ln for ln in head),
          "legacy/act_rollout_joint7.py: no RETIRED banner in the first 8 lines")
    body = open(leg).read()
    ck.ok("data.ctrl[:7] = " in body or "ctrl[:7]" in body,
          "legacy/act_rollout_joint7.py: does not look like the 7-D implementation")
    ck.ok(any(ln.startswith("import torch") for ln in body.splitlines()),
          "legacy/act_rollout_joint7.py: the 7-D copy should still import torch eagerly")


def check_trial_layout(ck):
    """The summary's top-level layout fields describe what the trials ran, not the scene default."""
    row = {"cube_xy": [0.474, 0.0435], "bowl_xy": [0.320, -0.280]}
    rows = [dict(row, trial=0), dict(row, trial=1)]
    cube_xy, bowl_xy, spread = ar.trial_layout(rows)
    ck.ok(bowl_xy == [0.320, -0.280],
          "trial_layout: a bowl at [0.32, -0.28] summarised as %r, the scene default %r"
          % (bowl_xy, list(ar.BOWL_XY)))
    ck.ok(cube_xy == [0.474, 0.0435], "trial_layout: cube summarised as %r" % (cube_xy,))
    ck.ok(spread is None, "trial_layout: spread %r for a shared bowl" % (spread,))
    mixed = [dict(rows[0]), dict(row, trial=2, bowl_xy=[0.300, -0.200])]
    cube_xy, bowl_xy, spread = ar.trial_layout(mixed)
    ck.ok(bowl_xy is None and spread is not None and spread["distinct"] == 2
          and spread["min"] == [0.300, -0.280] and spread["max"] == [0.320, -0.200],
          "trial_layout: two bowls summarised as %r / %r" % (bowl_xy, spread))
    ck.ok(ar.trial_layout([]) == (None, None, None),
          "trial_layout: no rows gave %r" % (ar.trial_layout([]),))


def check_summary(ck, path):
    if not os.path.exists(path):
        print("  (no summary at %s - artefact checks skipped)" % path)
        return
    with open(path) as fh:
        s = json.load(fh)
    place = s.get("task_kind") == "place"
    allowed = PLACE_OUTCOMES if place else ("success", "no_motion", "gripper_never_closed",
                                            "not_lifted", "slipped_out", "lift_insufficient",
                                            "lost_after_lift", "diverged")
    for key in ("success_rate", "per_trial", "trials", "device", "ckpt"):
        ck.ok(key in s, "summary: missing key %r" % key)
    ck.ok(len(s["per_trial"]) == int(s["trials"]),
          "summary: %d per_trial rows for trials=%r" % (len(s["per_trial"]), s["trials"]))
    wins = sum(1 for r in s["per_trial"] if r["success"])
    ck.ok(abs(float(s["success_rate"]) - wins / len(s["per_trial"])) < 1e-9,
          "summary: success_rate does not match the per_trial flags")
    for r in s["per_trial"]:
        tag = "trial %02d" % r["trial"]
        ck.ok(r["outcome"] in allowed, "%s: unknown outcome %r" % (tag, r["outcome"]))
        ck.ok((r["outcome"] == "success") == bool(r["success"]),
              "%s: outcome %r disagrees with success=%r" % (tag, r["outcome"], r["success"]))
        ck.ok(r["rise_m"] >= -0.05, "%s: rise %r means the cube fell through the table"
              % (tag, r["rise_m"]))
        if place:
            # the place verdict, checked against the recorded numbers rather than trusted
            for key in ("bowl_dxy_m", "in_bowl", "hold_err_m"):
                ck.ok(key in r, "%s: place trial without %r" % (tag, key))
            if r["outcome"] == "success":
                ck.ok(bool(r["in_bowl"]) is True,
                      "%s: success with in_bowl=%r" % (tag, r["in_bowl"]))
                # A success must land inside the bowl, in either mode. The tighter
                # MAX_BOWL_DXY_M (5 mm) is the *generator's* acceptance gate and is enforced
                # by test_act_dataset on meta.json; a replay of those actions still drifts
                # (measured 7.1 mm on the random dataset), so it is not reused here.
                ck.ok(float(r["bowl_dxy_m"]) <= IN_BOWL_RADIUS_M,
                      "%s: success (%s run) with bowl_dxy %r above %r m"
                      % (tag, s.get("mode"), r["bowl_dxy_m"], IN_BOWL_RADIUS_M))
                ck.ok(float(r["tool_offset_m"]) >= RELEASE_SEP_M,
                      "%s: success with the cube only %r m from tool0 (still held)"
                      % (tag, r["tool_offset_m"]))
                if r["hold_err_m"] is not None:
                    gate = MAX_HOLD_ERR_M if s.get("mode") == "replay" else POLICY_HOLD_ERR_M
                    ck.ok(float(r["hold_err_m"]) <= gate,
                          "%s: success (%s run) with hold_err %r above %r m"
                          % (tag, s.get("mode"), r["hold_err_m"], gate))
        if "ik_pos_max_m" in r:
            ck.ok(r["ik_pos_max_m"] <= IK_TOL_M,
                  "%s: max IK position residual %r exceeds %r m"
                  % (tag, r["ik_pos_max_m"], IK_TOL_M))
            if r["outcome"] == "success" and not place:
                ck.ok(r["rise_m"] >= SUCCESS_RISE,
                      "%s: success with rise %r below %r" % (tag, r["rise_m"], SUCCESS_RISE))
                ck.ok(r["slip_m"] <= CRITERIA["max_slip_m"],
                      "%s: success with slip %r above %r"
                      % (tag, r["slip_m"], CRITERIA["max_slip_m"]))
                ck.ok(r["tool_offset_m"] <= SUCCESS_TOOL,
                      "%s: success with tool offset %r above %r"
                      % (tag, r["tool_offset_m"], SUCCESS_TOOL))
    if "ee_layout" in s:
        ck.ok(s["ee_layout"] == ["x", "y", "z", "qw", "qx", "qy", "qz", "g"],
              "summary: the EE layout moved: %r" % (s["ee_layout"],))
        ck.ok(s["ee_site"] == "tool0", "summary: the EE site is not tool0")
        ck.ok(float(s["ik_pos_err_max_m"]) <= IK_TOL_M,
              "summary: max IK position residual %r exceeds %r m"
              % (s["ik_pos_err_max_m"], IK_TOL_M))
        if not place and s.get("rise_min_m") is not None:
            ck.ok(float(s["rise_min_m"]) >= SUCCESS_RISE,
                  "summary: rise_min_m %r below %r" % (s["rise_min_m"], SUCCESS_RISE))
        if not place and s.get("slip_max_m") is not None:
            ck.ok(float(s["slip_max_m"]) <= CRITERIA["max_slip_m"],
                  "summary: slip_max_m %r above %r"
                  % (s["slip_max_m"], CRITERIA["max_slip_m"]))
        if not place and s.get("tool_offset_max_m") is not None:
            ck.ok(float(s["tool_offset_max_m"]) <= SUCCESS_TOOL,
                  "summary: tool_offset_max_m %r above %r"
                  % (s["tool_offset_max_m"], SUCCESS_TOOL))
        if place:
            # place keeps rise/slip/tool_offset as measurements (slip is the cube's whole
            # trip to the bowl under the lift definition), so only the place fields gate it
            ck.ok(s["task"] == ar.PLACE_TASK, "summary: task %r is not the place task"
                  % (s["task"],))
            ck.ok("min_rise_m" not in s["criteria"],
                  "summary: the place criteria still list min_rise_m")
            ck.ok(abs(float(s["criteria"]["release_sep_m"]) - RELEASE_SEP_M) < 1e-12,
                  "summary: criteria.release_sep_m %r != RELEASE_SEP_M"
                  % (s["criteria"]["release_sep_m"],))
            if s.get("mode") == "replay":
                # A replay of recorded actions has no excuse: every trial must land inside.
                ck.ok(float(s["bowl_dxy_max_m"]) <= IN_BOWL_RADIUS_M,
                      "summary (replay): bowl_dxy_max_m %r above %r m"
                      % (s["bowl_dxy_max_m"], IN_BOWL_RADIUS_M))
                ck.ok(s["in_bowl_all"] is True
                      and int(s["in_bowl_count"]) == len(s["per_trial"]),
                      "summary: in_bowl %r/%r" % (s["in_bowl_count"], len(s["per_trial"])))
                if s["hold_err_max_m"] is not None:
                    ck.ok(float(s["hold_err_max_m"]) <= MAX_HOLD_ERR_M,
                          "summary (replay): hold_err_max_m %r above %r m"
                          % (s["hold_err_max_m"], MAX_HOLD_ERR_M))
            else:
                # A policy run is allowed to fail trials - that is what it is measuring. The
                # quality bound therefore applies per *successful* trial (checked above), and
                # here only the aggregates are held to their own definitions.
                ck.ok(int(s["in_bowl_count"])
                      == sum(1 for r in s["per_trial"] if r["in_bowl"]),
                      "summary (policy): in_bowl_count %r disagrees with per_trial"
                      % (s["in_bowl_count"],))
                # the summary keeps this aggregate rounded to 6 decimals (act_rollout.py:853),
                # so the comparison is against that rounding, not against float equality
                worst = round(max(float(r["bowl_dxy_m"]) for r in s["per_trial"]), 6)
                ck.ok(abs(float(s["bowl_dxy_max_m"]) - worst) <= 1e-12,
                      "summary (policy): bowl_dxy_max_m %r is not the max of per_trial (%r)"
                      % (s["bowl_dxy_max_m"], worst))
            ck.ok(s["table"].endswith("waypoints_cube_place.json"),
                  "summary: table %r is not the place table" % (s["table"],))
            ck.ok(list(s["anchored_labels"]) == ["carry", "lower", "release", "retreat"],
                  "summary: anchored_labels %r" % (s["anchored_labels"],))
        if s.get("state_dev0_max") is not None:
            ck.ok(float(s["state_dev0_max"]) <= 1e-9,
                  "summary: the first 8-D state is %r away from pose8_state[0]"
                  % s["state_dev0_max"])
        print("  summary: mode=%s rise %.4f..%.4f slip_max %.5f tool_max %.5f "
              "ik_max %.2e m  clip %d fb %d fail %d"
              % (s.get("mode"), s["rise_min_m"], s["rise_max_m"], s["slip_max_m"],
                 s["tool_offset_max_m"], s["ik_pos_err_max_m"], s.get("g_clip_ticks", 0),
                 s.get("ik_fallback_ticks", 0), s.get("ik_fail_ticks", 0)))
    print("  summary: %d trials, %d success, rate %.2f, %.1f ms/tick"
          % (len(s["per_trial"]), wins, float(s["success_rate"]),
             float(s.get("infer_ms_mean", float("nan")))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", default=os.path.join(REPO, "out", "act_rollout", "summary.json"))
    ap.add_argument("--place-summary",
                    default=os.path.join(REPO, "out", "act_rollout_place", "summary.json"))
    args = ap.parse_args(argv)

    ck = Checks()
    check_offsets(ck)
    print("  cube randomisation: in-box, reproducible, uses the extremes")
    check_runaway(ck)
    print("  runaway detector: limits, NaN/inf, ctrl bound, free-joint gate")
    check_classify(ck)
    print("  outcome classifier: every branch, in order")
    check_classify_place(ck)
    print("  place verdict: in bowl + released, rise/tool_offset never consulted")
    check_anchoring(ck)
    print("  place table: cube-side waypoints follow the offset, bowl-side ones anchored")
    check_trial_layout(ck)
    print("  summary layout: the top-level bowl/cube follows the trials, not the scene default")
    check_decode(ck)
    print("  action decode: quat normalised (3x -> same R), broken action flagged, g clipped")
    check_ctrl_row(ck)
    print("  ctrl row: feed-forward + clipped gripper target inside joint_7's range")
    check_state8(ck)
    print("  observation: tool0 pose, first-tick flip, dot<0 sign walk, measured opening")
    check_replay_plumbing(ck)
    print("  replay plumbing: episode discovery in name order, offset from meta.json")
    check_viewer_stub(ck)
    print("  viewer: stubbed - not opened unless --viewer, then scene/traces/hz/passes checked")
    check_sources(ck)
    print("  sources: lazy torch, --replay-demos present, retired 7-D copy in place")
    check_summary(ck, args.summary)
    check_summary(ck, args.place_summary)
    print("%d checks" % ck.count)
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Real-time hand teleoperation: webcam -> MediaPipe Hands -> wrist frame -> hand_to_ee
-> 8-D command -> IK -> MuJoCo position servos.

Usage (repo root, mujoco env python):
    python experiments/hand_teleop_live.py --probe       # camera chain only, no robot
    python experiments/hand_teleop_live.py --selftest    # no camera / no cv2 / no window
    python experiments/hand_teleop_live.py               # live teleoperation

Mapping. Nothing here assumes a hand-eye calibration, so hand_to_ee is called with
T_base_cam = identity: the camera frame IS the robot base frame. With X right, Y down,
Z away from the camera, moving the hand right moves the tool along +X, moving it down
moves the tool along +Y, and pushing it away from the camera moves the tool along +Z.

    position : p_cmd = p_neutral + gain * (anchor_now - anchor_ref)
    rotation : R_cmd = R_neutral @ (R_ref^T @ R_now)
    gripper  : ctrl  = g_to_ctrl(g_norm)         1 = open hand, 0 = pinch

The neutral pose (p_neutral, R_neutral) is the "approach" waypoint of
configs/waypoints_cube_lift.json -- the verified grasp-ready pose, position and Ry(pi)
orientation together. The home keyframe qpos is NOT a good neutral: forcing the grasp
orientation onto its FK leaves a 4 cm move along +X unsolvable (2.77 mm best residual),
while the approach pose solves in all six directions. The plant is teleported there at
startup, so the first tick commands exactly where the arm already is.

anchor_ref and R_ref are captured from the first usable frame, so whatever pose the hand
holds when the loop starts becomes neutral and the arm does not jump at startup.

A frame that is missing, NaN, or geometrically degenerate freezes the last command and
skips the IK entirely. Freezing is recoverable; a NaN written into ctrl is not, and the
gripper servos would take the whole plant with them.

Exit codes: 0 ok / 1 camera or MediaPipe unusable, or a selftest check failed /
2 the scene could not be loaded / 3 no window available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mujoco                                                        # noqa: E402
import mujoco.viewer                                                 # noqa: E402

from src.ee_retarget.hand_to_ee import hand_to_ee                    # noqa: E402
from src.ee_retarget.ik_jaka_zu3 import rotvec_to_rotmat             # noqa: E402
from src.hand_input import wrist_frame                               # noqa: E402
from src.hand_input.mediapipe_source import (CameraError, CameraHandSource,   # noqa: E402
                                             MissingDependency, build_frame, probe)
from src.pose_cmd.replay import ReplayError, Replayer, g_to_ctrl     # noqa: E402

# The verified cube-lift grasp orientation: configs/waypoints_cube_lift.json uses rotvec
# [0, pi, 0], i.e. Ry(pi) -- not the Rz(pi) mentioned in the task brief.
R_GRASP = rotvec_to_rotmat(np.array([0.0, np.pi, 0.0]))
WAYPOINTS_PATH = os.path.join(REPO_ROOT, "configs", "waypoints_cube_lift.json")
# Used only when the config cannot be read; it is the "approach" waypoint of that file.
FALLBACK_NEUTRAL_POS = np.array([0.474, 0.0435, 0.91])
WORKSPACE_RADIUS_M = 0.15     # a hand-relative offset is never allowed beyond this
MIN_TOOL_Z_M = 0.80           # table plane; the verified grasp depth is z = 0.81
TRACK_TOL_M = 0.005           # acceptance tolerance for the positional servo
TRACK_TOL_DEG = 2.0
NEUTRAL_TOL_M = 0.002         # the verified neutral pose solves to < 1 mm residual

HAND_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


def load_neutral_pose(path=WAYPOINTS_PATH):
    """Neutral (position, rotation) of the mapping: the cube-lift "approach" waypoint.

    The home keyframe qpos is a folded pose, and its FK makes a poor neutral once the grasp
    orientation is imposed on it: measured on this scene a 4 cm move along +X from there is
    unsolvable (2.77 mm best residual), while the approach waypoint solves in all 6
    directions. It is also the pose experiments/pose_replay.py starts the lift from.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            waypoints = json.load(handle)["waypoints"]
        picked = next((w for w in waypoints if w.get("label") == "approach"), waypoints[0])
        pos = np.asarray(picked["pos"], dtype=float)
        rot = rotvec_to_rotmat(np.asarray(picked["rotvec"], dtype=float))
        return pos, rot
    except (OSError, KeyError, IndexError, ValueError, TypeError):
        return FALLBACK_NEUTRAL_POS.copy(), np.array(R_GRASP, dtype=float)


def angle_between_deg(rot_a, rot_b):
    cos = (np.trace(np.asarray(rot_a, dtype=float).T @ np.asarray(rot_b, dtype=float)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def clamp_to_workspace(target, home, radius=WORKSPACE_RADIUS_M, min_z=MIN_TOOL_Z_M):
    """Keep a hand-driven offset inside a sphere around home and above the table guard."""
    target = np.asarray(target, dtype=float)
    delta = target - np.asarray(home, dtype=float)
    length = float(np.linalg.norm(delta))
    clamped = False
    if length > radius:
        delta *= radius / length
        clamped = True
    out = np.asarray(home, dtype=float) + delta
    if out[2] < min_z:
        out = out.copy()
        out[2] = min_z
        clamped = True
    return out, clamped


class HandTeleop:
    """Hand frame -> clamped base-frame target -> IK -> position actuator ctrl."""

    def __init__(self, replayer, gain=1.0, lost_timeout=0.5, radius=WORKSPACE_RADIUS_M,
                 min_z=MIN_TOOL_Z_M, neutral=None):
        self.rp = replayer
        self.gain = float(gain)
        self.lost_timeout = float(lost_timeout)
        self.radius = float(radius)
        self.min_z = float(min_z)
        home_q = np.array(replayer.home_qpos(), dtype=float)
        # Neutral from the verified waypoint file, never from the live state: the mapping
        # reference must not depend on when this object was constructed or where the arm is.
        if neutral is None:
            neutral = load_neutral_pose()
        self.home_pos = np.array(neutral[0], dtype=float)
        self.home_rot = np.array(neutral[1], dtype=float)
        self.home_q = home_q
        self.q = home_q.copy()
        self.q_cmd = home_q.copy()
        self.pos_cmd = np.array(self.home_pos, dtype=float)
        self.rot_cmd = np.array(self.home_rot, dtype=float)
        self.g_cmd = 1.0
        self.ref_anchor = None
        self.ref_rot_inv = None
        self.lost_since = None
        self.frozen = False
        self.calibrated = False
        self.frames_ok = 0
        self.frames_lost = 0
        self.clamps = 0
        self.ik_misses = 0

    def prepare(self, g=1.0):
        """Snap the plant onto the neutral command so the first tick does not jump.

        Without this the arm would swing to the neutral grasp pose on the first frame even
        though the hand has not moved yet. Returns False when the neutral pose is
        unreachable, and the caller starts from the keyframe instead.
        """
        res = self.rp.ik.solve(self.home_pos, self.home_rot, self.home_q)
        if not res.converged:
            res = self.rp.ik.solve_robust(self.home_pos, self.home_rot, self.home_q)
        if not res.converged:
            return False
        self.q = np.array(res.qpos, dtype=float)
        self.q_cmd = self.q.copy()
        self.pos_cmd = np.array(self.home_pos, dtype=float)
        self.rot_cmd = np.array(self.home_rot, dtype=float)
        self.rp.teleport(self.q, g)
        return True

    def calibrate(self, anchor, R_wrist):
        self.ref_anchor = np.array(anchor, dtype=float)
        self.ref_rot_inv = np.array(R_wrist, dtype=float).T
        self.calibrated = True

    def status(self, ok, now, reason="", clamped=False, target=None, res=None):
        lost_s = 0.0 if self.lost_since is None else float(now - self.lost_since)
        return {
            "ok": ok,
            "frozen": self.frozen,
            "stale": self.frozen and lost_s > self.lost_timeout,
            "lost_s": lost_s,
            "reason": reason,
            "clamped": clamped,
            "target": self.pos_cmd if target is None else target,
            "g": self.g_cmd,
            "pos_err_mm": 0.0 if res is None else 1000.0 * float(res.pos_err),
            "ang_err_deg": 0.0 if res is None else float(res.ang_err_deg),
        }

    def _hold(self, now, reason, clamped=False, target=None):
        if self.lost_since is None:
            self.lost_since = float(now)
        self.frozen = True
        self.frames_lost += 1
        self.rp.apply(self.q_cmd, self.g_cmd)      # keep pushing the last good command
        return self.status(False, now, reason, clamped=clamped, target=target)

    def step(self, frame, now):
        """One control tick: command the arm, or hold still if the hand is unusable."""
        if frame is None or not frame.valid:
            return self._hold(now, "no usable hand" if frame is None else frame.reason)
        ee8, info = hand_to_ee(frame.keypoints, frame.R_wrist)   # identity camera->base
        if not info["valid"]:
            return self._hold(now, "hand_to_ee rejected the frame")
        if not self.calibrated:
            self.calibrate(info["anchor_base"], frame.R_wrist)

        target = self.home_pos + self.gain * (info["anchor_base"] - self.ref_anchor)
        target_rot = self.home_rot @ (self.ref_rot_inv @ frame.R_wrist)
        target, clamped = clamp_to_workspace(target, self.home_pos, self.radius, self.min_z)
        res = self.rp.ik.solve(target, target_rot, self.q)
        if not res.converged:
            # a single warm-start miss is usually a bad seed, not an unreachable target; the
            # same restart fallback Replayer.solve_frame uses keeps one miss from freezing us
            res = self.rp.ik.solve_robust(target, target_rot, self.q)
        if not res.converged:
            self.ik_misses += 1
            return self._hold(now, "IK 未收敛 |dp| = %.2f mm / %.2f deg"
                              % (1000.0 * res.pos_err, res.ang_err_deg),
                              clamped=clamped, target=target)

        self.q = np.array(res.qpos, dtype=float)
        self.q_cmd = self.q.copy()
        self.pos_cmd = target
        self.rot_cmd = target_rot
        self.g_cmd = float(info["g_norm"])
        self.rp.apply(self.q, self.g_cmd)
        self.frames_ok += 1
        self.lost_since = None
        self.frozen = False
        if clamped:
            self.clamps += 1
        return self.status(True, now, clamped=clamped, res=res)

    def reset_calibration(self):
        self.ref_anchor = None
        self.ref_rot_inv = None
        self.calibrated = False


# --------------------------------------------------------------------------- #
# live teleoperation
# --------------------------------------------------------------------------- #
def draw_preview(cv2, sample, status, teleop):
    img = sample.frame
    if status["stale"]:
        color = (0, 0, 255)
    elif status["frozen"]:
        color = (0, 165, 255)
    else:
        color = (0, 255, 0)
    if sample.found:
        pts = [(int(lm[0] * sample.width), int(lm[1] * sample.height)) for lm in sample.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(img, p, 3, (255, 255, 0), -1)
    # ASCII only: cv2.putText cannot draw CJK and would render tofu boxes
    label = "LOST" if status["stale"] else ("HOLD" if status["frozen"] else "TRACK")
    cv2.putText(img, "%s  g=%.2f" % (label, teleop.g_cmd), (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img, "pos %.3f %.3f %.3f" % tuple(teleop.pos_cmd), (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if not status["ok"]:
        cv2.putText(img, status["reason"][:46], (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1)
    return img


def run_live(args):
    print("=" * 78)
    print("手部遥操作：camera %d -> mediapipe -> hand_to_ee -> IK -> 位置伺服"
          % int(args.camera))
    print("gain %.2f | %.0f Hz | lost-timeout %.2f s | preview %s"
          % (args.gain, args.control_hz, args.lost_timeout, "on" if args.preview else "off"))
    print("=" * 78)
    try:
        rp = Replayer(control_hz=float(args.control_hz))
    except ReplayError as exc:
        print("[FAIL] 场景加载失败：%s" % exc)
        return 2

    source = CameraHandSource(args.camera, args.width, args.height, keep_frame=bool(args.preview))
    try:
        source.start()
    except (MissingDependency, CameraError) as exc:
        print("[FAIL] %s" % exc)
        rp.close()
        return 1
    print("  opencv %s / mediapipe %s" % (source.cv2_version, source.version))

    try:
        viewer = mujoco.viewer.launch_passive(rp.model, rp.data)
    except Exception as exc:                                    # noqa: BLE001
        print("[FAIL] 打不开 MuJoCo passive viewer（%s）。\n"
              "  没有图形界面时请用 --probe 或 --selftest。" % exc)
        source.close()
        rp.close()
        return 3

    teleop = HandTeleop(rp, gain=args.gain, lost_timeout=args.lost_timeout)
    if not teleop.prepare():
        print("  警告: 中性位姿（home 位置 + 抓取朝向）IK 不可达，直接从 home 关节位开始。")
    smoother = wrist_frame.FrameSmoother(alpha=0.35)
    period = 1.0 / float(args.control_hz)
    last_tick = 0.0
    last_report = 0.0
    seq = -1
    rc = 0
    print("  运行中：关闭窗口或 Ctrl-C 退出。当前手姿将成为中性位。")
    try:
        while viewer.is_running():
            sample = source.wait_for_sample(seq, timeout=0.2)
            if sample is None:
                if source.error:
                    print("[FAIL] %s" % source.error)
                    rc = 1
                    break
                if not viewer.is_running():
                    break
                continue
            seq = sample.seq
            now = time.monotonic()
            if now - last_tick < period:
                continue                       # drop the frame, keep the tick rate bounded
            last_tick = now
            status = teleop.step(build_frame(sample, smoother), now)
            rp.step()
            viewer.sync()
            if args.preview and sample.frame is not None:
                import cv2                                    # already imported by the source
                cv2.imshow("hand_teleop", draw_preview(cv2, sample, status, teleop))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            if now - last_report >= 1.0:
                last_report = now
                state = "LOST" if status["stale"] else ("HOLD" if status["frozen"] else "TRACK")
                print("  %-5s pos %.3f,%.3f,%.3f g %.2f | ok %d lost %d clamp %d ik-miss %d"
                      % (state, teleop.pos_cmd[0], teleop.pos_cmd[1], teleop.pos_cmd[2],
                         teleop.g_cmd, teleop.frames_ok, teleop.frames_lost, teleop.clamps,
                         teleop.ik_misses))
    except KeyboardInterrupt:
        print("\n  (Ctrl-C)")
    finally:
        try:
            viewer.close()
        except Exception:                                  # noqa: BLE001
            pass
        source.close()
        if args.preview:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:                              # noqa: BLE001
                pass
        rp.close()

    print("  统计: 指令 %d 帧, 冻结 %d 帧, 钳制 %d 次, IK 失败 %d 次"
          % (teleop.frames_ok, teleop.frames_lost, teleop.clamps, teleop.ik_misses))
    return rc


# --------------------------------------------------------------------------- #
# self test: the whole chain on synthetic hands, no camera and no window
# --------------------------------------------------------------------------- #
def make_hand(anchor=(0.025, -0.0025, 0.0), R=None, d_t=0.06):
    """Synthetic 21-point hand in metres whose thenar anchor sits exactly at 'anchor'.

    Local layout, chosen so wrist_frame's construction is well posed: the wrist sits
    behind the knuckle line, index MCP at +Y and pinky MCP at -Y make
    cross(wrist->index, wrist->pinky) point along +Z, so the reconstructed R_wrist is
    exactly the requested R. The thumb tip is placed d_t away from the index tip because
    that distance is what hand_to_ee reads as the gripper opening.
    """
    rot = np.eye(3) if R is None else np.asarray(R, dtype=float)
    local = np.zeros((wrist_frame.NUM_LANDMARKS, 3))
    local[0] = (-0.055, 0.000, 0.0)          # wrist
    local[1] = (-0.028, 0.022, 0.0)          # thumb CMC
    local[2] = (-0.004, 0.030, 0.0)          # thumb MCP
    local[3] = (-0.010, 0.030, 0.0)          # thumb IP
    local[5] = (0.060, -0.035, 0.0)          # index MCP
    local[6] = (0.095, -0.035, 0.0)
    local[7] = (0.115, -0.035, 0.0)
    local[8] = (0.130, -0.035, 0.0)          # index tip
    local[9] = (0.060, 0.000, 0.0)           # middle MCP
    local[10] = (0.095, 0.000, 0.0)
    local[11] = (0.115, 0.000, 0.0)
    local[12] = (0.128, 0.000, 0.0)
    local[13] = (0.058, 0.016, 0.0)          # ring
    local[14] = (0.090, 0.016, 0.0)
    local[15] = (0.110, 0.016, 0.0)
    local[16] = (0.122, 0.016, 0.0)
    local[17] = (0.055, 0.035, 0.0)          # pinky MCP
    local[18] = (0.085, 0.035, 0.0)
    local[19] = (0.103, 0.035, 0.0)
    local[20] = (0.114, 0.035, 0.0)
    thumb_dir = np.array([-0.6, 0.8, 0.0])          # unit length, points at the thumb base
    local[4] = local[8] + d_t * thumb_dir    # thumb tip
    anchor_local = 0.5 * (local[3] + local[5])
    return local @ rot.T + (np.asarray(anchor, dtype=float) - rot @ anchor_local)


def to_landmarks(points, width, height):
    """Metric points -> the normalized landmark tuples MediaPipe would report."""
    px = np.asarray(points, dtype=float)
    return [(px[i, 0] / width + 0.5, px[i, 1] / height + 0.5, px[i, 2] / width)
            for i in range(wrist_frame.NUM_LANDMARKS)]


def selftest():
    print("=" * 78)
    print("hand teleop selftest: adapter -> hand_to_ee -> mapping -> IK -> ctrl -> mj_step")
    print("synthetic hands only; no camera, no cv2/mediapipe, no window")
    print("=" * 78)
    failures = []

    def check(name, condition, detail=""):
        print("  [%s] %s%s" % ("PASS" if condition else "FAIL", name,
                               "" if not detail else "   " + detail))
        if not condition:
            failures.append(name)

    width, height = 640, 480
    anchor0 = np.array([0.025, -0.0025, 0.0])

    def frame_of(hand):
        return wrist_frame.build(to_landmarks(hand, width, height), (width, height), hand)

    # -- 1. adapter --------------------------------------------------------- #
    print("\n  [adapter]")
    hand = make_hand()
    frame = frame_of(hand)
    check("adapter rebuilds the metric cloud",
          frame.valid and np.allclose(frame.keypoints, hand, atol=1e-9),
          "max|d| = %.2e m" % (np.max(np.abs(frame.keypoints - hand)) if frame.valid else np.nan))
    check("adapter rebuilds R_wrist = I", np.allclose(frame.R_wrist, np.eye(3), atol=1e-12))
    check("scale from world landmarks is 1.0 for a metric cloud", np.isclose(frame.scale, 1.0))
    fallback = wrist_frame.build(to_landmarks(hand, width, height), (width, height), None)
    palm_span = float(np.linalg.norm(hand[wrist_frame.INDEX_MCP_IDX]
                                     - hand[wrist_frame.PINKY_MCP_IDX]))
    check("scale without world landmarks uses the palm-width prior",
          np.isclose(fallback.scale, wrist_frame.PALM_WIDTH_M / palm_span, rtol=1e-9),
          "%.4f m/px = %.3f m / %.5f m" % (fallback.scale, wrist_frame.PALM_WIDTH_M, palm_span))
    R_test = rotvec_to_rotmat(np.array([0.3, -0.2, 0.9]))
    rotated = frame_of(make_hand(anchor=anchor0, R=R_test))
    check("R_wrist is rotation-equivariant", np.allclose(rotated.R_wrist, R_test, atol=1e-12),
          "max|dR| = %.1e" % np.max(np.abs(rotated.R_wrist - R_test)))
    collapsed = make_hand()
    # put the pinky MCP on the wrist->index-MCP bone so the palm normal is undefined
    collapsed[wrist_frame.PINKY_MCP_IDX] = 0.5 * (collapsed[wrist_frame.WRIST_IDX]
                                                 + collapsed[wrist_frame.INDEX_MCP_IDX])
    check("collinear palm bones -> invalid frame", not frame_of(collapsed).valid)
    check("NaN landmarks -> invalid frame",
          not wrist_frame.build([(np.nan, 0.1, 0.0)] * 21, (width, height)).valid)

    # -- 2. hand_to_ee handshake -------------------------------------------- #
    print("\n  [hand_to_ee]")
    ee8, info = hand_to_ee(frame.keypoints, frame.R_wrist)
    check("hand_to_ee accepts the adapter output", info["valid"] and np.all(np.isfinite(ee8)))
    check("commanded point is the thenar anchor",
          np.allclose(info["anchor_base"], anchor0, atol=1e-12),
          "anchor = [%.4f, %.4f, %.4f]" % tuple(info["anchor_base"]))
    nan_frame = wrist_frame.WristFrame(keypoints=np.full((21, 3), np.nan), R_wrist=np.eye(3),
                                      scale=1.0, valid=True)
    nan_ee, nan_info = hand_to_ee(nan_frame.keypoints, nan_frame.R_wrist)
    check("NaN hand is rejected by hand_to_ee, not zero-filled",
          not nan_info["valid"] and np.all(np.isnan(nan_ee)))

    # -- 3. closed loop against the plant ----------------------------------- #
    print("\n  [plant]")
    rp = Replayer(control_hz=50.0)
    teleop = HandTeleop(rp, gain=1.0, lost_timeout=0.1)
    neutral_pos, neutral_rot = load_neutral_pose()
    check("neutral pose is the verified cube-lift approach waypoint",
          np.allclose(teleop.home_pos, neutral_pos, atol=1e-12)
          and angle_between_deg(teleop.home_rot, neutral_rot) < 1e-9,
          "p = [%.4f, %.4f, %.4f]" % tuple(neutral_pos))
    check("grasp orientation equals the config rotvec [0, pi, 0]",
          angle_between_deg(R_GRASP, neutral_rot) < 1e-9)
    rp.teleport(rp.home_qpos(), 1.0)            # move the plant; the neutral must not follow
    check("neutral does not depend on the live robot state",
          np.allclose(HandTeleop(rp).home_pos, neutral_pos, atol=1e-12))
    prepared = teleop.prepare()
    start_pos, start_rot = rp.tool_pose()
    check("startup snaps the plant onto the neutral command",
          bool(prepared) and float(np.linalg.norm(start_pos - neutral_pos)) < NEUTRAL_TOL_M
          and angle_between_deg(start_rot, neutral_rot) < 1.0,
          "tool at [%.4f, %.4f, %.4f], %.2f deg off the neutral rotation"
          % (tuple(start_pos) + (angle_between_deg(start_rot, neutral_rot),)))

    clock = [0.0]

    def tick(fr):
        clock[0] += 0.02
        status = teleop.step(fr, clock[0])
        rp.step()
        return status

    status = tick(frame)
    check("first usable frame is neutral (no jump at startup)",
          status["ok"] and np.linalg.norm(teleop.pos_cmd - neutral_pos) < 1e-9
          and np.allclose(teleop.rot_cmd, neutral_rot)
          and float(np.linalg.norm(rp.tool_pose()[0] - neutral_pos)) < NEUTRAL_TOL_M,
          "|dp| = %.1e m" % np.linalg.norm(teleop.pos_cmd - neutral_pos))

    # assert on the mapping output so the check never depends on an IK seed of a second plant
    gain_teleop = HandTeleop(rp, gain=2.0, lost_timeout=0.1)
    gain_teleop.step(frame, 0.0)
    moved = make_hand(anchor=anchor0 + np.array([0.02, -0.01, 0.0]))
    g_status = gain_teleop.step(frame_of(moved), 0.02)
    delta_cmd = np.asarray(g_status["target"], dtype=float) - neutral_pos
    check("position increment = gain * hand displacement",
          bool(g_status["ok"]) and np.allclose(delta_cmd, 2.0 * np.array([0.02, -0.01, 0.0]),
                                               atol=1e-9),
          "|d| = %.4f m = gain 2 x %.4f m"
          % (np.linalg.norm(delta_cmd), np.linalg.norm([0.02, -0.01, 0.0])))

    track_hand = make_hand(anchor=anchor0 + np.array([0.03, 0.02, 0.01]))
    for _ in range(60):
        tick(frame_of(track_hand))
    moved_m = float(np.linalg.norm(teleop.pos_cmd - neutral_pos))
    pos_now, rot_now = rp.tool_pose()
    tracking_m = float(np.linalg.norm(pos_now - teleop.pos_cmd))
    print("  末端位置误差 %.2f mm (%.5f m) / 姿态误差 %.2f deg"
          % (1000.0 * tracking_m, tracking_m,
             angle_between_deg(rot_now, teleop.rot_cmd)))
    check("EE tracks the commanded position", tracking_m < TRACK_TOL_M and moved_m > 0.02,
          "%.2f mm < %.0f mm after a %.1f mm move"
          % (1000.0 * tracking_m, 1000.0 * TRACK_TOL_M, 1000.0 * moved_m))

    R_hand = rotvec_to_rotmat(np.array([0.0, 0.0, np.deg2rad(20.0)]))
    for _ in range(60):
        check_frame = frame_of(make_hand(anchor=anchor0, R=R_hand))
        clock[0] += 0.02
        teleop.step(check_frame, clock[0])
        rp.step()
    check("orientation command = R_grasp @ (R_ref^T R_now)",
          np.allclose(teleop.rot_cmd, neutral_rot @ R_hand, atol=1e-9),
          "%.2f deg about Z" % angle_between_deg(teleop.rot_cmd, neutral_rot))
    pos_now, rot_now = rp.tool_pose()
    tracking_deg = angle_between_deg(rot_now, teleop.rot_cmd)
    check("EE tracks the commanded orientation", tracking_deg < TRACK_TOL_DEG,
          "%.2f deg < %.1f deg" % (tracking_deg, TRACK_TOL_DEG))

    # -- 4. gripper --------------------------------------------------------- #
    print("\n  [gripper]")
    for d_t, expect_open in ((0.090, True), (0.025, False)):
        clock[0] += 0.02
        st = teleop.step(frame_of(make_hand(anchor=anchor0, d_t=d_t)), clock[0])
        rp.step()
        ctrl = float(rp.data.ctrl[6])
        check("d_t = %.3f m -> g = %.3f, ctrl = %.3f (%s)"
              % (d_t, st["g"], ctrl, "open" if expect_open else "pinch"),
              np.isclose(ctrl, g_to_ctrl(st["g"])) and st["g"] > 0.0)

    # -- 5. lost hand / NaN / clamping -------------------------------------- #
    print("\n  [safety]")
    # settle on a still command first, so the hold test measures the hold and not the servo
    # still catching up with where the previous command put it
    for _ in range(30):
        tick(frame_of(make_hand(anchor=anchor0, R=R_hand)))
    ctrl_before = np.array(rp.data.ctrl, dtype=float)
    q_before = teleop.q_cmd.copy()
    pos_before = rp.tool_pose()[0]
    lost = wrist_frame.invalid_frame("simulated loss")
    frozen, poses = [], []
    for _ in range(20):
        clock[0] += 0.02
        frozen.append(teleop.step(lost, clock[0]))
        rp.step()
        poses.append(rp.tool_pose()[0].copy())
    check("lost hand freezes the joint command",
          all(s["frozen"] and not s["ok"] for s in frozen)
          and np.array_equal(teleop.q_cmd, q_before),
          "q_cmd unchanged over 20 ticks")
    # the gravity feedforward reads the whole qpos, so ctrl may wiggle at the 1e-7 level while
    # the gripper jaw keeps settling; the joint command itself must not move at all
    check("frozen actuators keep the last command",
          np.allclose(rp.data.ctrl[:6], ctrl_before[:6], atol=1e-5)
          and float(rp.data.ctrl[6]) == float(ctrl_before[6]),
          "max|dctrl| = %.1e" % np.max(np.abs(rp.data.ctrl[:6] - ctrl_before[:6])))
    settle = float(np.linalg.norm(poses[9] - pos_before))
    hold = float(np.linalg.norm(poses[-1] - poses[9]))
    check("frozen arm settles and then holds",
          settle < 2e-3 and hold < 2e-4,
          "settle %.3f mm in 0.2 s, then %.3f mm over the next 0.2 s"
          % (1000.0 * settle, 1000.0 * hold))
    check("frozen longer than lost-timeout is flagged",
          frozen[-1]["stale"] and frozen[-1]["lost_s"] > teleop.lost_timeout,
          "lost_s = %.2f s > %.2f s" % (frozen[-1]["lost_s"], teleop.lost_timeout))
    clock[0] += 0.02
    nan_status = teleop.step(nan_frame, clock[0])
    rp.step()
    check("NaN keypoints never reach the actuators",
          nan_status["frozen"] and np.array_equal(teleop.q_cmd, q_before)
          and np.allclose(rp.data.ctrl[:6], ctrl_before[:6], atol=1e-5)
          and np.all(np.isfinite(rp.data.ctrl)) and np.all(np.isfinite(rp.data.qpos))
          and np.all(np.isfinite(rp.data.qvel)),
          "q_cmd held, ctrl/qpos/qvel finite")

    clock[0] += 0.02
    far = np.array([0.8, -0.6, 0.5])
    st = teleop.step(frame_of(make_hand(anchor=anchor0 + far)), clock[0])
    rp.step()
    clamp_delta = np.asarray(st["target"], dtype=float) - neutral_pos
    check("out-of-workspace target is clamped to the sphere",
          np.linalg.norm(clamp_delta) <= WORKSPACE_RADIUS_M + 1e-9 and st["clamped"],
          "|d| = %.4f m <= %.3f m" % (np.linalg.norm(clamp_delta), WORKSPACE_RADIUS_M))
    floor_teleop = HandTeleop(rp, gain=1.0, lost_timeout=0.1, radius=10.0)
    floor_teleop.step(frame_of(make_hand(anchor=anchor0)), 0.0)
    st = floor_teleop.step(frame_of(make_hand(anchor=anchor0 + np.array([0.0, 0.0, -0.4]))), 0.02)
    check("tool never goes below the table guard",
          st["target"][2] >= MIN_TOOL_Z_M - 1e-12 and st["clamped"],
          "z = %.4f m >= %.2f m" % (st["target"][2], MIN_TOOL_Z_M))

    rp.close()
    print("\n" + "=" * 78)
    if failures:
        print("FAILED %d check(s):" % len(failures))
        for name in failures:
            print("  - %s" % name)
        return 1
    print("ALL CHECKS PASSED")
    return 0


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="摄像头 -> MediaPipe 21 点 -> hand_to_ee -> IK -> MuJoCo 位置伺服",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV VideoCapture 索引")
    parser.add_argument("--gain", type=float, default=1.0, help="手部位移 -> 末端位移的比例")
    parser.add_argument("--lost-timeout", type=float, default=0.5,
                        help="手部丢失多久后标红（期间一直冻结）")
    parser.add_argument("--control-hz", type=float, default=30.0, help="控制频率")
    parser.add_argument("--width", type=int, default=640, help="摄像头请求宽度")
    parser.add_argument("--height", type=int, default=480, help="摄像头请求高度")
    parser.add_argument("--probe", action="store_true", help="只验证摄像头链路，不动机器人")
    parser.add_argument("--probe-seconds", type=float, default=3.0, help="--probe 采集时长")
    parser.add_argument("--selftest", action="store_true",
                        help="不开摄像头，用合成手跑完整链路自检")
    parser.add_argument("--preview", dest="preview", action="store_true", default=True,
                        help="显示 OpenCV 预览窗口")
    parser.add_argument("--no-preview", dest="preview", action="store_false",
                        help="不显示预览窗口")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    if args.probe:
        return probe(camera_index=args.camera, seconds=args.probe_seconds,
                     preview=args.preview, width=args.width, height=args.height)
    return run_live(args)


if __name__ == "__main__":
    rc = main()
    # GLFW teardown after a passive viewer session can segfault on some drivers; the
    # results are already on stdout, so skip interpreter shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)

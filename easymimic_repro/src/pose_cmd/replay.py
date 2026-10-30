"""Waypoint table executor: JAKA Zu3 IK + gravity feedforward + gripper + frames.

One frame = one control tick of meta.control_hz: solve the pose, write ctrl, step the plant
for exactly 1/control_hz seconds. The table is replayed, not tracked by a servo loop, so the
only feedback in the path is the plant itself.

Why the replay starts by teleporting the arm to the first waypoint instead of driving it
from the keyframe pose: driving there is a motion-planning problem (the keyframe pose is the
upward fold, the grasp pose points down and yawed), and letting it share the table would make
the first row do double duty. The teleport is reported (lead-in residual) so it is visible.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mujoco  # noqa: E402

from src.ee_retarget.ik_jaka_zu3 import (ANG_TOL_DEG, POS_TOL_M, JakaZu3IK,
                                         rotvec_to_rotmat)  # noqa: E402

SCENE_XML = os.path.join(REPO_ROOT, "assets", "jaka_zu3", "scene_cube_bowl.xml")
DEFAULT_TABLE = os.path.join(REPO_ROOT, "configs", "waypoints_cube_lift.json")
DEFAULT_FRAME_DIR = os.path.join(REPO_ROOT, "out", "t9_frames")

# The only place the g -> ctrl mapping lives. joint_7 is a hinge with range 0..0.9 rad and
# ctrl = q, so g = 1 (open) must land exactly on 0.9 and g = 0 (closed) on 0.
GRIPPER_CONFIG = {"joint": "joint_7", "ctrl_per_g": 0.9, "open_at_g": 1.0}

# Acceptance numbers. Hard-coded on purpose: a run that misses them is reported, not tuned.
CRITERIA = {
    "min_rise_m": 0.08,          # cube lifted at least this far above its rest height
    "max_slip_m": 0.005,         # horizontal cube drift inside the jaws after the grasp
    "min_pad_contacts": 1,       # contacts at the end, per pad
    "max_tool_offset_m": 0.040,  # cube centre this far from tool0 means it left the jaws
}

RENDER_EVERY = 5
RENDER_WIDTH = 760
RENDER_HEIGHT = 560

FAILS = []


def check(cond, msg):
    print("  [PASS] " + msg if cond else "  [FAIL] " + msg)
    if not cond:
        FAILS.append(msg)
    return bool(cond)


class ReplayError(RuntimeError):
    """The executor refuses to continue: an IK miss, a limit-hugging solution, a bad scene."""


@dataclass
class SegmentReport:
    label: str
    frames: int = 0
    iters: int = 0
    pos_err: float = 0.0
    ang_err_deg: float = 0.0
    max_pos_err: float = 0.0
    fallbacks: int = 0


@dataclass
class Report:
    segments: list = field(default_factory=list)
    cube_start: np.ndarray = None
    cube_end: np.ndarray = None
    cube_grasp: np.ndarray = None
    pad_contacts: dict = field(default_factory=dict)
    tool_offset: float = 0.0
    frames: int = 0
    sim_time: float = 0.0
    wall_s: float = 0.0
    frames_png: list = field(default_factory=list)
    ff_drop: dict = field(default_factory=dict)

    @property
    def rise(self):
        return float(self.cube_end[2] - self.cube_start[2])

    @property
    def slip(self):
        return float(np.linalg.norm(self.cube_end[:2] - self.cube_grasp[:2]))


def g_to_ctrl(g, cfg=None):
    """g in [0, 1] -> joint_7 ctrl, through the one configuration block."""
    cfg = GRIPPER_CONFIG if cfg is None else cfg
    return float(g) / float(cfg["open_at_g"]) * float(cfg["ctrl_per_g"])


def angle_between_deg(rot_a, rot_b):
    """Angle of rot_a^T rot_b in degrees. atan2 keeps precision near 0 and near pi, where
    acos(trace) loses it. This metric is used instead of a rotvec round-trip: a rotvec has
    two representatives at pi and the conversion is where the accuracy would be spent, so
    the check stays on matrices (ik_jaka_zu3.rotmat_to_rotvec does handle pi now, see
    ik_jaka_zu3.py:131-179, but nothing here depends on that)."""
    rel = np.asarray(rot_a, dtype=float).T @ np.asarray(rot_b, dtype=float)
    return math.degrees(math.atan2(0.5 * float(np.linalg.norm(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])),
        0.5 * (float(np.trace(rel)) - 1.0)))


def write_png(path, rgb):
    """Plain PNG writer: the runtime has mujoco + numpy only, no PIL."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    blob = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)


class Replayer:
    """Executes a Trajectory on the scene. Owns the plant, the IK, and the frames."""

    def __init__(self, scene_path=SCENE_XML, control_hz=50.0, use_ff=True,
                 frame_dir=DEFAULT_FRAME_DIR, render_every=RENDER_EVERY):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(scene_path))
        self.data = mujoco.MjData(self.model)
        # History: the class default used to be the upstream third_party model, which does not
        # load under mujoco 3.16 (its flexcomp rejects joint damping, "flex vertex/node body
        # 'rag_0' cannot have joint damping"). DEFAULT_XML_PATH is now assets/jaka_zu3/robot.xml
        # (ik_jaka_zu3.py:66), but the IK is still built on the SCENE model: the plant steps that
        # model, so joint ids, dof ids and limits must come from the same mjModel.
        # tcp_offset stays None: site tool0 sits at the pinch centre (robot.xml:142), so the
        # site pose IS the TCP and the 32.5 mm lever of the retired placeholder is gone.
        self.ik = JakaZu3IK(xml_path=os.path.abspath(scene_path), site_name="tool0")
        self.control_hz = float(control_hz)
        self.use_ff = bool(use_ff)
        self.timestep = float(self.model.opt.timestep)
        self.substeps = max(1, int(round((1.0 / self.control_hz) / self.timestep)))
        self.kp = np.array(self.model.actuator_gainprm[:6, 0], dtype=float)
        self.scratch = mujoco.MjData(self.model)
        self.site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "tool0")
        self.cube_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.pad_geoms = {"gripper_pad_fixed": self._id(mujoco.mjtObj.mjOBJ_GEOM,
                                                      "gripper_pad_fixed"),
                          "gripper_pad_moving": self._id(mujoco.mjtObj.mjOBJ_GEOM,
                                                        "gripper_pad_moving")}
        self.gripper_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_CONFIG["joint"])
        self.key_id = self._id(mujoco.mjtObj.mjOBJ_KEY, "home")
        self.frame_dir = frame_dir
        self.render_every = int(render_every)
        self._renderer = None
        self._cam = None
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)

    def _id(self, obj_type, name):
        found = mujoco.mj_name2id(self.model, obj_type, name)
        if found < 0:
            raise ReplayError("%s: the scene has no %s named %r"
                              % (SCENE_XML, str(obj_type), name))
        return int(found)

    def home_qpos(self):
        return np.array(self.model.key_qpos[self.key_id, :6], dtype=float)

    def tool_pose(self):
        pos = np.array(self.data.site_xpos[self.site_id], dtype=float)
        rot = np.array(self.data.site_xmat[self.site_id], dtype=float).reshape(3, 3)
        return pos, rot

    def cube_pos(self):
        return np.array(self.data.xpos[self.cube_body], dtype=float)

    def feedforward(self, q6):
        """qfrc_bias / kp: a position actuator carries no gravity model of its own, so
        without this the tool sags by whatever gravity torque that pose needs."""
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qpos[:6] = q6
        self.scratch.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.scratch)
        return np.array(self.scratch.qfrc_bias[:6], dtype=float) / self.kp

    def apply(self, q6, g):
        self.data.ctrl[:6] = q6 + (self.feedforward(q6) if self.use_ff else 0.0)
        self.data.ctrl[6] = g_to_ctrl(g)

    def solve_frame(self, label, pos, rotvec, q_seed):
        """One IK solve. A miss, or a solution pressed against a joint limit, stops the run:
        continuing with a limit-hugging answer would look like success in the numbers."""
        target_pos = np.asarray(pos, dtype=float).reshape(3)
        target_rot = rotvec_to_rotmat(np.asarray(rotvec, dtype=float).reshape(3))
        res = self.ik.solve(target_pos, target_rot, q_seed)
        fallback = 0
        if not res.converged:
            fallback = 1
            res = self.ik.solve_robust(target_pos, target_rot, q_seed)
        if not res.converged:
            raise ReplayError("waypoint %s: IK not converged, |dp|=%.4e m, |dR|=%.4f deg"
                              % (label, res.pos_err, res.ang_err_deg))
        if not self.ik.within_limits(res.qpos):
            raise ReplayError("waypoint %s: IK solution violates a joint limit "
                              "(q6 = %s), |dp|=%.4e m, |dR|=%.4f deg"
                              % (label, np.array2string(res.qpos, precision=4),
                                 res.pos_err, res.ang_err_deg))
        res.fallback = fallback
        return res

    def teleport(self, q6, g):
        """Place the arm at the first waypoint instead of driving it there (see module doc)."""
        self.data.qpos[:6] = q6
        self.data.qvel[:] = 0.0
        self.data.qpos[6] = g_to_ctrl(g)
        self.data.qpos[7:] = self.model.key_qpos[self.key_id, 7:]
        mujoco.mj_forward(self.model, self.data)
        self.apply(q6, g)

    def step(self, n=None):
        for _ in range(self.substeps if n is None else n):
            mujoco.mj_step(self.model, self.data)

    def pad_contacts(self):
        counts = {name: 0 for name in self.pad_geoms}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.cube_geom not in pair:
                continue
            for name, geom in self.pad_geoms.items():
                if geom in pair:
                    counts[name] += 1
        return counts


    def render_frame(self, path):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=RENDER_HEIGHT,
                                             width=RENDER_WIDTH)
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, cam)
            # Framed on the grasp column, not on the cube: the lift ends at z = 0.93 and a
            # 0.34 m ball centred at z = 0.86 pushed the jaws out of the top edge.
            cam.lookat[:] = [0.474, 0.0435, 0.885]
            cam.distance = 0.44
            cam.azimuth = 250.0
            cam.elevation = -14.0
            self._cam = cam
        self._renderer.update_scene(self.data, camera=self._cam)
        return write_png(path, self._renderer.render())

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def run(self, trajectory, render=True):
        """Replay the table once. Returns the measured report; raises ReplayError on an IK
        miss so a table that cannot be executed is never reported as a result."""
        rep = Report()
        t_wall0 = time.perf_counter()
        t_sim0 = float(self.data.time)
        first = np.asarray(trajectory.ee8[0], dtype=float)
        res0 = self.solve_frame(trajectory.label[0], first[0:3], first[3:6], self.home_qpos())
        self.teleport(np.array(res0.qpos, dtype=float), float(first[6]))
        pos, rot = self.tool_pose()
        lead = SegmentReport(label="0: teleport -> '%s'" % trajectory.label[0], frames=1,
                             iters=int(res0.iters),
                             pos_err=float(np.linalg.norm(first[0:3] - pos)),
                             ang_err_deg=angle_between_deg(rotvec_to_rotmat(first[3:6]), rot))
        lead.max_pos_err = lead.pos_err
        rep.segments.append(lead)
        rep.cube_start = self.cube_pos()

        # A segment is named after the waypoint it ends at: a row that reads "-> lift" is the
        # motion that arrives at lift, whatever the samples inside it were labelled with.
        destination = {}
        for k, label in enumerate(trajectory.label):
            destination[int(trajectory.seg[k])] = label
        rows = {}
        for k in range(len(trajectory)):
            ee8 = np.asarray(trajectory.ee8[k], dtype=float)
            seg = int(trajectory.seg[k])
            if seg not in rows:
                rows[seg] = SegmentReport(label="%d: -> '%s'" % (seg + 1, destination[seg]))
                rep.segments.append(rows[seg])
            row = rows[seg]
            res = self.solve_frame(trajectory.label[k], ee8[0:3], ee8[3:6],
                                   self.data.qpos[:6].copy())
            self.apply(np.array(res.qpos, dtype=float), float(ee8[6]))
            self.step()
            row.frames += 1
            row.iters += int(res.iters)
            row.fallbacks += int(getattr(res, "fallback", 0))
            pos, rot = self.tool_pose()
            row.pos_err = float(np.linalg.norm(ee8[0:3] - pos))
            row.ang_err_deg = angle_between_deg(rotvec_to_rotmat(ee8[3:6]), rot)
            row.max_pos_err = max(row.max_pos_err, row.pos_err)
            if rep.cube_grasp is None and float(ee8[6]) <= 0.0:
                # Cube pose when the table first asks for a closed gripper: everything the
                # cube does relative to this point is slip inside the jaws.
                rep.cube_grasp = self.cube_pos()
            if render and self.render_every > 0 and k % self.render_every == 0:
                path = os.path.join(self.frame_dir, "t9_f%04d.png" % k)
                try:
                    self.render_frame(path)
                except Exception as exc:  # GL failures are reported, never raised
                    check(False, "render frame %d failed: %s" % (k, exc))
                else:
                    rep.frames_png.append(path)

        rep.cube_end = self.cube_pos()
        rep.pad_contacts = self.pad_contacts()
        pos, rot = self.tool_pose()
        rep.tool_offset = float(np.linalg.norm(rot.T @ (rep.cube_end - pos)))
        rep.frames = len(trajectory)
        rep.sim_time = float(self.data.time) - t_sim0
        rep.wall_s = time.perf_counter() - t_wall0
        self.close()
        return rep


def droop_probe(table, use_ff=True, hold_s=1.0, control_hz=50.0):
    """Hold the first waypoint's pose for hold_s seconds and report where the tool ends up.
    Run twice (feedforward on and off) this is the droop the feedforward removes."""
    rp = Replayer(control_hz=control_hz, use_ff=use_ff, render_every=0)
    wp = table.waypoints[0]
    res = rp.solve_frame(wp.label, wp.ee8[0:3], wp.ee8[3:6], rp.home_qpos())
    rp.teleport(np.array(res.qpos, dtype=float), wp.g)
    for _ in range(max(1, int(round(hold_s * control_hz)))):
        rp.apply(np.array(res.qpos, dtype=float), wp.g)
        rp.step()
    pos, _ = rp.tool_pose()
    target = np.asarray(wp.ee8[0:3], dtype=float)
    rp.close()
    return {"use_ff": bool(use_ff), "pos_err": float(np.linalg.norm(target - pos)),
            "dz": float(pos[2] - target[2])}


def replay_table(table, render=True, control_hz=None, use_ff=True, frame_dir=None):
    """Plan + execute in one call: the path the CLI takes."""
    from src.pose_cmd.waypoints import plan_trajectory
    traj = plan_trajectory(table)
    rp = Replayer(control_hz=control_hz or table.control_hz, use_ff=use_ff,
                  frame_dir=frame_dir or DEFAULT_FRAME_DIR)
    try:
        return traj, rp.run(traj, render=render)
    finally:
        rp.close()

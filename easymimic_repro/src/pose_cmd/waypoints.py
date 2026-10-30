"""8-D waypoint table -> pose trajectory: schema, strict validation, interpolation.

The 8-D action is the one src/ee_retarget/hand_to_ee.py:258-262 emits:

    [pos(3) metres in the robot base frame, rotvec(3) radians, g (0 closed .. 1 open), 0]

Table shape (the executor reads nothing else):

    {"meta": {"frame": "base", "units": "m", "control_hz": 50},
     "waypoints": [{"label": str, "pos": [3], "rotvec": [3], "g": float,
                    "duration_s": float}, ...]}

A caller that already holds 8-D actions may give a waypoint as {"label", "ee8": [8 numbers],
"duration_s"} instead of pos/rotvec/g. The 8th number is reserved and ignored, exactly like
the consumer of the 8-D action ignores it.

Timing rule, so a table can be read without the code: duration_s is the time spent travelling
from THIS waypoint to the next one, and the duration_s of the LAST waypoint is the final
hold. After arriving at an intermediate waypoint the planner repeats that waypoint for
SETTLE_S, so the residual reported at the end of a segment is a steady-state number.

Why the orientation is not interpolated component-wise: a rotation vector is not unique. The
grasp pose is a 180 deg rotation, so (0, +pi, 0) and (0, -pi, 0) are the same matrix, and a
component-wise ramp between them sweeps the tool a full turn. The planner interpolates the
relative rotation instead and always takes the quaternion branch with w >= 0, whose angle is
in [0, pi] -- the shortest arc, whatever the two rotvec representatives are.

Why this module still goes through quaternions instead of calling rotmat_to_rotvec: the
argument used to be that ik_jaka_zu3.rotmat_to_rotvec divides by sin(theta) and collapses a
180 deg rotation to the zero vector, which would silently turn the grasp orientation into the
identity. That is no longer true of the current implementation (quaternion/Shepperd
extraction, see ik_jaka_zu3.py:131-179), so the reason is only about interpolation: a
component-wise ramp between two rotvec representatives sweeps the wrong arc, while a
quaternion slerp with w >= 0 is the shortest one. Nothing to fix here; the note stays so the
next reader does not re-introduce the old claim.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

# Repeated hold after arriving at an intermediate waypoint (seconds). Without it the servo
# is still catching up when the next segment starts and every "end of segment" residual
# would be a transient that says nothing about accuracy.
SETTLE_S = 0.25

# A runaway table (duration_s = 1e9) would allocate forever before the first frame runs.
MAX_FRAMES = 100000
MAX_CONTROL_HZ = 1000.0
SUPPORTED_FRAMES = ("base",)
SUPPORTED_UNITS = ("m",)
WAYPOINT_KEYS = ("label", "pos", "rotvec", "g", "duration_s")
META_KEYS = ("frame", "units", "control_hz")


class WaypointError(ValueError):
    """A table the executor refuses to run. The message is written for the person who
    edited the JSON: it names the waypoint index and the field, never a stack trace."""


@dataclass(frozen=True)
class Waypoint:
    label: str
    ee8: np.ndarray          # (8,) pos, rotvec, g, reserved
    duration_s: float

    @property
    def g(self):
        return float(self.ee8[6])


@dataclass(frozen=True)
class Table:
    source: str
    frame: str
    units: str
    control_hz: float
    waypoints: tuple


@dataclass(frozen=True)
class Trajectory:
    """Control-rate samples of a table, ready to be fed to the plant one at a time."""
    t: np.ndarray            # (N,) seconds since the first frame
    ee8: np.ndarray          # (N, 8)
    label: list              # (N,) waypoint the sample belongs to
    seg: np.ndarray          # (N,) index of the waypoint the sample's segment starts at
    arrive: np.ndarray       # (N,) True where the sample is an exact table waypoint
    control_hz: float

    def __len__(self):
        return int(self.ee8.shape[0])

    @property
    def duration_s(self):
        return float(self.t[-1]) if len(self) else 0.0


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _brief(value):
    if isinstance(value, (list, tuple)):
        return "a list of %d items" % len(value)
    if isinstance(value, dict):
        return "an object with keys %s" % sorted(value)
    return repr(value)[:60]


def _missing(mapping, key, source, where, hint=""):
    if key not in mapping:
        raise WaypointError("%s: %s.%s is missing%s" % (source, where, key, hint))
    return mapping[key]


def _number(value, where, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WaypointError("%s: %s must be a number, got %s"
                            % (source, where, _brief(value)))
    return float(value)


def _finite(value, where, source):
    out = _number(value, where, source)
    if not math.isfinite(out):
        raise WaypointError("%s: %s = %r is not a finite number" % (source, where, value))
    return out


def _vector3(value, where, source):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise WaypointError("%s: %s must be a list of 3 numbers, got %s"
                            % (source, where, _brief(value)))
    return np.array([_finite(v, "%s[%d]" % (where, i), source) for i, v in enumerate(value)])


def parse_table(doc, source="waypoint table"):
    """Validate a decoded table. Raises WaypointError with a message for the table's author."""
    if not isinstance(doc, dict):
        raise WaypointError("%s: the top level must be a JSON object with keys 'meta' and "
                            "'waypoints', got %s" % (source, _brief(doc)))

    meta = _missing(doc, "meta", source, "top level")
    raw_waypoints = _missing(doc, "waypoints", source, "top level")

    if not isinstance(meta, dict):
        raise WaypointError("%s: 'meta' must be an object, got %s" % (source, _brief(meta)))
    frame = _missing(meta, "frame", source, "meta")
    if frame not in SUPPORTED_FRAMES:
        raise WaypointError("%s: meta.frame = %s is not supported (expected 'base': the 8-D "
                            "action's pos and rotvec live in the robot base frame)"
                            % (source, _brief(frame)))
    units = _missing(meta, "units", source, "meta")
    if units not in SUPPORTED_UNITS:
        raise WaypointError("%s: meta.units = %s is not supported (expected 'm': metres for "
                            "pos, radians for rotvec)" % (source, _brief(units)))
    control_hz = _finite(_missing(meta, "control_hz", source, "meta"),
                         "meta.control_hz", source)
    if not (0.0 < control_hz <= MAX_CONTROL_HZ):
        raise WaypointError("%s: meta.control_hz = %r must be in (0, %g]"
                            % (source, control_hz, MAX_CONTROL_HZ))

    if not isinstance(raw_waypoints, (list, tuple)):
        raise WaypointError("%s: 'waypoints' must be a list, got %s"
                            % (source, _brief(raw_waypoints)))
    if not raw_waypoints:
        raise WaypointError("%s: 'waypoints' is empty, at least one waypoint is required"
                            % source)

    waypoints = []
    for i, raw in enumerate(raw_waypoints):
        where = "waypoints[%d]" % i
        if not isinstance(raw, dict):
            raise WaypointError("%s: %s must be an object, got %s"
                                % (source, where, _brief(raw)))
        label = raw.get("label", "")
        if not isinstance(label, str) or not label.strip():
            raise WaypointError("%s: %s.label must be a non-empty string, got %s"
                                % (source, where, _brief(label)))
        duration_s = _finite(_missing(raw, "duration_s", source, where),
                             where + ".duration_s", source)
        if duration_s <= 0.0:
            raise WaypointError("%s: %s.duration_s = %r must be > 0"
                                % (source, where, duration_s))

        has_fields = any(k in raw for k in ("pos", "rotvec", "g"))
        if "ee8" in raw:
            if has_fields:
                raise WaypointError("%s: %s gives both 'ee8' and pos/rotvec/g; use one form"
                                    % (source, where))
            raw8 = raw["ee8"]
            if not isinstance(raw8, (list, tuple)) or len(raw8) != 8:
                raise WaypointError("%s: %s.ee8 must be a list of 8 numbers (pos3, rotvec3, "
                                    "g, reserved), got %s" % (source, where, _brief(raw8)))
            for j in range(7):
                _finite(raw8[j], "%s.ee8[%d]" % (where, j), source)
            ee8 = np.array([_number(v, "%s.ee8[%d]" % (where, j), source)
                            for j, v in enumerate(raw8)])
            # ee8[7] is the reserved slot: accepted and never read, so it is not
            # range-checked -- the 8-D action's consumer ignores it too.
        else:
            pos = _vector3(_missing(raw, "pos", source, where, " (or the 8-D form 'ee8')"),
                           where + ".pos", source)
            rotvec = _vector3(_missing(raw, "rotvec", source, where,
                                       " (or the 8-D form 'ee8')"),
                              where + ".rotvec", source)
            g = _finite(_missing(raw, "g", source, where, " (or the 8-D form 'ee8')"),
                        where + ".g", source)
            if not (0.0 <= g <= 1.0):
                raise WaypointError("%s: %s.g = %r is outside [0, 1] (0 = closed, 1 = open)"
                                    % (source, where, g))
            ee8 = np.concatenate([pos, rotvec, [g, 0.0]])
        waypoints.append(Waypoint(label=label, ee8=ee8, duration_s=duration_s))

    return Table(source=source, frame=str(frame), units=str(units),
                 control_hz=control_hz, waypoints=tuple(waypoints))


def load_table(path):
    """Read + validate a table file. Every failure mode is a WaypointError, never a traceback."""
    if not os.path.isfile(path):
        raise WaypointError("%s: no such waypoint table" % path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except json.JSONDecodeError as exc:
        raise WaypointError("%s: not valid JSON: %s (line %d, column %d)"
                            % (path, exc.msg, exc.lineno, exc.colno)) from None
    except OSError as exc:
        raise WaypointError("%s: cannot read: %s" % (path, exc)) from None
    return parse_table(doc, source=path)


# --------------------------------------------------------------------------- #
# rotation helpers (quaternions, (w, x, y, z))
# --------------------------------------------------------------------------- #
def _quat_from_rotvec(rotvec):
    v = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * angle
    return np.concatenate([[math.cos(half)], math.sin(half) * (v / angle)])


def _quat_to_rotvec(q):
    """Shortest-arc rotation vector: the quaternion is taken with w >= 0, so the angle is
    in [0, pi] and a 180 deg rotation survives as a unit axis (sin(theta) would not)."""
    q = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.zeros(3)
    q = q / norm
    if q[0] < 0.0:
        q = -q
    sin_half = float(np.linalg.norm(q[1:]))
    if sin_half < 1e-12:
        return np.zeros(3)
    return q[1:] * (2.0 * math.atan2(sin_half, q[0]) / sin_half)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def interp_rotvec(rotvec_a, rotvec_b, alpha):
    """Shortest-arc interpolation between two rotvec representatives.

    The relative rotation is converted to the w >= 0 branch, which bounds its angle by pi,
    so two rotvecs that describe the same pose from opposite sides (0, +pi, 0) and
    (0, -pi, 0) do not make the tool spin a full turn.
    """
    if alpha <= 0.0:
        return np.asarray(rotvec_a, dtype=float).reshape(3).copy()
    if alpha >= 1.0:
        return np.asarray(rotvec_b, dtype=float).reshape(3).copy()
    qa = _quat_from_rotvec(rotvec_a)
    delta = _quat_mul(_quat_from_rotvec(rotvec_b), _quat_conj(qa))
    if delta[0] < 0.0:
        delta = -delta
    return _quat_to_rotvec(_quat_mul(_quat_from_rotvec(_quat_to_rotvec(delta) * alpha), qa))


def _blend(ee8_a, ee8_b, alpha):
    """One control sample between two 8-D actions: pos and g linear, rotation shortest-arc."""
    out = np.zeros(8)
    out[0:3] = ee8_a[0:3] + alpha * (ee8_b[0:3] - ee8_a[0:3])
    out[3:6] = interp_rotvec(ee8_a[3:6], ee8_b[3:6], alpha)
    out[6] = ee8_a[6] + alpha * (ee8_b[6] - ee8_a[6])
    # ee8[7] is the reserved slot of the 8-D action: carried as 0, never read.
    return out


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def plan_trajectory(table):
    """Sample a table at meta.control_hz. Raises WaypointError if the table is too long."""
    hz = float(table.control_hz)
    dt = 1.0 / hz
    n_settle = int(round(SETTLE_S * hz))
    wps = table.waypoints
    n_seg = len(wps) - 1
    if n_seg > MAX_FRAMES:
        raise WaypointError("%s: %d waypoints exceeds the %d-frame limit at meta.control_hz = %g"
                            % (table.source, len(wps), MAX_FRAMES, hz))

    ee8_rows, labels, segs, arrives, times = [], [], [], [], []

    def emit(row, label, seg, arrive, time_s):
        ee8_rows.append(np.asarray(row, dtype=float).copy())
        labels.append(label)
        segs.append(seg)
        arrives.append(bool(arrive))
        times.append(float(time_s))

    # Frame 0 is the first waypoint itself, and it belongs to the first segment as its start:
    # a frame is emitted once, so no sample is attributed to two segments.
    emit(wps[0].ee8, wps[0].label, 0, True, 0.0)
    t = 0.0
    for i in range(n_seg):
        a, b = wps[i].ee8, wps[i + 1].ee8
        n_frames = max(1, int(round(wps[i].duration_s * hz)))
        for k in range(1, n_frames + 1):
            t += dt
            row = b if k == n_frames else _blend(a, b, k / float(n_frames))
            emit(row, wps[i + 1].label, i, k == n_frames, t)
        # Repeat the waypoint so the reported residual is steady state, not a transient.
        if i + 1 < n_seg:
            for _ in range(n_settle):
                t += dt
                emit(b, wps[i + 1].label, i, False, t)

    last = wps[-1]
    n_hold = max(1, int(round(last.duration_s * hz)))
    for _ in range(n_hold):
        t += dt
        emit(last.ee8, last.label, n_seg - 1, False, t)

    if len(ee8_rows) > MAX_FRAMES:
        raise WaypointError("%s: the plan needs %d frames, more than the %d-frame limit "
                            "(shorten duration_s or lower meta.control_hz)"
                            % (table.source, len(ee8_rows), MAX_FRAMES))

    return Trajectory(t=np.array(times, dtype=float),
                      ee8=np.array(ee8_rows, dtype=float),
                      label=labels,
                      seg=np.array(segs, dtype=int),
                      arrive=np.array(arrives, dtype=bool),
                      control_hz=hz)

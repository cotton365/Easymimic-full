#!/usr/bin/env python3
"""Move the cube and/or the bowl on the table - and rewrite every place that says where they are.

The layout lives in exactly five places, and this tool is the only thing that should ever
touch them:

    1. assets/jaka_zu3/scene_cube_bowl.xml   <body name="cube" pos="...">
    2. the same file                          keyframe "home" qpos, the cube's free-joint tail
    3. the same file                          the 17 bowl geoms (bowl_bottom + bowl_wall_00..15)
    4. experiments/gen_act_dataset.py         CUBE_XY, CUBE_Z, BOWL_XY
    5. configs/waypoints_cube_{lift,place}.json  the grasp column at the cube, in the place
                                              table also carry/lower/release/retreat over the bowl

Default mode is a check: nothing is written, and every waypoint the tables actually use is
IK-solved at the new position (same criterion the generator applies - converged, inside the
joint limits, position residual <= 2e-3 m, orientation residual <= 2 deg), plus the
cube-to-bowl distance, the bowl-to-grasp-column clearance, and whether both objects are still
on the table slab.  The exit code is the verdict, so this is usable from a script.

--apply rewrites all five places (and refuses to write anything unless the check passes), then
re-runs the check on what it just wrote and prints how to look at the result.  Rewrites are
token-level: comments, indentation, key order and quoting stay byte-identical, a number whose
value did not change is left verbatim, and applying the current coordinates writes nothing.

What this tool does NOT check: self-collision (the arm against the table or itself) and the
dynamics of a grasp.  A reachable layout can still be a bad one, and IK-reachable is not
graspable: at 42 IK-reachable points around the default cube, 8 pushed the cube ~28 mm off the
gripper instead of holding it.  --grasp-grid therefore runs one real place episode per grid
point and prints the o/x map of the same gate the generator accepts an episode with (in_bowl,
bowl_dxy_m <= MAX_BOWL_DXY_M, hold_err_m <= MAX_HOLD_ERR_M).  Build a sampler on that map, not
on the IK table alone (docs/ACT_PIPELINE.md section 11.6).

--sample N prints N layouts from the very sampler the generator uses (continuous uniform over
CUBE_BOX / BOWL_BOX, at least MIN_SEP_M apart) together with the ranges it actually drew: the
cheapest way to see the distribution without running a single episode.  --sample-preview draws
one layout, puts the cube and the bowl there in memory (place_bowl + the generator's own cube
teleport) and opens MuJoCo's passive viewer on it.  That is the only window this tool ever
opens - the repo's other viewers (act_rollout.py --viewer, hand_teleop_live.py, legacy/) are
policy-replay and teleop paths, unrelated to layouts.  It runs only when the flag is passed,
it writes nothing, and no test ever takes it.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from experiments.gen_act_dataset import (ANCHORED_LABELS, BOWL_BOX,  # noqa: E402
                                         CONTROL_HZ, CUBE_BOX, MAX_BOWL_DXY_M,
                                         MAX_HOLD_ERR_M, MIN_SEP_M, SEED, place_bowl,
                                         place_cube_abs, run_attempt, sample_layout,
                                         table_task)
from src.ee_retarget.ik_jaka_zu3 import rotvec_to_rotmat  # noqa: E402
from src.pose_cmd.replay import Replayer  # noqa: E402
from src.pose_cmd.waypoints import load_table  # noqa: E402

SCENE = "assets/jaka_zu3/scene_cube_bowl.xml"
GEN = "experiments/gen_act_dataset.py"
TABLES = ("configs/waypoints_cube_lift.json", "configs/waypoints_cube_place.json")

CUBE_BODY = "cube"
KEY_NAME = "home"
BOWL_PREFIX = "bowl_"
BOWL_BOTTOM = "bowl_bottom"
TABLE_GEOM = "table_top"        # the slab: box 0.60 x 0.50 x 0.02, top face z = 0.80

BOWL_OUTER_R_M = 0.049          # wall ring centreline = the bowl's footprint radius
ARM_JOINTS = 6                  # the IK solves the arm, the gripper is held by q_seed
IK_POS_TOL_M = 2.0e-3           # gen_act_dataset.MAX_IK_ERR_M, i.e. what an episode may leave
IK_ANG_TOL_DEG = 2.0            # the measured reachable region was probed at the same bar
SLAB_MARGIN_M = 0.0             # the object centre must be inside the slab, not on its edge
MAX_GRID_POINTS = 400           # one real episode each, so 400 points is already ~10 min
GRID_SECONDS_PER_POINT = 1.5    # measured here; drives the cost estimate and the printout

VIEWER_CMD = "python -m mujoco.viewer --mjcf=%s" % SCENE


# ---------------------------------------------------------------- number formatting
def _num(v):
    """Shortest round-trip text - the style the scene, the JSON and the generator already use."""
    return repr(float(v))


def _num6(v):
    """Six decimals - the style the 17 bowl geoms are written in."""
    return "%.6f" % float(v)


def _tok(old_tok, new_val, fmt):
    """old_tok when the value did not change, else new_val in the requested style.

    Keeping the old token verbatim is what makes --apply idempotent and keeps a rewrite down
    to the numbers that actually moved."""
    if float(old_tok) == float(new_val):
        return old_tok
    return fmt(new_val)


_NUMBER = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def _splice(inner, values, fmt):
    """Replace the numbers inside a text fragment, leaving every separator byte alone."""
    hits = list(_NUMBER.finditer(inner))
    if len(hits) != len(values):
        raise SystemExit("set_layout: expected %d numbers in %r, found %d"
                         % (len(values), inner, len(hits)))
    out, last = [], 0
    for hit, value in zip(hits, values):
        out.append(inner[last:hit.start()])
        out.append(_tok(hit.group(0), value, fmt))
        last = hit.end()
    out.append(inner[last:])
    return "".join(out)


# ---------------------------------------------------------------- reading the five places
_SCENE_BODY = re.compile(r'<body name="%s" pos="([^"]*)"' % CUBE_BODY)
_SCENE_KEY = re.compile(r'<key name="%s" qpos="([^"]*)"' % KEY_NAME)
_SCENE_GEOM = re.compile(r'<geom name="([^"]+)"[^>]*?pos="([^"]*)"')
_GEN_PAIR = re.compile(r"^(?P<name>CUBE_XY|BOWL_XY) = \((?P<a>[^),]*), ?(?P<b>[^)]*)\)", re.M)
_GEN_Z = re.compile(r"^(?P<name>CUBE_Z) = (?P<v>[^\s#]+)", re.M)


def scene_cube(text):
    return tuple(float(v) for v in _SCENE_BODY.search(text).group(1).split())


def scene_key_cube(text):
    """The cube's free-joint coordinates out of the home keyframe (arm 7 + cube 7)."""
    toks = _SCENE_KEY.search(text).group(1).split()
    return tuple(float(v) for v in toks[7:10])


def scene_bowl(text):
    """{name: (x, y, z)} of the bowl's geoms, in document order."""
    return {name: tuple(float(v) for v in pos.split())
            for name, pos in _SCENE_GEOM.findall(text) if name.startswith(BOWL_PREFIX)}


def gen_layout(text):
    """(cube_xy, cube_z, bowl_xy) as experiments/gen_act_dataset.py spells them."""
    cube, bowl = _GEN_PAIR.search(text), None
    values = {m.group("name"): m for m in _GEN_PAIR.finditer(text)}
    cube = values["CUBE_XY"]
    bowl = values["BOWL_XY"]
    return ((float(cube.group("a")), float(cube.group("b"))),
            float(_GEN_Z.search(text).group("v")),
            (float(bowl.group("a")), float(bowl.group("b"))))


def read(path):
    with open(os.path.join(REPO, path)) as fh:
        return fh.read()


# ---------------------------------------------------------------- rewriting the five places
def rewrite_scene(text, cube_xyz, bowl_xy):
    """The scene XML with the cube body, the home keyframe and the 17 bowl geoms moved."""
    def sub(pattern, fn, text):
        def repl(m):
            return m.group(1) + fn(m.group(2)) + m.group(3)
        return re.sub(pattern, repl, text, count=1)

    def body(pos):
        return " ".join(_tok(tok, value, _num) for tok, value in zip(pos.split(), cube_xyz))

    def free_joint(qpos):
        toks = qpos.split()
        for i, value in enumerate(cube_xyz):
            toks[7 + i] = _tok(toks[7 + i], value, _num)
        return " ".join(toks)

    out = sub(r'(<body name="%s" pos=")([^"]*)(")' % CUBE_BODY, body, text)
    out = sub(r'(<key name="%s" qpos=")([^"]*)(")' % KEY_NAME, free_joint, out)

    # the bowl is 17 independent worldbody geoms: they all translate by the same delta, so the
    # ring keeps its shape (radii and euler angles are not touched)
    centre = scene_bowl(text)[BOWL_BOTTOM]
    shift = (float(bowl_xy[0]) - centre[0], float(bowl_xy[1]) - centre[1])

    def slide(pos):
        x, y, z = pos.split()
        return " ".join([_tok(x, float(x) + shift[0], _num6),
                         _tok(y, float(y) + shift[1], _num6), z])

    for name in scene_bowl(text):
        out = sub(r'(<geom name="%s"[^>]*?pos=")([^"]*)(")' % re.escape(name), slide, out)
    return out


def rewrite_gen(text, cube_xyz, bowl_xy):
    """The generator's CUBE_XY / CUBE_Z / BOWL_XY, with their comments and spacing intact."""
    def pair(m):
        target = cube_xyz if m.group("name") == "CUBE_XY" else bowl_xy
        return "%s = (%s, %s)" % (m.group("name"),
                                  _tok(m.group("a").strip(), target[0], _num),
                                  _tok(m.group("b").strip(), target[1], _num))

    def z(m):
        return "CUBE_Z = %s" % _tok(m.group("v"), cube_xyz[2], _num)

    return _GEN_Z.sub(z, _GEN_PAIR.sub(pair, text, count=2), count=1)


def rewrite_table(text, table, cube_xyz, bowl_xy, dz):
    """The waypoint table re-seated: the grasp column at the cube, the anchored rows at the bowl.

    Only the numbers inside "pos": [...] change; the separators, indentation and key order are
    the file's own.  dz shifts the cube column's heights when --cube-z moves the cube."""
    plan = []
    for w in table.waypoints:
        anchored = w.label in ANCHORED_LABELS
        xy = bowl_xy if anchored else cube_xyz[:2]
        plan.append((float(xy[0]), float(xy[1]), float(w.ee8[2]) + (0.0 if anchored else dz)))
    seen = []

    def repl(m):
        if len(seen) >= len(plan):
            raise SystemExit("set_layout: %s has more \"pos\" arrays than waypoints"
                             % getattr(table, "source", "?"))
        values = plan[len(seen)]
        seen.append(values)
        return m.group(1) + _splice(m.group(2), values, _num) + m.group(3)

    out = re.sub(r'("pos":\s*\[)([^\]]*)(\])', repl, text)
    if len(seen) != len(plan):
        raise SystemExit("set_layout: %s has %d \"pos\" arrays but %d waypoints"
                         % (getattr(table, "source", "?"), len(seen), len(plan)))
    return out


# ---------------------------------------------------------------- the check
def slab(model):
    """(x0, x1, y0, y1) of the table top, read from the slab geom itself."""
    import mujoco

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    if gid < 0:
        raise SystemExit("set_layout: the scene has no geom named %s" % TABLE_GEOM)
    cx, cy, _cz = model.geom_pos[gid]
    hx, hy, _hz = model.geom_size[gid]
    return cx - hx, cx + hx, cy - hy, cy + hy


def column_heights(tables, dz):
    """{column: {z: [labels]}} - the heights the tables actually command, at the new layout."""
    columns = {"cube": {}, "bowl": {}}
    for table in tables:
        for w in table.waypoints:
            anchored = w.label in ANCHORED_LABELS
            z = round(float(w.ee8[2]) + (0.0 if anchored else dz), 9)
            rows = columns["bowl" if anchored else "cube"].setdefault(z, [])
            if w.label not in rows:
                rows.append(w.label)
    return columns


def rotvec_of(tables, labels):
    """The orientation a row of waypoints commands (every waypoint agrees on it today)."""
    for table in tables:
        for w in table.waypoints:
            if w.label in labels:
                return np.asarray(w.ee8[3:6], dtype=float)
    raise SystemExit("set_layout: no waypoint labelled %s" % "/".join(labels))


def check(rp, model, cube_xyz, bowl_xy, dz, tables, places, must_match=False):
    """(ok, lines) for a candidate layout. Every line is one verdict.

    The IK verdicts are about the requested layout; places_agree is about the files, which is
    why check-only runs answer "is this layout reachable" without also demanding that the files
    already say so.  must_match=True is the post-apply reading: there the files have to agree."""
    lines = []
    ok = True
    q_seed = rp.home_qpos()
    columns = column_heights(tables, dz)
    for name, xy in (("cube", cube_xyz[:2]), ("bowl", bowl_xy)):
        pos_x = float(xy[0])
        pos_y = float(xy[1])
        for z in sorted(columns[name]):
            rotvec = rotvec_of(tables, columns[name][z])
            target = np.array([pos_x, pos_y, z], dtype=float)
            res = rp.ik.solve(target, rotvec_to_rotmat(rotvec), q_seed)
            if not res.converged:
                res = rp.ik.solve_robust(target, rotvec_to_rotmat(rotvec), q_seed)
            in_limits = bool(rp.ik.within_limits(np.asarray(res.qpos[:ARM_JOINTS], dtype=float)))
            good = (bool(res.converged) and in_limits
                    and float(res.pos_err) <= IK_POS_TOL_M
                    and float(res.ang_err_deg) <= IK_ANG_TOL_DEG)
            ok = ok and good
            lines.append("  %-4s z %.3f  %-20s  IK %-4s  %s  pos %.1e m  ang %.2f deg  "
                         "joints %s  rotvec %s"
                         % (name, z, "/".join(columns[name][z]), "ok" if good else "FAIL",
                            "conv" if res.converged else "MISS", float(res.pos_err),
                            float(res.ang_err_deg),
                            "ok" if in_limits else "LIMIT",
                            np.array2string(rotvec, precision=6, separator=", ")))

    dist = float(np.hypot(float(cube_xyz[0]) - float(bowl_xy[0]),
                          float(cube_xyz[1]) - float(bowl_xy[1])))
    clear = dist > BOWL_OUTER_R_M
    ok = ok and clear
    lines.append("  cube-bowl distance %.4f m  %s the bowl's %.3f m outer radius "
                 "(grasp column clears by %.4f m)"
                 % (dist, ">" if clear else "<= FAIL:", BOWL_OUTER_R_M,
                    dist - BOWL_OUTER_R_M)
                 if clear else
                 "  cube-bowl distance %.4f m  FAIL: <= the bowl's %.3f m outer radius "
                 "(the gripper would come down on the wall)"
                 % (dist, BOWL_OUTER_R_M))

    x0, x1, y0, y1 = slab(model)
    for name, xy in (("cube", cube_xyz[:2]), ("bowl", bowl_xy)):
        inside = (x0 + SLAB_MARGIN_M < float(xy[0]) < x1 - SLAB_MARGIN_M
                  and y0 + SLAB_MARGIN_M < float(xy[1]) < y1 - SLAB_MARGIN_M)
        ok = ok and inside
        lines.append("  %-4s on the table slab (x %.2f..%.2f, y %.2f..%.2f)  %s  "
                     "margins x %.3f/%.3f m, y %.3f/%.3f m"
                     % (name, x0, x1, y0, y1, "ok" if inside else "FAIL",
                        float(xy[0]) - x0, x1 - float(xy[0]),
                        float(xy[1]) - y0, y1 - float(xy[1])))

    agree, why = places_agree(places)
    same = (np.allclose(np.asarray(places["cube"], dtype=float), np.asarray(cube_xyz, dtype=float),
                        atol=1e-9, rtol=0.0)
            and np.allclose(np.asarray(places["bowl"], dtype=float),
                            np.asarray(bowl_xy, dtype=float), atol=1e-9, rtol=0.0))
    ok = ok and agree and (same or not must_match)
    lines.append("  the five places agree with each other  %s%s"
                 % ("ok" if agree else "FAIL", "" if agree else ": " + why))
    lines.append("  on disk: cube %s  bowl %s  %s"
                 % (np.array2string(np.asarray(places["cube"], dtype=float), precision=4,
                                    separator=", "),
                    np.array2string(np.asarray(places["bowl"], dtype=float), precision=4,
                                    separator=", "),
                    "that is what was asked for" if same else
                    ("DIFFERS from the layout asked for  FAIL" if must_match else
                     "differs from the layout checked above (--apply would rewrite it)")))
    return ok, lines


def places_agree(places):
    """Do the scene, the generator constants and the tables describe one layout?"""
    cube = np.array(places["cube"], dtype=float)
    bowl = np.array(places["bowl"], dtype=float)
    problems = []
    for name, value in (("scene cube body", places["scene_cube"]),
                        ("scene home keyframe", places["scene_key"]),
                        ("gen_act_dataset CUBE_XY/CUBE_Z", places["gen_cube"])):
        if not np.allclose(np.asarray(value, dtype=float), cube, atol=1e-9, rtol=0.0):
            problems.append("%s %s != %s" % (name, np.array(value), cube))
    if not np.allclose(np.asarray(places["gen_bowl"], dtype=float), bowl, atol=1e-9, rtol=0.0):
        problems.append("gen_act_dataset BOWL_XY %s != %s"
                        % (np.array(places["gen_bowl"]), bowl))
    if not np.allclose(np.asarray(places["scene_bowl"], dtype=float), bowl, atol=1e-9, rtol=0.0):
        problems.append("scene bowl %s != %s" % (np.array(places["scene_bowl"]), bowl))
    for path, rows in places["tables"].items():
        for label, xy, anchored in rows:
            want = bowl if anchored else cube[:2]
            if not np.allclose(np.asarray(xy, dtype=float), want, atol=1e-9, rtol=0.0):
                problems.append("%s: waypoint %s at %s, expected %s"
                                % (path, label, np.array(xy), want))
    return not problems, "; ".join(problems)


# ---------------------------------------------------------------- main
def disk_places(scene_text, gen_text, tables):
    """What the five places say right now, for the agreement check."""
    gen_cube, gen_z, gen_bowl = gen_layout(gen_text)
    bowl = scene_bowl(scene_text)[BOWL_BOTTOM][:2]
    return {"cube": scene_cube(scene_text), "bowl": bowl,
            "scene_cube": scene_cube(scene_text), "scene_key": scene_key_cube(scene_text),
            "gen_cube": (gen_cube[0], gen_cube[1], gen_z), "gen_bowl": gen_bowl,
            "scene_bowl": bowl,
            "tables": {path: [(w.label, tuple(float(v) for v in w.ee8[0:2]),
                               w.label in ANCHORED_LABELS) for w in table.waypoints]
                       for path, table in zip(TABLES, tables)}}


def _metres(v):
    return "n/a" if v is None else "%.4f m" % float(v)


def parse_grid(text):
    """"XMIN:XMAX:XSTEP,YMIN:YMAX:YSTEP" -> (xs, ys); a bad spec is a SystemExit, not a crash."""
    try:
        xpart, ypart = text.split(",")
        xmin, xmax, xstep = [float(v) for v in xpart.split(":")]
        ymin, ymax, ystep = [float(v) for v in ypart.split(":")]
    except (AttributeError, TypeError, ValueError):
        raise SystemExit("bad --grasp-grid %r: want XMIN:XMAX:XSTEP,YMIN:YMAX:YSTEP" % (text,))
    for axis, lo, hi, step in (("x", xmin, xmax, xstep), ("y", ymin, ymax, ystep)):
        if hi < lo or step <= 0.0:
            raise SystemExit("bad --grasp-grid %r: %s needs MIN <= MAX and STEP > 0"
                             % (text, axis))

    def axis(lo, hi, step):
        # rounded so 0.0385 + 2 * 0.005 prints and compares as 0.0485
        return [round(lo + i * step, 9) for i in range(int(round((hi - lo) / step)) + 1)]

    return axis(xmin, xmax, xstep), axis(ymin, ymax, ystep)


def grasp_report(xs, ys, bowl_xy):
    """One real place episode per grid point, then the o/x map of the generator's own gate.

    Exit code 0 whenever the grid ran: the map is the answer, and an x cell is a measurement
    rather than a failure of the tool.  Nothing is written outside a temporary directory, and
    no renderer is ever created."""
    base = load_table(os.path.join(REPO, TABLES[1]))
    if table_task(base) != "place":
        print("refusing: %s is not a place table, there is no bowl to measure against"
              % TABLES[1])
        return 2
    total = len(xs) * len(ys)
    print("grasp grid: %d x %d = %d points at bowl (%.4f, %.4f), ~%.1f s/point (~%.0f s)"
          % (len(xs), len(ys), total, bowl_xy[0], bowl_xy[1], GRID_SECONDS_PER_POINT,
             total * GRID_SECONDS_PER_POINT))
    verdicts = {}
    root = tempfile.mkdtemp(prefix="set_layout_grasp_")
    t_all = time.perf_counter()
    try:
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                t0 = time.perf_counter()
                res = run_attempt(base, (x, y), bowl_xy,
                                  os.path.join(root, "p%03d_%03d" % (iy, ix)), False)
                secs = time.perf_counter() - t0
                dxy, hold = res.get("bowl_dxy_m"), res.get("hold_err_m")
                good = (bool(res.get("in_bowl")) and dxy is not None and hold is not None
                        and float(dxy) <= MAX_BOWL_DXY_M and float(hold) <= MAX_HOLD_ERR_M)
                verdicts[(x, y)] = (good, dxy, hold)
                print("  (%8.4f, %8.4f)  %s  bowl_dxy %s  hold %s  %.2f s"
                      % (x, y, "o" if good else "x", _metres(dxy), _metres(hold), secs))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    wall = time.perf_counter() - t_all

    print("")
    print("  %8s " % "y\\x" + " ".join("%8.4f" % x for x in xs))
    for y in reversed(ys):
        print("  %8.4f " % y
              + " ".join("%8s" % ("o" if verdicts[(x, y)][0] else "x") for x in xs))
    ok = sum(1 for v in verdicts.values() if v[0])
    fails = [(x, y) + verdicts[(x, y)][1:] for y in reversed(ys) for x in xs
             if not verdicts[(x, y)][0]]
    print("")
    print("ok/total: %d/%d   (%.1f s for %d points, %.2f s/point)"
          % (ok, total, wall, total, wall / float(total or 1)))
    if fails:
        print("failing points (the cube is pushed off instead of held):")
        for x, y, dxy, hold in fails:
            print("  (%8.4f, %8.4f)  bowl_dxy %s  hold %s" % (x, y, _metres(dxy),
                                                              _metres(hold)))
    else:
        print("no failing point: every point passed the gate")
    print("gate: in_bowl and bowl_dxy_m <= %.3f m and hold_err_m <= %.3f m - the same numbers "
          "the generator accepts an episode with" % (MAX_BOWL_DXY_M, MAX_HOLD_ERR_M))
    return 0


def sample_report(rng, n):
    """N layouts from the generator's own sampler, plus the ranges they actually covered."""
    print("sampler: cube xy ~ U(%.3f..%.3f) x (%.3f..%.3f), bowl xy ~ U(%.3f..%.3f) x "
          "(%.3f..%.3f), cube-bowl >= %.3f m, continuous (no grid)"
          % (CUBE_BOX[0], CUBE_BOX[1], CUBE_BOX[2], CUBE_BOX[3],
             BOWL_BOX[0], BOWL_BOX[1], BOWL_BOX[2], BOWL_BOX[3], MIN_SEP_M))
    pairs = [sample_layout(rng) for _ in range(max(1, n))]
    for i, (cube, bowl) in enumerate(pairs):
        print("  %3d  cube (%8.4f, %8.4f)  bowl (%8.4f, %8.4f)  sep %.4f m"
              % (i, cube[0], cube[1], bowl[0], bowl[1],
                 math.hypot(cube[0] - bowl[0], cube[1] - bowl[1])))
    cxs = [c[0] for c, _b in pairs]
    cys = [c[1] for c, _b in pairs]
    bxs = [b[0] for _c, b in pairs]
    bys = [b[1] for _c, b in pairs]
    seps = [math.hypot(c[0] - b[0], c[1] - b[1]) for c, b in pairs]
    print("  drew: cube x %.4f..%.4f  y %.4f..%.4f | bowl x %.4f..%.4f  y %.4f..%.4f | "
          "sep %.4f..%.4f m"
          % (min(cxs), max(cxs), min(cys), max(cys), min(bxs), max(bxs), min(bys), max(bys),
             min(seps), max(seps)))
    print("  nothing written: --sample only prints")
    return 0


def sample_preview(rng):
    """Sample one layout, put both objects there in memory, and open the MuJoCo viewer."""
    cube_xy, bowl_xy = sample_layout(rng)
    print("preview layout: cube (%.4f, %.4f) m, bowl (%.4f, %.4f) m, separation %.4f m"
          % (cube_xy[0], cube_xy[1], bowl_xy[0], bowl_xy[1],
             math.hypot(cube_xy[0] - bowl_xy[0], cube_xy[1] - bowl_xy[1])))
    print("opening the viewer: drag to orbit, close the window to exit.  This is a static "
          "snapshot of the sampled layout - physics is not stepped, nothing is written.")
    rp = Replayer(scene_path=os.path.join(REPO, SCENE), control_hz=CONTROL_HZ, render_every=0)
    try:
        place_bowl(rp, bowl_xy)
        place_cube_abs(rp, cube_xy)
        import mujoco.viewer  # the only viewer this tool ever opens, and only from here
        with mujoco.viewer.launch_passive(rp.model, rp.data) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    finally:
        rp.close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cube", nargs=2, type=float, default=None, metavar=("X", "Y"),
                    help="cube centre on the table (m); required unless --grasp-grid says where")
    ap.add_argument("--bowl", nargs=2, type=float, default=None, metavar=("X", "Y"),
                    help="bowl centre on the table (m); default: the scene's current one")
    ap.add_argument("--cube-z", type=float, default=None,
                    help="cube centre height (m); default: the scene's current one")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the five places (refused unless the check passes)")
    ap.add_argument("--grasp-grid", default=None, metavar="XMIN:XMAX:XSTEP,YMIN:YMAX:YSTEP",
                    help="measure instead of writing: one real place episode per grid point, "
                         "then an o/x map (~%.1f s/point, at most %d points)"
                         % (GRID_SECONDS_PER_POINT, MAX_GRID_POINTS))
    ap.add_argument("--seed", type=int, default=SEED,
                    help="sampler seed for --sample and --sample-preview (default %d)" % SEED)
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="print N layouts from the generator's sampler and the ranges they "
                         "covered; writes nothing")
    ap.add_argument("--sample-preview", action="store_true",
                    help="draw one layout, put both objects there and open the MuJoCo viewer "
                         "on it; the only window this tool opens")
    args = ap.parse_args(argv)

    if args.sample_preview and args.grasp_grid is not None:
        print("refusing: --sample-preview shows one sampled layout and --grasp-grid measures "
              "a grid; run them separately (no file was touched)")
        return 2
    measuring = [flag for flag, on in (("--grasp-grid", args.grasp_grid is not None),
                                       ("--sample", args.sample > 0),
                                       ("--sample-preview", args.sample_preview)) if on]
    if args.apply and measuring:
        print("refusing: --apply rewrites files and %s only measure; run them separately "
              "(no file was touched)" % " / ".join(measuring))
        return 2
    rng = np.random.default_rng(args.seed)
    if args.sample > 0:
        return sample_report(rng, args.sample)
    if args.sample_preview:
        return sample_preview(rng)
    scene_text = read(SCENE)
    scene_cur = scene_cube(scene_text)
    bowl_xy = ((float(scene_bowl(scene_text)[BOWL_BOTTOM][0]),
                float(scene_bowl(scene_text)[BOWL_BOTTOM][1]))
               if args.bowl is None else (float(args.bowl[0]), float(args.bowl[1])))
    if args.grasp_grid is not None:
        xs, ys = parse_grid(args.grasp_grid)
        total = len(xs) * len(ys)
        if total > MAX_GRID_POINTS:
            print("refusing: the grid asks for %d points, the cap is %d; at ~%.1f s/point that "
                  "is about %.0f min (no episode was run)"
                  % (total, MAX_GRID_POINTS, GRID_SECONDS_PER_POINT,
                     total * GRID_SECONDS_PER_POINT / 60.0))
            return 2
        return grasp_report(xs, ys, bowl_xy)
    if args.cube is None:
        print("refusing: --cube X Y is required unless --grasp-grid gives the points "
              "(no file was touched)")
        return 2

    gen_text = read(GEN)
    cube_xy = (float(args.cube[0]), float(args.cube[1]))
    cube_z = scene_cur[2] if args.cube_z is None else float(args.cube_z)
    cube_xyz = (cube_xy[0], cube_xy[1], cube_z)
    dz = cube_z - scene_cur[2]

    tables = [load_table(os.path.join(REPO, path)) for path in TABLES]
    places = disk_places(scene_text, gen_text, tables)

    rp = Replayer(scene_path=os.path.join(REPO, SCENE), control_hz=20.0, render_every=0)
    try:
        model = rp.model
        print("layout: cube (%.4f, %.4f) z %.3f   bowl (%.4f, %.4f)   %s"
              % (cube_xyz[0], cube_xyz[1], cube_xyz[2], bowl_xy[0], bowl_xy[1],
                 "check only" if not args.apply else "check + apply"))
        ok, lines = check(rp, model, cube_xyz, bowl_xy, dz, tables, places)
        for line in lines:
            print(line)
        print("PASS" if ok else "FAIL")
        if not args.apply:
            print("nothing written (-check only; use --apply to rewrite the five places)")
            return 0 if ok else 1
        if not ok:
            print("refusing to write: fix the FAILs above first (no file was touched)")
            return 1

        new_scene = rewrite_scene(scene_text, cube_xyz, bowl_xy)
        writes = []
        if new_scene != scene_text:
            writes.append((SCENE, new_scene))
        new_gen = rewrite_gen(gen_text, cube_xyz, bowl_xy)
        if new_gen != gen_text:
            writes.append((GEN, new_gen))
        for path, table in zip(TABLES, tables):
            old = read(path)
            new = rewrite_table(old, table, cube_xyz, bowl_xy, dz)
            if new != old:
                writes.append((path, new))
        for path, text in writes:
            with open(os.path.join(REPO, path), "w") as fh:
                fh.write(text)
        print("wrote %d of the 5 places (%s)" % (len(writes), ", ".join(p for p, _ in writes))
              if writes else "wrote 0 of the 5 places: they already said exactly this")

        # re-run the check on what is now on disk
        tables = [load_table(os.path.join(REPO, path)) for path in TABLES]
        places = disk_places(read(SCENE), read(GEN), tables)
        ok, lines = check(rp, model, cube_xyz, bowl_xy, dz, tables, places, must_match=True)
        print("re-check on disk: %s" % ("PASS" if ok else "FAIL"))
        print("look at it:  %s" % VIEWER_CMD)
        return 0 if ok else 1
    finally:
        rp.close()


if __name__ == "__main__":
    sys.exit(main())

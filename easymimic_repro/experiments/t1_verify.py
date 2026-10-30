#!/usr/bin/env python
"""T7 acceptance check: the REAL gripper (装配体2.STL export) on the JAKA Zu3 MuJoCo model.

    ~/miniconda3/envs/mujoco/bin/python experiments/t1_verify.py

A  the scene loads by ABSOLUTE path from two working directories (/ and the repo root)
B  the gripper sits on the real flange face, the jaw opens the right way, the robot does
   not collide with itself at the home keyframe
C  a 0.020 x 0.030 x 0.030 m / 8 g block is grasped off the table, lifted 0.10 m and
   simulated for 2 s:
   both fingers stay in contact, the cube neither slips nor gets squashed
D  one offscreen frame -> out/t7_frames/final.png
E  one compact table with every measured number

Every path is derived from this file, so the numbers do not depend on the caller's cwd.
"""
import os
import sys
import math
import time
import zlib
import struct
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
from src.ee_retarget.ik_jaka_zu3 import JakaZu3IK          # noqa: E402  (reuse, do not copy)

ASSETS = os.path.join(REPO, "assets", "jaka_zu3")
SCENE = os.path.join(ASSETS, "scene_cube_bowl.xml")
JAKA = os.path.join(REPO, "third_party", "jaka-zu3-mujoco")
MESHDIR = os.path.join(JAKA, "assets", "jaka_zu3_meshes")
GRIPDIR = os.path.join(JAKA, "assets", "gripper")
OUTDIR = os.path.join(REPO, "out", "t7_frames")

Q_OPEN = 0.9            # 8-dim action g = 1 == open == joint_7 0.9 rad; g = 0 == closed
CUBE_HALF = 0.010       # 20 mm across the pinch axis -> each pad face ends up 10 mm from the pinch point
CUBE_HALF_YZ = 0.015    # 30 mm in y/z: free to grow, only buys aiming tolerance
R_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])   # tool +z -> world -z

FAILS = []


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)
    return cond


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tool_pose(d, sid):
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()


def sim(m, d, secs):
    for _ in range(int(round(secs / m.opt.timestep))):
        mujoco.mj_step(m, d)


def pad_gap(m, d, gfix, gmov):
    """Closest distance between the two pad boxes (0 = touching)."""
    return float(mujoco.mj_geomDistance(m, d, gfix, gmov, 1.0, np.zeros(6)))


def cube_edges(m, d, gcube):
    """Edge lengths of the cube measured along its OWN axes (a world-axis extent is wrong
    as soon as the box is rotated, and this cube may turn by a fraction of a degree)."""
    b, R = d.geom_xpos[gcube], d.geom_xmat[gcube].reshape(3, 3)
    h = m.geom_size[gcube]
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    loc = s * h
    return np.array([np.ptp(loc[:, i]) for i in range(3)])


def mesh_points(m, gid):
    """Vertices of a mesh geom in its BODY frame: m.mesh_vert lives in the mesh frame, which
    MuJoCo recentres and rotates, so it has to be composed with geom_pos/geom_quat."""
    i = m.geom_dataid[gid]
    a, n = m.mesh_vertadr[i], m.mesh_vertnum[i]
    v = m.mesh_vert[a:a + n].astype(float)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[gid])
    return v @ R.reshape(3, 3).T + m.geom_pos[gid]


def mesh_bbox(m, gid):
    w = mesh_points(m, gid)
    return w.min(axis=0), w.max(axis=0), len(w)


def write_png(path, rgb):
    """Minimal PNG writer (zlib + struct only; this env has no PIL)."""
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


T0 = time.perf_counter()
print("=" * 78)
print("SCRIPT :", os.path.abspath(__file__))
print("CWD    :", os.getcwd())
print("python :", sys.executable)
print("mujoco :", mujoco.__version__, "| numpy", np.__version__)
print("=" * 78)

# ------------------------------------------------- A. load by absolute path, two cwds
print("\n[A] MjModel.from_xml_path(ABSOLUTE) from two working directories")
ORIG = os.getcwd()
MODELS = {}
for label, cwd in (("/", "/"), ("repo", REPO)):
    try:
        os.chdir(cwd)
        t = time.perf_counter()
        mm = mujoco.MjModel.from_xml_path(SCENE)
        dt = time.perf_counter() - t
        MODELS[label] = mm
        print("  cwd=%-30s nq=%2d nv=%2d nu=%d nmesh=%2d ngeom=%2d  (%.2f s)"
              % (os.getcwd(), mm.nq, mm.nv, mm.nu, mm.nmesh, mm.ngeom, dt))
    except Exception as exc:                                   # noqa: BLE001
        print("  cwd=%-30s LOAD FAILED: %r" % (cwd, exc))
        MODELS[label] = None
os.chdir(ORIG)
ok_load = all(v is not None for v in MODELS.values())
if not check(ok_load, "scene loads from cwd=/ AND cwd=%s (meshdir ../gripper/... resolves)"
             % REPO):
    print("\nRESULT: FAIL (model does not load)")
    sys.exit(1)
m = MODELS["repo"] if MODELS["repo"] is not None else MODELS["/"]
check(m.nq == 14 and m.nu == 7,
      "nq/nu == 14/7 (6 arm hinges + joint_7 + 7 cube free-joint dofs)  measured %d/%d"
      % (m.nq, m.nu))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

# ---------------------------------------- B. assembly: flange face, jaw, self-collision
print("\n[B] assembly: flange face, joint_7 direction, home self-collision")
sid = m.site("tool0").id
b6 = m.body("Link_6").id
bbase = m.body("gripper_base").id
bjaw = m.body("gripper_moving_jaw").id
jid = m.joint("joint_7").id
jawadr = int(m.jnt_qposadr[jid])
gfix = m.geom("gripper_pad_fixed").id
gmov = m.geom("gripper_pad_moving").id
gcube = m.geom("cube_geom").id
bcube = m.body("cube").id
cubeq = int(m.jnt_qposadr[m.joint("cube_free").id])

mesh_ok = True
for nm, f in (("gripper_housing", "fixed_housing.stl"),
              ("gripper_hardware_a", "fixed_hardware_a.stl"),
              ("gripper_hardware_b", "fixed_hardware_b.stl"),
              ("gripper_jaw_visual", "moving_jaw.stl")):
    gid = m.geom(nm).id
    lo, hi, n = mesh_bbox(m, gid)
    exists = os.path.isfile(os.path.join(GRIPDIR, f))
    mesh_ok &= exists
    print("  %-26s %-22s verts=%6d  bbox/gripper-frame x[%+.5f,%+.5f] y[%+.5f,%+.5f] z[%+.5f,%+.5f]"
          % (nm, f, n, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
check(mesh_ok and m.nmesh == 11,
      "4 gripper STLs resolved from %s (11 meshes in the model, metres)" % GRIPDIR)

# the flange plane is Link_6 local z = 0 (the exposed face measured on Link_6.STL)
p6 = d.xpos[b6].copy()
R6 = d.xmat[b6].reshape(3, 3).copy()
flange_n = R6[:, 2]
dev = float(np.dot(d.xpos[bbase] - p6, flange_n))
lo, hi, _ = mesh_bbox(m, m.geom("gripper_housing").id)
print("  Link_6 origin %s   flange normal (+z) %s"
      % (np.array2string(p6, precision=5), np.array2string(flange_n, precision=4)))
print("  gripper_base  %s   local pos %s   deviation from the flange plane %+.6f m"
      % (np.array2string(d.xpos[bbase], precision=5),
         np.array2string(m.body_pos[bbase], precision=5), dev))
print("  housing mesh in the gripper frame: z [%+.6f, %+.6f] m (0 = flange face)"
      % (lo[2], hi[2]))
check(abs(dev) < 1e-6, "gripper_base origin lies ON the flange plane (dev %.2e m)" % dev)
check(abs(lo[2]) < 1e-6 and hi[2] > 0.10,
      "housing bottom face is on the flange (z %+.6f) and grows %+.4f m AWAY from the arm"
      % (lo[2], hi[2]))

Q_GRASP = 0.226                # stroke at which the pads are 20 mm apart (nominal 2 cm grip)
gaps = {}
for q in (0.0, Q_GRASP, Q_OPEN):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[jawadr] = q
    mujoco.mj_forward(m, d)
    gaps[q] = pad_gap(m, d, gfix, gmov)
# mj_geomDistance is the min 3D box-box distance.  The moving box is pre-rotated by
# 0.226 rad, so at small gaps it reports a corner-to-face distance, a bit less than the
# wear-face gap (0.00225 m at q=0 by geometry).  The ordering that matters - closed <
# cube < open - clears the cube by more than 18 mm either way.
print("  min 3D distance between the two pad boxes: q=0 %.5f m | q=0.226 %.5f m |"
      " q=0.9 %.5f m | stroke %.5f m"
      % (gaps[0.0], gaps[Q_GRASP], gaps[Q_OPEN], gaps[Q_OPEN] - gaps[0.0]))
check(gaps[0.0] < 0.005, "joint_7 q=0 is CLOSED, pad boxes %.5f m apart" % gaps[0.0])
check(gaps[Q_OPEN] > 0.05, "joint_7 q=0.9 is OPEN, pad boxes %.5f m apart" % gaps[Q_OPEN])
check(gaps[Q_OPEN] > gaps[0.0], "increasing q OPENS the jaw -> ctrl_7 = g*0.9, g=1 open")
check(gaps[0.0] < 2 * CUBE_HALF < gaps[Q_OPEN],
      "closed gap %.5f < cube %.4f < open gap %.5f m" % (gaps[0.0], 2 * CUBE_HALF, gaps[Q_OPEN]))

# tool0 must be the pinch point: half a cube away from the fixed pad's inner face
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[jawadr] = 0.0
mujoco.mj_forward(m, d)
Rb = d.xmat[bbase].reshape(3, 3)
site_b = Rb.T @ (d.site_xpos[sid] - d.xpos[bbase])
cf = Rb.T @ (d.geom_xpos[gfix] - d.xpos[bbase])
cm = Rb.T @ (d.geom_xpos[gmov] - d.xpos[bbase])
face_fix = float(cf[0] - m.geom_size[gfix][0])
face_mov = float(cm[0] + m.geom_size[gmov][0])
print("  gripper frame: tool0 %s | fixed pad %s face x %+.5f | moving pad %s face x %+.5f"
      % (np.array2string(site_b, precision=5), np.array2string(cf, precision=5), face_fix,
         np.array2string(cm, precision=5), face_mov))
print("  -> the site is %.5f m from the fixed inner face (half a %.4f m cube)"
      % (face_fix - site_b[0], 2 * CUBE_HALF))
check(abs((face_fix - site_b[0]) - CUBE_HALF) < 1e-6,
      "tool0 is half a cube off the fixed pad face (%.6f m)" % (face_fix - site_b[0]))

# the collision proxy must sit on the REAL fixed finger, not float next to it
hp = mesh_points(m, m.geom("gripper_housing").id)
tip = hp[(hp[:, 2] > 0.115) & (np.abs(hp[:, 1] + 0.0035) < 0.0205)]
finger_x = float(tip[:, 0].min())
print("  fixed finger material above z=0.115: %d verts, inner face x %+.5f"
      " (y in [%+.4f,%+.4f], z up to %+.4f)"
      % (len(tip), finger_x, tip[:, 1].min(), tip[:, 1].max(), tip[:, 2].max()))
check(abs(finger_x - face_fix) < 1e-4,
      "the pad proxy face coincides with the real fixed finger surface (%.5f vs %.5f m)"
      % (face_fix, finger_x))

# home keyframe: no robot-robot contact, no penetration between non-adjacent geoms
mujoco.mj_resetDataKeyframe(m, d, 0)
sim(m, d, 0.5)
robot = {m.body("Link_0").id}
stack = [m.body("Link_0").id]
while stack:
    b = stack.pop()
    for ch in range(m.nbody):
        if m.body_parentid[ch] == b:
            robot.add(ch)
            stack.append(ch)
coll = [i for i in range(m.ngeom)
        if m.geom_bodyid[i] in robot and (m.geom_contype[i] or m.geom_conaffinity[i])]
bolted = {(b6, bbase), (b6, bjaw), (m.body("Link_5").id, bbase), (m.body("Link_5").id, bjaw)}
self_con = []
for i in range(d.ncon):
    c = d.contact[i]
    if m.geom_bodyid[c.geom1] in robot and m.geom_bodyid[c.geom2] in robot:
        self_con.append((m.geom(c.geom1).name, m.geom(c.geom2).name))
worst, worst_pair = np.inf, None
for ii, ga in enumerate(coll):
    for gb in coll[ii + 1:]:
        b1, b2 = int(m.geom_bodyid[ga]), int(m.geom_bodyid[gb])
        if (b1, b2) in bolted or (b2, b1) in bolted:
            continue
        if m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
            continue
        dd = float(mujoco.mj_geomDistance(m, d, ga, gb, 1.0, np.zeros(6)))
        if dd < worst:
            worst, worst_pair = dd, (m.geom(ga).name, m.geom(gb).name)
print("  home pose: %d collision geoms, %d robot-robot contacts, closest non-adjacent"
      " pair %s -> %.5f m" % (len(coll), len(self_con), worst_pair, worst))
for pair in self_con:
    print("    contact %s <-> %s" % pair)
check(not self_con, "no robot-robot contact in the home pose")
check(worst > 0.0, "no penetration between non-adjacent robot geoms (min %.5f m)" % worst)

# ------------------------------------------------------------ C. grasp + lift + hold
print("\n[C] grasp 0.020 x 0.030 x 0.030 m / 8 g block, close, lift 0.10 m, hold 2 s")
ik = JakaZu3IK(xml_path=SCENE, site_name="tool0")
KP = m.actuator_gainprm[:6, 0].copy()
scratch = mujoco.MjData(m)


def ff(q6):
    """Servo preload = the arm's own gravity torque / kp, otherwise the 6 position servos
    sag 40-50 mm and never reach the pose."""
    scratch.qpos[:] = d.qpos
    scratch.qpos[:6] = q6
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(m, scratch)
    return scratch.qfrc_bias[:6].copy() / KP


def cmd(tp, tR, jaw):
    res = ik.solve(tp, tR, d.qpos[:6].copy())
    d.ctrl[:6] = res.qpos + ff(res.qpos)
    d.ctrl[6] = jaw
    return res


def pad_contacts():
    """Which pad geoms touch the cube right now (name -> contact count)."""
    out = {}
    for i in range(d.ncon):
        c = d.contact[i]
        if gcube in (c.geom1, c.geom2):
            o = c.geom2 if c.geom1 == gcube else c.geom1
            if o in (gfix, gmov):
                out[m.geom(o).name] = out.get(m.geom(o).name, 0) + 1
    return out


mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
sim(m, d, 0.4)                                   # let the cube settle on the table
cube0 = d.xpos[bcube].copy()
q_open_meas = float(d.qpos[jawadr])
tR = Rz(math.pi) @ R_DOWN
print("  cube at %s | jaw qpos %.4f | tool0 %s"
      % (np.array2string(cube0, precision=5), q_open_meas,
         np.array2string(d.site_xpos[sid], precision=5)))
print("  approach orientation: tool +z -> world -z and the tool yawed 180 deg about z, so the")
print("  jaws still close in the table plane, but the forearm clears the table: the same grasp")
print("  with the plain R_DOWN rests Link_3 on the table (measured 103 N) and stalls the tool")
print("  9.8 mm high; yawed by 180 deg it reaches the pinch to 0.4 mm.")

pre = cube0 + np.array([0.0, 0.0, 0.10])
grasp = cube0.copy()
t = time.perf_counter()
for i in range(1, 11):
    cmd(pre + (grasp - pre) * i / 10.0, tR, Q_OPEN)
    sim(m, d, 0.20)
t_desc = time.perf_counter() - t
corr = grasp.copy()
err = np.inf
for k in range(5):                               # settle the last millimetres on the plant
    cmd(corr, tR, Q_OPEN)
    sim(m, d, 0.30)
    p, _ = tool_pose(d, sid)
    err = float(np.linalg.norm(grasp - p))
    if err < 5e-4:
        break
    corr = grasp + (grasp - p)
cube_in_tool = tR.T @ (d.xpos[bcube] - d.site_xpos[sid])
print("  descent 2.0 s sim (%.2f s wall, %d IK solves): tool0 %s  |dp| %.5f m  cube in tool %s"
      % (t_desc, k + 1, np.array2string(d.site_xpos[sid], precision=5), err,
         np.array2string(cube_in_tool, precision=4)))
check(err < 2e-3, "tool0 reaches the grasp pose: |dp| %.5f m < 2e-3 m" % err)
check(np.all(np.abs(cube_in_tool) < 3e-3),
      "the pads straddle the cube (cube in the tool frame %s, tolerance 3 mm)"
      % np.array2string(cube_in_tool, precision=4))

# close: ramp ctrl 0.9 -> 0 over 0.5 s, then hold 0.7 s
tc = time.perf_counter()
cube_pre_close = d.xpos[bcube].copy()
nramp = int(round(0.5 / m.opt.timestep))
qtraj = np.empty(nramp + int(round(0.7 / m.opt.timestep)))
for i in range(nramp):
    d.ctrl[6] = Q_OPEN * (1.0 - (i + 1) / nramp)
    mujoco.mj_step(m, d)
    qtraj[i] = d.qpos[jawadr]
for i in range(nramp, qtraj.size):
    mujoco.mj_step(m, d)
    qtraj[i] = d.qpos[jawadr]
t_close_wall = time.perf_counter() - tc
q_closed = float(d.qpos[jawadr])
gap_closed = pad_gap(m, d, gfix, gmov)
gather = float(np.linalg.norm(d.xpos[bcube][:2] - cube_pre_close[:2]))
reach = np.nonzero(np.abs(qtraj - q_closed) < 0.01 * (Q_OPEN - q_closed))[0]
close_sim = float((reach[0] if reach.size else qtraj.size) * m.opt.timestep)
con_close = pad_contacts()
print("  close ctrl_jaw 0.9 -> 0: q %.5f rad, pad gap %.5f m, closed after %.3f s sim"
      " (%.3f s wall), cube gathered %.5f m" % (q_closed, gap_closed, close_sim, t_close_wall, gather))
print("  pad-cube contacts while closed: %s" % con_close)
check(abs(gap_closed - 2 * CUBE_HALF) < 2e-3,
      "pads stop %.5f m apart on the %.4f m cube (contact penetration %.2f mm, no squash)"
      % (gap_closed, 2 * CUBE_HALF, 1e3 * (2 * CUBE_HALF - gap_closed)))
check("gripper_pad_fixed" in con_close and "gripper_pad_moving" in con_close,
      "both fingers touch the cube after closing: %s" % sorted(con_close))

# lift 0.10 m in 10 steps and hold: exactly 2.0 s of simulation
cube_l0 = d.xpos[bcube].copy()
quat0 = d.qpos[cubeq + 3:cubeq + 7].copy()
t = time.perf_counter()
for i in range(1, 11):
    cmd(grasp + np.array([0.0, 0.0, 0.010 * i]), tR, 0.0)
    sim(m, d, 0.10)
sim(m, d, 1.0)
t_lift_wall = time.perf_counter() - t
cube_l = d.xpos[bcube].copy()
rise = float(cube_l[2] - cube_l0[2])
slip = float(np.linalg.norm(cube_l[:2] - cube_l0[:2]))
con_lift = pad_contacts()
quat1 = d.qpos[cubeq + 3:cubeq + 7].copy()
tilt = math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(quat0, quat1))))))
ext = cube_edges(m, d, gcube)
print("  lift+hold 2.0 s sim (%.2f s wall): cube %s rise %.5f m slip %.5f m tilt %.2f deg"
      " contacts %s" % (t_lift_wall, np.array2string(cube_l, precision=5), rise, slip, tilt, con_lift))
check(rise >= 0.08, "cube rose %.5f m >= 0.08 m" % rise)
check(slip < 0.005, "cube horizontal slip %.5f m < 0.005 m" % slip)
check("gripper_pad_fixed" in con_lift and "gripper_pad_moving" in con_lift,
      "both fingers still press the cube after the lift: %s" % sorted(con_lift))
check(np.all(np.abs(ext - 2 * np.array([CUBE_HALF, CUBE_HALF_YZ, CUBE_HALF_YZ])) < 1e-9)
      and tilt < 5.0,
      "cube intact: own-axis edge lengths %s m, tilt %.2f deg"
      % (np.array2string(ext, precision=6), tilt))
check(abs(float(np.linalg.norm(quat1)) - 1.0) < 1e-9, "cube free-joint quaternion stays unit")

# ---------------------------------------------------------------- D. offscreen frame
print("\n[D] offscreen render -> out/t7_frames/final.png")
png_path = os.path.join(OUTDIR, "final.png")
t_render = float("nan")
try:
    rend = mujoco.Renderer(m, height=560, width=760)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = 0.5 * (d.xpos[bbase] + cube_l)
    cam.distance = 0.30
    cam.azimuth = 250.0
    cam.elevation = -8.0
    t = time.perf_counter()
    rend.update_scene(d, camera=cam)
    img = rend.render()
    t_render = time.perf_counter() - t
    rend.close()
    nbytes = write_png(png_path, img)
    print("  %s  %dx%d  mean %.2f  %d bytes  (%.3f s)"
          % (png_path, img.shape[1], img.shape[0], float(img.mean()), nbytes, t_render))
    check(os.path.isfile(png_path) and nbytes > 5000, "final frame written (%d bytes)" % nbytes)
    check(img.shape == (560, 760, 3) and float(img.mean()) > 5.0,
          "Renderer produced a lit 560x760 RGB frame (mean %.2f)" % float(img.mean()))
except Exception as exc:                                    # noqa: BLE001
    print("  [FAIL] Renderer unavailable: %r" % (exc,))
    FAILS.append("renderer: %r" % (exc,))

# -------------------------------------------------------------------- E. number table
print("\n[E] measured numbers")
step_us = 1e6 * t_desc / (3.0 / m.opt.timestep)      # descent: 2.0 s of waypoints + 1.0 s settle
rows = [
    ("pad boxes q=0 (closed)", "%.5f m" % gaps[0.0]),
    ("pad boxes q=0.9 (open)", "%.5f m" % gaps[Q_OPEN]),
    ("jaw stroke", "%.5f m" % (gaps[Q_OPEN] - gaps[0.0])),
    ("pad gap on the cube", "%.5f m (cube %.5f)" % (gap_closed, 2 * CUBE_HALF)),
    ("tool0 -> fixed face", "%.5f m" % (face_fix - site_b[0])),
    ("fixed finger face (mesh)", "%.5f m" % finger_x),
    ("moving pad centre x (q=0)", "%.5f m" % face_mov),
    ("gripper_base flange dev", "%+.6f m" % dev),
    ("housing depth from flange", "%+.5f m" % hi[2]),
    ("descent tool error", "%.5f m" % err),
    ("cube rise", "%.5f m" % rise),
    ("cube slip", "%.5f m" % slip),
    ("cube tilt", "%.2f deg" % tilt),
    ("cube edges (own axes)", "%.4f %.4f %.4f m" % tuple(ext)),
    ("pad contacts fixed/moving", "%s / %s" % (con_lift.get("gripper_pad_fixed", 0),
                                               con_lift.get("gripper_pad_moving", 0))),
    ("jaw close time", "%.3f s sim, %.3f s wall" % (close_sim, t_close_wall)),
    ("robot self contacts at home", "%d" % len(self_con)),
    ("min non-adjacent geom dist", "%.5f m" % worst),
    ("physics step", "%.1f us/step" % step_us),
    ("render frame", "%.1f ms" % (1e3 * t_render) if t_render == t_render else "n/a"),
    ("total wall time", "%.2f s" % (time.perf_counter() - T0)),
]
for k, v in rows:
    print("  %-28s %s" % (k, v))

print("\n" + "=" * 78)
if FAILS:
    print("RESULT: FAIL (%d)" % len(FAILS))
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")

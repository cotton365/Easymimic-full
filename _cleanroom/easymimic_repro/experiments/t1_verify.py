#!/usr/bin/env python
"""T1 acceptance check: JAKA Zu3 MuJoCo assets + placeholder gripper + a real grasp.

    ~/miniconda3/envs/mujoco/bin/python easymimic_repro/experiments/t1_verify.py

Every path is derived from this file's own location, so the results are identical from any
working directory (the acceptance run uses cwd=/ and cwd=<repo root>).
"""
import os
import sys
import math
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets", "jaka_zu3"))
SCENE = os.path.join(ASSETS, "scene_cube_bowl.xml")
ROBOT = os.path.join(ASSETS, "robot.xml")
MESHDIR = os.path.abspath(os.path.join(ASSETS, "..", "..", "third_party", "jaka-zu3-mujoco",
                                       "assets", "jaka_zu3_meshes"))

FAILS = []


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)
    return cond


def rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    return ang / (2.0 * math.sin(ang)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def tool_pose(d, sid):
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()


def sim(m, d, secs):
    for _ in range(int(round(secs / m.opt.timestep))):
        mujoco.mj_step(m, d)


def ik_tool(m, d, sid, target_pos, target_R, qids, iters=400, lam=2e-3, step=0.35):
    """Damped least squares on joints qids, tool orientation held fixed."""
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        pos, R = tool_pose(d, sid)
        ep = target_pos - pos
        er = rotvec(target_R @ R.T)
        if np.linalg.norm(ep) < 1e-7 and np.linalg.norm(er) < 1e-7:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        J = np.vstack([jacp, jacr])[:, qids]
        dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6),
                                   np.concatenate([ep, er]))
        d.qpos[qids] += np.clip(dq, -step, step)
    mujoco.mj_kinematics(m, d)
    pos, R = tool_pose(d, sid)
    return np.linalg.norm(target_pos - pos), np.linalg.norm(rotvec(target_R @ R.T))


def servo_to(m, d, sid, target_pos, target_R, ctrl, qids, iters=10, secs=0.25):
    """Closed loop on the real plant: the position servos have a gravity droop, so the
    commanded joint targets are corrected from the measured tool pose until the tool
    actually sits on the target. Returns the number of 0.25 s segments simulated."""
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    used = 0
    for _ in range(iters):
        pos, R = tool_pose(d, sid)
        ep = target_pos - pos
        er = rotvec(target_R @ R.T)
        if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        J = np.vstack([jacp, jacr])[:, qids]
        dq = J.T @ np.linalg.solve(J @ J.T + 4e-6 * np.eye(6), np.concatenate([ep, er]))
        ctrl[qids] += np.clip(dq, -0.15, 0.15)
        ctrl[qids] = np.clip(ctrl[qids], m.actuator_ctrlrange[qids, 0],
                             m.actuator_ctrlrange[qids, 1])
        d.ctrl[:] = ctrl
        sim(m, d, secs)
        used += 1
    return used


print("=" * 78)
print("SCRIPT :", os.path.abspath(__file__))
print("CWD    :", os.getcwd())
print("python :", sys.executable)
print("mujoco :", mujoco.__version__, "| numpy", np.__version__)
print("=" * 78)

# ---------------------------------------------------------------- 1. load model
print("\n[1] load scene by ABSOLUTE path")
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
mr = mujoco.MjModel.from_xml_path(ROBOT)
print("  scene      :", SCENE)
print("  nq=%d nv=%d nu=%d nbody=%d ngeom=%d nmesh=%d njnt=%d"
      % (m.nq, m.nv, m.nu, m.nbody, m.ngeom, m.nmesh, m.njnt))
print("  robot.xml  : nq=%d nu=%d  (6 arm hinges + 1 gripper slide)" % (mr.nq, mr.nu))
check(m.nu == 7, "scene nu == 7 (6 arm + 1 gripper position actuator)")
check(mr.nq == 7, "robot subtree nq == 7 (6 arm hinges + 1 gripper slide)")
print("  scene nq   : %d = 7 robot DoF + 7 cube free-joint qpos (3 pos + 4 quat)." % m.nq)
print("               a graspable cube must be a free body, so a scene-level nq==7 is")
print("               physically impossible; the invariant the brief states (6 arm +")
print("               1 gripper) is asserted on the robot subtree (mr.nq=7).")

# ------------------------------------------------- 2. names, meshes, portability
print("\n[2] joints / actuators / meshes")
for i in range(m.njnt):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    print("  joint %d  %-14s type=%d qposadr=%d axis=%s range=[%.4f, %.4f]"
          % (i, n, m.jnt_type[i], m.jnt_qposadr[i],
             np.array2string(m.jnt_axis[i], precision=1),
             m.jnt_range[i][0], m.jnt_range[i][1]))
for i in range(m.nu):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0])
    print("  actuator %d  %-18s joint=%-12s kp=%.0f kv=%.0f ctrlrange=%s"
          % (i, n, jn, -m.actuator_biasprm[i][1], -m.actuator_biasprm[i][2],
             np.array2string(m.actuator_ctrlrange[i], precision=4)))

print("  meshdir declared  : ../../third_party/jaka-zu3-mujoco/assets/jaka_zu3_meshes")
print("  resolved against  : %s" % ASSETS)
print("  -> absolute       : %s" % MESHDIR)
mesh_ok = True
for i in range(m.nmesh):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i)
    f = os.path.join(MESHDIR, n.replace("_mesh", ".STL"))
    ok = os.path.isfile(f)
    mesh_ok &= ok
    adr, num = m.mesh_vertadr[i], m.mesh_vertnum[i]
    lo = m.mesh_vert[adr:adr + num].min(axis=0)
    hi = m.mesh_vert[adr:adr + num].max(axis=0)
    print("  mesh %d %-12s %s\n            exists=%s verts=%6d bbox_z=[%.4f, %.4f]"
          % (i, n, f, ok, num, lo[2], hi[2]))
check(mesh_ok and m.nmesh == 7, "7 STL meshes resolved to absolute existing files")

# --------------------------------------------------------- 3. grasp hard test
print("\n[3] grasp test: tool over the cube, jaw commanded closed, lift 0.10 m")
sid = m.site("tool0").id
print("  timestep=%.4f integrator=%d gravity=%s"
      % (m.opt.timestep, m.opt.integrator, np.array2string(m.opt.gravity, precision=2)))

qids = np.arange(6)
jnt_jaw = m.joint("gripper_jaw")
jawadr = int(m.jnt_qposadr[jnt_jaw.id])
gid_fix = m.geom("gripper_tip_fixed").id
gid_mov = m.geom("gripper_tip_moving").id
bid_cube = m.body("cube").id
cubeq = int(m.jnt_qposadr[m.joint("cube_free").id])

mujoco.mj_resetDataKeyframe(m, d, 0)          # keyframe "home": arm raised, jaw open
mujoco.mj_forward(m, d)
home_pos, home_R = tool_pose(d, sid)
print("  home tool0 = %s" % np.array2string(home_pos, precision=5))

# grasp with the tool axis vertical: that is the tool orientation at the arm's zero
# posture. The "home" keyframe leans the whole arm by ~8.7 deg (joint_2 = 0.15) and would
# tilt the pads, so only its horizontal position is reused.
d.qpos[:7] = 0.0
mujoco.mj_kinematics(m, d)
_, R_vert = tool_pose(d, sid)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
grasp_pos = np.array([home_pos[0], home_pos[1], 0.858])
ep, er = ik_tool(m, d, sid, grasp_pos, R_vert, qids)
print("  IK -> grasp %s (tool axis vertical) |dp|=%.2e |dR|=%.2e"
      % (np.array2string(grasp_pos, precision=5), ep, er))
check(ep < 1e-4 and er < 1e-4, "kinematic IK reaches the grasp pose over the cube")

# cube on the table, centred under the tool
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[cubeq:cubeq + 7] = np.array([grasp_pos[0], grasp_pos[1], 0.81, 1.0, 0.0, 0.0, 0.0])
ctrl = d.ctrl.copy()                          # jaw open (0.04) from the keyframe
mujoco.mj_forward(m, d)
sim(m, d, 0.3)                                # let the cube settle on the table
z_settle = float(d.xpos[bid_cube][2])
print("  cube settled on table at z=%.5f  ncon=%d" % (z_settle, d.ncon))

# 3a: bring the OPEN gripper down over the cube (closed loop on the real plant)
n1 = servo_to(m, d, sid, grasp_pos, R_vert, ctrl, qids)
pos_c, _ = tool_pose(d, sid)
print("  descended, tool0=%s (%.2f s)  jaw open=%.4f  ncon=%d"
      % (np.array2string(pos_c, precision=5), 0.25 * n1, d.qpos[jawadr], d.ncon))

# 3b: close the gripper: ctrl=0 is the closed command
ctrl[6] = 0.0
d.ctrl[:] = ctrl
sim(m, d, 0.5)
jaw_closed = float(d.qpos[jawadr])
z0 = float(d.xpos[bid_cube][2])
p_fix0 = d.geom_xpos[gid_fix].copy()
p_mov0 = d.geom_xpos[gid_mov].copy()
print("  3b closed: ctrl_jaw=0 -> jaw qpos=%.5f  cube z=%.5f  ncon=%d"
      % (jaw_closed, z0, d.ncon))
print("     tip_fixed =%s" % np.array2string(p_fix0, precision=5))
print("     tip_moving=%s" % np.array2string(p_mov0, precision=5))

# 3c: lift 0.10 m, exactly 2.0 s of simulation
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.10])
tool_z0 = float(pos_c[2])
n2 = servo_to(m, d, sid, lift_pos, R_vert, ctrl, qids, iters=8, secs=0.25)
sim(m, d, max(0.0, 2.0 - 0.25 * n2))          # top the lift up to a full 2.0 s
pos_l, _ = tool_pose(d, sid)
print("  lift commanded 0.100 m, then %.2f s servo + %.2f s hold = %.2f s simulated"
      % (0.25 * n2, max(0.0, 2.0 - 0.25 * n2), 2.0))
print("  tool rise %.5f m" % (float(pos_l[2]) - tool_z0))

z1 = float(d.xpos[bid_cube][2])
rise = z1 - z0
p_fix1 = d.geom_xpos[gid_fix].copy()
p_mov1 = d.geom_xpos[gid_mov].copy()
cube_pos = d.xpos[bid_cube].copy()
jaw_q = float(d.qpos[jawadr])

print("  after lift: cube z=%.5f (rise %.5f) jaw=%.5f ncon=%d"
      % (z1, rise, jaw_q, d.ncon))
print("     tip_fixed =%s" % np.array2string(p_fix1, precision=5))
print("     tip_moving=%s" % np.array2string(p_mov1, precision=5))
print("     cube      =%s" % np.array2string(cube_pos, precision=5))

ax = d.geom_xmat[gid_fix].reshape(3, 3)[:, 0]
ax = ax / np.linalg.norm(ax)                  # world jaw axis, pad half thickness 0.005
f_fix = float((p_fix1 + 0.005 * ax) @ ax)
f_mov = float((p_mov1 - 0.005 * ax) @ ax)
lo, hi = min(f_fix, f_mov), max(f_fix, f_mov)
proj = float(cube_pos @ ax)
print("  jaw axis (world) = %s" % np.array2string(ax, precision=4))
print("  pad inner faces on jaw axis: %.5f .. %.5f  (gap %.5f m)" % (lo, hi, hi - lo))
print("  cube centre on jaw axis    : %.5f" % proj)

check(rise >= 0.08, "cube lifted >= 0.08 m (measured %.5f m)" % rise)
check(lo + 0.0005 < proj < hi - 0.0005,
      "cube still between the fingers (proj %.5f inside (%.5f, %.5f))" % (proj, lo, hi))
check(abs(float(cube_pos[2] - p_fix1[2])) < 0.06,
      "cube still at fingertip height (dz=%.5f m)" % abs(float(cube_pos[2] - p_fix1[2])))
check(hi - lo < 0.0205,
      "pads are pressing the cube, not merely straddling it (gap %.5f m vs cube 0.02000)"
      % (hi - lo))
check(jaw_q < 0.002, "jaw servo settled closed on the cube (qpos %.5f m)" % jaw_q)

# ------------------------------------------------------------- 4. final report
print("\n[4] summary numbers")
print("  final cube z          : %.5f m  (start %.5f, rise %.5f)" % (z1, z0, rise))
print("  gripper_tip_fixed     : %s" % np.array2string(p_fix1, precision=5))
print("  gripper_tip_moving    : %s" % np.array2string(p_mov1, precision=5))
print("  gripper_jaw axis      : %s" % np.array2string(jnt_jaw.axis, precision=4))
print("  gripper_jaw range     : [%.4f, %.4f] m  (qpos now %.5f)"
      % (jnt_jaw.range[0], jnt_jaw.range[1], jaw_q))
print("  pad half-size         : fixed=%s moving=%s"
      % (np.array2string(m.geom_size[gid_fix], precision=5),
         np.array2string(m.geom_size[gid_mov], precision=5)))
print("  pad pos (local)       : fixed=%s moving=%s"
      % (np.array2string(m.geom_pos[gid_fix], precision=5),
         np.array2string(m.geom_pos[gid_mov], precision=5)))
print("  pad condim/friction   : %d / %s"
      % (m.geom_condim[gid_fix], np.array2string(m.geom_friction[gid_fix], precision=4)))
print("  pad contype/conaff    : %d / %d"
      % (m.geom_contype[gid_fix], m.geom_conaffinity[gid_fix]))
print("  collision geoms (group 3):")
for i in range(m.ngeom):
    if m.geom_group[i] == 3:
        print("    %-20s type=%d size=%s pos=%s"
              % (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i), m.geom_type[i],
                 np.array2string(m.geom_size[i], precision=4),
                 np.array2string(m.geom_pos[i], precision=4)))

# ---------------------------------------------------------------- 5. rendering
print("\n[5] offscreen render")
try:
    r = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = [0.474, 0.0435, 0.86]
    cam.distance = 0.55
    cam.azimuth = 130.0
    cam.elevation = -18.0
    r.update_scene(d, camera=cam)
    img = r.render()
    r.close()
    print("  frame shape=%s dtype=%s mean=%.2f min=%d max=%d"
          % (img.shape, img.dtype, float(img.mean()), int(img.min()), int(img.max())))
    check(img.shape == (480, 640, 3), "Renderer produced a 480x640x3 RGB frame")
    check(float(img.mean()) > 1.0, "frame is not blank (mean %.2f)" % float(img.mean()))
except Exception as exc:
    print("  [FAIL] Renderer unavailable: %r" % (exc,))
    FAILS.append("renderer: %r" % (exc,))

print("\n" + "=" * 78)
if FAILS:
    print("RESULT: FAIL (%d)" % len(FAILS))
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")

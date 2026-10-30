import os, numpy as np
REPO = "/home/zh/Desktop/EasyMimic/_cleanroom/easymimic_repro"
import mujoco

SCENE = os.path.join(REPO, "assets", "jaka_zu3", "scene_cube_bowl.xml")
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

sid = m.site("tool0").id
bid = m.body("gripper_pg2").id
print("site tool0 body_id=%d -> body name=%r" % (m.site_bodyid[sid], m.body(m.site_bodyid[sid]).name))
print("site tool0 local pos =", m.site_pos[sid], "| gripper_pg2 body local pos =", m.body_pos[bid])
print("delta site_world - body_world =", d.site_xpos[sid] - d.xpos[bid])

jnt = m.joint("gripper_jaw"); adr = int(m.jnt_qposadr[jnt.id])
d.qpos[adr] = 0.0
mujoco.mj_forward(m, d)
gfix = m.geom("gripper_tip_fixed").id
gmov = m.geom("gripper_tip_moving").id
ax = d.geom_xmat[gfix].reshape(3, 3)[:, 0]
f_fix = float((d.geom_xpos[gfix] + 0.005 * ax) @ ax)
f_mov = float((d.geom_xpos[gmov] - 0.005 * ax) @ ax)
print("pad inner faces at jaw=0: %.6f .. %.6f (gap %.6f)" % (f_fix, f_mov, f_mov - f_fix))
print("fixed geom size/pos =", m.geom_size[gfix], m.geom_pos[gfix])
print("moving body local pos =", m.body_pos[m.body("gripper_finger_b").id])
print("jaw joint type/axis/range =", m.jnt_type[jnt.id], m.jnt_axis[jnt.id], m.jnt_range[jnt.id])
print("gripper actuator ctrlrange =", m.actuator_ctrlrange[m.actuator("gripper_actuator").id],
      "gainprm0 =", m.actuator_gainprm[m.actuator("gripper_actuator").id][0],
      "biasprm[:2] =", m.actuator_biasprm[m.actuator("gripper_actuator").id][:2])
print("keyframe[0] name=%r qpos=%s" % (m.key(0).name, m.key_qpos[0]))
g = m.geom("table_top")
print("table_top size/pos =", m.geom_size[g.id], m.geom_pos[g.id], "-> top z = %.5f" % (m.geom_pos[g.id][2] + m.geom_size[g.id][2]))
print("cube geom size =", m.geom_size[m.geom("cube_geom").id], "condim =", m.geom_condim[m.geom("cube_geom").id])
print("pad condim =", m.geom_condim[gfix])

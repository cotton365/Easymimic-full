# 验收记录 · Verification

本文件记录**怎么复跑**、**实测输出长什么样**，以及**这些测试不能证明什么**。
原始输出未经修饰（只做了行裁剪与省略号标记）；凡是数字与 `README.md` §4 不一致，以本文件的复跑结果为准。
下面摘录里的 `/home/zh/...` 是作者本机的实测路径，用来证明输出没有被改写；换到你的机器上会变成你自己的路径，不必逐字对。

> **场景变更（真夹爪定稿之后）**：`assets/jaka_zu3/scene_cube_bowl.xml` 里的方块从 20 mm 立方改成 **2×3×3 cm 长方体**（捏合方向仍是 20 mm，因为 `tool0` 距固定指内面只有 10 mm，见 README §7）。下面 T1 摘录是 **20 mm 立方**那一次运行的原始输出，保留作为历史；当前重跑的实际值是 rise **0.09921 m**、slip 0.00037 m、pad gap on the cube **0.01871 m**、cube edges 0.0200/0.0300/0.0300 m，断言数量与判据都没有变。

环境快照（本次定稿实测）

| 项 | 值 |
|---|---|
| Python | 3.11.17（`/home/zh/miniconda3/envs/mujoco/bin/python`）|
| mujoco | 3.16.0 |
| numpy | 2.4.6 |
| 平台 | Linux x86_64 |
| 额外依赖 | 无（无 scipy / cv2 / torch / mink / pytest / pandas / pyarrow / imageio；`experiments/probe_env.py` 退出码 0，实测这 8 项全部 `MISSING`）|
| 上游模型 | jaka-zu3-mujoco @ `c592951`，`scripts/fetch_third_party.sh` 退出码 0 |
| 默认模型 | 仓库内 `assets/jaka_zu3/robot.xml`（`src/ee_retarget/ik_jaka_zu3.py:66`）|
| 图形 | `DISPLAY=:1`；离屏渲染通过。无 GL 环境未验证 |

本文档所有数字都来自本机一次完整的复跑，日志留在 `scratch/t8/`（不入库）。

---

## A. 8 维 EE + DLS-IK 验收

```bash
python tests_ee/test_ee_pipeline.py
```

退出码 **0**，末行 `ALL CHECKS PASSED`。本次实测 **88 条 `[PASS]`、0 条 `[FAIL]`**（`grep -c PASS` 会得到 89，因为末行 `ALL CHECKS PASSED` 也含 `PASS`）。

实测输出（摘录，省掉 `[1]` 里 21 条同构的逐字段断言）：

```text
==============================================================================
8-D EE action layer acceptance test (numpy + mujoco only)
src: /home/zh/Desktop/EasyMimic/easymimic_repro/src/ee_retarget
==============================================================================

[1] analytic synthetic hand (truth by construction)
  [PASS] g closed (d_min) == 0.0   g=0.000000000000000
  [PASS] g mid (d_mean) == 0.5   g=0.500000000000000
  [PASS] g open (d_max) == 1.0   g=1.000000000000000
  [PASS] rotvec round-trip mid (d_mean)   err=2.220e-16
  [PASS] T_base_cam position   err=0.000e+00
  [PASS] T_base_cam rotation   err=2.220e-16
  [PASS] palm-normal sign rule   err=2.220e-16  Z.ref=1.000000  z_flipped=False

[2] orthonormality / chirality
  [PASS] 50 random hand frames: |R^T R - I|, |det R - 1|, |X.Z| all < 1e-09   max |R^T R-I|=6.66e-16, max |det-1|=4.44e-16, max |X.Z|=9.63e-17

[3] NaN propagation (no silent zero-fill)
  [PASS] each of the 21 keypoints NaN -> all 8 outputs NaN
  [PASS] NaN R_wrist -> all-NaN output
  [PASS] NaN T_base_cam -> all-NaN output
  [PASS] finite input -> finite output
```

IK 部分。注意模型头两行：默认模型已经是仓库内的 `robot.xml`（`nq=7`），不再是带道具的上游场景（那是 `nq=20`）。

```text
  model: /home/zh/Desktop/EasyMimic/easymimic_repro/assets/jaka_zu3/robot.xml
  site 'tool0' at qpos0 = [0.49120, -0.11850, 0.82955]
  arm DoF columns [0, 1, 2, 3, 4, 5] (nq=7, nv=7; props have free joints, hence by name)

[4] IK convergence on 100 FK-sampled reachable targets
  warm start only (home seed, single DLS attempt): 66/100 converged
  position error  [m]    p50=6.615e-04  p95=9.673e-04  max=9.944e-04
  orientation err [deg]  p50=0.2166  p95=0.4105  max=0.4980
  success rate           100/100 = 100.0%
  mean DLS iterations    40.0  (mean extra restarts 1.06, max 16)
  [PASS] success rate >= 95%   measured 100.0%
  [PASS] converged solutions respect the tolerances

[5] joint limits respected
  [PASS] all 100 solutions inside the model joint limits   smallest distance to a limit = 0.000e+00 rad
  [PASS] within_limits() rejects an out-of-range vector
  joint limits  [rad]    [-6.28, 6.28], [-1.48, 4.62], [-3.05, 3.05], [-1.48, 4.62], [-6.28, 6.28], [-6.28, 6.28]

[5b] configurable TCP offset + analytic Jacobian vs finite differences
  [PASS] TCP offset composition   |dp|=0.000e+00
  [PASS] default solver is offset-free
  [PASS] Jacobian rows vs central differences   max |dJp|=1.02e-10  max |dJr|=2.78e-10
  [PASS] IK converges on a TCP-offset target   pos_err=3.148e-04 m ang_err=0.0805 deg iters=104

[6] end-to-end: synthetic hand -> EE(8) -> IK -> qpos -> FK
  trial 4: hand->EE residual 0.00e+00 m / 2.70e-06 deg | IK residual 9.359e-04 m / 0.2874 deg | converged=True iters=226 | g=0.500
  worst residual         9.359e-04 m / 0.2992 deg
```

十二组断言（按脚本打印的段号）：`[1]` 解析重建、`[2]` 右手系、`[3]` NaN 传播、`[4]` 100 个随机可达目标、`[5]` 限位、`[5b]` 可配置 TCP 偏移 + 中心差分雅可比、`[6]` 手→EE→关节→FK 回环、`[7]` 精确 180°、`[8]` π±1e-9 与 θ≈0 边界、`[9]` 5000 次随机 SO(3) 往返、`[10]` 与 `hand_to_ee` 版交叉对照、`[11]` IK 层遇到「同位置、姿态差 180°」的目标、`[12]` 默认模型与 `EASYMIMIC_JAKA_XML`。

### A2. `rotmat_to_rotvec` 的 π 分支（T10 的结果，`[7]`-`[12]`）

```text
[7] rotmat_to_rotvec at exactly 180 deg (before the fix: [0, 0, 0])
  diag(-1, 1, -1) -> v = [0.          3.141592654 0.         ]   (defective body returned [0. 0. 0.])
  [PASS] Rx(pi): |v| = pi and rotvec_to_rotmat(v) = R   |v|=3.14159265358979312  |v|-pi=+0.00e+00  max|R-R(v)|=1.225e-16  v=[3.141593 0.       0.      ]
  [PASS] 2000 random-axis exact 180 deg rotations   max ||v|-pi|=4.441e-16  max max|R-R(v)|=1.110e-15

[8] rotvec boundary: theta = pi +/- 1e-9, pi - 1e-12, and theta ~ 0
  [PASS] theta = 3.141592653590 (+/- 1e-12): finite, |v| <= pi, angle and matrix exact   max ||v|-theta|=8.88e-16  max|R-R(v)|=8.88e-16  finite=True  |v|<=pi=True
  [PASS] theta = 1e-16: finite and |v| = theta   max ||v|-theta|=2.47e-32  finite=True
  [PASS] identity -> [0, 0, 0] exactly   v = [0. 0. 0.]
  worst over the pi neighbourhood: max ||v|-theta|=1.332e-15  max|R-R(v)|=8.882e-16

[9] 5000 random SO(3) round-trips
  [PASS] max |R - rotvec_to_rotmat(rotmat_to_rotvec(R))| < 1e-12   measured 1.332e-15 over 5000 rotations
  [PASS] every |v| <= pi (no flipped half-turn)   0 of 5000 outside

[10] cross-check: ik_jaka_zu3.rotmat_to_rotvec vs hand_to_ee's
  [PASS] both describe the same rotation (max angle difference)   max 2.958e-06 deg over 5000 rotations
  [PASS] ik_jaka_zu3 round-trip on all 5000 is < 1e-12   max 1.110e-15
  [PASS] the reference (arccos + vee/sin, pi branch) is the weaker one   reference max 2.119e-09 vs 1.110e-15

[11] IK layer: target with the SAME position and a rotation 180 deg away
  Rx(pi)         pose_error = 0.000e+00 m / 179.999998 deg | rotvec = [3.141593 0.       0.      ] |v| = 3.141593
                 DLS orientation term = 0.628318531  (before the fix: 0.000000000)
                 solve() -> IKResult(converged=False, pos_err=1.921e-02 m, ang_err=31.1362 deg, iters=7, restarts=0)  |dq|_inf = 2.6075
  [PASS] Rx(pi): the attempt leaves the seed instead of stalling   iters=7 |dq|_inf=2.6075 (before the fix: iters=1 |dq|_inf=0.0)

[12] default model: no-arg JakaZu3IK() and EASYMIMIC_JAKA_XML
  [PASS] DEFAULT_XML_PATH is the repo asset   /home/zh/Desktop/EasyMimic/easymimic_repro/assets/jaka_zu3/robot.xml
  [PASS] DEFAULT_XML_PATH no longer points into third_party
  no-arg JakaZu3IK(): /home/zh/Desktop/EasyMimic/easymimic_repro/assets/jaka_zu3/robot.xml
  nq=7 nv=7 nu=7 njnt=7  site 'tool0'  loaded in 0.112 s
  [PASS] default model is arm + jaw only (nq = nu = njnt = 7, no free joint)   nq=7 nu=7 njnt=7
  [PASS] a relative xml_path is still rejected
  [PASS] EASYMIMIC_JAKA_XML overrides default_xml_path()   /home/zh/Desktop/EasyMimic/easymimic_repro/assets/jaka_zu3/scene_cube_bowl.xml
  [PASS] the old third_party default still fails to load under MuJoCo   Error: flex vertex/node body 'rag_0' cannot have joint damping; 

==============================================================================
ALL CHECKS PASSED
```

最后一条 `[PASS]` 解释了为什么默认模型必须换成本仓库的 `robot.xml`：上游那个模型在 mujoco 3.16.0 上根本加载不起来（`flexcomp` 不接受 joint damping）。

---

## B. MuJoCo 资产 + 抓取物理验收

```bash
python experiments/t1_verify.py
```

退出码 **0**，末行 `RESULT: ALL CHECKS PASSED`，24 条 `[PASS]` / 0 条 `[FAIL]`。实测输出（截断到关键行）：

```text
==============================================================================
SCRIPT : /home/zh/Desktop/EasyMimic/easymimic_repro/experiments/t1_verify.py
python : /home/zh/miniconda3/envs/mujoco/bin/python
mujoco : 3.16.0 | numpy 2.4.6
==============================================================================

[A] MjModel.from_xml_path(ABSOLUTE) from two working directories
  cwd=/                              nq=14 nv=13 nu=7 nmesh=11 ngeom=39  (0.24 s)
  cwd=/home/zh/Desktop/EasyMimic/easymimic_repro nq=14 nv=13 nu=7 nmesh=11 ngeom=39  (0.12 s)
  [PASS] scene loads from cwd=/ AND cwd=/home/zh/Desktop/EasyMimic/easymimic_repro (meshdir ../gripper/... resolves)
  [PASS] nq/nu == 14/7 (6 arm hinges + joint_7 + 7 cube free-joint dofs)  measured 14/7

[B] assembly: flange face, joint_7 direction, home self-collision
  [PASS] 4 gripper STLs resolved from /home/zh/Desktop/EasyMimic/easymimic_repro/third_party/jaka-zu3-mujoco/assets/gripper (11 meshes in the model, metres)
  [PASS] gripper_base origin lies ON the flange plane (dev 0.00e+00 m)
  [PASS] housing bottom face is on the flange (z -0.000000) and grows +0.1350 m AWAY from the arm
  min 3D distance between the two pad boxes: q=0 0.00049 m | q=0.226 0.01901 m | q=0.9 0.06329 m | stroke 0.06280 m
  [PASS] joint_7 q=0 is CLOSED, pad boxes 0.00049 m apart
  [PASS] joint_7 q=0.9 is OPEN, pad boxes 0.06329 m apart
  [PASS] increasing q OPENS the jaw -> ctrl_7 = g*0.9, g=1 open
  [PASS] closed gap 0.00049 < cube 0.0200 < open gap 0.06329 m
  gripper frame: tool0 [ 0.0172 -0.0035  0.127 ] | fixed pad [ 0.02885 -0.0035   0.127  ] face x +0.02720 | moving pad [ 0.02385 -0.0035   0.127  ] face x +0.02495
  -> the site is 0.01000 m from the fixed inner face (half a 0.0200 m cube)
  [PASS] tool0 is half a cube off the fixed pad face (0.010000 m)
  [PASS] the pad proxy face coincides with the real fixed finger surface (0.02720 vs 0.02722 m)
  [PASS] no robot-robot contact in the home pose
  [PASS] no penetration between non-adjacent robot geoms (min 0.04796 m)

[C] grasp 0.020 m / 8 g cube, close, lift 0.10 m, hold 2 s
  cube at [0.474   0.0435  0.80996] | jaw qpos 0.9000 | tool0 [ 0.49444 -0.1185   0.85437]
  descent 2.0 s sim (0.01 s wall, 1 IK solves): tool0 [0.47396 0.04354 0.81001]  |dp| 0.00008 m  cube in tool [-3.4224e-04 -3.4950e-05  4.9151e-05]
  [PASS] tool0 reaches the grasp pose: |dp| 0.00008 m < 2e-3 m
  [PASS] the pads straddle the cube (cube in the tool frame [-3.4224e-04 -3.4950e-05  4.9151e-05], tolerance 3 mm)
  close ctrl_jaw 0.9 -> 0: q 0.22220 rad, pad gap 0.01870 m, closed after 0.570 s sim (0.011 s wall), cube gathered 0.00031 m
  pad-cube contacts while closed: {'gripper_pad_fixed': 4, 'gripper_pad_moving': 4}
  [PASS] pads stop 0.01870 m apart on the 0.0200 m cube (contact penetration 1.30 mm, no squash)
  [PASS] both fingers touch the cube after closing: ['gripper_pad_fixed', 'gripper_pad_moving']
  lift+hold 2.0 s sim (0.02 s wall): cube [0.47312 0.04355 0.90785] rise 0.09790 m slip 0.00087 m tilt 0.11 deg contacts {'gripper_pad_fixed': 4, 'gripper_pad_moving': 4}
  [PASS] cube rose 0.09790 m >= 0.08 m
  [PASS] cube horizontal slip 0.00087 m < 0.005 m
  [PASS] both fingers still press the cube after the lift: ['gripper_pad_fixed', 'gripper_pad_moving']
  [PASS] cube intact: own-axis edge lengths [0.02 0.02 0.02] m, tilt 0.11 deg
  [PASS] cube free-joint quaternion stays unit

[D] offscreen render -> out/t7_frames/final.png
  /home/zh/Desktop/EasyMimic/easymimic_repro/out/t7_frames/final.png  760x560  mean 57.94  52951 bytes  (0.020 s)
  [PASS] final frame written (52951 bytes)
  [PASS] Renderer produced a lit 560x760 RGB frame (mean 57.94)

[E] measured numbers
  pad boxes q=0 (closed)       0.00049 m
  pad boxes q=0.9 (open)       0.06329 m
  jaw stroke                   0.06280 m
  pad gap on the cube          0.01870 m (cube 0.02000)
  tool0 -> fixed face          0.01000 m
  cube rise                    0.09790 m
  cube slip                    0.00087 m
  jaw close time               0.570 s sim, 0.011 s wall
  physics step                 7.9 us/step
  render frame                 20.4 ms
  total wall time              0.83 s

==============================================================================
RESULT: ALL CHECKS PASSED
```

### 路径可移植性（脚本自己测，不需要人肉跑两次）

`[A]` 段就是证据：同一次运行里从 `cwd=/` 与 `cwd=仓库根` 各加载一次，两次都是 `nq=14 nv=13 nu=7 nmesh=11 ngeom=39`、都 PASS。`assets/jaka_zu3/robot.xml:8` 的 `meshdir` 是相对自身文件目录的，所以与 cwd 无关。

> 历史遗留的结论要更正：**只有上游 MJCF 必须绝对路径加载**（顶层 `meshdir` 泄漏进 `<include>` 导致相对 mesh 解析失败）。现在默认模型是本仓库的 `robot.xml`，绝对路径要求仍写在 `src/ee_retarget/ik_jaka_zu3.py:220-224` 里（直接 `raise`），但它是保守约束，不是必需——`[12]` 里 `a relative xml_path is still rejected` 验证的是这条约束还在。

---

## C. 示教表回放（T9 的现役执行入口）

```bash
python experiments/pose_replay.py
```

退出码 **0**，末行 `RESULT: ALL CHECKS PASSED`。实测输出（全文，只有时间行会随机器波动）：

```text
== pose replay: configs/waypoints_cube_lift.json ==
[table ] 5 waypoints, meta.frame = base, meta.units = m, meta.control_hz = 50
[plan  ] 267 control frames, 5.320 s of table time, first frame |ee8 - wp[0]| = 0.0e+00

  segment                      frames   iters  |dp| end (m)  |dR| end (deg)  max |dp| (m)
  0: teleport -> 'approach'         1       9     7.71e-04         0.2999       7.71e-04
  1: -> 'descend'                  73     143     7.28e-04         0.3590       9.71e-03
  2: -> 'close'                    62      97     1.47e-03         0.2371       2.02e-03
  3: -> 'lift'                     42     111     1.13e-03         0.5787       2.28e-02
  4: -> 'hold'                     90      93     4.05e-04         0.4368       8.20e-04

[cube  ] z 0.81000 -> 0.92748 m, rise 0.11748 m
[slip  ] horizontal drift since the gripper closed: 0.000442 m
[pads  ] gripper_pad_fixed 4, gripper_pad_moving 4 contacts at the last frame
[hold  ] cube centre 0.00221 m from tool0 at the last frame
[droop ] gravity feedforward ON : |dp| 0.00077 m, dz +0.00040 m after 1.0 s hold
[droop ] gravity feedforward OFF: |dp| 0.04282 m, dz -0.04215 m after 1.0 s hold
[droop ] feedforward removes +0.04255 m of vertical sag
[time  ] 267 frames, 5.34 s sim, 0.78 s wall (2.9 ms per frame)
[frames] 54 PNG in out/t9_frames (last out/t9_frames/t9_f0265.png, 32175 bytes)

  criteria (hard-coded in src/pose_cmd/replay.py)
  [PASS] cube rise 0.11748 m >= 0.08000 m
  [PASS] cube horizontal slip 0.000442 m < 0.005000 m
  [PASS] gripper_pad_fixed contacts 4 >= 1 at the last frame
  [PASS] gripper_pad_moving contacts 4 >= 1 at the last frame
  [PASS] cube centre 0.00221 m from tool0 <= 0.04000 m (still in the jaws)
  [PASS] frames: 54 PNG for 267 control frames, one every 5 (54 expected)

[total ] 1.60 s wall
RESULT: ALL CHECKS PASSED
```

`--table` 换成显式路径跑同一条表：`python experiments/pose_replay.py --table configs/waypoints_cube_lift.json`，退出码 **0**，`[cube ] rise 0.11748 m`、`[slip ] 0.000442 m`、54 张 PNG，与上面逐位相同。

### C2. `--selftest`（不建场景，只跑表 schema + IK 反例）

```bash
python experiments/pose_replay.py --selftest
```

退出码 **0**，17 条 `[PASS]`，末行 `RESULT: ALL CHECKS PASSED`。摘录：

```text
== selftest: table validation, interpolation, IK failure ==
  schema
  [PASS] valid table accepted: 2 waypoints at 100 Hz
  [PASS] g = 1.5 rejected: waypoint table: waypoints[1].g = 1.5 is outside [0, 1] (0 = closed, 1 = open)
  [PASS] meta.frame = camera rejected: waypoint table: meta.frame = 'camera' is not supported (expected 'base': the 8-D action's pos and rotvec live in the robot base frame)
  interpolation
  [PASS] rotvec (0, +pi, 0) -> (0, -pi, 0) is the same pose at every sample (max |dR| = 2.4e-16)
  [PASS] near-pi pair sweeps 0.062832 rad, shortest arc 0.062832 rad (a component-wise ramp would sweep 6.2204)
  gripper mapping
  [PASS] g -> ctrl: g=1 -> 0.900, g=0 -> 0.000, g=0.5 -> 0.450
  outside the workspace
  error: waypoint x2: IK not converged, |dp|=1.2895e+00 m, |dR|=51.0903 deg
  [PASS] pos = [2.0, 0.0, 0.9] m is refused instead of replayed

[total ] 0.40 s wall
RESULT: ALL CHECKS PASSED
```

**这一节不能证明什么**

1. 只有一条轨迹、一个场景（2 cm 方块、桌面 z=0.80、固定起点）。换方块尺寸 / 桌面高度 / 起点没扫过，所以「成功率」是 1/1 条表，不是统计量。
2. 回放是开环的：逐帧解 IK、不读状态做反馈校正。段内最大瞬时偏差（第 3 段 2.28e-02 m）出现在段首目标跳变处，判据只看末帧。
3. 夹爪是位置指令不是力控：闭合到 g=0 时 pad gap 0.01870 m，对 2 cm 方块是 1.30 mm 穿透；8 g 方块能抬，更大 / 更软的物体没测过。
4. 姿态判据用 `atan2` 的矩阵夹角（`src/pose_cmd/replay.py:108-117`），刻意不走 rotvec 往返——π 附近 rotvec 有两个代表元，精度会花在转换上而不是判据上。

---

## D. 已退役：旧闭环（历史记录，勿再当验收命令）

**历史（已失效）**：`src/closed_loop/run_closed_loop.py` 曾把「合成手 → EE → IK → MuJoCo 抓取」串成闭环，判据是 `VERDICT: PASS 19/19`。它绑定的是 T7 之前的占位夹爪（`joint` 名 `gripper_jaw`、slide `range="0 0.04"`、`kp=300 kv=5`），并且在自己这边又补了一次 32.5 mm 的 TCP 偏移。

两条命令的实测结果（都退出码 1）：

```text
# 移动前（旧路径）
$ python src/closed_loop/run_closed_loop.py --no-render
KeyError: "Invalid name 'gripper_jaw'. Valid names: ['cube_free', 'joint_1', ..., 'joint_7']"   (src/closed_loop/run_closed_loop.py:409)
EXIT=1

# 现在（源码已搬到 legacy/，只加横幅，代码不修）
$ python legacy/closed_loop/run_closed_loop.py --no-render
ModuleNotFoundError: No module named 'hand_to_ee'   (legacy/closed_loop/run_closed_loop.py:54)
EXIT=1
```

移动后先挂 import（脚本按 `__file__` 往上找 `src/`，搬进 `legacy/` 后那个路径不存在），所以现在走不到 KeyError。撞 KeyError 之前它的 B 段已经 `[FAIL]`：0/75 单次收敛，max 位置残差 1.169e-02 m、max 姿态残差 8.108e+00 deg——根因就是「`site tool0` 已经是 TCP，却又补了一次偏移」。

退役方式：源码原样搬到 `legacy/closed_loop/`（`run_closed_loop.py` 804 行、`synth_hand.py` 189 行，各加 3 句横幅），不修、不删。现役替代是 §C。老数字（`0.09910 m` 抬升、`75/75` 单次收敛、`19/19` 判据）留在下面的历史段里，作为改资产之前的记录。

**这一节不能证明什么**：`legacy/` 里两个脚本在旧资产（占位夹爪）上是否还能跑，本次**没有验证**——旧资产已不在仓库里（git HEAD 里还能找到，但工作区已换成真夹爪）。

---

## E. 这些测试**不能**证明什么

- 不能证明真实视频输入可用：测试用的是闭式构造的合成手（真值由构造给出），`T_base_cam` 由测试注入（`tests_ee` 用单位阵）。
- 不能证明与真实手 / 真实相机的坐标系一致：`synth_hand` 是 `hand_to_ee` 的逆（同一套索引与符号约定），「重建误差 5.6e-17」证明的是两个函数的约定自洽。
- 不能证明真机可用：没有 JAKA 后端，没有手眼标定，没有真机跑过。
- 不能证明抓取统计意义上的鲁棒：所有抓取结论都来自 1 条轨迹 / 1 张示教表。
- 不能证明无 GL 环境下 `t1_verify.py` 的第 D 步（离屏渲染）能过：本次是在 `DISPLAY=:1` 上跑的。
- 不能证明 `hand_to_ee.rotmat_to_rotvec` 在 π 附近可用：它是弱实现（`src/ee_retarget/hand_to_ee.py:96-122`），`[10]` 实测它在 π 附近最差 2.119e-09、ik 版 1.110e-15；1.5e-3 rad 那个数字出自 `ik_jaka_zu3.py:134-141` 的 docstring（T10 记录），本次没有重测。

## F. 复现失败时先查这三件事

1. `third_party/` 是否已取回（`bash scripts/fetch_third_party.sh`，本机退出码 0、`third_party ready at c592951`）。`assets/jaka_zu3/robot.xml` 的 11 个 mesh 全靠它。
2. 是否在无 GL 的环境里跑 `t1_verify.py` 的最后一步渲染（前 4 步不需要 GL）。
3. 版本：mujoco 3.1.x 与 3.16.0 在 `mj_jacSite` 之外的细节上有差异，本记录只在 3.16.0 + numpy 2.4.6 上验证过。

---

## G. 8 维 ACT 策略闭环验收（真 checkpoint，50 次）

> 与 §A/§B/§C 不同，这一节不是资产或解码链的自测，而是**训出来的策略**在闭环里跑出的结果。
> checkpoint = `out/act_train_ee/checkpoints/last/pretrained_model`（= `002000`，2000 步，末段 loss 0.911）。
> 环境 = `~/miniconda3/envs/lerobot/bin/python`（torch + mujoco 3.1.6），`--device cpu`，策略跑满 200 拍或提前判成功。

```
export HF_HOME=$PWD/data/_scratch/hf_home HF_DATASETS_CACHE=$HF_HOME/datasets
/home/zh/miniconda3/envs/lerobot/bin/python experiments/act_rollout.py \
  --ckpt out/act_train_ee/checkpoints/last/pretrained_model \
  --trials 10 --seed <0..4> --device cpu --out scratch/ee_seed<n> --json-only --no-video
```

| seed | 成功 | rise (m) | slip 最大 (m) | tool offset 最大 (m) |
|---|---|---|---|---|
| 0 | 10/10 | 0.107243 – 0.12316 | 0.003552 | 0.012805 |
| 1 | 10/10 | 0.1119 – 0.1235 | 0.00525 | 0.00894 |
| 2 | 10/10 | 0.1075 – 0.1231 | 0.00370 | 0.01345 |
| 3 | 10/10 | 0.1072 – 0.1238 | 0.00338 | 0.01359 |
| 4 | 10/10 | 0.1041 – 0.1235 | 0.00381 | 0.01665 |

5 个 seed × 10 次 = **50/50 成功**；`ik_pos_err_max` 全程 ≤ 9.98e-04 m（0.45 deg），`ik_fallback_ticks` /
`ik_fail_ticks` / `ik_limit_ticks` / `bad_action_ticks` 全为 0，`infer_ms_mean` 1.3 ms/拍，seed 0 十次
总 wall 8.7 s。seed 1 那一次 `slip 0.00525` 是 50 次里唯一越过 §C 的 5 mm 记录线的一次。

### G1. 同一组扰动下 8 维 vs 7 维（可配对）

同一个 `--seed` 生成同一组方块初始偏移，所以两边是**逐次配对**的比较（`legacy/act_rollout_joint7.py`
用 `out/act_train/checkpoints/last`，即旧的关节 7 维策略）：

| 指标 | 8 维（EE） | 7 维（关节） |
|---|---|---|
| 成功 | 10/10 | 10/10 |
| slip 均值 | **2.80 mm** | 6.23 mm |
| slip 最大 | **3.55 mm** | 9.50 mm |
| slip > 5 mm 的次数 | **0/10** | **6/10** |
| rise | 0.1072 – 0.1232 m | 0.1262 – 0.1272 m |
| tool offset 最大 | 12.81 mm | 3.00 mm |

7 维逐次 slip（mm）：8.3 / 5.1 / 8.0 / 9.5 / 2.4 / 4.8 / 4.9 / 3.8 / 9.2 / 6.3。
8 维逐次 slip（mm）：3.3 / 1.2 / 3.5 / 3.0 / 2.8 / 3.3 / 2.9 / 3.6 / 3.3 / 1.1。

**要点：`classify()` 判成功只查 rise 与 tool offset，从不查 slip**（`experiments/act_rollout.py:41-42`、
`:140` 起的 `classify()` 只比较 `rise_m` 与 `tool_offset_m`；`max_slip_m` 只写进 `criteria` 字段留档）。
所以「10/10」的意思是「10 次都抬起来了」，不是「10 次都抓干净」——按 5 mm 抓取质量线算，旧 7 维策略是
4/10，8 维是 10/10。**比成功率之前先看 slip，否则成功率是虚的。**

### G2. 开环评测仍是 FAIL，但它 FAIL 在一个通道上

`scripts/eval_act_openloop.py --demos out/act_demos_ee --holdout-ep last`（107 步，留出 ep019）：

| | model | mean 基线 | persistence 基线 |
|---|---|---|---|
| overall MSE | 0.000317 | 0.025268（**79.8x**） | 0.000060（**0.190x**） |
| 其中 `g` 通道 | 0.002496 | 0.199308 | 0.000467 |

- 出口码 1 / `判定: FAIL` 是**判据设计**的结果，不是策略退化：`g` 一个通道就占了 overall 的
  **98.4%**（0.002496/8 ≈ 0.000312 vs 0.000317）。剔掉 `g`，其余 7 维平均 **5.7e-06**，persistence
  是 1.9e-06 —— 位姿通道已经贴到 persistence 的量级。
- 同一判据下 7 维基线是 0.001403，8 维是 0.000317，**好 4.4 倍**。
- 机制：episode 只有 108 拍，`chunk_size=100 / n_action_steps=100` 等于一次预测吃掉几乎整条轨迹，
  而 `g` 是只在第 ~46 拍翻转的阶跃；模型给的是被平滑过的翻转。§7 的 persistence 说明在这里同样适用。

### G3. 已知边界（别把这些当已验证）

- **`g` 头系统偏出 `[0,1]`**：闭环里 **1305/2000 拍（65%）的 `g` 被 `decode_pose8` 的 clip 拉回**。
  行为上等于饱和（顶到全开/全夹），所以当前无害；但说明 `g` 头没学会落在边界内，将来要做精细开度
  （软物、多档夹持）必须处理。
- **8 维夹得偏中心但夹得紧**：tool offset 12.81 mm vs 旧 3.00 mm（判据 40 mm 仍安全），slip 反而小
  2.2 倍。只观察到现象，没有做接触力分析。
- **只有位姿随机化，没有视觉随机化**：`RANDOM_X_M=0.02 / RANDOM_Y_M=0.005` 只移动方块，光照、纹理、
  噪声一律固定。仿真训练 → 真机执行时这一层是空的，需要单独做域差测量（比如挡掉一路相机看掉多少成功率）。
- **单场景、单方块、单物体尺寸**；`approach` 起点固定；没有失败样本，所以看不到失败模式。
- 全部在 CPU 上跑；`--video` 未开，本节不含 PNG。

---

## H. 放置任务（夹起 → 放进碗里）验收

> 这一节只有**数据层 + 回放层**，没有训练，也没有策略在环。两条来源分开写：
> **H1 是上游（建表的人）用自己的 scratch 探针跑出来的数字**，我一行代码都没参与；
> **H2 是我在本仓库用正式 CLI 复跑的数字**。两边不是同一份脚本，吻合是结论，不吻合就得查。

### H1 上游实测（表由上游建好并验证，我未改任何数字）

表 `configs/waypoints_cube_place.json`：8 航点、全部 `rotvec=[0,π,0]`、@20 Hz 共 **181 tick**
（`approach → descend → close → lift → carry → lower → release → retreat`，release 高度 **0.870 m**）。
碗几何：中心 `(0.300, -0.200)`、内半径 0.045 m、壁顶 z=0.840、底 z=0.802；方块静止中心 z=0.817。

| 项 | 上游数字 |
|---|---|
| 随机偏移下入碗 | **20/20** |
| `bowl_dxy`（末帧方块 xy 到碗心） | 0.0021 – 0.0046 m |
| 搬运全程是否掉块 | 不掉（末段 `joint_7` 仍 0.222） |
| IK 残差 | ≤ 9.9e-04 m |
| `rise_m` | ≈ +0.0020 m（放置任务抬升本来就≈0） |

### H2 本次复跑（本仓库正式 CLI，2026-11，CPU）

生成（mujoco env，**20/20 accepted，29.67 s**）：

```
$ /home/zh/miniconda3/envs/mujoco/bin/python experiments/gen_act_dataset.py \
    --table configs/waypoints_cube_place.json --out out/act_demos_place
20/20 episodes in 29.7 s, 20 attempts, rise 0.00196..0.00196 m, slip <= 0.316043 m
ik residual <= 9.97e-04 m, FK check <= 0.00e+00 m
in_bowl 20/20, bowl_dxy 0.00207..0.00466 m, hold_err <= 0.00119 m      EXIT=0
```

| 项 | 本次数字 | 门槛 | 与 H1 对照 |
|---|---|---|---|
| 条数 / `num_frames` | 20 条，每条 **181**（集合 `{181}`） | 20 / 181 | 一致 |
| `in_bowl` | **20/20 true** | 全 true | 一致 |
| `bowl_dxy_m` | **0.002069953829557699 – 0.004658764143423505** | ≤ 0.005 | 与 0.0021–0.0046 吻合 |
| `hold_err_m` | max **0.001191087275688795** | ≤ 0.002 | H1 未报此量 |
| `ik_err` max | **0.0009969513475666465** | ≤ 1.2e-3 | 与 ≤9.9e-04 吻合 |
| `rise_m` | 0.0019600533116050922 – 0.0019600533116057584 | 不参与判据 | 与 ≈+0.0020 吻合 |
| `slip_m` max | **0.31604300775805133** | — | 见下方注意 |

> **`slip_m` 在 place 里不是抓取质量**：它沿用 lift 的定义（夹紧之后的位移），而放置任务要把方块搬
> ~0.4 m，所以 0.28–0.32 m 是**正常值**，别拿 5 mm 线去卡。抓取质量看 `hold_err_m`（≤1.19 mm）。

导出（lerobot env，51 s）：`data/lerobot/act_pick_place`，读回 **num_episodes=20 / num_frames=3620**、
`action (8,)`、`observation.state (8,)`、`meta/info.json total_frames 3620`、
`meta/tasks.parquet` 单条任务 `pick up the red block and put it in the bowl`。

回放（mujoco env，1.9 s，不吃 policy）：

```
$ /home/zh/miniconda3/envs/mujoco/bin/python experiments/act_rollout.py \
    --replay-demos out/act_demos_place --task place --json-only --no-video \
    --out scratch/rollout_place_replay
summary: 20 trials, 20 success, rate 1.00                             EXIT=0
```

| 项 | 本次数字 | 门槛 |
|---|---|---|
| 成功 | **20/20**（`outcomes {"success": 20}`） | 20/20 |
| `bowl_dxy_m` | **0.004606 – 0.004751** | ≤ 0.005 |
| `in_bowl` | 20/20 true | 全 true |
| `hold_err_m` max | **0.0012182886434837516** | ≤ 0.002 |
| `ik_pos_err_max_m` | **0.0009992832630732616** | ≤ 2e-3 |
| 末拍指令 g | 20/20 = **1.0** | 测量值（判据只看方块是否脱离 tool0） |
| `state_dev0_max` | **0.0** | 0（首拍观测自洽） |

> 回放的 `bowl_dxy`（0.0046–0.0048）比生成的（0.0021–0.0047）**略大**：同一条位姿由 IK 走出不同的
> 关节解，落点跟着差零点几毫米，仍在 5 mm 门内。这不是数据不一致。

**lift 通路没被碰**：同一条回放命令把 `--task` 去掉（默认 `lift`）跑 `out/act_demos_ee`，`summary.json`
的顶层字段集合与改动前**逐项相同**（无新增/缺失字段），值只差 `per_trial` 里的 `ik_s/ik_ms/wall_s/pngs`
计时与渲染字段；20/20 成功、rise 0.1175–0.1187 m。旧产物三处的文件数与总字节数改动前后完全一致：
`out/act_demos_ee` 4362 / 45815859，`data/lerobot/act_pick_cube_ee` 5 / 45330518，
`out/act_train_ee` 44 / 2478498105。

测试（两条都是自测脚本，不是 pytest）：

| 命令 | checks | 退出码 |
|---|---|---|
| `tests_ee/test_act_dataset.py`（默认 lift 目录） | 14317 | 0 |
| `tests_ee/test_act_dataset.py --dir out/act_demos_place --episodes 20` | 23425 | 0 |
| `tests_ee/test_act_rollout.py --place-summary scratch/rollout_place_replay/summary.json` | 546 | 0 |

新增的 place 断言里有一条**负向性质**：造一个 `rise_m=0.12`（足够过 lift 判据）但 `in_bowl=False` 的
record，`classify_place()` 必须判 `not_in_bowl`，而同一个 record 喂给 lift 的 `classify()` 是 `success`
—— 判据不认 `rise`，这条就是证据。

### H3 负向变异（证明断言真的在管人管的事）

| 变异 | 结果 |
|---|---|
| `ANCHORED_LABELS` 改成 `()` | **EXIT=1**，`AssertionError: shifted_table: 'carry' moved by (+0.0170, +0.0040), expected (+0.0000, +0.0000) - the bowl does not follow the cube offset` |

变异后已用备份还原，`md5sum` 与备份逐字节相同，重跑回到 546 checks / EXIT=0。
`tests_ee/test_act_dataset.py` 不 import 生成器（只验磁盘产物），所以这个变异对它无效是**设计如此**。

### H4 这一节不能证明什么

- **没有训练、没有策略闭环**：`out/act_train_place` 不存在，`--task place` 的策略路径一次都没跑过。
  20/20 是**示教表回放**，证明的是表 + 物理 + 判据自洽，不证明 ACT 能学会。
- `--viewer` 的窗口**没有真的打开**（用户桌面上不许弹窗），只做了 stub：把 `play_in_viewer` 换成记录器，
  断言不带 `--viewer` 时**不被调用**、带 `--viewer` 时**被调用一次**且参数为
  `(SCENE, 1 条 (181,14) 的 trace, CONTROL_HZ, --viewer-passes=3)`。
- 随机化只有方块 xy（±20/±5 mm）与方块 yaw；碗固定、光照/纹理/相机噪声全固定，`in_bowl` 因此只是个
  几何判据，不代表真机上"放进去"的成功率。
- 只有一种方块尺寸、一个固定碗；`release` 高度 0.870 m 是手调的，没有扫过参数。

### H5 真策略闭环（20k 训练，10 个检查点 × 20 次）与判据修正

前面 H1–H4 说的都是**脚本化示教**（表 + IK + 物理）。这一节是**真 checkpoint**：`out/act_train_place_20k`
（20000 步，loss 平台 ≈2.37、`grdn` ≈30 不降），10 个检查点各跑 20 次、同 20 组扰动、seed 0、CPU。

判据先后有两版，差别是本节的核心：

| 版本 | "已松手"的证据 | step 6000 的结果 |
|---|---|---|
| 旧（作废） | 末拍**指令** `g ≥ 0.9` | **0/20**（`never_released` 20） |
| 新（现行） | 末帧方块中心到 `tool0` 距离 ≥ `RELEASE_SEP_M = 0.05 m`（松手后退开的物理证据） | **20/20** |

旧判据错在把「策略自己报的开度」当行为证据：那 20 次里末拍 `g` 只有 **0.639–0.766**，而方块
实际离 `tool0` **0.111–0.123 m**、全部躺在碗里（`in_bowl` 20/20）。g 头系统性偏低是训练出来的
性质，不是失败。现行 `classify_place`（`experiments/act_rollout.py`）只查物理脱离 + `in_bowl`，
`cmd_g_last` 降级为测量值。

| step | 002000 | 004000 | **006000** | 008000 | 010000 | 012000 | 014000 | 016000 | **018000** | 020000 |
|---|---|---|---|---|---|---|---|---|---|---|
| 成功 | 0/20 | 0/20 | **20/20** | 5/20 | 0/20 | 0/20 | 7/20 | 5/20 | **20/20** | 14/20 |

多 seed 复核（每检查点 4 个 seed × 20 次）：

| 检查点 | 成功 | 最差 `bowl_dxy_m` | 最差 `hold_err_m` | 备注 |
|---|---|---|---|---|
| 006000 | **80/80 (100%)** | 0.0295 | 0.0251 | 无一次失手 |
| 018000 | 77/80 (96%) | 0.2824 | 0.2948 | 3 次根本没抓到（`rise_peak` 0.001–0.004 m） |

- **训练不单调**：0 → 0 → 20/20 → 5 → 0 → 0 → 7 → 5 → 20/20 → 14/20。同一个 run 里 step 6000 与
  18000 都好、10000–12000 全废，而 loss 一直卡在 2.37。选权重必须按闭环扫描选，不能看 loss 或
  只看 `last`。
- **`bowl_dxy` 与 `hold_err` 的尺度变了**：策略放置的横向误差到 **29.5 mm**（仍在碗内，碗内半径
  45 mm，判据半径 30 mm），而脚本化示教只有 4.6 mm；`hold_err` 到 **25.1 mm**（示教 1.2 mm）。
  所以 `tests_ee/test_act_rollout.py` 对 policy 与 replay 用两个门：`mode=replay` 走 0.005 m /
  0.002 m（表生成的验收门），`mode=policy` 走 `IN_BOWL_RADIUS_M` / 0.030 m。
- 测试计数：`test_act_rollout.py --place-summary scratch/p20k2_006000/summary.json` → 547 checks；
  默认（重新生成的 `out/act_rollout_place/summary.json`）→ 339 checks，均 `ALL CHECKS PASSED` EXIT=0。

这一节**不能**证明：真机上的放置成功率；换碗/换方块尺寸后的泛化；视觉域差（训练与评测同一台
渲染器、同一光照）；`--viewer` 那条路径我只做了函数级 stub（用户自己开窗）。

---

## 历史验收（改资产之前的复跑记录）

> 下面这些数字来自 **T7 换真夹爪之前**的占位夹爪资产（`gripper_pg2.xml`：slide `0..0.04 m`、`kp=300 kv=5`、`site tool0` 在法兰上）。同一条命令现在跑出来的数字与它们不同，**不可复现是预期的**，保留只为记录当时的判据与踩过的坑。

- T2（8 维 EE + IK，模型还是上游 `nq=20` 场景）：`ALL CHECKS PASSED`，最差端到端 9.328e-04 m / 0.0863 deg，单次热启动 65/100，位置残差 p50 5.733e-04 / max 9.922e-04 m。
- T1（资产 + 抓取，占位夹爪）：两种 cwd 各一次 `EXIT=0`，抬升 0.09806 m（阈值 0.08），夹持中水平滑移 0.099 mm（<5 mm 断言），两指内面 gap 0.01987 m vs 方块 0.02000 m，离屏渲染 480×640×3 均值 71.22。
  > 旧的「滑移 1.96 mm」是 T1 开发期的一次口述记录，任何命令都复现不出来，已作废。
- T3（离线闭环，占位夹爪）：`VERDICT: PASS 19/19`（`--no-render` 为 15/15），75 帧 @30 Hz，逐帧 |Δq|∞ p50 0.0114 / max 0.0170 rad，抬升 +0.09910 m，产物 `out/t3_frames/{direct,servo}/` 各 16 张 PNG。
- 验收期间发现并修掉的真实故障（保留记录，防止回退）：
  1. 从 `qpos0` 起步时方块每侧预穿透 0.75 mm，9.9 g 的下颌被弹开到 0.0222 m → 改成「先张开下降，再 `ctrl=0` 伺服闭合」。
  2. `Link_5` 碰撞胶囊堵住腕部工具孔，下颌卡在 0.02225 m → 剪短胶囊 + 4 组 `Link_5`/`Link_6` 与夹爪的 contact exclude。
  3. 曾要求 `scene nq == 7`，物理上不可能（可抓取方块必须是 free joint，`+7` qpos）→ 判据改为 `robot 子树 nq == 7` 且 `scene nu == 7`。
  4. 姿态判据曾写成 `< 1e-9 deg`，低于它自己的量尺分辨率（`arccos((tr-1)/2)` 在 1 ulp 差异上报 0 或 ~1.2e-6 deg）→ 改成 `< 1e-4 deg` 并补一条矩阵路径判据。

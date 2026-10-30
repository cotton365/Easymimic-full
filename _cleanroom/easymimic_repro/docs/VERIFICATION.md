# 验收记录 · Verification

本文件记录**怎么复跑**、**实测输出长什么样**，以及**这些测试不能证明什么**。
原始输出未经修饰（只做了行裁剪）；凡是数字与 `README.md` §4 不一致，以本文件的复跑结果为准，并请提 issue。

环境快照（2026-10 实测）

| 项 | 值 |
|---|---|
| Python | 3.11.17 |
| mujoco | 3.16.0 |
| numpy | 2.4.6 |
| 平台 | Linux x86_64 |
| 额外依赖 | 无（无 scipy / cv2 / torch / mink / pytest） |
| 上游模型 | jaka-zu3-mujoco @ `7c40fec`，经 `scripts/fetch_third_party.sh` 取回 |
| 图形 | 离屏渲染需要 GL，本机 `DISPLAY=:0` 通过 |

---

## A. 8 维 EE + DLS-IK 验收

```bash
python tests_ee/test_ee_pipeline.py
```

期望最后一行 `ALL CHECKS PASSED`，退出码 0。实测输出（摘录）：

```
[1] analytic synthetic hand (truth by construction)
  [PASS] g closed (d_min) == 0.0   g=0.000000000000000
  [PASS] g mid (d_mean) == 0.5   g=0.500000000000000
  [PASS] g open (d_max) == 1.0   g=1.000000000000000
  [PASS] rotvec round-trip mid (d_mean)   err=2.220e-16
  [PASS] T_base_cam position   err=0.000e+00
  [PASS] T_base_cam rotation   err=2.220e-16
  [PASS] palm-normal sign rule   err=2.220e-16  Z.ref=1.000000  z_flipped=False

[2] orthonormality / chirality
  [PASS] 50 random hand frames: |R^T R - I|, |det R - 1|, |X.Z| all < 1e-09
         max |R^T R-I|=6.66e-16, max |det-1|=4.44e-16, max |X.Z|=9.63e-17

[3] NaN propagation (no silent zero-fill)
  [PASS] each of the 21 keypoints NaN -> all 8 outputs NaN
  [PASS] NaN R_wrist -> all-NaN output
  [PASS] NaN T_base_cam -> all-NaN output
  [PASS] finite input -> finite output

  site 'tool0' at qpos0 = [0.47400, 0.04350, 0.83305]
  arm DoF columns [12, 13, 14, 15, 16, 17] (nq=20, nv=18; props have free joints, hence by name)

[4] IK convergence on 100 FK-sampled reachable targets
  warm start only (home seed, single DLS attempt): 65/100 converged
  position error  [m]    p50=5.733e-04  p95=9.648e-04  max=9.922e-04
  orientation err [deg]  p50=0.0828  p95=0.4374  max=0.4977
  mean DLS iterations    45.2  (mean extra restarts 1.34, max 16)

  trial 4: hand->EE residual 0.00e+00 m / 3.19e-06 deg | IK residual 9.328e-04 m / 0.0863 deg
  worst residual         9.328e-04 m / 0.0863 deg
ALL CHECKS PASSED
```

七组断言：`test_1_analytic` 解析重建、`test_2_orthonormal` 右手系、`test_3_nan` NaN 传播、`test_4_ik` 100 目标、`test_5_limits` 限位、`test_5b_tcp_and_jacobian` TCP 力臂 + 中心差分雅可比、`test_6_end_to_end` 手→EE→关节→FK 回环。

---

## B. MuJoCo 资产 + 抓取物理验收

```bash
python experiments/t1_verify.py
```

期望 `RESULT: ALL CHECKS PASSED`，退出码 0。实测输出（摘录）：

```
  robot.xml  : nq=7 nu=7  (6 arm hinges + 1 gripper slide)
  [PASS] scene nu == 7 (6 arm + 1 gripper position actuator)
  [PASS] robot subtree nq == 7 (6 arm hinges + 1 gripper slide)
  scene nq   : 14 = 7 robot DoF + 7 cube free-joint qpos (3 pos + 4 quat).
  actuator 6  gripper_actuator   joint=gripper_jaw  kp=300 kv=5 ctrlrange=[0. 0.04]

[3] grasp test: tool over the cube, jaw commanded closed, lift 0.10 m
  cube settled on table at z=0.80996  ncon=4
  descended, tool0=[0.48614 0.0435  0.85799] (1.25 s)  jaw open=0.0400  ncon=4
  3b closed: ctrl_jaw=0 -> jaw qpos=0.00137  cube z=0.80996  ncon=12
  tool rise 0.10002 m
  after lift: cube z=0.90802 (rise 0.09806) jaw=0.00137 ncon=8
  pad inner faces on jaw axis: 0.48072 .. 0.50059  (gap 0.01987 m)
  cube centre on jaw axis    : 0.49066
  [PASS] cube lifted >= 0.08 m (measured 0.09806 m)
  [PASS] cube still between the fingers (proj 0.49066 inside (0.48072, 0.50059))
  [PASS] cube still at fingertip height (dz=0.01744 m)
  [PASS] pads are pressing the cube, not merely straddling it (gap 0.01987 m vs cube 0.02000)

[5] offscreen render
  frame shape=(480, 640, 3) dtype=uint8 mean=71.22 min=0 max=234
  [PASS] Renderer produced a 480x640x3 RGB frame
  [PASS] frame is not blank (mean 71.22)
RESULT: ALL CHECKS PASSED
```

### 路径可移植性（曾专门测过）

在 `cwd=/` 与 `cwd=仓库根` 各跑一次，两次均 `EXIT=0` / `ALL CHECKS PASSED`。

> 历史遗留的结论要更正：**只有上游 MJCF 必须绝对路径加载**（顶层 `meshdir` 泄漏进 `<include>` 导致相对 mesh 解析失败）。本仓库 `assets/jaka_zu3/robot.xml` 的 `meshdir` 是相对自身文件目录的，不 include 上游任何文件，两种 cwd 下都正常。

---

## C. 这些测试**不能**证明什么

- 不能证明真实视频输入可用：测试用的是闭式构造的合成手（真值由构造给出），`T_base_cam` 由测试注入。
- 不能证明轨迹连续：所有 IK 目标都是**独立采样**的，逐帧 |Δq| 抖动**没测**。
- 不能证明真机可用：没有 JAKA 后端，没有手眼标定，没有真机跑过。
- 不能证明 `src/closed_loop/` 可用：那部分代码**从未运行**（见 README §8.1）。

## D. 复现失败时先查这三件事

1. `third_party/` 是否已取回（`bash scripts/fetch_third_party.sh`），或 `EASYMIMIC_JAKA_XML` 指向某个绝对路径。
2. 是否用了无 GL 的环境跑 `t1_verify.py` 的最后一步渲染（前面 4 步不需要 GL）。
3. 版本：mujoco 3.1.x 与 3.16.0 在 `mj_jacSite` 之外的细节上有差异，本记录只在 3.16.0 上验证过。

---

## 历史验收（同一套判据，先前的复跑记录）

- T2（8 维 EE + IK）：`ALL CHECKS PASSED`，最差端到端 9.328e-04 m / 0.0863 deg。
- T1（资产 + 抓取）：两种 cwd 各一次 `EXIT=0`，抬升 0.09806 m（阈值 0.08），滑移 1.96 mm。
- 验收期间发现并修掉的真实故障（保留记录，防止回退）：
  1. 从 `qpos0` 起步时方块每侧预穿透 0.75 mm，9.9 g 的下颌被弹开到 0.0222 m → 改成"先张开下降，再 `ctrl=0` 伺服闭合"。
  2. `Link_5` 碰撞胶囊堵住腕部工具孔，下颌卡在 0.02225 m → 剪短胶囊 + 4 组 `Link_5`/`Link_6` 与夹爪的 contact exclude。
  3. 曾要求 `scene nq == 7`，物理上不可能（可抓取方块必须是 free joint，`+7` qpos）→ 判据改为 `robot 子树 nq == 7` 且 `scene nu == 7`。

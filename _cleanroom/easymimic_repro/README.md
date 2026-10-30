# EasyMimic 复现 · 裸手示教 → JAKA Zu3 末端指令

> **English summary.** A partial, self-contained reproduction of the *action-representation* half of EasyMimic: 21 hand keypoints → 8-D end-effector command (position + rotation vector + gripper opening) → six joint angles for a JAKA Zu3 arm, verified numerically and by a physical grasp inside MuJoCo. Video ingestion, camera / hand-eye calibration, dataset writing and the learned policy are **not** implemented. Two acceptance suites pass; the numbers are in §4 and `docs/VERIFICATION.md`.

**一句话**：本仓库把"一段视频"里最靠后、也最容易做错的两段——**打点表示**与**指令下发**——单独拆出来做对了、并留下可复跑的证据；前两段（收到视频、识别手部）还没有代码。

**状态**：离线数学层与 MuJoCo 执行层已验收（见 §4）；视频入口、手眼标定、真机后端、策略层均未开始（见 §8）。

---

## 1. 这是什么，不是什么

用户最初的设想是一条四段流水线：

```
视频  →  手部特征(21 关键点)  →  打点表示  →  机器人末端+夹爪指令
```

| 段 | 在本仓库 | 状态 |
|---|---|---|
| ① 接收视频 | 无代码 | ❌ 未开始 |
| ② 手部特征识别（HaMeR，21×3 + 腕部 6DoF） | 无代码；假定已就绪 | ⚠️ 外部依赖，未移植 |
| ③ 打点表示 | `src/ee_retarget/hand_to_ee.py` | ✅ 已验收 |
| ④ 末端指令（→ 关节角 → 夹爪） | `src/ee_retarget/ik_jaka_zu3.py` | ✅ 已验收 |

**是**：一个可独立运行、可复跑的"手 → 8 维末端动作 → 6 关节角"数学与执行层，外加一套能在 MuJoCo 里真正夹起 2 cm 方块的 JAKA Zu3 资产。
**不是**：不是完整的 EasyMimic，不含策略网络、不含数据集、不含视频解码、不含标定工具，**也没有在真机上跑过一次**。

---

## 2. 数据流（读代码时按这个顺序）

```
21 keypoints (21,3) + R_wrist (3,3) + T_base_cam (4,4)
        │
        │  hand_to_ee.hand_to_ee()                 ← 本仓库 · 已验收
        ▼
8 维动作 ee8 = [pos(3), rotvec(3), g(1), reserved(1)]  in robot base frame
        │
        │  ik_jaka_zu3.JakaZu3IK.solve_robust()     ← 本仓库 · 已验收
        ▼
6 关节角 q (rad)  ──►  position actuators ──►  MuJoCo / 真机
                       jaw ctrl = g * 0.04 m
```

- `pos`：**thenar 锚点**（大鱼际点，拇指与食指根部之间）在基座系下的位置，不是手腕。
- `rotvec`：手掌平面的姿态，X 轴 = 食指 MCP→PIP，Z 轴 = 5 点拟合平面的法向。
- `g`：**g = 1 张开，g = 0 闭合**（与 `hand-6dof-pipeline/retarget/so101.py:110` 的 `norm` 语义一致）。
- 第 8 维保留位恒为 0，给将来加"第二指/吸盘/力控"留位。
- 任一关键点为 NaN 时，整帧 8 维全 NaN（不补零、不插值）——下游靠 `info["valid"]` 过滤。

关节角是**执行层**，不是动作空间：6 轴臂的 IK 有多解，把它塞进学习信号会污染数据集。

---

## 3. 快速开始

### 3.1 环境

```bash
conda create -n mujoco python=3.11 -y
conda activate mujoco
pip install -r requirements.txt      # 只有 mujoco + numpy
```

整个已验证代码路径只依赖 **numpy + mujoco**（无 scipy / cv2 / torch / mink）。请保持这一点：显式 IK 用的是 `mj_jac` 上的阻尼最小二乘，不需要 QP 求解器。

### 3.2 取回上游模型（不进版本库）

```bash
bash scripts/fetch_third_party.sh    # clone jaka-zu3-mujoco @ 7c40fec 到 third_party/
```

为什么要单独拿：网格文件来自 JAKA 源码副本，**重新分发许可不明**，且整个 checkout 约 100 MB（其中 75 MB 是两个本项目根本不用的 Objaverse 道具）。我们自己的代码是干净的，网格不是我们能替你分发的。详见 §9。

### 3.3 跑起来（两条验收命令）

```bash
python tests_ee/test_ee_pipeline.py    # 8 维 EE + IK 全部断言
python experiments/t1_verify.py        # MuJoCo 资产 + 抓取 + 离屏渲染
```

两条都以 `ALL CHECKS PASSED` 结尾、退出码 0。它们不依赖 pytest（环境里没有），是普通脚本 + `assert`，故意做成零依赖。

- `t1_verify.py` 最后一步会做离屏渲染；无显示环境下需要可用的 GL（本机 `DISPLAY=:0` 通过）。
- 模型路径默认是"仓库内 `third_party/jaka-zu3-mujoco/jaka_zu3_mujoco.xml`"，可用环境变量 `EASYMIMIC_JAKA_XML` 覆盖。**该 XML 必须用绝对路径加载**（上游模型顶层 `meshdir` 会泄漏进 include，相对路径必报 `Error opening file 'assets/jaka_zu3_meshes/assets/objaverse/...'`）；代码内部已 resolve，构造函数对相对路径直接抛错。

---

## 4. 验证结果（全部可复跑）

| 项 | 判据 | 实测 | 结论 |
|---|---|---|---|
| 合成手 → EE 解析重建 | ≤1e-9 | 2.2e-16 | ✅ |
| 夹爪归一化端点 | 精确 0 / 0.5 / 1 | 一致 | ✅ |
| 姿态正交性（50 帧） | ‖RᵀR−I‖∞ | 6.66e-16；det−1 = 4.44e-16 | ✅ |
| IK 成功率（100 随机可达目标） | 100% | 100/100 | ✅ |
| IK 位置残差 | <1e-3 m | p50 5.73e-4 / p95 9.65e-4 / max 9.92e-4 m | ✅ |
| IK 姿态残差 | <0.5° | p50 0.083° / max 0.498° | ✅ |
| 端到端（手→EE→关节→FK） | <1e-3 m / 0.5° | 9.33e-4 m / 0.086° | ✅ |
| 抓取抬升 | ≥0.08 m | 0.09806 m | ✅ |
| 抬升后滑移 | 不掉 | 1.96 mm | ✅ |
| 两指内面 vs 2 cm 方块 | 压紧 | 0.01987 m vs 0.02000 m | ✅ |
| 离屏渲染 | 非空白 | 480×640×3，均值 71.22 | ✅ |

**必须一起读的两个数字**：IK 的 100% 成功率是 `solve_robust()` 最多 16 次重启换来的，**单次热启动只有 65/100**；位置残差最大值 9.92e-4 m 距 1e-3 m 判据只剩 8% 余量。实时控制若只允许单次求解，这两条就是天花板。

复跑记录与原始输出片段见 `docs/VERIFICATION.md`；决策过程与被否决的方案见 `docs/DESIGN_LOG.md`。

---

## 5. 目录结构

```
easymimic_repro/
├── README.md                     ← 本文件
├── requirements.txt              ← mujoco + numpy，仅此
├── .gitignore                    ← third_party/、scratch/、out/ 不入库
├── assets/jaka_zu3/              ← 自写 MJCF（7 个 STL 从 third_party 引用，不复制）
│   ├── robot.xml                 ← 臂 + 碰撞代理 + 7 个 position 执行器
│   ├── gripper_pg2.xml           ← 占位平行二指（被 robot.xml include）
│   └── scene_cube_bowl.xml       ← 桌面 + 2 cm 方块 + 碗
├── src/ee_retarget/
│   ├── hand_to_ee.py             ← 21 点 → 8 维 EE（机器人无关）
│   └── ik_jaka_zu3.py            ← 8 维 EE → 6 关节角（DLS IK）
├── src/closed_loop/              ← ⚠️ 未验证，见 §8
│   ├── synth_hand.py             ← hand_to_ee 的逆：给定位姿反解 21 点
│   └── run_closed_loop.py        ← 合成手 → EE → IK → MuJoCo 抓取抬升
├── tests_ee/test_ee_pipeline.py  ← 7 组断言（解析/正交/NaN/IK/限位/TCP/端到端）
├── experiments/
│   ├── t1_verify.py              ← 资产 + 抓取 + 渲染验收
│   └── probe_env.py              ← 环境自查：打印哪些包在、哪些缺
├── scripts/fetch_third_party.sh  ← 取回上游 JAKA 模型
├── docs/DESIGN_LOG.md            ← 决策日志（含被否决方案与理由）
├── docs/VERIFICATION.md          ← 验收命令与实测输出
├── third_party/                  ← 不入库，脚本获取
└── scratch/                      ← 不入库，现场探针
```

---

## 6. 关键设计决策（改了就会坏的原因）

1. **动作空间 = 8 维 EE，不是 6 关节。** JAKA 真机本来吃笛卡尔指令；6 轴 IK 多解，且 joint_2/joint_4 限位只有 −1.48~4.62 rad，把 IK 结果当学习目标等于把求解器的不确定性写进数据集。
2. **锚点是 thenar（大鱼际），不是手腕。** 论文的 thenar 锚点 = 拇指 PIP 与食指 MCP 的中点。21 点约定里没有"拇指 PIP"，取**拇指 IP（idx 3）**代替，所有 landmark 索引都做成模块常量，换手模型时只改表。
3. **平面法向用 SVD，X 轴用食指 MCP→PIP。** 5 个点（食指 5,6,7,8 + 拇指 3）拟合平面，最小奇异向量取法向；法向有 ±歧义，用腕部参考轴 `R_wrist[:,2]` 消歧，翻转后 Y 由 `Z×X` 重算，保证 det 恒为 +1（右手系，IK 里不可省）。
4. **g 的方向钉死为 "1 = 张开"。** 与 `hand-6dof-pipeline/retarget/so101.py:110` 的 `norm` 一致；仿真里 `ctrl = g * 0.04`（0 = 闭合、0.04 m = 全开）。反向就是这次交接最容易犯的错。
5. **TCP 必须加 32.5 mm 力臂。** 夹爪两指中心在贴附坐标系里是 `(0, -0.0325, 0)`，`site tool0` 在法兰上。IK 直接对准 `tool0` 会让指尖比目标低 3.25 cm——抓取因此差之毫厘。代码里 `Jp = Jp - skew(R@t)@Jr` 就是这个力臂的雅可比修正。
6. **阻尼是相对的。** `λ_abs = λ_rel * sqrt(trace(JJᵀ)/6)`。绝对 `λ = 1e-3` 在近奇异位形上实测单步 |dq| 达 127.86 rad、成功率掉到 48/100。
7. **NaN 全帧传播，绝不补零。** 补零会造出一条"机器人回原点"的假轨迹，训练时比丢帧更毒。
8. **不用 mink / qpsolvers。** 依赖只有 mujoco + numpy；DLS 在一百个随机目标上已经打满，装 QP 求解器换不来精度。

---

## 7. 坐标系与几何（改资产前必看）

- 世界 +z 向上，桌面顶面精确在 `z = 0.80` m，机器人基座也在这一层。
- `site tool0` 位于 body `gripper_pg2` 原点，该 body 在 `Link_6` 内、`pos="0 0 -0.035"`。
- 夹爪系：**+y = 世界上方**，**−y = 接近方向**，**x = 闭合轴**。两指内面在 `x = ±0.00925`，全闭 gap = 0.0185 m（对 2 cm 方块是 0.75 mm 预载，靠 `condim=4` + 摩擦夹住）。
- `gripper_jaw` 是 slide joint，`axis="1 0 0"`，`range="0 0.04"`，position 执行器 `kp=300 kv=5`。
- 视觉网格 `contype=0 conaffinity=0`；碰撞用 capsule/cylinder 代理，`group=3`、`contype=1 conaffinity=1`。整 STL 当碰撞体在 8 GB 机器上又慢又不稳。
- home keyframe `qpos="0 0.15 0 0 0 0 0.04"`：全零位时法兰会插进桌面。

**MuJoCo 两个静默坑**（踩过）：
- `<include>` 只 splice 子元素，单个 body 根会被静默丢弃 → `gripper_pg2.xml` 里的 `<gripper_pg2_fragment>` 包裹是必须的。
- `<actuator>` 只能写在文档根，写进被 include 的 body 会静默丢弃 → 夹爪执行器声明在 `robot.xml`。

---

## 8. 已知问题与未完成（交接重点）

**未验证 / 未开始**
1. `src/closed_loop/`（`synth_hand.py` 189 行 + `run_closed_loop.py` 685 行）是"合成手 → EE → IK → MuJoCo 抓取抬升"的离线闭环，写得像完成品，但**从未跑完过、没有输出、没有验收**。写它的人（子代理）在运行前被中断。请把它当草稿读，第一件事是跑通并出图。
2. 视频入口（解帧、多相机同步、ChArUco 内外参、尺度）**一行都没有**。采集侧已知条件是**多个 USB 相机**，尺度可由多视角三角化恢复——这比单目视角省掉"骨长先验"的猜测，但需要内参/外参标定与时间戳对齐。
3. 手眼标定 `T_base_cam` 没有做过；目前所有测试都是手工注入这一矩阵（闭环测试直接用单位阵）。
4. 真机后端没有：不走 lerobot，需要 JAKA SDK（TCP/IP）新后端。

**已知缺陷**
5. `ik_jaka_zu3.py` 的 `rotmat_to_rotvec()` **缺 θ≈π 分支**（`hand_to_ee.py` 里那份是完整的三支实现）。θ→π 时 `sin θ→0` 会除爆。目前不触发（步长被 `MAX_STEP_RAD` 截断、候选步被"残差必须下降"拒掉），属埋雷，**接线前请补上**。
6. 单次热启动 IK 只到 65/100；残差余量 8%。轨迹连续性（逐帧独立求解的 |Δq|）**没测**——这是闭环验证的第一优先项。
7. 每次迭代算 2~3 次正运动学（`ik_jaka_zu3.py` 的 `:264`/`:293`/jacobian 内`_sync`），纯性能问题，不影响正确性。
8. 夹爪是**占位**平行二指。真实夹爪资产后补；接口按"指尖点 + 开合量"设计，替换时只改 `gripper_pg2.xml` 与 `TCP 偏移`。
9. `assets/jaka_zu3/robot.xml` 的碰撞代理是手调的 capsule/cylinder，抓取姿态附近做过 4 组 contact exclude；改机械臂姿态范围前请重跑 `t1_verify.py`。

---

## 9. 外部依赖与许可

| 依赖 | 用途 | 许可 |
|---|---|---|
| 上游 [jaka-zu3-mujoco](https://github.com/yy2006-maker/jaka-zu3-mujoco) @ `7c40fec` | 机械臂 STL 网格（本项目只引用，不复制） | **不明**（上游 `package.xml` 写 TODO）→ 不随本仓库分发，用 `scripts/fetch_third_party.sh` 取 |
| [hand-6dof-pipeline](https://github.com/pass-opc/hand-6dof-pipeline) | 参考实现（HaMeR 后端、mink retarget、真机回放） | 上游仓库，**不在本仓库内**，仅作对照 |
| [MuJoCo](https://mujoco.org) | 仿真与 `mj_jac` | Apache-2.0 |

本仓库自身的 LICENSE **尚未选择**（交接方待定）。在选定之前，请把它当"内部可见"，不要对外宣称可再分发。

---

## 10. 接手须知

**第一小时读这三份**：`README.md` §2/§6 → `src/ee_retarget/hand_to_ee.py` 顶部 docstring（它逐条写了"为什么这样定义"）→ `docs/DESIGN_LOG.md` §8（决策与被否决方案）。

**按这个顺序往下做**（每一步都有可独立验收的产物）：
1. 跑通 `src/closed_loop/run_closed_loop.py`，出帧图，量化 |Δq| 轨迹连续性与抓取成功率 —— 补上最欠的那块证据。
2. 修 `rotmat_to_rotvec` 的 θ≈π 分支。
3. 视频入口：解帧 + ChArUco 内外参 + 多相机同步 + 三角化出公制尺度，验收线定在 **20~30 mm**（单目/多视角重建的现实精度，别照抄论文的 10 mm）。
4. 手眼标定 `T_base_cam`，替换掉测试里手工注入的单位阵。
5. 数据集（LeRobot v3，8 维 EE schema）→ 策略层。到这一步才需要 GPU；原始 EasyMimic 的 GR00T N1.7 路线要 16 GB VRAM，8 GB 卡上要另想办法。

**三个别踩的坑**
- 别把夹爪方向改成 "1 = 闭合"：下游 `ctrl = g * 0.04` 全部反了。
- 别用相对路径加载上游 MJCF（见 §3.3）。
- 别把 IK 结果当动作空间存进数据集（见 §6.1）。

---

## 11. 致谢

方法来自 EasyMimic 论文（thenar 锚点 + 平面拟合 + 8 维动作的重定向部分）；机械臂模型来自 yy2006-maker/jaka-zu3-mujoco；参考实现来自 pass-opc/hand-6dof-pipeline。本仓库只包含我们新写的代码与 MJCF。

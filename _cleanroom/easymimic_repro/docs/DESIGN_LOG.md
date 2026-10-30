# EasyMimic 复现 · 决策日志（DESIGN_LOG）

> **这份文件是过程记录，不是说明书。** 入口文档是 `README.md`；可复跑的验收证据在
> `docs/VERIFICATION.md`。下面 §0-§7 是当时的计划草案（保留决策理由与被否决方案），
> §8 起是决策定稿与实际进展。
>
> 现状核对（2026-10，整理交接时更新）：
> - D1/D2/D3 已定稿（见 §8），不再是"待决策"。
> - T1、T2 已验收；T3 代码存在但**未运行、未验收**；T4/T5/T6 未开始。
> - 本文中"待用户提供 G2 硬件清单"（§5）已部分回答：采集设备是**多个 USB 相机**（§10 R4）。
> - `ik_jaka_zu3.py` 的 `DEFAULT_XML_PATH` 已改为仓库相对路径 + `EASYMIMIC_JAKA_XML` 覆盖，
>   不再硬编码某台机器的绝对路径。

生成: 新会话（工作目录 /home/zh/Desktop/EasyMimic）
当时状态: 分析已核验；3 个范围决策待用户回答后定稿

## 0. 写权限（已实测，上一会话死点已解除）
- `/home/zh/Desktop/EasyMimic` 可写；`hand-6dof-pipeline/` 可写（git 干净 @ c2a1f46）。
- 产出落点: `/home/zh/Desktop/EasyMimic/easymimic_repro/`。

## 1. 用户描述的"基础流程"逐段核验
目标链: 接收视频 -> 手部特征识别 -> 打点表示 -> 转换为机器人末端夹爪指令

| 段 | 现状 | 结论 |
|---|---|---|
| 接收视频 | 全仓 `grep -rn 'VideoCapture'` 0 命中；唯一入口 `hand-6dof-pipeline/scripts/01_hand_track.py:41-58` 的 `--r3d/--r3d-dir`，走 `utils/iphone/r3d_reader.py`（iPhone Record3D，ARKit 附带深度+位姿） | **不存在，需新写**（G2），不是接线 |
| 手部特征识别 | `utils/hand_tracker/hamer_backend.py:221` 取 `pred_keypoints_3d` (21,3) + `pred_cam_t/rot` 得腕 6DoF | 已可用 |
| 打点表示 | v3 schema 已存 `*_hand_keypoints_world` / `*_action_wrist_pose_world`，但是**腕部锚点**，非论文 thenar + 5 点平面 | 已可用；论文版属 G3 |
| 点 -> 夹爪指令 | `retarget/so101.py`: pinch 中点（拇指尖+食指尖）+ 腕姿态 -> mink 软 IK 打到 site `gripperframe` -> 6 关节角(rad)；夹爪 `retarget/so101.py:110` `deg = open_deg + (1.0 - norm) * (close_deg - open_deg)` | **产出关节空间，不是 8 维 EE 位姿** |

## 2. 三个纠正（相对 handoff §5 的草案）

1. **第 4 段形态**: 现有链路是"关节+夹爪"，论文是"8 维 EE+gripper"。仓库内目前无任何代码产基座系 EE 位姿 -> G3 (`retarget/easymimic_ee.py`) 是全新模块。二者是不同目标，必须先定。
2. **下发端点已存在**: `replay/real/so101.py:76` `from lerobot.robots.so_follower import SO100Follower`，用法 `python -m replay --output real --port COMx`。代价 = 依赖 lerobot 包 -> 要么在 lerobot env 只跑不改，要么 em-pipe 另装。**故 G2 验收放 mujoco，真机后置。**
3. **G2 难点是尺度不是解帧**: 丢深度同时丢外参来源（`utils/cam_pose/` 不存在，`ls` 确认）与米制尺度；`utils/hand_tracker/depth_correction.py` 原本用 `scale = z_lidar / z_est` 补这个洞。替代 = 骨长先验 + 桌面平面。**验收线建议 20–30mm，不采用 handoff 的 10mm。**

**G1 可缩水**: 真崩点是 `retarget/so101.py:54` `from optimize.filters.quat_one_euro import _ScalarOneEuro`（`optimize/` 目录不存在，用于 `:341`），约 40 行可补。778 顶点导出只服务 G4 视觉对齐，**基础流程不需要**，移出 G1。

## 3. 建议路线（待确认）

`G0 环境` -> `G1 只补 one-euro` -> `M0 离线最小闭环（GT 点 -> retarget -> mujoco mp4）` -> `G2 真视频入口` -> `G3 EE 动作空间` -> `G4 数据集+视觉对齐` -> `G5 策略层` -> `G6 A 轨`

理由: M0 不依赖相机/标定/HaMeR 权重，先用一天钉死"点->指令"的数学，再换真输入；跳过它等于把"数学对不对"和"标定准不准"两个未知叠一起调。

## 4. 待用户决策（blocking）

- **D1 指令形态**: (A) 关节+夹爪先跑通，G3 后补 [建议] (B) 直接 8 维 EE (C) 只要关节，不碰 EE 消融
- **D2 第一个证据**: (A) M0 离线最小闭环 [建议] (B) 真视频最小闭环 (C) 先只修现有链路+13 测试
- **D3 终点硬件**: (A) 仅 mujoco [建议] (B) 有 SO-101 实物 (C) 暂无实物

## 5. G2 启动前仍需用户提供（handoff §6，未收到）
USB 摄像头型号/分辨率/帧率 + 能否固定安装；能否打印 A4 ChArUco 标定板；有无 SO-101 实物；有无 Joy-Con；`nvidia-smi`、`conda --version`、`pip config list`、`ls -lh hand-6dof-pipeline/assets`（MANO / hamer.ckpt 是否就位）。

## 6. 硬约束（不随计划变化）
- 产出只落 `/home/zh/Desktop/EasyMimic`；不在 `裸手示教` 写。
- 不动 lerobot conda env。
- 网络/GPU/conda 安装类命令交用户执行，我读输出。
- 每 gate 停一次给证据（测试输出/数值/图）。
- 不得把 GR00T SO100 示例的 6 维关节空间当作论文 8 维 EE 动作空间。
## 7. JAKA Zu3 资产评估（已实测，2026-10）

来源: https://github.com/yy2006-maker/jaka-zu3-mujoco @ 7c40fec（2 commits）
克隆位置: `easymimic_repro/third_party/jaka-zu3-mujoco`（100MB，其中 objaverse mug/plate 占 75MB）

### 7.1 实测事实
- **加载必须用绝对路径**。用相对路径 `from_xml_path('jaka_zu3_mujoco.xml')` 会报
  `Error opening file 'assets/jaka_zu3_meshes/assets/objaverse/mug_5/collision/model_normalized_collision_23.obj'`
  （顶层 `<compiler meshdir="assets/jaka_zu3_meshes">` 泄漏到 include 里的相对 mesh 路径）。用 `str(Path(...).resolve())` 即正常。
- 绝对路径加载成功（mujoco 3.16.0 / py3.11 env `mujoco`）: `nq=6 nu=6 ngeom=75`，6 关节 + 6 position actuator（kp 1200/1200/1000/800/500/300, kv 对应），`site tool0` 在 qpos0 时 = `[0.474, 0.0435, 0.8331]`。
- 关节限位: `joint_1 [-6.28,6.28]`, `joint_2 [-1.48,4.62]`, `joint_3 [-3.05,3.05]`, `joint_4 [-1.48,4.62]`, `joint_5 [-6.28,6.28]`, `joint_6 [-6.28,6.28]`；base Link_0 在 z=0.80（桌面高度）；qpos0 下 tool0 距基座 0.478m（Zu3 标称 626mm，一致）。
- 去掉 4 个 include 后单独加载机器人: `nq=6 nu=6 ngeom=7`，**全部 geom `contype=0, conaffinity=0`** → 纯视觉，无碰撞。
- **无夹爪**。README 原文: "There is no gripper yet, and the arm does not collide with the table or objects. The current stage is suitable for viewing and testing joint motion, not for a pick-and-place task or policy training."
- 上游 license 不明: "The JAKA source copy ... has no clear redistribution license (package.xml says TODO)". → mesh **不拷进产出**，只在 `third_party/` 内引用；不得对外分发。

### 7.2 集成方案（建议）
1. **不要直接 include 上游主 XML**。抽三份文件到 `easymimic_repro/assets/jaka_zu3/`：
   - `robot.xml`: 仅 7 个 STL `<asset>` + Link_0..Link_6 链 + `<actuator>`，`meshdir` 指向上游绝对目录（不复制网格）。
   - `scene_cube_bowl.xml`: `include robot.xml` + 自制桌面（box）+ cube（2cm box）+ bowl（开口 cylinder），全部带碰撞。
   - `gripper_pg2.xml`: 挂在 `tool0` 的平行二指（2 box + 1 slide joint 0~0.04m + 1 position actuator）。
2. **弃用 objaverse mug/plate**：75MB + license 不明 + 与 cube→bowl 无关；primitive 场景更可控、碰撞更便宜。→ 作为默认假设执行，不占用决策。
3. **夹爪是最大缺口**：无夹爪 = 8 维动作第 8 维没有接收方，`g clip((d_t-d_min)/(d_max-d_min))` 无处可发，G3 验收（能抓住）不成立。指尖 geom 必须 `contype/conaffinity=1` + `condim=4`/摩擦，否则夹不住。
4. **碰撞简化**：Link_1..Link_6 visual geom 保持 `contype=0`，另加 capsule/cylinder 简化碰撞体（整 STL 碰撞在 8GB 机器上慢且不稳）。
5. **retarget 侧**：`retarget/so101.py` 的 mink IK 核心通用，需换的是 profile（joint_1..6 / site tool0 或指尖中点 / 6 自由度而非 5+夹爪）。
6. **真机**：JAKA 不走 lerobot/feetech，`replay/real/so101.py` 接不上；真机需 JAKA SDK（TCP/IP）新后端，`replay/sim/mujoco_so101.py` 里硬编码的 SO-101 XML 合成也要替换。

### 7.3 对决策 D1/D3 的影响（待用户确认后定稿）
终点若是 6 轴工业臂 + 外置夹爪，则论文的 **8 维 EE+gripper 是最自然的动作空间**（JAKA 真机本来就吃笛卡尔），关节空间退化为下游执行层（EE -> IK -> 6 关节）。**D1 建议从 A 改为 B**；D3 的验收终点从 "SO-101 动起来" 改为 "Zu3+夹爪 在 cube->bowl 场景动起来"。
## 8. 决策定稿（用户确认，2026-10）

| 编号 | 决策 | 依据 |
|---|---|---|
| D1 | **动作空间 = 8 维 EE+gripper（pos3+rot3+g1+reserved1），关节空间降为执行层** | 用户选"改判"；6 轴 IK 多解 + joint_2/4 限位仅 -1.48~4.62 rad，把 IK 塞进动作表示会污染学习信号；JAKA 真机本就吃笛卡尔指令 |
| D3 | **终点 = JAKA Zu3 实物（用户确认有实物）**，仿真先行为验证手段 | 用户答"有 Zu3 实物" |
| 夹爪 | 实物资产**后补**（用户原话: "夹爪资产文件后续会有的"）→ 现在用**占位平行二指**，接口按可替换设计 | 用户自定义回答 |
| 环境 | 不在 lerobot env 安装/改动；仅在其中只读运行。仿真路径优先用 `~/miniconda3/envs/mujoco`（仅 numpy+mujoco，**无 scipy/cv2/imageio/pandas**） | 实测两个 env 的包清单 |

### 8.1 记为假设（会把我拉回来的条件）
- **A1 首个证据 = Zu3 上的离线最小闭环**（合成手 GT 点 -> EE -> IK -> mujoco 回放），不等真视频入口。→ 若你要求第一份证据就必须是真摄像头输入，我会改序。
- **A2 IK 不引入 mink**：用 mujoco `mj_jac` 的阻尼最小二乘（DLS）自实现，保持 env 零新增依赖。→ 若 DLS 在 100 个随机可达目标上成功率 < 95%，我会回来申请装 mink/qpsolvers。
- **A3 场景物体用 primitive**（2cm cube + 开口 cylinder bowl + box 桌面），弃用上游 objaverse mug/plate。→ 若你需要与论文/GR00T 数据集里的物体类别对齐，我会换。
- **A4 论文"拇指 PIP"在 21 点约定下按拇指 IP(idx=3) 实现，且 landmark 索引全部做成可配置**（MANO/HaMeR: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky）。
- **A5 真机后端（JAKA SDK / TCP-IP）留到 G5 之后**，现在只在接口层预留。

### 8.2 任务台账
| ID | 任务 | 依赖 | 状态 |
|---|---|---|---|
| T1 | Zu3 资产落位：robot.xml / scene_cube_bowl.xml / gripper_pg2.xml + 碰撞 + 抓取验证 | — | 已下发 |
| T2 | EE 动作空间层：hand->EE 数学（机器人无关）+ Zu3 DLS IK 执行层 + 数值验收 | — | 已下发 |
| T3 | 离线最小闭环：合成 GT 点 -> EE -> IK -> mujoco 回放序列 | T1,T2 | pending |
| T4 | 真视频入口 G2：mp4 解帧 + ChArUco 外参 T_base_cam + 尺度先验 | T2 | pending |
| T5 | LeRobot v3 写 8 维 EE schema + 双相机 + 778 顶点视觉对齐 | T3 | pending |
| T6 | 策略层（共享冻结 VLM + DiT + 双动作编解码器） | T5 | pending |

## 9. 验收记录（协调者独立复跑，不采信子代理自述）

### T2 · 8 维 EE 动作层 + DLS-IK —— 2025 通过
- 交付: easymimic_repro/src/ee_retarget/hand_to_ee.py(310), ik_jaka_zu3.py(387), easymimic_repro/tests_ee/test_ee_pipeline.py(407)
- 独立复跑: `~/miniconda3/envs/mujoco/bin/python easymimic_repro/tests_ee/test_ee_pipeline.py` -> exit=0, ALL CHECKS PASSED, 端到端最差 9.328e-04 m / 0.0863 deg
- 代码抽查结论（读源码确认，非采信报告）:
  - hand_to_ee.py:253-256 cam->base 对位置**和**姿态都做了变换 (R_base = R_base_cam @ R_plane_cam)，非只转位置。
  - hand_to_ee.py:181-183 R=[X, Y=Z×X, Z] 右手系；:162-163 SVD 最小奇异向量作法向。
  - hand_to_ee.py:169-171 法向符号用 R_wrist[:,2] 消歧，翻转 Z 后 Y 由 cross 重算 -> det 恒 +1。
  - hand_to_ee.py:196-197 g=(d_t-d_min)/(d_max-d_min) 后 clip -> **g=1 为张开**，与 hand-6dof-pipeline/retarget/so101.py:110 的 norm 语义一致（norm=1->open_deg）。
  - hand_to_ee.py:242-246 任一 NaN -> 8 维全 NaN，info["valid"]=False，无补零。
  - ik_jaka_zu3.py:63-65 按 joint_1..joint_6 **名字**取 dofadr（该 MJCF 含 2 个 free joint 道具，nq=20 nv=18，绝不按"前 6 个"取）。
  - ik_jaka_zu3.py 的 DEFAULT_XML_PATH 由 `__file__` 推出仓库内绝对路径（可用 `EASYMIMIC_JAKA_XML` 覆盖），构造函数仍显式拒绝相对路径。整理交接时已修，修后两套验收均复跑通过。
- 遗留风险（记入 T3 必查）:
  1. 100% 成功率来自 solve_robust() 16 次重启；单次热启动仅 65/100。真机在线控制若只允许单次求解，需接受 65% 或改初值策略。
  2. 残差贴着阈值（max 9.92e-4 m 对 1e-3 m 判据），实际精度余量很小；T3 里需测"连续帧轨迹"上的 IK 连续性，逐帧独立求解可能抖动。
  3. 相机系->基座系外参 T_base_cam 目前只能手工给，真机手眼标定尚未做（属 G2 之后）。
- T1 未完成；assets/jaka_zu3/ 现有 robot.xml(7965B)/gripper_pg2.xml(1943B)/scene_cube_bowl.xml(5114B) 但伴随 _frag*.xml/_tA/_tB/_tC/_x_v*.xml 共 13 个草稿文件，验收时须要求清理。

### T1 · Zu3 MuJoCo 资产 + 占位夹爪 —— 2025 通过
- 交付: assets/jaka_zu3/{robot.xml 146 行, gripper_pg2.xml 35 行, scene_cube_bowl.xml 65 行}, experiments/t1_verify.py 307 行。目录内无草稿残留（此前 13 个 _frag/_tA/_tB/_tC/_x_v* 已清）。
- 独立复跑: 在 cwd=/ 与 cwd=仓库根 各跑一次 t1_verify.py -> 两次均 EXIT=0 / ALL CHECKS PASSED，抓取上升 0.09806 m（阈值 0.08），滑移 1.96 mm，下颌 qpos 0.00137，两指内面间隙 0.01987 vs cube 0.02000（确实在挤压而非虚夹）。
- 抽查结论:
  - robot.xml:8 meshdir="../../third_party/.../jaka_zu3_meshes" 相对自身目录；7 个 STL 在两种 cwd 下解析到同一绝对路径 -> 我原先的"相对路径必炸"结论只对**上游**模型成立（其顶格 meshdir 泄漏进 include），新资产不 include 上游任何文件，故不复现。
  - robot.xml:22-24 collision 类 group=3 contype=1 conaffinity=1；robot.xml:34-38 pg2_pad 类 condim=4 friction="1.5 0.02 0.002"。
  - 视觉 geom 保持 contype=0，碰撞用 capsule/cylinder 代理（type 5=cylinder, 3=capsule）。
- **我给的验收判据有误，子代理拒绝造假并顶了回来**：我要求 scene nq=7，但可抓取的 cube 必须是 free joint（+7 qpos）-> scene nq=14 物理上不可能等于 7。它改为断言 robot 子树 mr.nq==7、scene nu==7，并在输出里写明理由。此为正确处置，判据作废、以子树断言 + nu 断言为准。
- 过程中两个真实失败（已定位，不属于美化）:
  1. 从 qpos0 起步时 cube 每侧预穿透 0.75 mm，9.9 g 的下颌被求解器弹开到 0.0222 m -> 改为"先张开下降，再 ctrl=0 伺服闭合"。
  2. Link_5 碰撞胶囊堵住腕部工具孔，下颌卡在 0.02225 m -> 剪短胶囊 + 4 组 Link_5/Link_6 与夹爪的 contact exclude。
- 关键几何（T3 要用）: 两指 pad 中心在 gripper_pg2 体坐标系 = (0, -0.0325, 0)，pad 半尺寸 (0.005, 0.0225, 0.011)，闭合内面 gap=0.0185 m；body gripper_pg2 pos="0 0 -0.035" 即 site tool0。**-> 夹爪 TCP 偏移 = tool0 + (0,-0.0325,0)，指沿 -y 伸出、沿 x 闭合。**
- 关节/执行器: joint_1..6 + gripper_jaw(slide, axis 1 0 0, 0~0.04)，7 个 position 执行器；home keyframe qpos="0 0.15 0 0 0 0 0.04"（全零位时法兰会插进桌面）。

## 10. 待决/风险清单（滚动）
- R1 单次热启动 IK 仅 65/100，100% 靠 16 次重启 -> 实时控制预算待 T3 量化。
- R2 IK 残差贴着 1e-3 阈值，逐帧独立求解的轨迹连续性未测 -> T3 必测 |Δq|。
- R3 手眼外参 T_base_cam 只能手工给，真机标定未做。
- R4 采集设备：用户答复为**多个 USB 相机**（多视角）-> G2 可用多视角三角化恢复公制尺度，不必单目尺寸先验；但需做多相机内参+外参标定与帧同步（时间戳对齐），且 EasyMimic 原论文用双 RGB，多视角需确认是否超集兼容。

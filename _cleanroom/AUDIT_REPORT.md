# README 字面合规校验报告（clean-room）

- 校验对象（只读）：/home/zh/Desktop/EasyMimic/easymimic_repro（README.md 219 行）
- 副本：/home/zh/Desktop/EasyMimic/_cleanroom/easymimic_repro
- 建副本命令：`rsync -a --exclude 'scratch/' easymimic_repro/ _cleanroom/easymimic_repro/`（保留 .gitignore/requirements.txt/scripts/docs/assets/src/tests_ee/experiments，排除 scratch/）
- third_party 复现方式：**整目录拷贝**（非软链；`test -L third_party/jaka-zu3-mujoco` -> REALDIR，HEAD 仍为 7c40fec78fe127030cb67f21f536620d7448bfa0）
- 解释器：/home/zh/miniconda3/envs/mujoco/bin/python（python 3.11.17 / mujoco 3.16.0 / numpy 2.4.6）
- 未执行 scripts/fetch_third_party.sh（联网 + 100 MB），按任务要求改为通读判定
- 日志与探针：_cleanroom/logs/*.log、_cleanroom/probe_relative.py、probe_slip.py、probe_geom.py、mjtest/

## A. §3.3 两条验收命令（README:78-83）

命令与结果（cwd = 副本根目录，DISPLAY=:0）：
1. `/home/zh/miniconda3/envs/mujoco/bin/python tests_ee/test_ee_pipeline.py` -> **EXIT=0**，86 行，末行 `ALL CHECKS PASSED`（与 README 一致）
2. `/home/zh/miniconda3/envs/mujoco/bin/python experiments/t1_verify.py` -> **EXIT=0**，105 行，末行 `RESULT: ALL CHECKS PASSED`（含 README 所说的 `ALL CHECKS PASSED` 子串）

补充：两条命令换 cwd（cwd=/home/zh/Desktop/EasyMimic/_cleanroom，命令行写 `easymimic_repro/tests_ee/test_ee_pipeline.py` 等）同样 EXIT=0 并打印同样末行 -> README 未写 cwd 前置条件，但该前置条件实际不成立（好的一面：不依赖 cwd）。
注意：PATH 上的裸 `python` 是 conda base 3.13.11，`import mujoco` 失败；README §3.3 的字面命令必须先在 §3.1 的 `conda activate mujoco` 下运行——README 已按此顺序给出，故不算不一致。

## B. git 入库清单与 .gitignore 效果

命令：`git init -q . && git add -A && git status --short`（副本内，并预置了 scratch/probe.txt、out/run.png、tmp_fig.png 三个诱饵文件）
实际入库 **16 个文件**：
README.md / requirements.txt / .gitignore / assets/jaka_zu3/{robot.xml,gripper_pg2.xml,scene_cube_bowl.xml} / src/ee_retarget/{hand_to_ee.py,ik_jaka_zu3.py} / src/closed_loop/{synth_hand.py,run_closed_loop.py} / tests_ee/test_ee_pipeline.py / experiments/{t1_verify.py,probe_env.py} / scripts/fetch_third_party.sh / docs/{DESIGN_LOG.md,VERIFICATION.md}
`git ls-files` 与之一致（16 条）；`git ls-files | wc -l` =16。

逐条核对（`git check-ignore -v --no-index`）：
| 目标 | 命中规则 | 结论 |
|---|---|---|
| third_party/jaka-zu3-mujoco/jaka_zu3_mujoco.xml | .gitignore:9:third_party/ | ✓ 不入库 |
| scratch/probe.txt | .gitignore:12:scratch/ | ✓ 不入库 |
| out/run.png | .gitignore:13:out/ | ✓ 不入库 |
| src/ee_retarget/__pycache__/*.cpython-311.pyc | .gitignore:2:__pycache__/ | ✓ 不入库 |
| assets/jaka_zu3/robot.xml | 未命中（不忽略） | ✓ 在库内 |
结论：**无漏排、无误排**。唯一「做了但没说」是 .gitignore:14 的 `*.png` 全仓库生效（README:118 只声明 third_party/、scratch/、out/），将来放 assets/*.png 会被静默忽略——目前无 tracked PNG，非阻塞。

## C. §4 表格逐格核对（README:94-104）

| README 行 | README 数字 | 测试输出原文 | 判定 |
|---|---|---|---|
| 94 | 2.2e-16 | `[PASS] R three axes closed (d_min)   err=2.220e-16` | ✓ |
| 95 | 0 / 0.5 / 1 | `g closed (d_min) == 0.0  g=0.000000000000000` / `g mid (d_mean) == 0.5  g=0.500000000000000` / `g open (d_max) == 1.0  g=1.000000000000000` | ✓ |
| 96 | 6.66e-16；det−1=4.44e-16 | `[2] orthonormality / chirality  max |R^T R-I|=6.66e-16  max |det-1|=4.44e-16` | ✓ |
| 97 | 100/100 | `success rate           100/100 = 100.0%` | ✓ |
| 98 | p50 5.73e-4 / p95 9.65e-4 / max 9.92e-4 | `position error  [m]    p50=5.733e-04  p95=9.648e-04  max=9.922e-04` | ✓ |
| 99 | p50 0.083° / max 0.498° | `orientation err [deg]  p50=0.0828  p95=0.4374  max=0.4977` | ✓ |
| 100 | 9.33e-4 m / 0.086° | `[6] worst residual ... 9.328e-04 m / 0.0863 deg` | ✓ |
| 101 | 0.09806 m | `[PASS] cube lifted >= 0.08 m (measured 0.09806 m)` | ✓ |
| 102 | **1.96 mm** | **无任何命令输出该数字**（见下） | ✗ |
| 103 | 0.01987 vs 0.02000 | `[PASS] pads are pressing the cube ... (gap 0.01987 m vs cube 0.02000)` | ✓ |
| 104 | 480×640，均值 71.22 | `[5] offscreen render  frame shape=(480, 640, 3) dtype=uint8 mean=71.22 min=0 max=234` | ✓ |
| 106 | 「只剩 8% 余量」 | 9.922e-4 对 1e-3：(1e-3−9.922e-4)/1e-3 = **0.78%** | ✗（差 10 倍） |

关于 1.96 mm：`grep -rn '1\.96'`（排除 third_party/scratch）只命中 README.md:102、docs/VERIFICATION.md:127、docs/DESIGN_LOG.md:135，**三处都是叙述，没有任何脚本打印滑移量**（t1_verify.py 只记录 z、指尖位置、cube_pos、gap、渲染均值）。用探针脚本在 t1_verify 源码里于方块闭合后与抬升后各打一次 cube xy：
`SLIP_PROBE cube xy at close=[0.486977 0.043497] after lift=[0.487076 0.043498] |dxy|=0.000099 m = 0.10 mm`，脚本仍 `RESULT: ALL CHECKS PASSED`。
即：README §4 唯一不可复跑的一格是滑移，实测约 **0.099 mm**（比声称的 1.96 mm 好 20 倍），但 1.96 mm 无法由任何文档化命令复现。

## D. 事实性断言核对

✓ 成立：
- §3.3 (README:86)「该 XML 必须用绝对路径加载 / 构造函数对相对路径直接抛错」：`JakaZu3IK('third_party/jaka-zu3-mujoco/jaka_zu3_mujoco.xml')` -> `ValueError | xml_path must be absolute: ... got 'third_party/jaka-zu3-mujoco/jaka_zu3_mujoco.xml'`（src/ee_retarget/ik_jaka_zu3.py:168-172）；直接 `mujoco.MjModel.from_xml_path(相对)` -> `ValueError | Error opening file 'assets/jaka_zu3_meshes/third_party/jaka-zu3-mujoco/assets/objaverse/mug_5/collision/model_normalized_collision_22.obj'`。README 引的报错串是 `assets/jaka_zu3_meshes/assets/objaverse/...`（cwd=上游模型目录时形态），机制一致、字面略有前缀差异。`EASYMIMIC_JAKA_XML` 指向相对路径同样抛错 ✓。
- §7 (README:157) 桌面顶面：table_top pos z=0.78 + size z=0.02 -> **0.80000** ✓（机器人基座 body Link_0 pos z=**0.79999**，比 0.80 低 10 µm，「精确在这一层」差 1e-5）
- §7 (README:158) site tool0 与 body gripper_pg2 原点重合：探针 `delta site_world - body_world = [0. 0. 0.]` ✓；（caveat：site 的父 body 是 **Link_6**，声明在 robot.xml:105，不是 gripper_pg2；改 gripper_pg2 的 pos 会与 site 脱钩，README 未提）
- §7 (README:159) 两指内面 x=±0.00925、全闭 gap=0.0185：探针 jaw=0 时内面 `0.606819 .. 0.625319 (gap 0.018500)` ✓；交叉验证 0.01987−0.00137=0.018500 ✓
- §7 (README:160) gripper_jaw slide axis=[1,0,0] range=[0,0.04]、执行器 kp=300 kv=5 ✓（robot.xml:129）
- §7 (README:162) home keyframe qpos=`0 0.15 0 0 0 0 0.04` ✓（assets/jaka_zu3/scene_cube_bowl.xml:63，MuJoCo 补零到 nq=14）
- §5 (README:114-138) 目录树 16 条路径**全部存在，无缺失项**（README.md、requirements.txt、.gitignore、assets 3 个 xml、src 两个包 4 个 py、tests_ee、experiments 2 个 py、scripts、docs 2 个 md）
- §8.5 (README:179) ik_jaka_zu3.py 的 rotmat_to_rotvec 只有 theta<1e-7 一支（:122-130），hand_to_ee.py 的是三支实现 ✓
- §6.5 (README:148) `Jp = Jp - _skew(lever) @ Jr`（:249-251，代码里函数名是 `_skew`）✓
- §8.9 (README:183)「4 组 contact exclude」✓：恰好 4 条是 Link_5/Link_6 × gripper_pg2/gripper_finger_b（另 6 条是相邻连杆）；碰撞代理 5 capsule + 2 cylinder ✓
- §3.2 (README:74)「checkout 约 100 MB，其中 75 MB 是 Objaverse 道具」✓：`du -sh third_party/jaka-zu3-mujoco` = 100M，`assets/objaverse` = 75M（.git 18M、assets 82M）
- §2 (README:49) 第 8 维恒为 0 ✓（RESERVED_VALUE = 0.0）；`info["valid"]` 存在 ✓
- §9 (README:195) 本仓库无 LICENSE ✓；third_party 内无 LICENSE ✓
- 脚本取回：`scripts/fetch_third_party.sh` 的一条 `git clone` + `checkout 7c40fec` **确实产出** `third_party/jaka-zu3-mujoco/jaka_zu3_mujoco.xml`（本地 clone HEAD 已核对为 7c40fec78fe1）；但脚本的校验只有 (a) 7 位短 sha checkout、(b) XML 文件存在、(c) 打印 `rev-parse --short HEAD`，**没有断言 HEAD==7c40fec、没有校验 STL/网格**；若默认分支不含该 commit，`set -e` 下 checkout 会硬失败（干净失败分支成立）。

✗ 不成立 / 与实测不符：
- §7 (README:165-166) 两条 MuJoCo「静默坑」在 pinned 的 mujoco 3.16.0 上**不成立**：`<include>` 的单 `<body>` 根**不会**被丢弃（探针 main.xml：bodies = ['world','arm','b_included']）；`<actuator>` 写进被 include 的 body **不是静默丢弃**而是硬报错 —— `ValueError | XML Error: Schema violation: unrecognized element Element 'actuator'`（写在 body 内、以及作为被 include 文件根的兄弟节点，两种写法都报错）。同一说法也复制在 assets/jaka_zu3/gripper_pg2.xml:1-16 注释里。功能无影响（资产照样加载、两条验收照样通过），但「静默 + 必须包裹」的因果是错的。
- §8.7 (README:181) 行号失效：指到的 :264 实为 `hi = self.joint_limits[:, 1]`、:293 实为 `continue`；真正的两次 FK 在 :269 与 :298，jacobian 内的 `_sync` 在 :243。机制（每次迭代 2 次 FK + 1 次 _sync）正确，行号全错。
- §3.3 (README:83)「普通脚本 + `assert`」：两个验收脚本 **assert 语句数为 0**（grep -c 'assert ' -> 0），用的是 `check()` 辅助函数；「不依赖 pytest」✓（环境里确实没有 pytest，ModuleNotFoundError）。
- §4 (README:102) 1.96 mm / §4 (README:106) & §8 (README:180)「余量 8%」：见 C 节，前者不可复跑（实测 0.099 mm），后者应为 0.78%。
- §5 (README:118)「.gitignore ← third_party/、scratch/、out/ 不入库」漏述 `*.png`、`__pycache__/`、`*.py[cod]`、`.pytest_cache/`、`*.egg-info/`；其中 `*.png` 是唯一会影响未来资产入库的一条。

未验证（不猜）：
- §2 (README:48) 与 §6.4 (README:147) 对 `hand-6dof-pipeline/retarget/so101.py:110` 的 `norm` 语义引用：该仓库在硬边界外（禁止触碰），且不在 easymimic_repro 内（README §9 自己也说「不在本仓库内」），无法核对。
- §6.6 (README:149) 绝对 λ=1e-3 的 `|dq| = 127.86 rad`、`48/100`：当前代码只有相对阻尼 `lam = lam_rel * sqrt(trace(JJt)/6)`（:285），无该参数化路径，属历史测量，无法复现。
- §8.1 (README:173) `src/closed_loop/`「从未跑完过」：否定性断言，未尝试运行（会超出本任务范围且无法证明「从未」）。
- §9 (README:191)「上游 `package.xml` 写 TODO」：third_party/jaka-zu3-mujoco **没有 package.xml**（`find -name package.xml` 为空），只有上游 README.md:43 的第二手说法「package.xml says TODO」-> 本仓库内不可核对。
- §3.1 (README:61-63) `conda create` / `pip install -r requirements.txt`：禁止建环境/装包且无网络，未执行；仅核对需求文件内容（mujoco==3.16.0、numpy==2.4.6）与现成环境版本一致。
- §9 (README:193) MuJoCo Apache-2.0：公开事实，离线未复核。

## E. §0 English summary 与中文正文一致性
`Two acceptance suites pass` ✓（A 节）；`21 hand keypoints -> 8-D end-effector ... -> six joint angles` ✓（§2）；`Video ingestion, camera / hand-eye calibration, dataset writing and the learned policy are not implemented` ✓（§8.2-8.4）；**无夸大**，唯一措辞风险是 `verified ... by a physical grasp inside MuJoCo`：这里是**仿真接触抓取**（无硬件），中文 §1 明确「没有在真机上跑过一次」，建议改成 `contact-based grasp`。

## F. 结论
**PASS（可据此交接）**，附带必须修正的不一致（均非阻塞）：
1. README:102 的 1.96 mm —— 换成可复跑的数字（实测 0.099 mm）或标注为历史测量；这是全表唯一违反「全部可复跑」的一格。
2. README:106 / 180 的「8% 余量」-> 0.78%（10 倍数量级错误）。
3. README:165-166 两条 MuJoCo 静默坑的事实描述错误（改为「`<actuator>` 写进被 include 的 body 会报 schema violation」并删掉「单 body 根被静默丢弃」）。
4. README:181 的行号 :264/:293 -> :269/:298，jacobian 内 `_sync` -> :243。
5. README:83 的 `assert` 措辞 -> `check()` 辅助函数。
6. README:118 补上 `*.png`（以及其它三条忽略规则）。
阻塞级不一致：**0 条**（两条验收命令、git 入库清单、§5 目录树、§7 几何数字均与实测一致；1.96 mm 一格若按「§4 全表可复跑」的严格口径计，则升级为 FAIL 的单格，但实测行为优于声称值，不影响后续任何操作）。
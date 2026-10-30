# legacy/ — 已退役的 T3 离线闭环（只作历史参照：不运行、不修、不参与验收）

## 移动清单

| 旧路径 | 新路径 | 行数（移动前 -> 现在） |
|---|---|---|
| `src/closed_loop/run_closed_loop.py` | `legacy/closed_loop/run_closed_loop.py` | 804 -> 808（+4 行横幅） |
| `src/closed_loop/synth_hand.py` | `legacy/closed_loop/synth_hand.py` | 189 -> 193（+4 行横幅） |

两个文件的模块 docstring 顶部各加了 3 行 `RETIRED` 横幅，其余字节未动。

## 为什么退役

T7 把占位夹爪换成了真夹爪（`assets/jaka_zu3/gripper_real.xml`），这套脚本赖以成立的三个前提**同时**失效：

1. **joint 名**：`gripper_jaw`（slide，`range="0 0.04"`）-> `joint_7`（hinge，`range="0 0.9"`，`assets/jaka_zu3/gripper_real.xml:68`）。脚本按名字取 `model.joint("gripper_jaw")`，在当前资产上必然 KeyError。
2. **site `tool0` 的语义**：git HEAD 的 `assets/jaka_zu3/robot.xml:105` 里 `tool0` 在法兰上（`pos="0 0 -0.035"`），所以脚本靠 `TCP_OFFSET = (0, -0.0325, 0)` 补 32.5 mm 力臂；当前资产把 `tool0` 直接放到指尖捏合中心（`robot.xml:142`，`pos="0.0172 -0.0035 0.127"`），再补一次力臂就是重复补偿。
3. **夹爪控制量**：`jaw ctrl = g * 0.04`（米）-> `joint_7 ctrl = g * 0.9`（弧度，`src/pose_cmd/replay.py:40`）。

## 实测：两条命令，退出码都不是 0

移动前（文件还在 `src/closed_loop/` 时，字节与本目录这两份减去横幅后一致）：

```
$ python src/closed_loop/run_closed_loop.py --no-render
EXIT=1
KeyError: "Invalid name 'gripper_jaw'. Valid names: ['cube_free', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']"
```

抛出点是 `src/closed_loop/run_closed_loop.py:409`（同一语句现在位于 `legacy/closed_loop/run_closed_loop.py:413`，差的 4 行就是横幅）。

撞 KeyError 之前，A/B/C 三段已经不符：`[FAIL] B all 75 frames reach the IK tolerance`（单次收敛 0/75，max position residual 1.169e-02 m，max orientation residual 8.108e+00 deg），根因是上面第 2 条——`tool0` 已经是 TCP，力臂被补了两次。

移动后：

```
$ python legacy/closed_loop/run_closed_loop.py --no-render
EXIT=1
ModuleNotFoundError: No module named 'hand_to_ee'
```

位置 `legacy/closed_loop/run_closed_loop.py:54`。原因：脚本用 `_SRC = dirname(dirname(__file__))` 把"上一级目录"当仓库根去拼 `sys.path`，搬进 `legacy/` 后 `_SRC` 变成 `legacy/`，指向不存在的 `legacy/ee_retarget`。**按本次约定，legacy 里的代码不修**，所以它到此为止，能坏就坏着。

## 它当年验过什么（占位夹爪时代，当前资产上无法复现）

`python src/closed_loop/run_closed_loop.py --modes direct,servo` -> `EXIT=0`，末行 `VERDICT: PASS   19/19 checks passed`；75 帧 @30 Hz；`|dq|_inf` p50 0.0114 / max 0.0170 rad（0/74 超限）；方块抬升 +0.09910 m（direct）/ +0.09912 m（servo）；两指内面 gap 0.01987 m vs 方块 0.02000 m；每个模式 16 张 PNG。原始记录在 `docs/DESIGN_LOG.md` §11。

## 现役替代

`experiments/pose_replay.py` + `configs/waypoints_cube_lift.json`：8 维 waypoint 表、50 Hz、`joint_7`、`tcp_offset = None`。`EXIT=0`，末行 `RESULT: ALL CHECKS PASSED`，rise 0.11748 m，slip 0.000442 m，54 张 PNG。逐条数字见 `README.md` §4 与 `docs/VERIFICATION.md` §C。

## 不要做的事

- 不要修它来"跑起来"：它验的是占位夹爪的几何，修好了也不证明当前资产。
- 不要 import 它：全仓没有 Python 文件 import `closed_loop` 或 `synth_hand`（唯一的路径提及是 `src/ee_retarget/ik_jaka_zu3.py:77` 的一句注释，那份文件本次不在可改范围内）。
- 不要把它写回 quick start。

## 未验证

- **在 git HEAD 的资产上它是否还能 19/19：未验证。** HEAD 的 `robot.xml:110` 确实 `<include file="gripper_pg2.xml"/>`、`tool0` 也仍在法兰上，看起来正是它当年验过的组合；但要真跑得把工作树的资产回退到 HEAD 版本，那会破坏 T7 的交付物，本次没做。
- `synth_hand.py` 被单独 import 时行为如何：未验证（它只被同目录的 `run_closed_loop.py` import，而那个 import 现在先挂在第 54 行）。

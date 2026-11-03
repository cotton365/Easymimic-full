# ACT 全流程：示教数据 → LeRobot 数据集 → 训练 → 开环评测

本文档记录**已实测跑通**的三段命令。所有 CLI 参数名均来自 `--help` 实测，不是凭记忆写的。
验证环境：/home/zh/miniconda3/envs/lerobot/bin/python（py3.10、lerobot 0.4.4、torch 2.13.0+cu132、numpy 2.2.5、mujoco 3.1.6、**无 torchcodec**）。

## 0. 三条硬约束（先看，否则后面每条命令都会踩）

1. **必须用 lerobot 的 python**：/home/zh/miniconda3/envs/lerobot/bin/python。系统 python 没有 lerobot。
2. **torchcodec 缺失** → 数据集只能 `use_videos=False`（逐帧 PNG）。导出脚本已经写死这条，不要改。
3. **HF datasets 缓存必须指到可写目录**。默认缓存在 /home/zh/.cache/huggingface/datasets，在本机是**只读**的，任何"读回数据集"的动作（训练、校验）都会立刻挂：

```
OSError: [Errno 30] Read-only file system: '/home/zh/.cache/huggingface/datasets'
```

所以**每个新 shell 先跑这段**（放在仓库根目录 /home/zh/Desktop/EasyMimic/easymimic_repro 下）：

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
export PY=/home/zh/miniconda3/envs/lerobot/bin/python
export HF_HOME=$PWD/data/_scratch/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_DATASETS_CACHE"
```

> 只跑导出脚本时不碰这个缓存；但训练 / 读回校验一定需要，为了不来回切 shell，统一先 export。

---

## 1. 导出：示教目录 → LeRobot v3 数据集

**在哪个 env**：lerobot env（`$PY`），仓库根目录。

```bash
$PY scripts/export_lerobot_dataset.py \
  --demos out/act_demos \
  --out   data/lerobot/act_pick_cube
```

输入契约（上游导出器必须满足）：

```
out/act_demos/index.json              {fps, task, camera_names, state_dim, action_dim}
out/act_demos/ep000/data.npz          qpos(N,7) act(N,7) t(N,) cube_pos(3,) cube_quat(4,) rise_m slip_m
out/act_demos/ep000/front/000000.png  256x256 RGB，6 位零填充序号
out/act_demos/ep000/wrist/000000.png
```

输出 feature（**恰好这四个**，多一个少一个都会在 `add_frame` 里报 Feature mismatch）：

| key | dtype | shape |
|---|---|---|
| `observation.state` | float32 | (7,) |
| `action` | float32 | (7,) |
| `observation.images.front` | image | (256, 256, 3) |
| `observation.images.wrist` | image | (256, 256, 3) |

**预期输出关键行**（fixture 实测，2 条 episode 共 34 帧，耗时约 4s）：

```
[export] out=.../data/lerobot/act_pick_cube_fixture repo_id=local/act_pick_cube fps=20 task='pick up the cube' episodes=2
[export] ep000: 18 帧 x 2 相机 已写入
[export] ep001: 16 帧 x 2 相机 已写入
[export] 完成: num_episodes=2 num_frames=34 root=.../data/lerobot/act_pick_cube_fixture
```

磁盘产物（`use_videos=False` 时图像不进 mp4，而是逐帧 PNG）：

```
data/lerobot/act_pick_cube/data/chunk-000/file-000.parquet
data/lerobot/act_pick_cube/images/observation.images.front/<ep>/<frame>.png
data/lerobot/act_pick_cube/images/observation.images.wrist/...
data/lerobot/act_pick_cube/meta/{info,stats,episodes,tasks}
```

### 常见报错

- **FileExistsError / 输出目录已存在**：`LeRobotDataset.create` 的 root **必须不存在**，它不会覆盖。本脚本会先自己拦一道并给中文提示：
  `[export] 错误: 输出目录已存在，LeRobotDataset.create 不会覆盖: <path>`
  处置：`rm -rf data/lerobot/act_pick_cube` 后重跑，或换 `--out`。
- **ModuleNotFoundError: torchcodec**：有人把 `use_videos` 改回 True 了。保持 `use_videos=False`。
- **Feature mismatch in frame dictionary / Extra features**：`add_frame` 的 key 集合必须与 features 完全一致；别多塞 `timestamp` / `frame_index`（这些是自动补的）。
- **图像形状/类型报错**：必须是 `uint8 (256,256,3)`，脚本逐张校验并指出文件名。

### 读回校验（可选，但建议每次导出后跑一次）

```bash
$PY - <<'EOF'
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="local/act_pick_cube", root="data/lerobot/act_pick_cube")
print("num_episodes =", ds.num_episodes, "num_frames =", ds.num_frames, "fps =", ds.fps)
batch = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
for k, v in batch.items():
    print(f"  {k:28s}", tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
EOF
```

**预期输出**（fixture 实测）：

```
num_episodes = 2 num_frames = 34 fps = 20
  observation.state            (4, 7)
  action                       (4, 7)
  observation.images.front     (4, 3, 256, 256)
  observation.images.wrist     (4, 3, 256, 256)
  timestamp                    (4,)
  frame_index                  (4,)
  episode_index                (4,)
  index                        (4,)
  task_index                   (4,)
  task                         list
```

注意图像是 `float32` CHW 且已归一化到 `[0,1]`（实测 min 0.102 / max 0.886）——这正是策略要吃的东西，自己拼 batch 时不要重复除 255。

---

## 2. 训练

两个版本：**CPU 冒烟**（只验证链路通）与 **GPU 正式**（桌面会话跑）。
两者参数名一致，差别只有 `--policy.device`、步数、batch 与 workers。

### 2.1 通用前置（实测踩过的两个坑，两条命令都必须带）

- `--policy.repo_id` **必填**，否则会报：`ValueError: 'policy.repo_id' argument missing. Please specify it to push the model to the hub.`（本地训练也要填，给个 local/xxx 名字即可）。
- `--policy.push_to_hub=false` **必填**。它默认是 True，训练结束会去 huggingface.co/api/repos/create 建仓；本机无外网，会在收尾阶段炸掉并**白等约 140s 重试**：
  `requests.exceptions.ConnectionError: ... Failed to establish a new connection: [Errno 101] Network is unreachable`
  实测对比：带 push_to_hub=false 全程 **9s / exit 0**；不带则 **145s / exit 1**（checkpoint 其实已写好，但退出码是 1，脚本化流程会误判）。

### 2.2 CPU 冒烟：3 步（耗时约 9s，fixture 实测）

```bash
$PY -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/act_pick_cube \
  --dataset.root=data/lerobot/act_pick_cube_fixture \
  --policy.type=act \
  --policy.repo_id=local/act_pick_cube_smoke \
  --policy.push_to_hub=false \
  --policy.device=cpu \
  --policy.chunk_size=8 \
  --policy.n_action_steps=8 \
  --output_dir=out/act_train_smoke \
  --job_name=act_smoke \
  --batch_size=4 \
  --num_workers=0 \
  --steps=3 \
  --log_freq=1 \
  --save_freq=3 \
  --wandb.enable=false
```

**预期输出关键行**（实测原文）：

```
INFO ot_train.py:336 num_learnable_params=51552135 (52M)
INFO ot_train.py:423 step:1 smpl:4 ep:0 epch:0.12 loss:76.517 grdn:1259.855 lr:1.0e-05 updt_s:0.562 data_s:0.176
INFO ot_train.py:423 step:2 smpl:8 ep:0 epch:0.24 loss:60.017 grdn:1136.252 lr:1.0e-05 updt_s:0.487 data_s:0.077
INFO ot_train.py:423 step:3 smpl:12 ep:1 epch:0.35 loss:49.532 grdn:907.036 lr:1.0e-05 updt_s:0.546 data_s:0.081
INFO ot_train.py:443 Checkpoint policy after step 3
INFO ot_train.py:514 End of training
```

判据：**有 loss: 打印、无 Traceback、出现 End of training、exit 0**。CPU 单步约 0.5–0.6s（updt_s）。

产物：

```
out/act_train_smoke/checkpoints/000003/pretrained_model/{config.json, model.safetensors,
    policy_preprocessor.json, policy_preprocessor_step_3_normalizer_processor.safetensors,
    policy_postprocessor.json, policy_postprocessor_step_0_unnormalizer_processor.safetensors,
    train_config.json}
out/act_train_smoke/checkpoints/last -> 000003      # 符号链接
```

> policy_preprocessor.json / policy_postprocessor.json 这两个文件**必须跟 checkpoint 一起保留**：归一化统计量在里面，评测脚本靠它把 raw 动作还原。只拷 model.safetensors 是没法评测的。

### 2.3 GPU 正式版：2000 步 / batch 8（桌面会话跑）

**前置**：在**你自己的桌面会话**（不是本沙箱）里跑；本沙箱 nvidia-smi 直接失败，只能 --policy.device=cpu。

```bash
$PY -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/act_pick_cube \
  --dataset.root=data/lerobot/act_pick_cube \
  --policy.type=act \
  --policy.repo_id=local/act_pick_cube \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=out/act_train \
  --job_name=act_pick_cube \
  --batch_size=8 \
  --num_workers=8 \
  --steps=2000 \
  --log_freq=50 \
  --save_freq=500 \
  --seed=1000 \
  --wandb.enable=false
```

- `--policy.chunk_size` / `--policy.n_action_steps` 留空即 ACT 默认 **100/100**（实测确认默认 chunk_size: 100，且 episode 短于 100 帧时 LeRobot 自动用末帧 pad，不报错）。
- 显存吃紧：加 `--policy.use_amp=true`，或把 `--batch_size` 降到 4。
- 2000 步会在 checkpoints/000500、001000、001500、002000 各存一个，最终产物同样是 checkpoints/last/pretrained_model。
- 无外网别开 `--wandb.enable=true`（会卡在 wandb 认证）。

### 常见报错

- **CUDA 不可用**：`AssertionError: Torch not compiled with CUDA enabled` 或 `RuntimeError: Found no NVIDIA driver` → 本环境 torch 虽标 +cu132，但沙箱无驱动，一律 --policy.device=cpu；GPU 那版只能在桌面会话跑。
- **OSError: [Errno 30] Read-only file system: /home/zh/.cache/huggingface/datasets** → 忘了 export HF_HOME / HF_DATASETS_CACHE（见第 0 节）。
- **ValueError: 'policy.repo_id' argument missing** → 补 --policy.repo_id。
- **训练结束才炸、huggingface.co 连接失败** → 忘了 --policy.push_to_hub=false。
- **torchcodec 相关报错** → 数据集被以视频模式创建了；重导时保持 use_videos=False。

---

## 3. 开环评测（不碰仿真器）

**在哪个 env**：lerobot env，仓库根目录，`--device cpu` 即可（本脚本**完全不走 torchcodec**：日志/图像直接读 PNG + npz）。

```bash
$PY scripts/eval_act_openloop.py \
  --ckpt      out/act_train/checkpoints/last/pretrained_model \
  --demos     out/act_demos \
  --holdout-ep last \
  --device    cpu
```

- `--ckpt` 可以传 .../pretrained_model，也可以直接传训练输出根目录（如 out/act_train），脚本会自动解析到 checkpoints/last/pretrained_model。
- `--holdout-ep`：last（默认，最后一条）/ 序号 19 / 名字 ep019。**至少要 2 条 episode**，否则没有"留出"可言，脚本直接报错退出。
- 每个时刻先 `policy.reset()` 再 `select_action()`，保证是"逐步单帧推理"，不是复用 action chunk 队列。

**评测口径**：三方（模型 / mean 基线 / persistence 基线）都在**同一批时刻 t=1..N-1** 上比。
t=0 没有"上一帧动作"，persistence 无定义，故一律排除——否则基线会被这一帧白送分。

**PASS 判据**：模型总体 MSE 同时低于两个基线 → exit 0；否则 FAIL → exit 1。

**预期输出**（fixture + 3 步冒烟 checkpoint，实测原文，节选）：

```
[eval] ckpt=.../out/act_train_smoke/checkpoints/000003/pretrained_model
[eval] demos=.../out/act_demos_fixture holdout=ep001 frames=16 task='pick up the cube' device=cpu
[eval] loaded ACTPolicy  chunk_size=8 n_action_steps=8

[eval] 评测步数 = 15 (t=1..15, 与 persistence 同口径)
     dim |  model MSE  model MAE |   mean MSE |  persis MSE
--------------------------------------------------------------
  joint1 |   0.000000   0.000000 |   0.000000 |    0.000000
  joint2 |   0.002859   0.045573 |   0.002668 |    0.000152
  joint3 |   0.000000   0.000000 |   0.000000 |    0.000000
  joint4 |   0.006796   0.067521 |   0.006002 |    0.000343
  joint5 |   0.000000   0.000000 |   0.000000 |    0.000000
  joint6 |   0.000159   0.010744 |   0.000167 |    0.000010
 gripper |   0.000008   0.002311 |   0.000007 |    0.000000
--------------------------------------------------------------
 overall |   0.001403   0.018021 |   0.001263 |    0.000072

[eval] 总体 MSE  model=0.001403  mean-baseline=0.001263  persistence-baseline=0.000072
[eval] 相对 mean  基线: 0.900x
[eval] 相对 persis基线: 0.051x
[eval] 判定: FAIL  (模型总体 MSE 须同时低于 mean 基线 0.001263 与 persistence 基线 0.000072)
```

退出码实测 **1**（FAIL）。**3 步冒烟必然 FAIL，这是正常的**：loss 才从 76.5 降到 49.5，模型基本还停在均值附近。换 2000 步的 checkpoint 重跑这条命令，PASS/FAIL 才有业务含义。

### 常见报错

- **`[eval] 错误: ...只有 1 条 episode`** → 开环评测需要留出集，先补 episode 或换 --demos。
- **`[eval] 错误: ... 下找不到 config.json`** → --ckpt 指错层级；传 .../checkpoints/last/pretrained_model 或训练输出根目录。
- **KeyError: 'policy_preprocessor.json' / 归一化后动作离谱** → checkpoint 目录缺 policy_preprocessor.json（被手工拷漏）。这两个 json+safetensors 必须与模型同目录。
- **ModuleNotFoundError: torchcodec** → 说明给 --demos 传了 data/lerobot/... 数据集路径。本脚本要的是**示教原始目录**（index.json + epNNN/），不是 LeRobot 数据集。
- **FileNotFoundError: .../data.npz** → --demos 指到了单条 ep000 目录，应指上一层的示教根目录。

---

## 4. 为什么本轮**不做**闭环仿真评测

`lerobot-eval` 的闭环评测路径要求 --env 装配一个 gym 环境。本仓库的 MuJoCo 场景在 lerobot 侧的 mujoco **3.1.6** 下加载不进去，实测报错：

```
invalid qpos size, expected length 14
```

即场景 XML 里 <key> 记录的 qpos 只有 7 维（6 轴臂 + 夹爪），而模型编译后 nq=14（多了 mocap / 自由物体自由度），mujoco 在两者不匹配时直接拒绝加载 keyframe，因此 lerobot 的环境装配在第一步就失败，任何 --env.type 组合都跑不到 env.reset()。

要闭环评测，需要先改本仓库场景 XML 的 <key>（把 qpos 补到 14 维，或改用无 keyframe 的 reset），**这属于 assets/ 侧的改动，不在本次任务边界内**，故本轮只交付开环评测。场景修好后再用 lerobot-eval --policy.path=out/act_train/checkpoints/last/pretrained_model 接对应 --env.type。

---

## 5. 本轮自测记录（fixture，真数据未到）

| 环节 | 命令 | 结果 |
|---|---|---|
| 造 fixture | out/act_demos_fixture（ep000 18 帧 / ep001 16 帧，256x256，fps 20） | 2 条 episode 齐备 |
| 导出 | export_lerobot_dataset.py --demos out/act_demos_fixture --out data/lerobot/act_pick_cube_fixture | num_episodes=2 num_frames=34，约 4s |
| 读回 | DataLoader(batch_size=4) | observation.state (4,7)、两路图像 (4,3,256,256) |
| 训练冒烟 | 见 2.2 | loss 76.517 → 60.017 → 49.532，9s，exit 0 |
| 开环评测 | 见 3 | 总体 MSE 0.001403 vs mean 0.001263 / persis 0.000072 → FAIL，exit 1 |
| 重复导出 | 同上再跑一次 | 中文报错"输出目录已存在"，exit 1 |

真数据（out/act_demos 20 条）到位后，把文档里的 out/act_demos_fixture → out/act_demos、act_pick_cube_fixture → act_pick_cube、act_train_smoke → act_train 换掉即可，命令结构不用改。

---

## 6. 真数据实测记录（out/act_demos 20 条）

| 环节 | 命令/配置 | 实测 |
|---|---|---|
| 导出 | export_lerobot_dataset.py --demos out/act_demos --out data/lerobot/act_pick_cube | num_episodes=20 num_frames=2160，44 MB |
| 读回 | LeRobotDataset(root=data/lerobot/act_pick_cube) | len=2160、fps=20、state (7,)、action (7,)、两路 (3,256,256) |
| CPU 冒烟 | 200 步 / batch 8 / chunk 32 / n_action 8 / device cpu | 约 4 分钟，checkpoint 落在 out/act_train_cpu200/checkpoints/last |
| 开环评测 | 同上，--ckpt out/act_train_cpu200 --demos out/act_demos | 总体 MSE 0.020980 vs mean 0.034825 / persis 0.000108 → FAIL，exit 1 |

persistence 基线（复制上一帧动作）在 20 Hz 的平滑航点示教上极强：末段速度趋零时相邻动作几乎相同。
200 步的模型只赢过 mean 基线、输给 persistence 是预期结果。这也说明「判据 = 同时低于两个基线」
在轨迹内插值这件事上没有难度，真要衡量泛化得靠留出 episode 或闭环评测。

另：§4 提到的 <key name="home" qpos="7 个数"> 与 nq=14 不匹配，是 lerobot-eval 闭环评测加载本仓库场景的
第一个硬报错（mujoco 3.1.6 直接拒绝加载 keyframe）。修它属于 assets/ 侧改动，留给下一轮。

## 7. 闭环 rollout 与 MuJoCo 窗口回放

策略在环里真的驱动机械臂（不是开环比 MSE）：每一拍取两路相机 + `qpos[:7]` 喂策略，
它的 7 维动作直接写进 `data.ctrl[:7]`，跑 25 substeps × 0.002 s = 20 Hz。

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
export HF_HOME=$PWD/data/_scratch/hf_home HF_DATASETS_CACHE=$HF_HOME/datasets
export DISPLAY=:1
PY=/home/zh/miniconda3/envs/lerobot/bin/python

# 立刻看：一条 trial 跑完后立刻弹出 MuJoCo 窗口，20 Hz 回放，关窗口才退出
$PY experiments/act_rollout.py --trials 1 --viewer

# 三种起点（决定你看到的是"照轨迹走"还是"真在纠偏"）
$PY experiments/act_rollout.py --trials 1 --viewer --start approach   # 训练分布，10/10
$PY experiments/act_rollout.py --trials 1 --viewer --start nominal    # 臂不动只挪方块，4/10
$PY experiments/act_rollout.py --trials 1 --viewer --start keyframe   # home 起步，3/10
```

`--viewer` 的两段式设计（不是偷懒，是 GL 限制）：一个进程里同时持有 viewer 的 GL 上下文和
相机用的离屏上下文在这台机器上不可靠，所以先无头把策略跑完（6~7 ms/拍），再另起一个只有
viewer 的上下文按 20 Hz 回放记录的 `qpos`（14 维：6 轴 + 夹爪 + 方块 freejoint）。回放是
**播记录**，不是第二次仿真，所以不会发散；看到的就是策略跑出的那条轨迹，方块和接触都在。

`--viewer-passes N` 控制循环遍数（0 = 一直循环到关窗口）。`--no-video` 可省掉 mp4 编码。

产物：`out/act_rollout/summary.json`（每 trial 的 `dx dy rise_m slip_m grasp_tick outcome`）、
`trial_XX.mp4`（512×256，front|wrist 并排）、每 25 拍一张 PNG。
`outcome` 取值：`diverged / no_motion / gripper_never_closed / slipped_out / not_lifted /
lift_insufficient / lost_after_lift / success`。


---

## 8. 8 维 EE 接口（阶段 A：数据 + 导出 + 开环评测）

§1–§7 是**关节空间**的 7 维接口（state = `qpos`，action = `act`）。这一节是并行的另一套：
观测/动作都换成**末端位姿**，为将来接真机与手部示教做准备。物理、控制频、substeps、随机化、
渲染分辨率**一行没改**——同一 `--seed 0` 下新数据的 rise/slip 与旧数据逐位相同
（rise 0.11833811413533357..0.11895673974152277 m、slip ≤ 0.0005283970146368739 m），
说明只换了「记录什么」，没换「跑什么」。

### 8.1 契约表

一组 8 维向量，单位米，在机械臂 base 系（本场景 base == world），参考点是 IK 正在打的
`tool0` site（`assets/jaka_zu3/robot.xml:142`，夹爪指尖中心），位姿由 MuJoCo site 位姿读出，
**不是**自己拼的 FK 链：

```
[x, y, z, qw, qx, qy, qz, g]        # pos3 + quat4(标量在前) + g
```

| 槽位 | 含义 | 单位/范围 |
|---|---|---|
| `[0:3]` | tool0 位置 | m |
| `[3:7]` | tool0 姿态四元数，**标量在前** (`qw` 第一)，单位长度 | 无量纲 |
| `[7]` | 夹爪开度 `g`：1 = 张开，0 = 闭合 | `[0, 1]`（实测 state 可微微越界 6.5e-6） |

⚠️ **不要再写成 `[pos3, rotvec3, g, 0]`**：那是航点表 `configs/waypoints_cube_lift.json` 的
8 维 ee8 约定（rotvec + 保留位），与本接口是两套东西，混用会让 quat 的第 2/3 个分量被当成
欧拉角喂给 IK。本接口不做 rotvec / 6D / 矩阵转换。

| 记录 | 语义 | 与仿真步的关系 |
|---|---|---|
| `pose8_state[k]` | 第 k 拍动作**执行前**的 tool0 位姿 + 当时夹爪开度（由 `qpos[k][6]` 反解） | 就是第 k 帧渲染时的状态，与该帧 PNG 严格配对 |
| `pose8_action[k]` | 第 k 拍动作**执行后**的 tool0 位姿 + **该拍指令的 g** | `[0:7]` 等于 `pose8_state[k+1][0:7]`（逐位相同）；`g` 取指令值而不是执行后的实际开度 |

最后一个 tick（`k = N-1`）的 `pose8_action` 取最后一次 `step()` 之后的采样，所以整条
episode 的 `pose8_action` 长度与 `pose8_state` 相同，`pose8_action[N-1]` 没有对应的
`pose8_state[N]`。

`data.npz` 键（旧键保留作调试）：

```
pose8_state (N,8) float64     ← 策略观测
pose8_action(N,8) float64     ← 策略动作（回归目标）
qpos (N,7) act (N,7)          ← 关节空间调试行，语义同 §1
ik_err (N,)                   ← 每拍 IK 位置残差（m），门限仍是 2.0e-3
t (N,) cube_pos(3,) cube_quat(4,) rise_m slip_m
```

`index.json` 的 `state_dim` / `action_dim` 都是 **8**。`meta.json` 键集不变。

### 8.2 四元数符号规范（必须逐帧看，否则相邻帧在网络里是“反向旋转”）

`q` 与 `-q` 是同一个姿态，但作为回归目标时是两个不同的数。规范：

1. episode 首帧取 `qw >= 0`；
2. 之后逐帧，若 `dot(q_k, q_{k-1}) < 0` 则 `q_k` 整体取负（比较对象是**已经写出去的** `q_{k-1}`）；
3. `pose8_action[k]` 的四元数**不独立做符号游走**，而是取与 `pose8_state[k]` 同符号——
   否则同一拍的两个四元数可能差一个负号，模型会学到「动作 ≈ -观测」。

实测（20 条 episode，2160 帧）：相邻帧最小点积 **0.999997873**，state/action 同拍最小点积
**0.999997873**，两者都 ≥ 0。`tests_ee/test_act_dataset.py` 逐条 episode 断言这两条。

### 8.3 g 换算

唯一配置块与执行器都在 `src/pose_cmd/replay.py`：

```python
GRIPPER_CONFIG = {"joint": "joint_7", "ctrl_per_g": 0.9, "open_at_g": 1.0}
g_to_ctrl(g) = g / open_at_g * ctrl_per_g        # g=1 -> ctrl 0.9 (张开), g=0 -> ctrl 0
g_from_q7(q) = q * open_at_g / ctrl_per_g        # joint_7 位置反解, 即 q / 0.9
```

`act[6]` 永远等于 `pose8_action[:,7]` 经 `g_to_ctrl` 的值（生成器与测试都断言这条恒等式，
所以改 `GRIPPER_CONFIG` 时不会两边漂移）。

往返精度（arm-only 模型、位置伺服 `kp=18 kv=0.8`、8 s 收敛，重力下垂是唯一残差）：

| g | joint_7 | 反解 g | 误差 |
|---|---|---|---|
| 0.00 | -0.000088 | -0.000098 | 9.8e-5 |
| 0.25 | 0.224831 | 0.249812 | 1.9e-4 |
| 0.50 | 0.449805 | 0.499783 | 2.2e-4 |
| 0.75 | 0.674789 | 0.749765 | 2.3e-4 |
| 1.00 | 0.899784 | 0.999760 | 2.4e-4 |

最大 **2.40e-4** ≪ 1e-3 判据（`tests_ee/test_act_dataset.py` 里断言）。
注意 `state8[:,7]` 是**实际**开度：夹住方块时指令 g=0 而 `joint_7` 只合到 0.222，
`state` 因此读到 0.247（物理正确）；`action8[:,7]` 才是指令的 0。

### 8.4 重训命令（四个环境变量照 §0）

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
export PY=/home/zh/miniconda3/envs/lerobot/bin/python
export HF_HOME=$PWD/data/_scratch/hf_home HF_DATASETS_CACHE=$HF_HOME/datasets

# 1) 生成 8 维示教（mujoco env！lerobot env 的 mujoco 不要用来生成）
/home/zh/miniconda3/envs/mujoco/bin/python experiments/gen_act_dataset.py --seed 0
#    -> out/act_demos_ee/ 20 条 x 108 tick

# 2) 导出
$PY scripts/export_lerobot_dataset.py --demos out/act_demos_ee --out data/lerobot/act_pick_cube_ee
#    -> num_episodes=20 num_frames=2160

# 3) 训练（CPU 200 步冒烟；正式版把 device/steps/batch/out 换成 §2.3 那组）
$PY -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/act_pick_cube_ee --dataset.root=data/lerobot/act_pick_cube_ee \
  --policy.type=act --policy.repo_id=local/act_pick_cube_ee_200 --policy.push_to_hub=false \
  --policy.device=cpu --policy.chunk_size=32 --policy.n_action_steps=8 \
  --output_dir=out/act_train_ee_200 --job_name=act_pick_cube_ee_200 \
  --batch_size=8 --num_workers=0 --steps=200 --log_freq=50 --save_freq=200 --wandb.enable=false

# 4) 开环评测（出口码 1 = FAIL 属正常，见下）
$PY scripts/eval_act_openloop.py --ckpt out/act_train_ee_200 --demos out/act_demos_ee \
  --holdout-ep last --device cpu

# 5) 契约测试（lerobot env，会真跑 MuJoCo FK 与 gripper 往返）
$PY tests_ee/test_act_dataset.py
```

实测（200 步 / batch 8 / chunk 32 / n_action 8 / cpu，约 4 分 52 秒）：

```
step:200 smpl:2K ep:15 epch:0.74 loss:3.378 grdn:96.501 lr:1.0e-05 updt_s:0.972 data_s:0.448
INFO ot_train.py:514 End of training
```

```
     dim |  model MSE  model MAE |   mean MSE |  persis MSE
       z |   0.002258   0.033495 |   0.002642 |    0.000013
      qw |   0.000007   0.001762 |   0.000011 |    0.000000
       g |   0.065408   0.147103 |   0.199308 |    0.000467
 overall |   0.008460   0.023105 |   0.025268 |    0.000060
[eval] 判定: FAIL  ...  exit 1
```

200 步必然 FAIL：模型总体 MSE 0.008460 已经赢过 mean 基线（3.0x），但输给 persistence
（0.000060）——20 Hz 的平滑航点示教里相邻动作几乎相同，persistence 在**轨迹内插值**这件事上
几乎不可战胜。这不是本接口的问题，判据本身不该为此放宽（§6 的 7 维数据也是同一结论：
0.020980 vs 0.000108）。

> 8 维比 7 维多了一个「结构事实」：overall MSE 是 8 个量纲不同的维度的等权平均（米 / 无量纲），
> 只能做**相对**比较；诊断要看逐维表。另外 `g` 一维就占了模型 MSE 的 97%（0.065408/8 ≈ 0.0082
> 的总体贡献），因为它在 close 那一段要跳变而模型只学会了均值附近。

### 8.5 兼容性：**旧 7 维 checkpoint 与新数据不能混用**

两个方向都不行。脚本层面都会**响亮地**失败（下面每条都是实测原文），所以真正的风险只有
「有人为了让它跑起来手工补维度」那一种：

- **新 checkpoint / 新脚本喂旧 7 维数据**：旧 `data.npz` 没有 `pose8_state`/`pose8_action`，
  两个脚本在读第一帧之前就退出：

  ```
  [export] 错误: 本导出器只支持 8 维 state/action，index.json 声明为 state_dim=7 action_dim=7
  [eval] 错误: ep000/data.npz 缺少 'pose8_state'（旧 7 维示教没有 EE 数组，请用 ... 重新生成）
  ```
- **旧 checkpoint 喂新数据**：LeRobot 的 `policy_preprocessor.json` 里存着归一化统计量
  （均值/方差形状 `(7,)`）、`config.json` 里 `input_features.observation.state.shape=[7]`。
  实测（拿 `out/act_train/checkpoints/last/pretrained_model` 去评 `out/act_demos_ee`）：

  ```
  RuntimeError: The size of tensor a (8) must match the size of tensor b (7) at non-singleton dimension 1
  ```

  真正的坑在于「为了让它跑起来手工补维度」：把 7 维 checkpoint 的归一化统计量补齐成 8 维，
  模型就会拿 `qw` 的统计量去解释第 4 个关节角，输出**静默错**。所以：

> 7 维 checkpoint（`out/act_train/checkpoints/{002500,005000,last}`）只配 7 维数据
> （`out/act_demos` + `data/lerobot/act_pick_cube`）；8 维必须从 §8.4 的第 2 步重新导出、
> 第 3 步重新训练。旧产物一律保留未动（sha256 校验：`out/act_demos/index.json` =
> `3117132a…`、`data/lerobot/act_pick_cube/meta/info.json` = `bb63168c…`）。

闭环 `experiments/act_rollout.py` 本阶段**没有**改，仍然是 7 维：它读的是旧 checkpoint 与
`qpos`，跑得通，但不认识 `pose8_*`。8 维闭环是阶段 B 的事。

### 8.6 一条旧结论已过期

§4 说「lerobot 侧的 mujoco 3.1.6 加载不进本仓库场景（`invalid qpos size, expected length 14`）」
——现在不成立：`assets/jaka_zu3/scene_cube_bowl.xml:78-83` 的 `<key>` 已经把 qpos 补到 14 维，
3.1.6 能正常加载（nq=14 / nsite=1），本轮 `tests_ee/test_act_dataset.py` 就是在 lerobot env
（mujoco 3.1.6）里做的真 FK 与真 gripper 往返。§4 的闭环决策是否要重开，留给阶段 B 评估。

---

## 9. 阶段 B：闭环 8 维接口

§8.5 说「闭环 `experiments/act_rollout.py` 本阶段**没有**改，仍然是 7 维」。这一节就是那笔账：
现役 `experiments/act_rollout.py` 已改成 **策略输出 8 维 EE → IK → 关节 ctrl**（384 → 706 行），
旧的 7 维实现原样搬到 `legacy/act_rollout_joint7.py`（384 → 389 行，多出的 5 行是 RETIRED 横幅）。

### 9.1 解码链（一行一行的契约）

```
a8 = [x, y, z, qw, qx, qy, qz, g]        # 与 pose8_* 同一布局，base(=world) 系，米，1=张开
  -> |q| 归一化                          # |q| < 0.5 视为坏动作：计数、复用上一拍 ctrl，不抛异常
  -> R = mju_quat2Mat(q)
  -> q6 = ik.solve(pos, R, q_seed=data.qpos[:6])；不收敛再 ik.solve_robust(...)（16 个确定性种子）
  -> 仍不收敛 / 不在关节限位内 -> 计数 + 复用上一拍 ctrl
  -> ctrl[:6] = q6 + feedforward(q6)     # use_ff=True，src/pose_cmd/replay.py:203-205
     ctrl[6]  = g_to_ctrl(clip(g, 0, 1)) # GRIPPER_CONFIG: ctrl_per_g=0.9, open_at_g=1.0
```

观测侧 `observation.state`（8 维）= `fk_tool(tool0)` 的 pos3 + quat4 + 实测开度 `qpos[6]/0.9`，
其中位姿走的是 `gen_act_dataset.fk_tool`（私有 MjData + `mj_kinematics`，不碰 plant），
**不是** `data.site_xpos`——和造数据时同一个函数，B4 才能逐位相等。
四元数符号与训练一致：首拍先翻到 `qw >= 0`，之后逐拍 `dot(q_k, q_{k-1}) < 0` 就整帧取负
（`gen_act_dataset.sign_walk` 的逐拍版；q 与 -q 是同一个姿态，但网络吃的是数字）。

两处「必须」，都是实测出来的：

- **`mju_quat2Mat` 不归一化**。把动作四元数乘 3 直接喂进去，矩阵元素偏 **8.000**；先归一化再算，
  3q 与 q 的差是 **4.441e-16**（B3 的判据是 ≤1e-12）。
- **g 是回归量**，冒烟模型实测首帧写 1.1003。不 clip 就会算出 `ctrl[6] = 0.99`，越过 `joint_7`
  的 `[0, 0.9]`（`assets/jaka_zu3/gripper_real.xml:68`）。clip 之后 ctrl[6] 严格落在 [0, 0.9]。

### 9.2 `--replay-demos`：没有 8 维 checkpoint 时唯一能做的真实验证

不吃策略、不吃 checkpoint：把 `data.npz` 的 `pose8_action` 当作"策略输出"逐拍喂进上面那条解码链，
闭环跑完整个 episode，用同一套 `classify()` 判据判成功。episode 的 cube offset 从 `meta.json` 的
`cube_pos` 反推（`dx = cube_pos[0] - CUBE_XY[0]`），所以 `--start approach` 落点就是该 episode
录制时的第一帧。

**在哪个 env**：mujoco env（`/home/zh/miniconda3/envs/mujoco/bin/python`，mujoco 3.16 —— 造数据用的
就是它，物理版本必须一致；这个 env **没有 torch**，所以 `import torch` 被推迟进 `load_policy()`，
模块顶层没有它）。

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
/home/zh/miniconda3/envs/mujoco/bin/python experiments/act_rollout.py --replay-demos out/act_demos_ee
```

20 条全跑，实测（`out/act_rollout/summary.json`）：

| 判据 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| `rise_m`（20 条） | ≥ 0.08 | **min 0.117544 / max 0.118662** | PASS（20/20 success） |
| `slip_m` | ≤ 0.005 | **max 0.000993** | PASS |
| `tool_offset_m` | ≤ 0.040 | **max 0.001699** | PASS |
| IK 位置残差 | ≤ 2.0e-3 m | **max 9.959e-04** | PASS |
| IK 角度残差 | — | max 0.499 deg | 记录 |
| IK 失败 / 回退 / 越限 | 0 | **0 / 0 / 0**（restarts 0） | 全部一次收敛 |
| 坏动作 / g clip | — | 0 / 0（示教动作本身合法） | 记录 |
| 首拍 state 偏差 | ≤ 1e-9 | **0.0（20 条全是 0）** | PASS |
| 中途 state 最大偏差 | — | 0.019647 m | 预期内，见下 |

整轮 11.6 s，IK 0.4 ms/拍。

**为什么重放不是逐位复现**：同一个位姿有多个关节解，IK 从当前 qpos 温启动会解到另一组关节角，
于是末端虽然到了同一个位姿，中间过程与录制差了最多 1.96 cm。判据因此是那几条验收数字，不是
与录制 qpos 的逐位相等——但**第一拍**是逐位相等的（同一 reset、同一 fk_tool）。

**不许作弊**：这条链里没有任何一条「把录制的 qpos 直接写进 ctrl」的旁路，动作只能从
`pose8_action` 经「归一化 → 旋转矩阵 → IK → 前馈」变成 ctrl。

### 9.3 两个环境的分工（`import torch` 只在 `load_policy()` 里）

| 要跑什么 | 用哪个 env | 为什么 |
|---|---|---|
| `--replay-demos`（离线验证解码链） | mujoco env | 物理 3.16 与造数据一致；不需要 torch |
| 策略闭环（`--ckpt ...`） | lerobot env | 需要 lerobot + torch |

三种出错姿势都实测过，都是**一句话退出**，不是 traceback：

```
$ /home/zh/miniconda3/envs/mujoco/bin/python experiments/act_rollout.py --trials 1
[roll] this interpreter has no torch, so it cannot run the policy.
       the policy path needs the lerobot env; --replay-demos <dir> exercises the
       same decode chain in the mujoco env (mujoco 3.16, the demos' version).

$ $PY experiments/act_rollout.py --ckpt out/act_train/checkpoints/last/pretrained_model --trials 1
[roll] .../out/act_train/checkpoints/last/pretrained_model has a 7-dim action head; this rollout
       drives the 8-D EE interface and needs 8 (x,y,z,qw,qx,qy,qz,g).
       train/export the 8-D checkpoint into out/act_train_ee, or run the 7-D one with
       legacy/act_rollout_joint7.py
```

`av` 只装在 lerobot env，所以 replay 在 mujoco env 里**不写 mp4**（打印一行提示），PNG 关键帧照写
（`src/pose_cmd/replay.py:120` 的 PNG 写入只用 zlib）。要看视频就在 lerobot env 里跑，或者只收 PNG。

策略路径的冒烟（用已有的 8 维冒烟 checkpoint，40 拍，只为证明管线通、不证明成功率）：

```
$ $PY experiments/act_rollout.py --ckpt out/act_train_ee/checkpoints/last/pretrained_model \
    --trials 1 --max-steps 40 --json-only --no-video
[roll] ckpt=... chunk=100 n_action=100
[trial 00] ... steps 40 ... ik 9.43e-04 m 0.49 deg  fb 0 fail 0 clip 27 ... gripper_never_closed
```

`clip 27`：40 拍里有 27 拍的 `g` 越界被 clip——就是 §8.4 那个「冒烟模型写 1.1003」的实测复现。
没有这层 clip，`ctrl[6]` 会去 0.99。

### 9.4 CLI 与产物

| 参数 | 阶段 A 之前 | 现在 |
|---|---|---|
| `--ckpt` | `out/act_train/...`（7 维） | 默认 `out/act_train_ee/checkpoints/last/pretrained_model`（8 维）；旧路径不是硬依赖，找不到就一句话退出 |
| `--trials` | 10 | 默认 `None`：策略路径 10，`--replay-demos` 是「目录里找到几条跑几条」（20） |
| `--replay-demos DIR` | — | 新增，离线验证模式，不吃策略 |
| `--device/--out/--seed/--max-steps/--start/--json-only/--no-video/--viewer/--viewer-passes` | | 语义未变 |

控制频率（20 Hz）、substeps（25）、`MAX_STEPS`、`GRASP_G`、`CRITERIA` 全部未动。
`summary.json` 在原有字段上多了 `mode / replay_demos / ee_layout / ee_site / rise_min_m / rise_max_m /
slip_max_m / tool_offset_max_m / ik_pos_err_max_m / ik_ang_err_max_deg / ik_{fallback,fail,limit,restart}
ticks / bad_action_ticks / g_clip_ticks / state_dev0_max / state_dev_max / ik_ms_mean`，
每 trial 多了 `ik_pos_max_m / ik_ang_max_deg / ik_fallback / ik_fail / ik_limit / g_clip /
bad_actions / state_dev0 / state_dev_max / state8_0 / mode`。

### 9.5 测试

`tests_ee/test_act_rollout.py`（200 → 458 行，245 checks）在原有「随机化 / runaway / 判据分支」
之上加了 8 维那半边：解码（3q 归一化 → 同一 R；坏动作被标记而不是被放大成旋转；g 两侧 clip）、
ctrl 行（前馈 + clip 后的 `ctrl[6]` 落在 `joint_7` 的 `[0, 0.9]`）、观测（首拍翻号、dot<0 逐拍取负、
开度 = `hinge_q(qpos[6])`）、replay 管路（episode 按名排序、offset 从 `meta.json` 反推）、
以及两条源码契约（模块顶层无 `import torch`、legacy 副本带 RETIRED 横幅）。

负向变异都实测会红（变异文件在 /tmp，仓库内不动）：

| 变异 | 结果 |
|---|---|
| `decode_pose8` 去掉 `np.clip(g_raw, 0, 1)` | `EXIT=1`，`AssertionError: decode: g=1.1 was not clipped to 1.0` |
| `decode_pose8` 去掉 `quat / norm` | `EXIT=1`，`AssertionError: ... changed the rotation by 8.000000000000002` |

### 9.6 旧 7 维通路没被弄坏

`legacy/act_rollout_joint7.py` = HEAD 的 `experiments/act_rollout.py` 逐字节 + 5 行横幅。
lerobot env 实跑：

```
$ $PY legacy/act_rollout_joint7.py --ckpt out/act_train/checkpoints/last/pretrained_model \
    --trials 1 --max-steps 40 --json-only --no-video --out out/act_rollout_joint7
[trial 00] ... steps 40 ... grasp None ... gripper_never_closed      EXIT=0
```

40 拍只够走到接近段（这一拍的 grasp 还没发生），所以判定是 `gripper_never_closed`；
跑满 200 拍同一条 trial 是 `success`（rise +0.1250，slip 0.0083，tool 0.0028，EXIT=0），
这就是用户那份 10/10 的来源，通路未变。

> 注意：这两条命令都额外传了 `--out out/act_rollout_joint7`。不传的话它按 7 维时代的默认写
> `out/act_rollout/summary.json`，会**覆盖**阶段 B 的 8 维产物。

### 9.7 未验证 / 已知边界

- **闭环成功率已验（2026-11-01）**：`out/act_train_ee` 的 2000 步 checkpoint 跑到 **50/50 成功**
  （5 seed × 10 次，slip 最大 5.25 mm，含 8 维 vs 7 维的逐次配对对比）——见 `docs/VERIFICATION.md` §G。
  §9.2 验的仍然只是解码链，不是策略。
- **`success` 里没有 slip**：`classify()` 只查 `rise_m` 与 `tool_offset_m`（`experiments/act_rollout.py:140`），
  抓取质量必须另看 summary 的 `slip_max_m`，否则成功率是虚的（见 §G1）。
- **`joint_7` 是软限位**：`ctrl[6]` 严格在 `[0, 0.9]` 内（实测 0.900000000000000），但 MuJoCo 的
  关节限位是软约束，实测关节会停在 `0.9` 上方 1.04e-06（录制数据自己也是 8.24e-06）。判定用
  仓库既有的 `JNT_TOL_RAD = 1e-3`（`tests_ee/test_act_rollout.py` 里对 7.48e-06 也有断言）。
- **旧 10/10 已整体复跑（10 次，2026-11-01）**：`10/10` 成立，但按 5 mm slip 线只有 **4/10**；
  逐次数字在 `docs/VERIFICATION.md` §G1。
- **mp4 在 mujoco env 里没有**（无 `av`），replay 只落 PNG；`--viewer` 的窗口在本轮没有被打开
  （用户桌面上不许弹窗），只有 `--help` 与参数解析被验证。

---

## 10. 阶段 C：放置任务（夹起 → 放进碗里）

第二条数据/评测链，和 §8–§9 的 lift 任务**并行**：同一套 8 维 EE 接口、同一个场景、同一套
20 Hz / 25 substeps / 随机化，只是示教表换了一张、判据换了一套。

### 10.1 表从哪来：`configs/waypoints_cube_place.json`

8 个航点，全部 `rotvec=[0,π,0]`，`meta.control_hz=50`（和 lift 表一样是 50 Hz 写表，生成器按
`CONTROL_HZ=20` 重采样），@20 Hz 共 **181 拍**：

| # | label | xyz (m) | duration_s | g |
|---|---|---|---|---|
| 0 | approach | 0.474, 0.0435, 0.910 | 1.2 | 1 |
| 1 | descend | 0.474, 0.0435, 0.815 | 1.0 | 1 |
| 2 | close | 0.474, 0.0435, 0.815 | 0.6 | 0 |
| 3 | lift | 0.474, 0.0435, 0.935 | 1.0 | 0 |
| 4 | carry | 0.300, −0.200, 0.945 | 1.4 | 0 |
| 5 | lower | 0.300, −0.200, 0.870 | 0.8 | 0 |
| 6 | release | 0.300, −0.200, 0.870 | 0.6 | 1 |
| 7 | retreat | 0.300, −0.200, 0.935 | 0.9 | 1 |

碗的几何（`assets/jaka_zu3/scene_cube_bowl.xml`）：中心 `(0.300, −0.200)`、内半径 0.045、
壁顶 `z=0.840`、底 `z=0.802`；方块静止时中心 `z=0.817`（那 0.870 的释放高度让方块从 ~5 cm
自由落体进碗，不是贴上去放的）。

### 10.2 与 lift 的三处差别（都是必须的，不是风格问题）

1. **判据**：place 不看 `rise_m`。放置任务本来就不抬升——脚本化数据实测 `rise ≈ +0.0020 m`，
   `CRITERIA["min_rise_m"]=0.08` 永远不可能满足。place 的成功是「末帧方块在碗里 **且** 末拍
   指令 `g ≥ 0.9`（已松手）」。
2. **随机化锚点**：见 10.3。lift 表整表跟着方块偏移；place 表只让 `approach/descend/close/lift`
   跟着走。
3. **三个新指标**：`bowl_dxy_m` / `in_bowl` / `hold_err_m`，写进 `data.npz`、`meta.json`、
   `index.json`、`summary.json`（缺一不可：导出与读回都靠它们自证）。

> **`slip_m` 在这里换了含义，别拿 5 mm 线卡它**：`slip_m` 的定义是「夹爪闭合之后方块的水平位移」，
> 而 place 把方块搬了 ~0.4 m，所以实测 `slip_m ≈ 0.28–0.32 m`。旧定义一行没改（lift 的
> `slip ≤ 5.3e-4 m` 逐位不变），但 place 的 `slip_m` 是**搬运距离**，不是抓取质量指标。
> 抓取质量在 place 里由 `hold_err_m` 承担（松手前最后一拍方块离 tool0 多远）。

> **「已松手」怎么判：物理脱离，不看指令值**（2026-11-01 修正）。原判据要求末拍指令
> `g ≥ 0.9`，但真实策略的 g 头系统性偏低：step 6000 权重在 80 次物理成功的试次里末拍
> `g` 只有 **0.62–0.80**，于是 20/20、80/80 全被判成 `never_released`。现在的证据是
> `tool_offset_m`（末帧方块中心到 tool0 的距离）：夹住时 0.013–0.025 m，松手退开后
> 0.11–0.32 m，阈值 `RELEASE_SEP_M = 0.05 m`。`cmd_g_last` 仍写进 summary 当测量值。

### 10.3 为什么要「锚定」（`ANCHORED_LABELS`）

方块 xy 每个 episode 随机偏移 `dx ∈ ±0.020 m`、`dy ∈ ±0.005 m`；**碗在世界坐标里是固定的**
（它是场景资产，不在随机化范围内）。如果整张表跟着偏移走，`carry/lower/release/retreat` 会
整体漂 ±20 mm，于是方块落在碗沿上或掉在碗外——offset 越大越离谱，而成功率统计会把这种
「表设计错误」伪装成「策略泛化不好」。

所以 `experiments/gen_act_dataset.py` 里有一条常量：

```python
ANCHORED_LABELS = ("carry", "lower", "release", "retreat")
```

`shifted_table(base, dx, dy)` 只给**不在**这个元组里的航点加 `(dx, dy)`。lift 表里一个都没有，
所以 lift 数据的每一个字节都没变（§10.8 的回归里有逐位对比）。

### 10.4 生成（mujoco env，20 条约 30 s）

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
/home/zh/miniconda3/envs/mujoco/bin/python experiments/gen_act_dataset.py \
  --table configs/waypoints_cube_place.json \
  --out out/act_demos_place
```

- `--table` 不给时仍然是 `configs/waypoints_cube_lift.json`，`--out` 不给时仍然是
  `out/act_demos_ee`——**默认行为逐位未变**；`--out` 不给但 `--table` 是 place 表时，产物落
  `out/act_demos_place`。
- 任务是**从表的 label 推出来的**（`table_task(base)`：有 `ANCHORED_LABELS` 里的 label 就是
  place），不额外加 `--task` 开关，免得命令行和表打架。
- 收条门槛（place）：`in_bowl` 为真、`bowl_dxy_m ≤ 5 mm`、`hold_err_m ≤ 2 mm`、`ik_err ≤ 2e-3 m`；
  lift 的 `rise ≥ 0.08 m` 那条门槛对 place 不适用，换成了上面这组。
- 产物：`ep000..ep019/{data.npz,meta.json,front/,wrist/}`、`index.json`、`summary.json`。
  `data.npz` 在原来 10 个键之上多 `bowl_dxy_m`(float64 标量)、`in_bowl`(bool 标量)、
  `hold_err_m`(float64 标量)。

实测（本轮）：`20/20 episodes in 29.7 s, 20 attempts`，`in_bowl 20/20`，
`bowl_dxy 0.00207..0.00466 m`，`hold_err ≤ 0.00119 m`，`ik residual ≤ 9.97e-04 m`，
每条都是 181 拍。

### 10.5 导出（lerobot env）

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
export HF_HOME=$PWD/data/_scratch/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
/home/zh/miniconda3/envs/lerobot/bin/python scripts/export_lerobot_dataset.py \
  --demos out/act_demos_place \
  --out data/lerobot/act_pick_place \
  --repo-id act_pick_place
```

导出的 task 字符串取 `index.json` 的 `task`，place 数据是
`pick up the red block and put it in the bowl`（`meta/tasks.parquet` 里就这一条）。
读回实测：`num_episodes=20 num_frames=3620`（181×20）、`action (8,)`、`observation.state (8,)`、
两路相机 `observation.images.front/wrist`。

### 10.6 训练（用户自己的 GPU 会话，本轮**没有**跑）

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
export HF_HOME=$PWD/data/_scratch/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
/home/zh/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/act_pick_place \
  --dataset.root=data/lerobot/act_pick_place \
  --policy.type=act \
  --policy.repo_id=local/act_pick_place \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=out/act_train_place \
  --job_name=act_pick_place \
  --batch_size=8 \
  --num_workers=8 \
  --steps=2000 \
  --log_freq=50 \
  --save_freq=500 \
  --seed=1000 \
  --wandb.enable=false
```

参数与 §2.3 完全一致（`--policy.repo_id`、`--policy.push_to_hub=false` 两个坑照踩），只换了
数据集、输出目录与 job 名：**`out/act_train_place` 是 `act_rollout.py --task place` 的默认
checkpoint 路径**，训完不用再传 `--ckpt`。181 拍 < ACT 默认 `chunk_size=100`… 反过来：episode
比 chunk 长，不存在 pad 问题。

### 10.7 回放 / 看过程

**A. 没有 checkpoint 时（脚本示教回放，验证数据本身）**

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
/home/zh/miniconda3/envs/mujoco/bin/python experiments/act_rollout.py \
  --replay-demos out/act_demos_place --task place --json-only --no-video \
  --out scratch/rollout_place_replay
```

实测：`20/20 success`，`bowl_dxy 0.004606..0.004751 m`、`hold_err ≤ 0.00122 m`、
`ik ≤ 9.99e-04 m`，1.9 s 跑完。

**B. 训练好之后，开 MuJoCo 窗口看过程**

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
export HF_HOME=$PWD/data/_scratch/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export DISPLAY=:1
/home/zh/miniconda3/envs/lerobot/bin/python experiments/act_rollout.py \
  --task place --trials 1 --viewer
```

`--task place` 会把默认 checkpoint 指到 `out/act_train_place/checkpoints/last/pretrained_model`、
默认产物目录指到 `out/act_rollout_place/`，并按 place 判据打标签；`--viewer` 是 §7 那个两段式
（先无头跑完，再另起窗口 20 Hz 回放 14 维 `qpos`）。**注意两条命令用的 python 不同**：回放
`--replay-demos` 不需要 torch，用 mujoco env；吃 checkpoint 的那条必须用 lerobot env。

`--task place` 的 `outcome` 值：`diverged / no_motion / gripper_never_closed /
never_released / not_in_bowl / success`。`summary.json` 里除原有字段外多了
`task_kind / table / anchored_labels / bowl_xy_m / bowl_dxy_min_m / bowl_dxy_max_m /
in_bowl_count / in_bowl_all / hold_err_max_m`，`criteria` 换成 place 的那一套
（`release_sep_m / in_bowl_radius_m / in_bowl_z / max_ik_err_m / grasp_g_below / ctrl_limit`，
**没有** `min_rise_m`），每 trial 多 `bowl_dxy_m / in_bowl / released / hold_err_m / cmd_g_last`。

`bowl_xy_m` 和 `layout_m` 里的 `cube_xy0 / bowl_xy0` 描述的是**这批 trial 实际用的**
布局：所有 trial 站在同一个 bowl 时就是那一对坐标；若某些 trial 的 bowl 不同（`--replay-demos`
混了不同布局的数据集就会这样），`bowl_xy_m` 换成 `{"min", "max", "distinct"}`、
`cube_xy0 / bowl_xy0` 给 `null`——逐 trial 的真值只认 `per_trial[i].cube_xy / .bowl_xy`。

不带 `--task` 时（`lift`）产物与字段和 §9.4 逐位一致。

### 10.8 测试与本轮负向变异

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
/home/zh/miniconda3/envs/mujoco/bin/python tests_ee/test_act_dataset.py                                  # lift, 14378 checks
/home/zh/miniconda3/envs/mujoco/bin/python tests_ee/test_act_dataset.py --dir out/act_demos_place         # place, 23486 checks
/home/zh/miniconda3/envs/mujoco/bin/python tests_ee/test_act_rollout.py \
  --place-summary scratch/rollout_place_replay/summary.json                                              # 546 checks
```

`test_act_dataset.py` **不 import 生成器**（它只验盘上的产物），所以它按 `index.json` 里的
`task_kind` 切契约：place 目录多要求 3 个键、`num_frames == 181`、`in_bowl` 为真、`bowl_dxy ≤ 5 mm`、
`hold_err ≤ 2 mm`、末拍 `g ≥ 0.9`、`g` 曾到 0。lift 目录的要求一条没少。

负向变异（实测，变异完已按 md5 还原）：

| 变异 | 结果 |
|---|---|
| `ANCHORED_LABELS = ()` | `EXIT=1`，`AssertionError: shifted_table: 'carry' moved by (+0.0170, +0.0040), expected (+0.0000, +0.0000) - the bowl does not follow the cube offset` |
| place 判据认 `rise`（造 rise=0.12 但 `in_bowl=False`） | 断言要求 `not_in_bowl`，同时要求同一份 rec 在 lift 判据下是 `success`——两条互相咬死，`rise` 一旦混进 place 判据必红 |

`--viewer` 在本轮**没有真的开过窗口**（用户桌面上不许弹窗）：`check_viewer_stub()` 把
`play_in_viewer` 换成记录参数的 stub，验证「不带 `--viewer` 时不被调用」+「带时收到
`(scene, traces, control_hz, passes)`、1 条 trace、形状 `(181, 14)`、`passes==3`」。

## 11. 换桌子：cube / bowl 的坐标住在哪儿，怎么改

前面每一节都默认 cube 在 `(0.474, 0.0435)`、碗在 `(0.300, -0.200)`。这一节说的是**换位置**：
坐标一共住在五个地方，必须一起改，漏一处就会变成「场景按新坐标、表格按旧坐标」，IK 不报错，
生成出来的数据全是错的。

### 11.1 五个地方，一个都不能漏

| # | 位置 | 内容 |
|---|---|---|
| 1 | `assets/jaka_zu3/scene_cube_bowl.xml:36` | `<body name="cube" pos="0.474 0.0435 0.815">`：cube 的初始位姿 |
| 2 | 同文件 `:82` | 关键帧 `home` 的 qpos 尾巴：7 个手臂关节 + cube 自由关节的 `x y z qw qx qy qz` |
| 3 | 同文件 `:47-64` | 碗的 17 个 geom：`bowl_bottom`（圆柱）+ `bowl_wall_00..15`（盒子，中心线半径 0.049，euler 绕 z 每 22.5°） |
| 4 | `experiments/gen_act_dataset.py:118-119,139` | `CUBE_XY`、`CUBE_Z`、`BOWL_XY`，生成器和 `act_rollout.py` 都从这里取 |
| 5 | `configs/waypoints_cube_lift.json`、`configs/waypoints_cube_place.json` | 抓取列（`approach/descend/close/lift/hold`）在 cube xy 上；place 表里 `carry/lower/release/retreat` 在碗 xy 上 |

三件事容易踩：

* 第 2 处是第 1 处的副本。只改 body 不改关键帧，`Replayer.home_qpos()` 会把 cube 拽回旧位置。
* 第 3 处写的是**世界坐标**，不是 body 局部坐标（`<body name="bowl" pos="0 0 0">` 是空壳），
  所以「搬碗」= 把这 17 个 `pos` 用同一个 delta 平移，`euler` 和 `size` 一位不动。
* 五处由 `experiments/set_layout.py` 一次改完（§11.3），别手改。

### 11.2 可达区（实测，只有运动学）

判据：抓取姿态 `rotvec = [0, π, 0]`，site = `tool0`，IK 收敛（`solve` 不行再 `solve_robust`）
+ 六个关节都在限位内 + 位置残差 ≤ 2e-3 m + 角度残差 ≤ 2°。在 0.815 / 0.870 / 0.945 三个高度上
扫出来的是**同一个集合**（绑定的是最高的 0.945）：

| x | y 的可行范围 |
|---|---|
| 0.10 – 0.38 | ±0.36（探针边界，可能更宽） |
| 0.40 | ±0.34 |
| 0.42 | ±0.32 |
| 0.44 | ±0.30 |
| 0.46 | ±0.26 |
| 0.48 | ±0.22 |
| 0.50 | −0.18 … +0.16 |
| 0.52 | ±0.10 |
| ≥ 0.54 | 不可达 |

台面 x −0.25…0.95、y −0.46…0.54、顶面 z = 0.80。现在的 cube `(0.474, 0.0435)` 和碗
`(0.300, -0.200)` 都在里面。

**这张表只有运动学，而且只是必要条件**：没算自碰撞（手臂撞台面、撞自己）、没算动力学，也
不保证夹得住——0.50 附近 IK 全过，但夹爪会把方块推着走（见 11.6 的实测）。所以默认采样器
**不按这张表取点**，用的是 §11.4 那两个按"真跑一集"量出来的安全盒；这张表仍然是
`set_layout.py` 挡明显不可能的布局、以及 `--grasp-grid` 圈范围时的第一道闸。

### 11.3 `experiments/set_layout.py`

八个开关：`--cube X Y`（必填，除非要的是 `--grasp-grid` / `--sample` / `--sample-preview`）、
`--bowl X Y`（默认取场景里 `bowl_bottom` 的当前值）、`--cube-z Z`（默认取场景当前值，即 0.815）、
`--apply`、只测量不写盘的 `--grasp-grid SPEC`（见 11.6），以及给采样器看的 `--sample N`、
`--sample-preview`、`--seed`（见 11.7）。

默认**只看不写**：对 cube 列（0.815 / 0.910 / 0.935）和碗列（0.870 / 0.935 / 0.945）逐高度打
一行 IK 判词（`conv/MISS`、位置残差、角度残差、关节是否在限位、rotvec），再打 cube-bowl 距离
vs 碗外径 0.049 m、两个物体是否都还在台面上、五处是否自洽，最后一行 `PASS` / `FAIL`。
**退出码就是判定**（0 = 过，非 0 = 不过），所以能直接塞进脚本。

`--apply` 先跑同一套检查，`FAIL` 直接拒绝、一个字节都不写；过了才重写五处，重写完从磁盘重读、
再检查一遍（`re-check on disk: PASS`），最后打印 `python -m mujoco.viewer --mjcf=...` 给你看。
工具自己**从不调用 `mujoco.viewer`、不开任何窗口**——例外只有你显式点名的那条
`--sample-preview`（见 11.7）；检查模式、`--grasp-grid`、`--sample` 依旧一个窗口都不开。

改写是**按 token 换数字**：注释、缩进、JSON 键顺序、行内数组 vs 每行一个数的展开写法全部原样
保留，数值没变的位置连 token 都不动——所以拿当前坐标 `--apply` 是零 diff（幂等），第二次
`--apply` 会直接写 `wrote 0 of the 5 places`。碗的 17 个 geom 用同一个 delta 平移、仍写 6 位小数
（源文件就是 6 位，环半径本身因此有 ≤ 3.3e-7 的舍入残差，平移一位不差地保留它）。

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro
# 1) 只问「能不能放」，什么都不写；碗沿用场景里的当前值
/home/zh/miniconda3/envs/mujoco/bin/python experiments/set_layout.py --cube 0.500 0.100

# 2) cube 和碗一起挪，检查通过才落盘，落盘后再自检一遍
/home/zh/miniconda3/envs/mujoco/bin/python experiments/set_layout.py \
  --cube 0.500 0.100 --bowl 0.320 -0.280 --apply

# 3) 搬回默认布局；搬回去之后再跑同样的命令就是 0 个文件被写
/home/zh/miniconda3/envs/mujoco/bin/python experiments/set_layout.py \
  --cube 0.474 0.0435 --bowl 0.300 -0.200 --apply
```

### 11.4 `sample_layout`：训练和评测必须同一个分布

生成器和 `act_rollout.py` 的随机化只能从一个函数出：

```python
sample_layout(rng) -> (cube_xy, bowl_xy)      # 两个绝对坐标，单位 m
```

默认实现是**连续均匀 + 一次拒绝**：cube 在 `CUBE_BOX` 里、碗在 `BOWL_BOX` 里各按轴独立
`rng.uniform`（先 x 后 y），两个中心的距离小于 `MIN_SEP_M` 就把**两个一起**重抽（同一对 rng
流，同种子同结果；`SAMPLE_MAX_TRIES = 1000` 次还抽不到就抛错）。三个常量：

| 常量 | 值（m） | 为什么是这个框 |
|---|---|---|
| `CUBE_BOX` | x 0.455…0.485，y −0.090…0.130 | 抓取图上最大的全通过矩形是 x 0.450…0.490、y −0.100…0.140，四边再各收 5 mm / 10 mm。**x 停在 0.485**：再往右夹爪会推着方块走而不是夹住（IK 到 0.52 都收敛，见 11.6）；框内实测 `hold_err_m = 0.0006…0.0016 m` |
| `BOWL_BOX` | x 0.310…0.415，y −0.235…0.295 | IK 图（0.870 / 0.935 / 0.945 三个高度）上最大的全通过矩形是 x 0.200…0.420、y −0.320…0.320，再各收 5 mm / 10 mm；**y 的下限另被前视相机裁掉**（y = −0.320 的碗角投到 v = 258 px，已经出了 256 px 画面）；**x 从 0.310 起**：碗越往小 x 放，release 段的沉降误差越大 |
| `MIN_SEP_M` | 0.15 | 碗外径 0.049 m，0.15 让抓取列离碗壁还有 10 cm 量级 |

这两条"往回缩"都不是洁癖，是量出来的：第一版连续采样用碗框 x 0.220…0.420 / y −0.240…0.300，
12 次抽样只接受 6 次——6 个失败全部是**抓得好、放得歪**（`hold_err_m = 0.0004…0.0016 m`，
但 `bowl_dxy_m = 0.0052…0.1101 m`），而且 6 个失败无一例外 `bowl x < 0.30`，成功的都在
`x >= 0.30`。收到 x 0.300 之后 16 次抽样 16/16 通过，按上面这组框再抽 24 个布局真跑也是 24/24
（其中故意包含 cube 框四个角、碗框四个角、12 个内部连续抽样、4 个正好压在
`0.15 ± 0.004 m` 上的配对），四个角的投影也都还在前视画面里、至少留 12 px。

"连续"是重点：**任何**落在框里的实数对都合法——没有网格、没有四舍五入、没有固定点表。自己写
2-D 采样器时可以直接引用这三个常量，别退化成格子。

* `experiments/act_rollout.py` 里的 `sample_offsets(rng)` 只是 `sample_layout(rng)` 的包装
  （返回 `layout_offsets(cube_xy)`），**不许自己再写一份采样**，否则训练和评测会慢慢漂开。
* 配套两个函数：`layout_table(base, cube_xy, bowl_xy)` 把抓取列搬到 `cube_xy`、把
  `ANCHORED_LABELS`（`carry/lower/release/retreat`）搬到 `bowl_xy`，z 和时长一格不动；
  `layout_table(base, CUBE_XY, BOWL_XY)` 与 `shifted_table(base, 0.0, 0.0)` 逐字段相等。
  `place_bowl(rp, bowl_xy)` 平移碗的 17 个 `model.geom_pos` 再 `mj_forward`，中心没变就是空操作。
* `meta.json` / `index.json` / `summary.json` 多记了 `cube_xy`、`bowl_xy`（旧字段 `dx`、`dy`、
  `cube_pos` 含义不变）；`index.json` / `summary.json` 另外记了 `cube_box_m`、`bowl_box_m`、
  `min_sep_m`。注意 `index.json` 里的 `cube_xy` / `bowl_xy` / `bowl_xy_m` 是**场景里写死的中心**
  （XML 里那一份），不是某一集抽出来的位置——每集的真值在各 `ep%03d/meta.json` 里。
  `--replay-demos` 用 meta 的 `cube_xy` / `bowl_xy` 还原布局；老 `meta.json` 没有这两个键时回落到
  `cube_pos` 和 `BOWL_XY`，所以冻结的数据集照样能回放（它们的分布当然还是旧的 ±0.02）。
* `RANDOM_X_M` / `RANDOM_Y_M`（±0.020 / ±0.005）**已经不是默认采样器的框**，只有 `legacy/`
  还 import 它们写自己的 `randomization_m` 块。现行 `act_rollout.py` 的 summary 写的是
  `layout_m: {cube_box_m, bowl_box_m, min_sep_m, cube_xy0, bowl_xy0}`，直接来自
  `gen_act_dataset.CUBE_BOX / BOWL_BOX / MIN_SEP_M`。
* 想换成自己的 2-D 采样器：**只改 `sample_layout` 这一处**，生成和回放同时变，别处不用动。

### 11.5 本节新增的测试

`tests_ee/test_act_dataset.py` 每跑一次多做 61 条检查（14317 → 14378；place 目录 23425 → 23486）：

| 检查 | 内容 |
|---|---|
| `layout_table` 等价 | 两张表上 `layout_table(base, CUBE_XY, BOWL_XY)` 与 `shifted_table(base, 0, 0)` 逐字段（含 ee8 精确相等）一致 |
| anchored 跟随 | `ANCHORED_LABELS` 的行落在 `bowl_xy`、其余落在 `cube_xy`，z / g / 四元数 / 时长不动 |
| `place_bowl` | 17 个 geom 的 `model.geom_pos` 精确等于 `pos + delta`，环半径前后差 ≤ 1e-12，中心不变时是空操作 |
| `sample_layout` | 2000 次抽样：每个 `cube_xy` 落在 `CUBE_BOX`、每个 `bowl_xy` 落在 `BOWL_BOX`、最小间距 ≥ `MIN_SEP_M`、cube 位置按 1e-6 取整后去重 ≥ 1990 个；同种子两次抽到的第一对相同 |
| `--sample` | 子进程验证：`--sample 3` 退出 0 且正好打印 3 对布局（什么都不写） |
| `set_layout.py` | 子进程验证：当前布局 `PASS` 退出 0；`--cube 0.60 0.00` 退出码非 0 且点名失败高度（0.815） |
| 抓取网格 | 子进程验证：单点 `--grasp-grid` 退出 0 且打印 `1/1` 与单点耗时；`--grasp-grid` + `--apply` 被拒绝（退出码 2） |

### 11.6 抓取质量网格：IK 通了不等于抓得住

11.2 那张表只回答"IK 能不能到"，不回答"抓住之后会不会被甩掉"。实测：在默认立方体周围 42 个
IK 全部通过（残差 < 1e-3 m）的点上各跑一条真实 place 轨迹，其中 **8 个把立方体推出了约
28 mm**——要么 `hold_err_m` 超过 `MAX_HOLD_ERR_M = 0.002`，要么落点离碗心超过
`bowl_dxy_m = MAX_BOWL_DXY_M = 0.005`。所以写 2-D 采样器时**要建在这张网格上，不是 11.2 的
IK 表上**。

`--grasp-grid` 就是同一判据的离线版：

```bash
python experiments/set_layout.py --grasp-grid "0.42:0.52:0.02,-0.08:0.16:0.04" --bowl 0.320 -0.280
```

* 每个格点跑一条真实轨迹（`gen_act_dataset.run_attempt(base, cube_xy=格点, bowl_xy=…,
  render=False)`，不建渲染器，永远不开窗口），判定条件和生成器接受一集时**完全一样**：
  `in_bowl` 且 `bowl_dxy_m <= MAX_BOWL_DXY_M` 且 `hold_err_m <= MAX_HOLD_ERR_M`。这几个常量
  直接从 `gen_act_dataset` 取，没有第二份数字。
* 输出：每个点的 `o` / `x`、`bowl_dxy_m`、`hold_err_m`、单点耗时；一行一个 y 的 `o/x` 地图
  （y 从大到小）；`ok/total`；失败点清单。上面那 42 个点里有 8 个 `x`，例如
  `(0.500, 0.100)` 是 `x`（`hold_err_m = 0.0286 m`，立方体被推走）、`(0.420, 0.120)` 是 `x`
  （`bowl_dxy_m = 0.0055 m`，落在碗外），而 `(0.499, 0.100)`、`(0.460, 0.040)` 是 `o`。
* 代价：每个点是一条完整轨迹，预算按 ~1.5 s/点算（本机实测 0.3–0.5 s/点，1.5 是留余量的
  估计），因此**上限 400 点**，超了直接拒绝并打印点数和预估时间；400 点约 10 分钟。
* 只测量、不落盘：每个点用 `tempfile.mkdtemp()` 的临时目录，跑完删掉；`out/`、`data/` 一律
  不碰，`assets/`、`configs/` 也不碰（网格模式不写文件）。`--apply` 与 `--grasp-grid` 同时给
  会直接拒绝（退出码 2）。
* 退出码：网格跑完就是 0，哪怕有 `x`——地图本身就是答案，`x` 是测量结果而不是工具失败。
  原来的检查模式（PASS/FAIL）和 `--apply` 的退出码不变。

顺序建议：先用 `--grasp-grid` 圈出"够得到又抓得住"的区域，再在这个区域里写你自己的
`sample_layout`（见 11.4）。

### 11.7 在 MuJoCo 里看随机布局

采样器长什么样，最快的办法是让它抽几个给你看：

```bash
cd /home/zh/Desktop/EasyMimic/easymimic_repro

# 1) 抽 5 个布局只打印：每对的 cube / 碗坐标、实际抽到的范围、最小间距
/home/zh/miniconda3/envs/mujoco/bin/python experiments/set_layout.py --sample 5

# 2) 换种子再抽 20 个
/home/zh/miniconda3/envs/mujoco/bin/python experiments/set_layout.py --sample 20 --seed 3

# 3) 抽 1 个布局并直接开窗口：cube 和碗已经被摆到抽到的位置上
/home/zh/miniconda3/envs/mujoco/bin/python experiments/set_layout.py --sample-preview
```

`--sample N` 的每一对都来自生成器那份 `sample_layout`（不是另抄一份），打完最后一行会给实际
抽到的 cube x/y、碗 x/y 范围和最小间距；**什么都不写**，纯打印。种子相同、N 相同，输出就相同。

`--sample-preview` 抽一个布局，先把坐标打出来，然后在内存里把它摆上去（`place_bowl()` 平移碗的
17 个 geom，cube 用生成器同一个 `place_cube_abs()` 落到 0.815 m），再开
`mujoco.viewer.launch_passive`。你会看到台面上一个方块、一个碗，位置就是打印出来的那一对；
鼠标拖动转视角，关窗口就退出。它是**静态快照**：物理不步进、不生成 episode、不写任何文件。

这是 `set_layout.py` 里唯一会开窗口的路径，只在显式给了 `--sample-preview` 时才跑；`--apply` 和
`--grasp-grid` 都不能跟它一起用（会被直接拒绝，退出码 2），测试里也没有任何一条会调用它。
（仓库别处本来就有窗口路径——`experiments/act_rollout.py --viewer`、
`experiments/hand_teleop_live.py`、`legacy/act_rollout_joint7.py`——那些是策略回放和示教用的，
跟布局工具无关。）

看的时候请记住这些框**没有**保证什么：

* **不看自碰撞**：手臂会不会撞台面、撞碗、撞自己，这里完全没有建模。框里的点都是"IK 可达
  且真跑一集通过"的经验结果，不是"不可能撞"的保证。
* **只在 cube z = 0.815 上验证过**（台面 0.80 + 半个方块 0.015）。换 `--cube-z` 之后要重新量。
* **前视相机**只保证了 12 px 的边缘余量；相机改一点、或者碗压着 `BOWL_BOX` 的 y 下限，画面里
  就可能只剩半个碗。
* 框是**真跑一集**量出来的，量的是"这一套 waypoint 表 + 这一组速度"。改了 waypoint 表（时长、
  途经点、`rotvec`）之后，要先用 `--grasp-grid` 重新圈区域再定框。


## 12. 随机布局：生成 → 训练 → 评测（现行主线）

§11 的采样器落地之后，方块与碗每集都在 `CUBE_BOX`/`BOWL_BOX` 里连续均匀重抽，间距 ≥
`MIN_SEP_M`。这一节是与之配套的训练与评测口径。

### 12.1 为什么固定布局训出来的权重不能直接用

`out/act_train_place_20k/checkpoints/006000`（固定布局时代的最佳检查点，固定布局 80/80）
在新随机分布上跑 20 次：**0/20**，全部 `not_in_bowl`，`bowl_dxy` **0.1569–0.4738 m**，
`hold_err` 最大 0.3703 m，`released` 20/20。也就是说它每次都把方块搬到**旧的**碗位置
(0.300, −0.200)，方块照样被夹住、松开，只是放错了地方。固定布局的表现不能外推到随机布局，
必须用随机布局数据重训。

### 12.2 数据

| 项 | 值 |
|---|---|
| 目录 | `out/act_demos_place_rand`（60 条 × 181 帧） |
| lerobot | `data/lerobot/act_pick_place_rand`（`num_episodes 60`、`num_frames 10860`、20 fps） |
| 接口 | `action (8,)` / `observation.state (8,)`，两路 256×256 相机 |
| 生成验收 | 60/60 首次尝试即接受，`in_bowl 60/60`、`bowl_dxy 0.00193–0.00488`、`hold_err ≤0.00098`、IK ≤9.99e-04 |

```bash
experiments/gen_act_dataset.py --table configs/waypoints_cube_place.json \
    --out out/act_demos_place_rand --episodes 60
scripts/export_lerobot_dataset.py --demos out/act_demos_place_rand \
    --out data/lerobot/act_pick_place_rand --repo-id local/act_pick_place_rand
```

### 12.3 训练（一条命令）

`scripts/train_act.py` 把路径与 20K 配方固定在脚本里（`dataset.root` 绝对路径、`HF_HOME`
指向仓库内缓存、`push_to_hub=false`），命令行只留会调的旋钮：

```bash
# 全量 20K（GPU）
/home/zh/miniconda3/envs/lerobot/bin/python scripts/train_act.py --device cuda

# 只打印将要执行的 lerobot 命令
/home/zh/miniconda3/envs/lerobot/bin/python scripts/train_act.py --dry-run
```

默认值：`--steps 20000 --batch-size 8 --num-workers 8 --save-freq 2000 --seed 1000
--out-dir out/act_train_place_rand`；学习率用 lerobot 默认 1e-5（CLI 里只有
`--policy.optimizer_lr`，没有 lr_scheduler）。

两条硬约束：

* **输出目录不覆盖**。lerobot 在 `resume=False` 下遇到非空 `output_dir` 会直接抛错，
  `scripts/train_act.py` 提前拦下并提示换目录，不会静默续训。
* **训练前先看 `--dry-run` 打印的 `episodes/frames`**：数据目录指错时唯一不报错的症状就是
  它悄悄训了另一套数据。

### 12.4 评测与可视化（训练完的权重）

```bash
# 20 次随机布局，写 mp4 + PNG 关键帧，打印汇总
/home/zh/miniconda3/envs/lerobot/bin/python experiments/act_rollout.py --task place \
    --ckpt out/act_train_place_rand/checkpoints/last/pretrained_model \
    --trials 20 --device cpu

# 跑完在 MuJoCo 窗口里按 20 Hz 重放这条轨迹（窗口自己开，跑完关窗退出）
/home/zh/miniconda3/envs/lerobot/bin/python experiments/act_rollout.py --task place \
    --ckpt out/act_train_place_rand/checkpoints/last/pretrained_model \
    --trials 1 --device cpu --viewer
```

每次 trial 的布局是当场用 `sample_layout` 抽的，逐 trial 的真值在 `summary.json.per_trial[*]
.cube_xy/.bowl_xy`，汇总的框在 `summary.json.layout_m`；产物默认落
`out/act_rollout_place/`（`trial_XX.mp4` + `trial_XX_{0000,0025,…}.png`，左 front、右 wrist）。

### 12.5 看结果看哪几个字段

* `successes/trials` + `outcomes`：判据是 `in_bowl` 且 `released`（末帧方块离 tool0 ≥
  `RELEASE_SEP_M` 0.05 m），**不看**指令 g 的末值——策略的 g 头系统性偏低。
* `bowl_dxy_min_m/max_m`：落点误差，碗内半径 0.045 m、判据 `IN_BOWL_RADIUS_M` 0.030 m。
* `hold_err_max_m`：最后一个"指令 g==0"拍上方块与 tool0 的水平距离，抓取质量的独立探针。
* `release_sep_m` / `criteria`：本次用的阈值，别用文档里的数字替代 summary 里的。

随机布局下的成功率天然低于固定布局（同一批权重、同一套判据），比较时必须**同分布同 trial 数**
才有意义；`--seed` 固定抽样序列。

### 12.6 本节没有保证的东西

只验证过一条 waypoint 表（`configs/waypoints_cube_place.json`）与 z=0.815 的方块；自碰撞没建模；
前视相机只留 12 px 边缘余量；`--viewer` 的窗口循环没有在本仓库的 CI 里跑过（需要人开窗口）。

### 12.7 检查点扫描：`last` 不是最好的那个

`out/act_train_place_rand` 训完 20000 步（loss 2.4→0.099，grdn ~12）后，把 10 个检查点各跑
20 次同批随机布局（`--seed 0`，布局逐 trial 完全相同，已验证），再对前几名加到 4 个 seed ×
20 次 = 80 次。判据是"方块落在碗里（`in_bowl`，见 §12.8 的半径口径）且已脱离夹爪"，
括号里的 **seated** 是更严的一档：落在 `IN_BOWL_SEATED_R_M = 0.030` 内（贴地而不是靠着碗壁）。

| checkpoint | seed 0 判据成功 | 4 seed 合计（seated） | `bowl_dxy` 均值 / 最大 | `hold_err` 最大 |
|---|---|---|---|---|
| 002000 | 14/20 | 41/80 51% (26) | 0.0517 / 0.1967 | 0.2626 |
| **006000** | **19/20** | **67/80 84% (43)** | **0.0217 / 0.0619** | **0.0256** |
| 010000 | 17/20 | 58/80 72% (37) | 0.0315 / 0.0909 | 0.3014 |
| 014000 | 19/20 | 61/80 76% (44) | 0.0201 / 0.0474 | 0.1918 |
| 018000 | 17/20 | — | 0.0351 / 0.0494 | 0.0250 |
| 020000 (`last`) | 13/20 | — | 0.0523 / 0.3100 | 0.2666 |

结论与口径：

* **选 006000**：4 seed 下进碗率最高（84%），判据成功率与 014000 打平，但 `hold_err` 最大只有
  0.026 m（014000 0.192 m，出现过中途脱手），落点也没有 014000 那种 0.19 m 的离群值。
* **训练 loss 与闭环成功率脱钩**：loss 一路降到 0.099，但 16k/18k/20k 只有 13–18/20（seed 0）。
  后段检查点变差不是"过拟合到 loss"，而是闭环里越走越偏；**用 `last` 会挑到较差的之一**。
* **seed 方差很大**（006000 逐 seed 19/10/9/9 在新口径下）。任何单 seed 的 20 次成绩都不能当结论。
* 默认 checkpoint 是 `out/act_train_place_rand/checkpoints/006000/pretrained_model`
  （`PLACE_CKPT`，见 `experiments/act_rollout.py:107`），`--task place` 不传 `--ckpt` 就是它。
* summary 里同时给两个数：`in_bowl_count`（碗内半径口径）与 `seated_count`（30 mm 坐实口径），
  逐 trial 还有 `in_bowl` / `seated`。

### 12.8 判据与测试口径（这一轮一并改的）

* **判据半径改了（2026-11-01，用户拍板）**：`IN_BOWL_RADIUS_M` 从 0.030 改成
  `BOWL_INNER_R_M`（0.045），理由是把方块滑到碗里、靠着碗壁也算进碗。更严的"坐实在地上"
  保留为 `IN_BOWL_SEATED_R_M = 0.030`，summary 与逐 trial 都两个都给。006000 因此在 seed 0 是
  19/20、4 seed 84%（严格口径 54%）。
* `tests_ee/test_act_rollout.py` 里"判成功就该落在碗里"这道门，从 `MAX_BOWL_DXY_M`（5 mm）改成
  `IN_BOWL_RADIUS_M`（30 mm）：5 mm 是**生成器**的接受门，由 `test_act_dataset.py` 对着
  `meta.json` 卡；重放同一批动作仍会漂（随机数据集实测 7.1 mm），拿它卡回放会误报。
* 同一文件里策略模式的**汇总级**检查不再要求每个 trial 都成功（策略跑失败是它要测量的东西），
  只要求：`in_bowl_count` 与 per-trial 一致、`bowl_dxy_max_m` 等于 per-trial 的最大值（按
  summary 的 6 位小数取整比较）。成功 trial 的质量门仍在 per-trial 里逐条查。
* `tests_ee/test_act_dataset.py` 里 place episode 的 `rise` 下界从 `0` 放到 `-1e-3`：碗随随机布局
  移动，方块落到碗底时可能比起点低 0.04 mm（60 条实测 `-0.00004 .. +0.00196`）。

复验：默认（策略 1 trial）338 checks、示教重放 20 条 553 checks、策略 20 条（含失败）531 checks、
最差检查点 020000 491 checks，全部 EXIT 0；把某个成功 trial 的 `in_bowl` 改成 False 的变异 EXIT 1。

### 12.9 修掉的坑：窗口里碗没跟着搬

`--viewer` 的窗口是**另起一个进程内模型**（`mujoco.MjModel.from_xml_path(SCENE)`），而每集
trial 的碗位置是运行时改 `model.geom_pos` 得到的。旧实现只把 qpos 帧喂给窗口，没有把碗搬到
那一集的布局上，于是：**随机布局下，窗口里的碗一直待在场景默认 (0.300, −0.200)，方块按轨迹
飞到它自己那一集的碗位——看起来就像"从来没放进碗里"，而 summary 里 in_bowl 是 True**。
mp4 / PNG 不受影响（它们在 rollout 进程里渲染，碗是搬过的）。

修法：`play_in_viewer(scene, traces, control_hz, passes)` 的 `traces` 改成 `(bowl_xy, frames)`
二元组，播每一集之前先 `place_bowl`（只用到 `.model`/`.data`，窗口模型够用）。

复验（headless，不开窗）：窗口模型里碗底 geom 从 `0.30000 −0.20000` 移到 `0.39570 −0.23350`
（17 个碗 geom 全平移，误差 0.00e+00）；同一 trial 的末帧方块 `[0.4034, −0.2531, 0.8149]` 到
**窗口里**的碗心距离 0.0211 m，与 summary 记录的 `bowl_dxy_m 0.0210` 一致；策略路径同样成立
（2 次均 success，窗口碗位 = trial 碗位）。测试里加了回归断言：`traces[0][0]` 必须等于该
trial summary 里的 `bowl_xy`。

正确的看窗口方式：

```bash
/home/zh/miniconda3/envs/lerobot/bin/python experiments/act_rollout.py --task place \
    --trials 20 --device cpu --viewer --viewer-passes 1
```

`--viewer-passes 1` = 20 集依次各播一遍然后退出；`--viewer-passes 0`（默认）= 一直循环到关窗。
窗口标题栏和终端第一行 `[roll] ckpt=...` 都会打印用的是哪个权重。

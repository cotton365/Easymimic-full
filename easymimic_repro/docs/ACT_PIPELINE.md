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


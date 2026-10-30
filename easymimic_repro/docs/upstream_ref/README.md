# 上游参考代码（只读，不参与本仓库构建）

来源：[yy2006-maker/jaka-zu3-mujoco](https://github.com/yy2006-maker/jaka-zu3-mujoco) @ `c592951`
（commit "Add reproducible hand-camera teleoperation and metric retargeting"）。

这些文件由 `git show c592951:<path>` 原样导出，**一个字节都没改**，列为只读参考：

| 文件 | 上游路径 | 是什么 |
|---|---|---|
| `hand_to_ee.py` | `scripts/hand_to_ee.py` | 上游的 MediaPipe 版 21 点 → 8 维重定向（336 行，numpy only）。对照本仓库 `src/ee_retarget/hand_to_ee.py` |
| `hand_camera_teleop.py` | `scripts/hand_camera_teleop.py` | 摄像头实时遥操作（807 行，mediapipe + opencv + MuJoCo viewer），带 MuJoCo 自带界面 |
| `retarget_hand_sequence.py` | `scripts/retarget_hand_sequence.py` | 把录下的一段手部关键点序列（npz）离线重定向成关节轨迹 |
| `check_hand_pipeline.py` / `check_hand_to_ee.py` | 同名 | 上游自检脚本（126 / 234 行） |
| `HAND_TO_EE.md` | `docs/HAND_TO_EE.md` | 上游对同一段数学的说明 |
| `requirements-hand-teleop.txt` / `constraints-hand-teleop-py310.txt` | 同名 | `mediapipe==0.10.21`、`opencv-contrib-python==4.11.0.86` 及其钉死约束 |

## 为什么只放参考、不接入

1. 依赖没装：本机 `~/miniconda3/envs/mujoco` 没有 mediapipe / opencv，装它们是网络+编译操作（由用户执行）。
2. 上游用 MediaPipe 单手 21 点，**没有腕部旋转**（`hand_to_ee.py:224` 自述），法向符号只能靠图像平面的近似；本仓库 `src/ee_retarget/hand_to_ee.py` 用的是腕部帧 Z 消歧，语义不同，不能混用。
3. 本仓库的 IK / 航点回放 / 抓取验收全部已在 MuJoCo 里跑绿，换成上游管线属于回归风险，需单独一轮验收。

## 若将来接入，要注意

- 上游 c592951 在 `jaka_zu3_mujoco.xml` 的夹爪 body 里加了一个 `<site name="tcp" pos="0 0 0.115">` 作为工具中心点；本仓库 `assets/jaka_zu3/robot.xml` 是手写副本，用的是 `site tool0`，两者不是同一个点，换算见 `src/ee_retarget/ik_jaka_zu3.py`。
- 上游 `hand_camera_teleop.py` 依赖它自己的 `MODEL_PATH`，直接跑会加载上游 `jaka_zu3_mujoco.xml`（该文件含 `<flexcomp name="rag">`，MuJoCo 3.16 加载失败）；本仓库因此改用 `assets/jaka_zu3/robot.xml`。

# 21 点手部关键点到 JAKA Zu 3 七关节目标

实现依据为用户提供的 `ALGORITHM_HAND_TO_EE.md` 和 `hand_to_ee_demo.py` 中已核对的公式及案例。这里的 A–F 检查是依据参考定义重建的本地测试，未直接执行原演示脚本，也未验证其全部导入路径、常量名或辅助接口兼容性。标准算法、自检和仿真运行均不依赖原附件。本项目没有引入 EasyMimic 的源代码、权重或完整训练流水线。

## 标准米制接口

模块为 `scripts/hand_to_ee.py`，仅依赖 NumPy：

```python
from scripts.hand_to_ee import hand_to_ee, ee_to_pose

ee8, info = hand_to_ee(kp, R_wrist, T_base_cam, d_min=0.02, d_max=0.10)
if info["valid"]:
    position_base, rotation_base, gripper = ee_to_pose(ee8)
```

| 输入/输出 | 形状 | 含义 |
| --- | --- | --- |
| `kp` | `(21, 3)` | 摄像头坐标系中的三维点，单位米；须包含手相对摄像头的位置 |
| `R_wrist` | `(3, 3)` | 腕部各坐标轴在同一摄像头坐标系下的方向，列向量构成右手旋转矩阵 |
| `T_base_cam` | `(4, 4)` | 摄像头坐标到机器人基座坐标的刚体变换，平移单位米 |
| `d_min`, `d_max` | 标量 | 指尖闭合/张开距离，单位米，`d_max > d_min` |
| `ee8` | `(8,)` | `[x,y,z,rx,ry,rz,g,0]`：米、旋转向量弧度、0–1 开度、保留位 |
| `info` | 字典 | `valid`；有效时含 `p_camera`、`R_hand`、`gap_m`、`plane_residual_m` |

腕部旋转必须与关键点使用同一坐标约定。相机画面若被镜像，坐标和旋转需一致处理；反射矩阵不是合法旋转。不同三维手部模型的 wrist/world/camera 坐标也不能混用。

计算顺序：

1. 三维 thenar 锚点：`p_camera = (kp[3] + kp[5]) / 2`，保留 XYZ 三个分量。
2. 对 `[5,6,7,8,3]` 五点中心化后做 SVD，取最小奇异值对应方向作为 Z 轴，以 `R_wrist[:,2]` 确定法线正负。
3. 将 `kp[6]-kp[5]` 投影到平面并归一化得到 X；Y = Z × X，组成 `R_hand=[X Y Z]`。
4. `gap = norm(kp[4]-kp[8])`；`g = clip((gap-d_min)/(d_max-d_min), 0, 1)`。
5. `p_base = R_bc @ p_camera + t_bc`；`R_base = R_bc @ R_hand`。
6. 将 `R_base` 转为旋转向量，输出 EE8。旋转向量只有三个数；不是四元数。

非有限数处理检查全部 21 点，包括不参与几何公式的点。任何点、腕部旋转或外参中有 NaN/Inf 时，返回全 NaN 的 EE8 和 `valid=False`。形状错误、有限但不是刚体的变换、共线平面或退化食指方向抛出 `ValueError`。`ee_to_pose` 拒绝无效 EE8，避免它进入 IK。

`T_base_cam` 只表示坐标变换。若还需固定的手部坐标到工具坐标旋转偏置，应在应用层明确建模；此标准接口严格按上述公式，没有暗中添加工具偏置。

## 与现有摄像头控制的衔接

两条路径共享 `_hand_geometry`、EE8 编解码以及 `ee8_to_joint_targets`：

```text
米制 kp + R_wrist + T_base_cam → hand_to_ee ───────────┐
                                                    ├→ EE8 → IK → 6 个臂关节 + 1 个夹爪关节
MediaPipe → 共用几何 → 相对位移/姿态适配 ─────────────┘
```

笔记本摄像头仍使用 `hand_camera_teleop.py`。图像 thenar 位移与掌面尺寸变化估计相对 XYZ；MediaPipe hand-relative world landmarks 提供相对几何，上一帧法线提供方向参考。按 Space 捕获手部/机器人初始姿态，C/O 标定指尖间距。固定方向映射 `CAMERA_TO_BASE` 是近似值。

没有实现可靠的手心/手背分类。连续法线参考可以减少相邻帧翻转，但不能保证快速翻掌、遮挡或重新检测后的朝向正确。标准接口中的腕部参考测试通过，不等于摄像头已经能准确观测该腕部参考。

MediaPipe 的 world landmarks 以手部为中心；手平移后原点会随手移动，不能靠外参平移把它变成绝对相机位置。当前摄像头路径没有伪造 `R_wrist` 或将这些点冒充标准米制输入。启动命令不变：

```bash
.venv-handcam/bin/python scripts/hand_camera_teleop.py --camera 0
```

共享 IK 使用上一组关节角作为初值、阻尼最小二乘和模型关节限位。位置残差需 ≤ 0.0005 m，姿态残差需 ≤ 0.005 rad，否则不输出新臂关节解。夹爪 `g` 线性映射为 `joint_7` 的 `0–0.9 rad`。摄像头路径原有的臂关节/夹爪限速仍生效；夹爪也可以在臂 IK 保持时继续响应指尖开度。求解不含碰撞约束或随机多初值搜索，不保证全工作空间内都能收敛。

## 米制数据的可运行入口

如果上游已能提供符合上述契约的三维数据，可保存为 NumPy NPZ：

```python
np.savez("observations.npz",
         kp=kp_sequence,              # (N,21,3)，绝对 camera-frame 米制点
         R_wrist=wrist_sequence,      # (N,3,3)
         T_base_cam=measured_transform)  # (4,4)
```

从项目根目录运行：

```bash
.venv-handcam/bin/python scripts/retarget_hand_sequence.py \
  --input observations.npz --output joint_targets.npz
```

输出包含 `ee8`、`joint_targets`（N×7，弧度）、`accepted`、`status`、`position_error_m` 和 `rotation_error_rad`。坏帧保持上一组七关节目标，第一帧失败时保持仿真的默认抬起姿态。必须检查 `accepted`，保持值不是成功求解。全部成功时退出码为 0；存在被拒绝帧时仍保存结果，退出码为 1。输出路径不能已存在。

此入口仅离线求解，不启动摄像头或执行动作。结果没有时间参数化、速度限制或避障验证，不能直接当成真实机械臂轨迹下发。

## 可复现检查与本次结果

```bash
.venv-handcam/bin/python scripts/check_hand_to_ee.py
.venv-handcam/bin/python scripts/check_hand_pipeline.py
.venv-handcam/bin/python scripts/hand_camera_teleop.py --check-ik
```

- A：标准手形输出位置 `[0,.025,0]`、单位姿态和 `g=.5`。
- B：相机绕 Z 轴旋转 90°、平移 `[.4,0,.8]` 后得到位置 `[.375,0,.8]`。
- C/C2：同一翻转手形使用不同腕部参考，正确选择平面法线方向。
- D：退化食指方向、共线五点被拒绝。
- E：未参与公式的点、腕部或外参含 NaN/Inf，均使整个 EE8 无效。
- F：固定随机种子的 200 次旋转往返，并额外检查零角、微小角和接近 180° 的姿态。
- 额外数学检查：改变相机坐标系但同步修改外参时，基座输出不变；检查开度裁剪、数据契约和相对摄像头适配。
- 5 组端到端合成输入：从已知机器人姿态构造手部输入，经过完整变换与 IK 后，以独立正运动学核对 TCP；同时检查夹爪从 0 到 0.9 rad。最大位置残差约 **0.284 mm**、姿态残差约 **0.241°**。
- 控制边界检查：无效/退化/不可达帧保持，恢复有效输入后继续；求解不改写仿真关节状态或执行器指令；NPZ 命令行结果可复现。
- 13 个局部 IK 位置检查：中心及 XYZ 各轴 ±2/±4 cm，全部通过新的收敛要求。

这些是 Python 3.10、MuJoCo 3.1.6 下的合成数据和运动学结果。没有进行实时摄像头精度测量，也没有证明动态跟随、接触情况下的执行精度或整个 20 cm 活动范围的可达性。参考文档中其他项目的 88 项检查、多初值 IK 和训练/轨迹流程不能算作本项目已完成。

下一阶段需要获取有尺度、位置定义明确的相机三维观测，测得摄像头到仿真/机器人基座的外参，并用已知位移、静态抖动和旋转角度来评估相机输入误差。标准接口已提供接入位置；仅修改几何公式无法消除单目深度的不确定性。

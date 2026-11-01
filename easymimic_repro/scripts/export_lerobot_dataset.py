#!/usr/bin/env python
"""把仿真示教目录 (out/act_demos_ee) 导出为 LeRobot v3 数据集 (parquet + 逐帧 PNG)。

输入契约 (8 维 EE 接口, 见 docs/ACT_PIPELINE.md §8):
  <demos>/index.json                 {fps, task, camera_names, state_dim=8, action_dim=8}
  <demos>/<ep>/data.npz             pose8_state(N,8) pose8_action(N,8) ...
                                    [x,y,z,qw,qx,qy,qz,g]: state 是该拍动作执行前的
                                    tool0 位姿+夹爪开度, action 是执行后位姿+该拍指令 g
  <demos>/<ep>/<cam>/<000000.png>   256x256 RGB uint8

输出契约 (LeRobot v3, use_videos=False 因为本环境无 torchcodec):
  observation.state  float32[8]
  action             float32[8]
  observation.images.front  image (256,256,3)
  observation.images.wrist  image (256,256,3)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "local/act_pick_cube_ee"
IMAGE_SIZE = 256
STATE_DIM = 8
ACTION_DIM = 8
# 只导出契约里允许的四类 feature；顺序固定，便于下游比对
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
    "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    "observation.images.front": {
        "dtype": "image",
        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "image",
        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", type=Path, default=Path("out/act_demos_ee"), help="示教目录，含 index.json")
    p.add_argument("--out", type=Path, default=Path("data/lerobot/act_pick_cube_ee"),
                   help="数据集输出根目录")
    p.add_argument("--repo-id", type=str, default=REPO_ID, help="数据集 repo_id 标识")
    return p.parse_args()


def load_index(demos: Path) -> dict:
    idx_path = demos / "index.json"
    if not idx_path.is_file():
        sys.exit(f"[export] 错误: 找不到 {idx_path}，请确认 --demos 指向示教根目录")
    index = json.loads(idx_path.read_text())
    for key in ("fps", "task", "camera_names"):
        if key not in index:
            sys.exit(f"[export] 错误: {idx_path} 缺少字段 '{key}'")
    if int(index.get("state_dim", STATE_DIM)) != STATE_DIM or int(index.get("action_dim", ACTION_DIM)) != ACTION_DIM:
        sys.exit(f"[export] 错误: 本导出器只支持 {STATE_DIM} 维 state/action，index.json 声明为 "
                 f"state_dim={index.get('state_dim')} action_dim={index.get('action_dim')}")
    missing = [c for c in index["camera_names"] if c not in ("front", "wrist")]
    if missing:
        sys.exit(f"[export] 错误: features 只声明了 front/wrist 两个相机，index.json 多了 {missing}")
    return index


def list_episodes(demos: Path) -> list[Path]:
    eps = [p for p in demos.iterdir() if p.is_dir() and re.fullmatch(r"ep\d+", p.name)]
    if not eps:
        sys.exit(f"[export] 错误: {demos} 下没有任何 epNNN 目录")
    return sorted(eps, key=lambda p: int(p.name[2:]))


def load_images(ep_dir: Path, cam: str) -> list[np.ndarray]:
    cam_dir = ep_dir / cam
    if not cam_dir.is_dir():
        sys.exit(f"[export] 错误: {ep_dir.name} 缺少相机目录 {cam}/")
    files = sorted(cam_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not files:
        sys.exit(f"[export] 错误: {cam_dir} 里没有 png")
    frames = []
    for f in files:
        img = np.asarray(Image.open(f).convert("RGB"))
        if img.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or img.dtype != np.uint8:
            sys.exit(f"[export] 错误: {f} 形状/类型为 {img.shape}/{img.dtype}，要求 "
                     f"({IMAGE_SIZE},{IMAGE_SIZE},3)/uint8")
        frames.append(img)
    return frames


def main() -> None:
    args = parse_args()
    demos = args.demos.expanduser().resolve()
    out = args.out.expanduser().resolve()

    index = load_index(demos)
    fps = int(index["fps"])
    task = str(index["task"])
    episodes = list_episodes(demos)
    cams = list(index["camera_names"])

    if out.exists():
        sys.exit(
            f"[export] 错误: 输出目录已存在，LeRobotDataset.create 不会覆盖: {out}\n"
            f"         请先删除它 (rm -rf {out}) 或换一个 --out。"
        )

    print(f"[export] demos={demos}")
    print(f"[export] out={out} repo_id={args.repo_id} fps={fps} task={task!r} episodes={len(episodes)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    # use_videos=False: 本环境没有 torchcodec，视频编码路径不可用，只能逐帧写 PNG。
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=FEATURES,
        root=out,
        use_videos=False,
    )

    for ep_dir in episodes:
        npz_path = ep_dir / "data.npz"
        if not npz_path.is_file():
            sys.exit(f"[export] 错误: 缺少 {npz_path}")
        data = np.load(npz_path)
        for key in ("pose8_state", "pose8_action", "t"):
            if key not in data:
                sys.exit(f"[export] 错误: {npz_path} 缺少数组 '{key}'"
                         f"（旧 7 维示教没有 EE 数组，请用 experiments/gen_act_dataset.py 重新生成）")
        # 8 维 EE 契约: state 在动作前, action 在动作后 (见 §8), 不做任何坐标变换
        state = data["pose8_state"].astype(np.float32)
        action = data["pose8_action"].astype(np.float32)
        if state.shape[1] != STATE_DIM or action.shape[1] != ACTION_DIM:
            sys.exit(f"[export] 错误: {ep_dir.name} pose8_state/pose8_action 形状为 "
                     f"{state.shape}/{action.shape}，要求 (N,{STATE_DIM})/(N,{ACTION_DIM})")
        n = state.shape[0]
        if action.shape[0] != n:
            sys.exit(f"[export] 错误: {ep_dir.name} pose8_state 与 pose8_action 帧数不一致: "
                     f"{n} vs {action.shape[0]}")

        images = {cam: load_images(ep_dir, cam) for cam in cams}
        for cam, frames in images.items():
            if len(frames) != n:
                sys.exit(f"[export] 错误: {ep_dir.name}/{cam} 有 {len(frames)} 张图，data.npz 有 {n} 帧")

        for i in range(n):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task,
            }
            for cam in cams:
                frame[f"observation.images.{cam}"] = images[cam][i]
            ds.add_frame(frame)
        ds.save_episode()
        print(f"[export] {ep_dir.name}: {n} 帧 x {len(cams)} 相机 已写入")

    ds.finalize()
    print(f"[export] 完成: num_episodes={ds.num_episodes} num_frames={ds.num_frames} root={out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""ACT 开环推理评测：在留出 episode 上逐步推理，与两个基线比 MSE/MAE。

基线:
  1) mean  : 全数据集动作均值 -> 常数预测
  2) persis: 复制上一帧动作 (persistence)

比较口径: 三者在同一批时刻上比较，即 t = 1..N-1。
  t=0 没有"上一帧动作"，persistence 无定义，故一律排除，保证公平。
判定: 模型总体 MSE 同时低于两个基线 -> PASS (exit 0)，否则 FAIL (exit 1)。

只读 PNG + npz + LeRobot checkpoint，不依赖 torchcodec，纯 CPU 可跑。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DIM_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
IMAGE_SIZE = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="checkpoint 路径：训练输出根目录或 .../pretrained_model 目录")
    p.add_argument("--demos", type=Path, default=Path("out/act_demos"), help="示教目录，含 index.json")
    p.add_argument("--holdout-ep", type=str, default="last", help="留出 episode：'last'、序号 (1) 或名字 (ep001)")
    p.add_argument("--device", type=str, default="cpu", help="推理设备，本沙箱只能用 cpu")
    return p.parse_args()


def resolve_ckpt(ckpt: Path) -> Path:
    """允许传训练输出根目录，自动定位到 checkpoints/last/pretrained_model。"""
    ckpt = ckpt.expanduser().resolve()
    if (ckpt / "config.json").is_file():
        return ckpt
    for cand in (ckpt / "checkpoints" / "last" / "pretrained_model",
                 ckpt / "pretrained_model"):
        if (cand / "config.json").is_file():
            return cand
    sys.exit(f"[eval] 错误: {ckpt} 下找不到 config.json；请传 .../checkpoints/last/pretrained_model")


def pick_holdout(demos: Path, spec: str) -> tuple[list[Path], Path]:
    eps = sorted([p for p in demos.iterdir() if p.is_dir() and re.fullmatch(r"ep\d+", p.name)],
                 key=lambda p: int(p.name[2:]))
    if len(eps) < 2:
        sys.exit(f"[eval] 错误: {demos} 下只有 {len(eps)} 条 episode，开环评测至少需要 2 条（1 条留出）")
    if spec == "last":
        return eps, eps[-1]
    if spec.isdigit():
        idx = int(spec)
        if idx >= len(eps):
            sys.exit(f"[eval] 错误: --holdout-ep={idx} 越界，共 {len(eps)} 条 (0..{len(eps) - 1})")
        return eps, eps[idx]
    for p in eps:
        if p.name == spec:
            return eps, p
    sys.exit(f"[eval] 错误: --holdout-ep={spec!r} 不匹配任何 episode ({[p.name for p in eps]})")


def load_rgb(path: Path) -> torch.Tensor:
    """PNG -> (1,3,H,W) float32 in [0,1]，与 LeRobotDataset.__getitem__ 的输出一致。"""
    img = np.asarray(Image.open(path).convert("RGB"))
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        sys.exit(f"[eval] 错误: {path} 形状 {img.shape}，要求 ({IMAGE_SIZE},{IMAGE_SIZE},3)")
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div_(255.0)
    return t.unsqueeze(0)


def load_episode(ep_dir: Path, cams: list[str]) -> dict:
    data = np.load(ep_dir / "data.npz")
    state = data["qpos"].astype(np.float32)
    action = data["act"].astype(np.float32)
    n = state.shape[0]
    frames = {cam: sorted((ep_dir / cam).glob("*.png"), key=lambda p: int(p.stem)) for cam in cams}
    for cam, fs in frames.items():
        if len(fs) != n:
            sys.exit(f"[eval] 错误: {ep_dir.name}/{cam} 有 {len(fs)} 张图，data.npz 有 {n} 帧")
    return {"state": state, "action": action, "frames": frames, "n": n}


def per_dim_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return ((pred - gt) ** 2).mean(axis=0)


def per_dim_mae(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.abs(pred - gt).mean(axis=0)


def main() -> None:
    args = parse_args()
    demos = args.demos.expanduser().resolve()
    index = json.loads((demos / "index.json").read_text())
    task = str(index["task"])
    cams = list(index["camera_names"])
    ckpt_dir = resolve_ckpt(args.ckpt)
    episodes, holdout = pick_holdout(demos, args.holdout_ep)

    # 1) 全数据集动作均值（含留出 episode，符合"全数据集均值"口径）
    all_actions = np.concatenate([np.load(ep / "data.npz")["act"].astype(np.float32) for ep in episodes])
    mean_action = all_actions.mean(axis=0).astype(np.float32)

    hold = load_episode(holdout, cams)
    gt = hold["action"]
    n = hold["n"]
    if n < 2:
        sys.exit(f"[eval] 错误: {holdout.name} 只有 {n} 帧，无法评测")
    state = hold["state"]

    print(f"[eval] ckpt={ckpt_dir}")
    print(f"[eval] demos={demos} holdout={holdout.name} frames={n} task={task!r} device={args.device}")

    # 2) 加载策略 + 训练时保存的归一化 pipeline
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(ckpt_dir))
    policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    print(f"[eval] loaded ACTPolicy  chunk_size={policy.config.chunk_size} "
          f"n_action_steps={policy.config.n_action_steps}")

    # 3) 逐步推理（t=1..n-1，与 persistence 同口径）
    preds = []
    for t in range(1, n):
        obs = {
            "observation.state": torch.from_numpy(state[t]).unsqueeze(0).float(),
            "task": [task],
        }
        for cam in cams:
            obs[f"observation.images.{cam}"] = load_rgb(hold["frames"][cam][t])
        policy.reset()  # 清空 action queue，保证每帧都重新前向一次
        batch = preprocessor(obs)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = postprocessor(action)
        preds.append(action.squeeze(0).detach().cpu().numpy().astype(np.float32))
    pred = np.stack(preds)          # (n-1, 7)
    target = gt[1:]                  # (n-1, 7)
    mean_pred = np.tile(mean_action, (n - 1, 1))
    persis_pred = gt[:-1]            # 复制上一帧动作

    # 4) 指标
    m_mse, m_mae = per_dim_mse(pred, target), per_dim_mae(pred, target)
    b1_mse = per_dim_mse(mean_pred, target)
    b2_mse = per_dim_mse(persis_pred, target)
    model_all, b1_all, b2_all = m_mse.mean(), b1_mse.mean(), b2_mse.mean()

    print(f"\n[eval] 评测步数 = {n - 1} (t=1..{n - 1}, 与 persistence 同口径)")
    print(f"{'dim':>8} | {'model MSE':>10} {'model MAE':>10} | {'mean MSE':>10} | {'persis MSE':>11}")
    print("-" * 62)
    for i, name in enumerate(DIM_NAMES):
        print(f"{name:>8} | {m_mse[i]:10.6f} {m_mae[i]:10.6f} | {b1_mse[i]:10.6f} | {b2_mse[i]:11.6f}")
    print("-" * 62)
    print(f"{'overall':>8} | {model_all:10.6f} {m_mae.mean():10.6f} | {b1_all:10.6f} | {b2_all:11.6f}")

    print(f"\n[eval] 总体 MSE  model={model_all:.6f}  mean-baseline={b1_all:.6f}  "
          f"persistence-baseline={b2_all:.6f}")
    print(f"[eval] 相对 mean  基线: {b1_all / max(model_all, 1e-12):.3f}x")
    print(f"[eval] 相对 persis基线: {b2_all / max(model_all, 1e-12):.3f}x")

    ok = bool(model_all < b1_all and model_all < b2_all)
    print(f"[eval] {holdout.name} 首帧 GT={np.round(gt[1], 4).tolist()}")
    print(f"[eval] {holdout.name} 首帧 预测={np.round(pred[0], 4).tolist()}")
    print(f"\n[eval] 判定: {'PASS' if ok else 'FAIL'}  "
          f"(模型总体 MSE 须同时低于 mean 基线 {b1_all:.6f} 与 persistence 基线 {b2_all:.6f})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

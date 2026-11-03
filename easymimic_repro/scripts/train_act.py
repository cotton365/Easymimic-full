#!/usr/bin/env python
"""一条命令训练 ACT（随机布局数据集），把 20K 配方与路径固定在这里。

为什么要有这个包装：lerobot 的训练入口是 `python -m lerobot.scripts.lerobot_train`，
它的十来个 flag 里有多处必须与仓库路径一致（dataset.root 要绝对路径、HF_HOME 必须在
读数据前指向仓库内的缓存目录、repo_id 只是本地缓存键）。这些抄错任何一处都不会报错，
而是安静地训练一个空/旧数据集。所以路径与环境在这里固定，命令行只暴露真正会调的旋钮。

默认配方（与 docs/ACT_PIPELINE.md §12 一致）:
  data   : data/lerobot/act_pick_place_rand  (60 条连续随机布局, 10860 帧, 20 fps)
  输出   : out/act_train_place_rand
  steps  : 20000, batch 8, lr 1e-5 (lerobot 默认), save_freq 2000, seed 1000

只读 dataset 的 meta/info.json 与 meta/episodes，不 import torch/lerobot；
真正的训练由 ${python} -m lerobot.scripts.lerobot_train 子进程执行。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "lerobot" / "act_pick_place_rand"
DEFAULT_DEMOS = REPO_ROOT / "out" / "act_demos_place_rand"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                   help="lerobot 数据集根目录（含 meta/info.json）")
    p.add_argument("--repo-id", default="local/act_pick_place_rand",
                   help="数据集缓存键，与导出一致即可，不联网")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out" / "act_train_place_rand",
                   help="训练输出根目录（checkpoints/ 与 train_config.json 落在这里）")
    p.add_argument("--job-name", default=None, help="默认取 out-dir 的目录名")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-freq", type=int, default=2000)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--hf-home", type=Path, default=REPO_ROOT / "data" / "_scratch" / "hf_home",
                   help="HF_HOME；数据集缓存必须留在仓库内，否则会去写只读路径")
    p.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    return p.parse_args()


def read_info(dataset: Path) -> dict:
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(
            "[train] 找不到 %s\n"
            "[train] 先生成并导出数据：\n"
            "  experiments/gen_act_dataset.py --table configs/waypoints_cube_place.json "
            "--out out/act_demos_place_rand --episodes 60\n"
            "  scripts/export_lerobot_dataset.py --demos out/act_demos_place_rand "
            "--out data/lerobot/act_pick_place_rand --repo-id local/act_pick_place_rand"
            % info_path)
    info = json.loads(info_path.read_text())
    for key in ("action", "observation.state"):
        if key not in info.get("features", {}):
            raise SystemExit("[train] %s 缺少特征 %s" % (info_path, key))
    shape = tuple(info["features"]["action"].get("shape", ()))
    if shape != (8,):
        raise SystemExit("[train] action 维度是 %s，期望 (8,)（8 维 EE 接口）" % (shape,))
    if int(info.get("total_frames", 0)) <= 0 or int(info.get("total_episodes", 0)) <= 0:
        raise SystemExit("[train] %s 里 total_frames/total_episodes 为 0" % info_path)
    return info


def build_command(args: argparse.Namespace, dataset: Path) -> list[str]:
    job = args.job_name or args.out_dir.name
    return [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--dataset.repo_id=%s" % args.repo_id,
        "--dataset.root=%s" % dataset,
        "--policy.type=act",
        "--policy.repo_id=%s_act" % args.repo_id,
        "--policy.push_to_hub=false",
        "--policy.device=%s" % args.device,
        "--output_dir=%s" % args.out_dir,
        "--job_name=%s" % job,
        "--batch_size=%d" % args.batch_size,
        "--num_workers=%d" % args.num_workers,
        "--steps=%d" % args.steps,
        "--log_freq=%d" % args.log_freq,
        "--save_freq=%d" % args.save_freq,
        "--seed=%d" % args.seed,
        "--wandb.enable=false",
    ]


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    info = read_info(dataset)
    args.out_dir = args.out_dir.resolve()

    hf_home = args.hf_home.resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HF_HOME"] = str(hf_home)
    env["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    env["TOKENIZERS_PARALLELISM"] = "false"

    cmd = build_command(args, dataset)
    print("[train] 数据集 %s" % dataset, flush=True)
    print("[train]   episodes %d  frames %d  fps %s  action %s"
          % (info["total_episodes"], info["total_frames"], info.get("fps"),
             tuple(info["features"]["action"]["shape"])), flush=True)
    print("[train] 输出 %s   steps %d  batch %d  device %s"
          % (args.out_dir, args.steps, args.batch_size, args.device), flush=True)
    print("[train] HF_HOME %s" % hf_home, flush=True)
    if args.dry_run:
        print("[train] dry-run: " + " ".join(cmd), flush=True)
        return 0

    # lerobot refuses an output dir that already has anything in it (resume=False), so the
    # dir is created by the child, never here — and a rerun gets a readable message instead
    # of a FileExistsError traceback from deep inside the trainer.
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("[train] 输出目录非空：%s\n[train] 换一个 --out-dir，或者先把它移走"
                         "（lerobot 不做覆盖，也不自动 resume）" % args.out_dir)
    print("[train] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    sys.exit(main())

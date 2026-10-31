#!/usr/bin/env python3
"""Contract test for the scripted ACT dataset written by experiments/gen_act_dataset.py.

Pure numpy + file inspection: no mujoco, no rendering, no imports from src/ - the point is
that the artefact on disk can be checked by a machine that cannot run the simulator.

Checks, per episode:
  * meta.json / data.npz carry exactly the agreed keys, with the agreed shapes and dtypes;
  * num_frames == len(t) == the number of PNGs in each camera directory, PNGs are 8-bit
    RGB 256x256 named 000000.png .. and the frame clock is k / fps;
  * qpos/pts are physically sane: qpos[6] and act[6] inside the hinge range, |dq| per tick
    bounded, act - qpos bounded, cube quaternion unit length, spawn z on the table;
  * meta and npz agree, rise >= 0.08 m and grasp_ok is true;
and across episodes: index.json is consistent with the directories it describes.

Usage:  python tests_ee/test_act_dataset.py [--dir out/act_demos] [--episodes 20]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

META_KEYS = {"episode", "fps", "num_frames", "camera_hw", "cube_pos", "cube_yaw",
             "rise_m", "slip_m", "grasp_ok", "scene", "control_hz"}
NPZ_KEYS = {"qpos", "act", "t", "cube_pos", "cube_quat", "rise_m", "slip_m"}
INDEX_KEYS = {"num_episodes", "fps", "control_hz", "task", "camera_names", "state_dim",
              "action_dim", "seed", "mujoco_version", "randomization", "generated_by"}
CAMERAS = ("front", "wrist")
HW = (256, 256)

MIN_RISE_M = 0.080
MAX_TICK_STEP_RAD = 0.15     # measured max per-tick joint step 0.045 rad at 20 Hz
MAX_ACT_QPOS_RAD = 0.35      # measured max |act - qpos| 0.222 rad (the hinge's own lag)
HINGE_RANGE = (0.0, 0.9)
HINGE_SLACK = 1e-3
CUBE_Z = 0.815               # table top 0.80 + half height 0.015, +-1 mm


class Checks:
    def __init__(self):
        self.count = 0

    def ok(self, cond, msg):
        self.count += 1
        if not cond:
            raise AssertionError(msg)


def read_png_header(path):
    """(width, height, bit_depth, colour_type) from the IHDR chunk; raises if not a PNG."""
    with open(path, "rb") as fh:
        head = fh.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise AssertionError("%s: not a PNG (bad signature or missing IHDR)" % path)
    width, height, depth, colour = struct.unpack(">IIBB", head[16:26])
    return width, height, depth, colour


def check_episode(ck, ep_dir, index):
    ck.ok(os.path.isdir(ep_dir), "%s: missing episode directory" % ep_dir)
    meta_path = os.path.join(ep_dir, "meta.json")
    npz_path = os.path.join(ep_dir, "data.npz")
    ck.ok(os.path.isfile(meta_path), "%s: missing meta.json" % ep_dir)
    ck.ok(os.path.isfile(npz_path), "%s: missing data.npz" % ep_dir)
    with open(meta_path) as fh:
        meta = json.load(fh)

    ck.ok(set(meta) == META_KEYS,
          "%s: meta.json keys %s != %s" % (ep_dir, sorted(set(meta)), sorted(META_KEYS)))
    ck.ok(isinstance(meta["episode"], int), "%s: meta.episode is not an int" % ep_dir)
    ck.ok(meta["scene"] == "assets/jaka_zu3/scene_cube_bowl.xml",
          "%s: meta.scene %r" % (ep_dir, meta["scene"]))
    ck.ok(os.path.isfile(os.path.join(REPO, meta["scene"])),
          "%s: meta.scene does not exist under the repo" % ep_dir)
    ck.ok(meta["camera_hw"] == list(HW), "%s: meta.camera_hw %r" % (ep_dir, meta["camera_hw"]))
    ck.ok(float(meta["fps"]) == float(meta["control_hz"]),
          "%s: one frame per control tick, so fps (%r) must equal control_hz (%r)"
          % (ep_dir, meta["fps"], meta["control_hz"]))
    ck.ok(float(meta["control_hz"]) == float(index["control_hz"]),
          "%s: control_hz disagrees with index.json" % ep_dir)
    ck.ok(meta["grasp_ok"] is True, "%s: grasp_ok is not true" % ep_dir)
    ck.ok(float(meta["cube_yaw"]) == 0.0, "%s: this round does not randomize yaw" % ep_dir)
    ck.ok(float(meta["rise_m"]) >= MIN_RISE_M,
          "%s: rise %.5f m is below %.3f m" % (ep_dir, float(meta["rise_m"]), MIN_RISE_M))
    ck.ok(0.0 <= float(meta["slip_m"]) < 1e-2, "%s: slip %.6f m" % (ep_dir, float(meta["slip_m"])))

    data = np.load(npz_path)
    ck.ok(set(data.files) == NPZ_KEYS,
          "%s: data.npz keys %s != %s" % (ep_dir, sorted(set(data.files)), sorted(NPZ_KEYS)))
    qpos, act, t = data["qpos"], data["act"], data["t"]
    n = int(meta["num_frames"])
    ck.ok(qpos.shape == (n, 7), "%s: qpos shape %s != (%d, 7)" % (ep_dir, qpos.shape, n))
    ck.ok(act.shape == (n, 7), "%s: act shape %s != (%d, 7)" % (ep_dir, act.shape, n))
    ck.ok(t.shape == (n,), "%s: t shape %s != (%d,)" % (ep_dir, t.shape, n))
    for key in ("qpos", "act", "t", "cube_pos", "cube_quat"):
        ck.ok(data[key].dtype == np.float64, "%s: %s dtype %s" % (ep_dir, key, data[key].dtype))
    for key in ("rise_m", "slip_m"):
        ck.ok(data[key].shape == () and data[key].dtype == np.float64,
              "%s: %s is %s %s, expected a float64 scalar"
              % (ep_dir, key, data[key].shape, data[key].dtype))
        ck.ok(abs(float(data[key]) - float(meta[key])) < 1e-12,
              "%s: %s disagrees between data.npz and meta.json" % (ep_dir, key))
    ck.ok(data["cube_pos"].shape == (3,), "%s: cube_pos shape %s" % (ep_dir, data["cube_pos"].shape))
    ck.ok(data["cube_quat"].shape == (4,), "%s: cube_quat shape %s"
          % (ep_dir, data["cube_quat"].shape))
    ck.ok(np.allclose(data["cube_pos"], np.asarray(meta["cube_pos"], dtype=float), atol=1e-12),
          "%s: cube_pos disagrees between data.npz and meta.json" % ep_dir)
    ck.ok(abs(float(data["cube_pos"][2]) - CUBE_Z) < 1e-3,
          "%s: cube spawn z %.4f is not on the table" % (ep_dir, float(data["cube_pos"][2])))
    ck.ok(abs(float(np.linalg.norm(data["cube_quat"])) - 1.0) < 1e-9,
          "%s: cube_quat is not unit length" % ep_dir)
    ck.ok(abs(abs(float(data["cube_quat"][0])) - 1.0) < 1e-6,
          "%s: cube_quat is not the identity (rotation exists but yaw is not randomized)" % ep_dir)

    # frame clock: t[k] = k / fps, so num_frames ticks cover num_frames / fps seconds
    ck.ok(abs(float(t[0])) < 1e-12, "%s: t[0] = %r, expected 0" % (ep_dir, float(t[0])))
    ck.ok(np.allclose(t, np.arange(n) / float(meta["fps"]), atol=1e-9),
          "%s: t is not k / fps" % ep_dir)
    ck.ok(abs(t[-1] - (n - 1) / float(meta["fps"])) < 1e-9,
          "%s: t[-1] does not match (num_frames - 1) / fps" % ep_dir)

    # physics sanity
    ck.ok(np.all(np.isfinite(qpos)) and np.all(np.isfinite(act)),
          "%s: non-finite values in qpos/act" % ep_dir)
    # joint_7 declares range 0..0.9 but is not `limited`, so the measured angle settles a
    # hair outside it (0.900006 was the worst case); 1 mrad of slack, not "anything goes".
    ck.ok(np.all(qpos[:, 6] >= HINGE_RANGE[0] - HINGE_SLACK)
          and np.all(qpos[:, 6] <= HINGE_RANGE[1] + HINGE_SLACK),
          "%s: gripper hinge %.6f..%.6f leaves %s"
          % (ep_dir, qpos[:, 6].min(), qpos[:, 6].max(), HINGE_RANGE))
    ck.ok(np.all(act[:, 6] >= HINGE_RANGE[0] - HINGE_SLACK)
          and np.all(act[:, 6] <= HINGE_RANGE[1] + HINGE_SLACK),
          "%s: gripper target %.6f..%.6f leaves %s"
          % (ep_dir, act[:, 6].min(), act[:, 6].max(), HINGE_RANGE))
    step = float(np.abs(np.diff(qpos, axis=0)).max()) if n > 1 else 0.0
    ck.ok(step <= MAX_TICK_STEP_RAD,
          "%s: a joint jumps %.4f rad in one tick (bound %.3f)" % (ep_dir, step, MAX_TICK_STEP_RAD))
    gap = float(np.abs(act - qpos).max())
    ck.ok(gap <= MAX_ACT_QPOS_RAD,
          "%s: |act - qpos| reaches %.4f rad (bound %.3f)" % (ep_dir, gap, MAX_ACT_QPOS_RAD))
    ck.ok(qpos[:, 6][0] > qpos[:, 6][-1],
          "%s: the gripper never closed (hinge %.3f -> %.3f)"
          % (ep_dir, qpos[0, 6], qpos[-1, 6]))

    # frames: one PNG per tick per camera, 8-bit RGB, right size, contiguous numbering
    for cam in CAMERAS:
        cam_dir = os.path.join(ep_dir, cam)
        ck.ok(os.path.isdir(cam_dir), "%s: missing %s/ directory" % (ep_dir, cam))
        names = sorted(os.listdir(cam_dir))
        expect = ["%06d.png" % k for k in range(n)]
        ck.ok(names == expect,
              "%s/%s: %d files, expected %d named 000000.png..%06d.png (first mismatch %r)"
              % (ep_dir, cam, len(names), n, n - 1,
                 next((a for a, b in zip(names, expect) if a != b), None)))
        for name in names:
            path = os.path.join(cam_dir, name)
            ck.ok(os.path.getsize(path) > 200, "%s: suspiciously small PNG" % path)
            width, height, depth, colour = read_png_header(path)
            ck.ok((width, height) == HW, "%s: %dx%d, expected %dx%d" % (path, width, height, *HW))
            ck.ok((depth, colour) == (8, 2), "%s: bit depth %d / colour type %d, expected 8 / 2 (RGB)"
                  % (path, depth, colour))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=os.path.join(REPO, "out", "act_demos"))
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args(argv)
    root = args.dir if os.path.isabs(args.dir) else os.path.join(REPO, args.dir)

    ck = Checks()
    index_path = os.path.join(root, "index.json")
    assert os.path.isfile(index_path), "%s: missing index.json" % root
    with open(index_path) as fh:
        index = json.load(fh)
    ck.ok(set(index) == INDEX_KEYS,
          "index.json keys %s != %s" % (sorted(set(index)), sorted(INDEX_KEYS)))
    ck.ok(index["task"] == "pick up the red block and lift it", "index.task %r" % index["task"])
    ck.ok(index["camera_names"] == list(CAMERAS), "index.camera_names %r" % index["camera_names"])
    ck.ok(index["state_dim"] == 7 and index["action_dim"] == 7,
          "index.state_dim/action_dim must both be 7")
    ck.ok(isinstance(index["seed"], int), "index.seed is not an int")
    ck.ok(float(index["fps"]) == float(index["control_hz"]), "index.fps != index.control_hz")
    ck.ok(isinstance(index["randomization"], str) and "cube xy" in index["randomization"],
          "index.randomization %r" % index["randomization"])
    ck.ok(index["generated_by"] == "experiments/gen_act_dataset.py",
          "index.generated_by %r" % index["generated_by"])
    ck.ok(index["mujoco_version"].startswith("3.16"), "index.mujoco_version %r"
          % index["mujoco_version"])

    eps = sorted(d for d in os.listdir(root)
                 if d.startswith("ep") and os.path.isdir(os.path.join(root, d)))
    ck.ok(eps == ["ep%03d" % k for k in range(args.episodes)],
          "episode directories %s != ep000..ep%03d" % (eps, args.episodes - 1))
    ck.ok(int(index["num_episodes"]) == len(eps),
          "index.num_episodes %r != %d directories found" % (index["num_episodes"], len(eps)))

    for k, name in enumerate(eps):
        ep_dir = os.path.join(root, name)
        check_episode(ck, ep_dir, index)
        with open(os.path.join(ep_dir, "meta.json")) as fh:
            meta = json.load(fh)
        ck.ok(meta["episode"] == k, "%s: meta.episode %r != %d" % (ep_dir, meta["episode"], k))
        print("  %s  %3d frames  rise %.5f m  slip %.6f m  cube xy %+0.3f %+0.3f"
              % (name, int(meta["num_frames"]), float(meta["rise_m"]), float(meta["slip_m"]),
                 float(meta["cube_pos"][0]), float(meta["cube_pos"][1])))

    print("%d checks over %d episodes in %s" % (ck.count, len(eps), root))
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

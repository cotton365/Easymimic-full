"""Acceptance test for the hand input layer (easymimic_repro/src/hand_input/wrist_frame.py).

Pure asserts, no pytest: numpy only. It must run where no camera stack is installed, so
nothing here imports cv2, mediapipe or mujoco -- section 1 proves that stays true:

    ~/miniconda3/envs/mujoco/bin/python easymimic_repro/tests_ee/test_hand_input.py

Sections
  1. import hygiene           -> importing the layer pulls in no cv2 / mediapipe
  2. image_points             -> normalized -> pixel triple, x by width, y by height
  3. metric_scale             -> world-landmark span vs the palm-width prior
  4. build round trip         -> metric cloud -> landmarks -> metric cloud
  5. wrist_rotation           -> R^T R = I, det = +1, X fingers / Z palm normal
  6. degeneracies             -> collinear palm, zero bones, NaN, wrong shape
  7. invalid_frame contract   -> NaN geometry, valid False, never zeros
  8. FrameSmoother            -> EMA, state survives a lost frame, reset, alpha guard
  9. validators               -> as_landmarks / as_image_size reject bad input
 10. build robustness         -> bad landmark count or image size -> invalid, no raise
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.hand_input import wrist_frame as W          # noqa: E402

WIDTH, HEIGHT = 640, 480

failures = []


def check(name, condition, detail=""):
    print("  [%s] %s%s" % ("PASS" if condition else "FAIL", name,
                           "" if not detail else "   " + detail))
    if not condition:
        failures.append(name)


def rodrigues(rotvec):
    """Axis-angle -> rotation matrix (kept local so the test needs numpy only)."""
    v = np.asarray(rotvec, dtype=float)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def make_hand(rotation=None, palm_scale=1.0):
    """21 metric keypoints in the MANO layout: palm in the z = 0 plane, fingers along +x.

    X_wrist = +x (wrist -> middle MCP), palm normal = +z, so a cloud built this way has
    R_wrist = I and any known rotation applied to it must come straight back out.
    """
    p = np.zeros((W.NUM_LANDMARKS, 3))
    mcps = {W.INDEX_MCP_IDX: (0.060, -0.035), W.MIDDLE_MCP_IDX: (0.085, 0.0),
            13: (0.078, 0.022), W.PINKY_MCP_IDX: (0.055, 0.035)}
    chains = {W.INDEX_MCP_IDX: (6, 7, 8), W.MIDDLE_MCP_IDX: (10, 11, 12),
              13: (14, 15, 16), W.PINKY_MCP_IDX: (18, 19, 20)}
    for base, rest in chains.items():
        x0, y0 = mcps[base]
        p[base] = (x0, y0, 0.0)
        for k, idx in enumerate(rest):
            p[idx] = (x0 + 0.025 * (k + 1), y0 + 0.004 * (k + 1), 0.0)
    for k, idx in enumerate((1, 2, 3, 4)):            # thumb, lifting off the palm plane
        p[idx] = (0.020 + 0.018 * k, -0.045 - 0.005 * k, 0.010 * k)
    p *= float(palm_scale)
    if rotation is not None:
        p = p @ np.asarray(rotation, dtype=float).T
    return p


def to_landmarks(points, width=WIDTH, height=HEIGHT):
    """Metric points -> the normalized triples MediaPipe reports (inverse of image_points)."""
    px = np.asarray(points, dtype=float)
    return [(px[i, 0] / width + 0.5, px[i, 1] / height + 0.5, px[i, 2] / width)
            for i in range(W.NUM_LANDMARKS)]


def palm_span(points):
    return float(np.linalg.norm(np.asarray(points)[W.INDEX_MCP_IDX]
                                - np.asarray(points)[W.PINKY_MCP_IDX]))

def test_1_import_hygiene():
    print("\n  [import hygiene]")
    loaded = [name for name in ("cv2", "mediapipe", "mujoco") if name in sys.modules]
    check("importing the layer pulls in no camera stack", not loaded,
          "loaded: %s" % (loaded or "none"))
    check("wrist_frame exposes the adapter entry points",
          all(hasattr(W, attr) for attr in ("build", "frame_from_points", "image_points",
                                            "metric_scale", "wrist_rotation", "invalid_frame",
                                            "FrameSmoother", "WristFrame", "PALM_WIDTH_M")))


def test_2_image_points():
    print("\n  [image_points]")
    hand = make_hand()
    lm = to_landmarks(hand)
    px = W.image_points(lm, (WIDTH, HEIGHT))
    check("image_points inverts the MediaPipe normalization",
          np.allclose(px, hand, atol=1e-12),
          "max|d| = %.1e m" % np.max(np.abs(px - hand)))
    center = W.image_points([(0.5, 0.5, 0.0)] * W.NUM_LANDMARKS, (WIDTH, HEIGHT))
    check("the image center maps to the origin", np.allclose(center, 0.0, atol=0.0))
    corner = W.image_points([(1.0, 1.0, 0.1)] * W.NUM_LANDMARKS, (WIDTH, HEIGHT))[0]
    check("x is normalized by width and y by height",
          np.allclose(corner, [WIDTH / 2.0, HEIGHT / 2.0, 0.1 * WIDTH], atol=1e-12),
          "[%.1f, %.1f, %.1f] px" % tuple(corner))


def test_3_metric_scale():
    print("\n  [metric_scale]")
    hand = make_hand()
    px = W.image_points(to_landmarks(hand), (WIDTH, HEIGHT))
    span = palm_span(hand)
    check("world landmarks give metres per pixel from the 3-D palm width",
          np.isclose(W.metric_scale(px, hand), span / palm_span(px), rtol=1e-12),
          "%.4f m/px" % W.metric_scale(px, hand))
    check("without world landmarks the palm width prior is used",
          np.isclose(W.metric_scale(px, None), W.PALM_WIDTH_M / span, rtol=1e-12),
          "%.4f m/px = %.3f m / %.5f m" % (W.metric_scale(px, None), W.PALM_WIDTH_M, span))
    scaled = W.metric_scale(px, make_hand(palm_scale=2.0))
    check("a hand twice as wide doubles the scale exactly",
          np.isclose(scaled, 2.0 * W.metric_scale(px, hand), rtol=1e-12),
          "%.4f m/px vs %.4f m/px" % (scaled, W.metric_scale(px, hand)))
    dead = np.zeros((W.NUM_LANDMARKS, 3))
    try:
        W.metric_scale(dead, None)
        raised = False
    except ValueError:
        raised = True
    check("a degenerate palm span raises instead of returning inf", raised)


def test_4_build_round_trip():
    print("\n  [build]")
    hand = make_hand()
    lm = to_landmarks(hand)
    frame = W.build(lm, (WIDTH, HEIGHT), hand)
    check("build rebuilds the metric cloud from landmarks + world landmarks",
          frame.valid and np.allclose(frame.keypoints, hand, atol=1e-12),
          "max|d| = %.1e m" % np.max(np.abs(frame.keypoints - hand)))
    check("a metric cloud has scale 1.0", np.isclose(frame.scale, 1.0, rtol=1e-12))
    prior = W.build(lm, (WIDTH, HEIGHT), None)
    check("without world landmarks the cloud is rescaled to the palm-width prior",
          prior.valid and np.isclose(palm_span(prior.keypoints), W.PALM_WIDTH_M, rtol=1e-12),
          "palm span = %.5f m" % palm_span(prior.keypoints))
    shape_err = float(np.max(np.abs(prior.keypoints / palm_span(prior.keypoints)
                                  - hand / palm_span(hand))))
    check("rescaling keeps the hand shape", shape_err < 1e-11,
          "max|d| = %.1e (relative)" % shape_err)
    check("frame_from_points marks a good cloud valid",
          W.frame_from_points(hand, 1.0).valid)


def test_5_wrist_rotation():
    print("\n  [wrist_rotation]")
    hand = make_hand()
    R = W.wrist_rotation(hand)
    check("the reference hand has R_wrist = I", np.allclose(R, np.eye(3), atol=1e-12),
          "max|dR| = %.1e" % np.max(np.abs(R - np.eye(3))))
    check("R^T R = I", np.allclose(R.T @ R, np.eye(3), atol=1e-12))
    check("det R = +1", np.isclose(float(np.linalg.det(R)), 1.0, rtol=1e-12),
          "det = %.12f" % float(np.linalg.det(R)))
    check("column 2 is the palm normal",
          np.allclose(R[:, 2], np.cross(hand[W.INDEX_MCP_IDX] - hand[W.WRIST_IDX],
                                        hand[W.PINKY_MCP_IDX] - hand[W.WRIST_IDX])
                       / np.linalg.norm(np.cross(hand[W.INDEX_MCP_IDX] - hand[W.WRIST_IDX],
                                                 hand[W.PINKY_MCP_IDX] - hand[W.WRIST_IDX])),
                      atol=1e-12))
    check("column 0 follows the fingers, column 1 closes the frame",
          np.dot(R[:, 0], R[:, 1]) < 1e-12 and R[0, 0] > 0.99,
          "X = [%.3f, %.3f, %.3f]" % tuple(R[:, 0]))
    R_test = rodrigues([0.3, -0.2, 0.9])
    check("R_wrist is rotation-equivariant",
          np.allclose(W.wrist_rotation(make_hand(rotation=R_test)), R_test, atol=1e-12),
          "max|dR| = %.1e" % np.max(np.abs(W.wrist_rotation(make_hand(rotation=R_test)) - R_test)))

def test_6_degeneracies():
    print("\n  [degeneracies]")
    hand = make_hand()

    collinear = hand.copy()
    collinear[W.PINKY_MCP_IDX] = 0.5 * (collinear[W.WRIST_IDX] + collinear[W.INDEX_MCP_IDX])
    check("collinear palm bones are rejected",
          not W.build(to_landmarks(collinear), (WIDTH, HEIGHT), collinear).valid)

    flat = hand.copy()
    flat[W.PINKY_MCP_IDX] = flat[W.WRIST_IDX]
    check("a zero-length palm bone is rejected",
          not W.build(to_landmarks(flat), (WIDTH, HEIGHT), flat).valid)

    along_normal = hand.copy()
    normal = W.wrist_rotation(hand)[:, 2]
    along_normal[W.MIDDLE_MCP_IDX] = along_normal[W.WRIST_IDX] + 0.01 * normal
    check("fingers parallel to the palm normal are rejected", not W.build(
        to_landmarks(along_normal), (WIDTH, HEIGHT), along_normal).valid)

    nan_hand = hand.copy()
    nan_hand[7, 1] = np.nan
    frame = W.build(to_landmarks(nan_hand), (WIDTH, HEIGHT), nan_hand)
    check("a NaN landmark is caught before any geometry runs",
          not frame.valid and "non-finite" in frame.reason, frame.reason)
    check("wrist_rotation raises on a bad shape",
          _raises(lambda: W.wrist_rotation(np.zeros((20, 3))), ValueError))


def test_7_invalid_frame():
    print("\n  [invalid_frame]")
    frame = W.invalid_frame("simulated loss")
    check("invalid_frame is NaN geometry, never zeros",
          np.all(np.isnan(frame.keypoints)) and np.all(np.isnan(frame.R_wrist))
          and not frame.valid and frame.reason == "simulated loss")
    check("its scale is NaN, so nothing downstream can silently use it",
          np.isnan(frame.scale))
    check("palm_normal of an invalid frame is NaN",
          np.all(np.isnan(frame.palm_normal)))
    built = W.build([(np.nan, 0.0, 0.0)] * W.NUM_LANDMARKS, (WIDTH, HEIGHT))
    check("build never returns None for an unusable hand", built is not None
          and not built.valid and np.all(np.isnan(built.keypoints)))


def test_8_smoother():
    print("\n  [FrameSmoother]")
    hand = make_hand()
    shifted = hand + np.array([0.02, 0.0, 0.0])
    f0 = W.frame_from_points(hand, 1.0)
    f1 = W.frame_from_points(shifted, 1.0)

    passthrough = W.FrameSmoother(alpha=1.0).update(f0)
    check("alpha = 1 passes the cloud through unsmoothed",
          np.allclose(passthrough.keypoints, hand, atol=0.0))

    smoother = W.FrameSmoother(alpha=0.5)
    smoother.update(f0)
    half = smoother.update(f1)
    check("alpha = 0.5 moves half way to the new cloud",
          np.allclose(half.keypoints, 0.5 * (hand + shifted), atol=1e-12),
          "max|d| = %.1e m" % np.max(np.abs(half.keypoints - 0.5 * (hand + shifted))))
    check("the smoothed cloud still has a proper R_wrist",
          np.allclose(half.R_wrist.T @ half.R_wrist, np.eye(3), atol=1e-12)
          and np.isclose(float(np.linalg.det(half.R_wrist)), 1.0, rtol=1e-12))

    lost = smoother.update(W.invalid_frame("lost"))
    check("a lost frame comes back invalid instead of being smoothed", not lost.valid)
    after = smoother.update(f1)
    expected = 0.25 * hand + 0.75 * shifted      # unchanged state, then one more half step
    check("the smoother state survives a lost frame",
          np.allclose(after.keypoints, expected, atol=1e-12),
          "max|d| = %.1e m" % np.max(np.abs(after.keypoints - expected)))
    check("update(None) returns an invalid frame", not smoother.update(None).valid)

    smoother.reset()
    check("reset forgets the state", np.allclose(smoother.update(f1).keypoints, shifted,
                                                 atol=0.0))
    check("alpha outside (0, 1] is rejected",
          _raises(lambda: W.FrameSmoother(alpha=0.0), ValueError)
          and _raises(lambda: W.FrameSmoother(alpha=1.5), ValueError))


def test_9_validators():
    print("\n  [validators]")
    hand = make_hand()
    check("as_landmarks passes a good array through",
          np.array_equal(W.as_landmarks(hand), hand))
    check("as_landmarks rejects the wrong count",
          _raises(lambda: W.as_landmarks(np.zeros((20, 3))), ValueError))
    check("as_landmarks rejects a (21, 2) array",
          _raises(lambda: W.as_landmarks(np.zeros((21, 2))), ValueError))
    check("as_image_size returns floats",
          W.as_image_size((WIDTH, HEIGHT)) == (float(WIDTH), float(HEIGHT)))
    check("as_image_size rejects a zero or negative size",
          _raises(lambda: W.as_image_size((0, HEIGHT)), ValueError)
          and _raises(lambda: W.as_image_size((WIDTH, -1)), ValueError)
          and _raises(lambda: W.as_image_size((WIDTH,)), ValueError))


def test_10_build_robustness():
    print("\n  [build robustness]")
    hand = make_hand()
    short = to_landmarks(hand)[:20]
    frame = W.build(short, (WIDTH, HEIGHT))
    check("too few landmarks give an invalid frame, not an exception",
          not frame.valid and frame.reason != "", frame.reason)
    frame = W.build(to_landmarks(hand), (0, HEIGHT))
    check("a zero image size gives an invalid frame", not frame.valid and frame.reason != "",
          frame.reason)


def _raises(fn, exc_type):
    try:
        fn()
    except exc_type:
        return True
    except Exception:                                   # noqa: BLE001
        return False
    return False


def main():
    print("=" * 78)
    print("hand input layer: landmark -> metric cloud + R_wrist (numpy only)")
    print("=" * 78)
    t0 = __import__("time").monotonic()
    test_1_import_hygiene()
    test_2_image_points()
    test_3_metric_scale()
    test_4_build_round_trip()
    test_5_wrist_rotation()
    test_6_degeneracies()
    test_7_invalid_frame()
    test_8_smoother()
    test_9_validators()
    test_10_build_robustness()

    print("\n" + "=" * 78)
    if failures:
        print("FAILED %d check(s):" % len(failures))
        for name in failures:
            print("  - %s" % name)
        return 1
    print("ALL CHECKS PASSED   (%.2f s)" % (__import__("time").monotonic() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

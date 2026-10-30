"""Camera + MediaPipe Hands wrapper: the only module that knows about cv2 / mediapipe.

Both dependencies are imported lazily inside the functions that need them, so
--selftest and the numpy unit tests stay runnable in an environment that has neither.
A missing package raises MissingDependency carrying the exact pip command for this
interpreter, instead of an ImportError traceback the user has to decode.

Threading model: one reader thread owns the camera handle and runs detection; the control
loop only ever reads the newest completed sample under a lock. A slow control tick
therefore drops frames instead of building a latency queue -- for teleoperation the fresh
frame matters, and a queued one is worse than useless.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import wrist_frame

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
MAX_READ_FAILURES = 30          # consecutive failed grabs before the reader gives up
READ_BACKOFF_S = 0.05


class MissingDependency(RuntimeError):
    """A camera-stack package needed here is not importable."""


class CameraError(RuntimeError):
    """The camera itself could not be opened or is not delivering frames."""


def pip_hint(packages="mediapipe opencv-python"):
    return "%s -m pip install %s" % (sys.executable, packages)


def import_cv2():
    try:
        import cv2
    except Exception as exc:                                # noqa: BLE001 - any import failure
        raise MissingDependency(
            "无法导入 OpenCV（%s）。\n"
            "  安装：  %s\n"
            "  说明：  脚本不联网、不自动安装；装好后先跑 --probe 验证摄像头链路。"
            % (exc, pip_hint())) from exc
    return cv2


def import_mediapipe():
    try:
        import mediapipe as mp
    except Exception as exc:                                # noqa: BLE001
        raise MissingDependency(
            "无法导入 mediapipe（%s）。\n"
            "  安装：  %s\n"
            "  说明：  python %s / numpy %s。若报错指向 numpy，mediapipe 官方 wheel 通常要 numpy<2。\n"
            "  装好后先跑 --probe 验证摄像头链路。"
            % (exc, pip_hint(), sys.version.split()[0], np.__version__)) from exc
    return mp


@dataclass(frozen=True)
class HandSample:
    """One detection result; landmarks are MediaPipe-normalized (x, y in [0, 1], z ~ width)."""

    landmarks: object          # list of 21 (x, y, z), or None when no hand was found
    world_landmarks: object    # list of 21 (x, y, z) in metres, or None
    frame: object              # BGR uint8 image, or None when keep_frame is off
    width: int
    height: int
    mono: float                # time.monotonic() of the capture
    seq: int

    @property
    def found(self):
        return self.landmarks is not None and len(self.landmarks) == wrist_frame.NUM_LANDMARKS

    @property
    def image_size(self):
        return (self.width, self.height)


def open_camera(cv2, index, width, height):
    """Open VideoCapture and prove it delivers: a camera that opens but never reads is common."""
    cap = cv2.VideoCapture(int(index))
    if not cap.isOpened():
        raise CameraError(
            "打不开摄像头 index=%d。\n"
            "  排查：  1) 确认没有别的程序（浏览器 / 会议软件）占着相机；\n"
            "          2) 确认设备节点存在： ls -l /dev/video*\n"
            "          3) 换一个 index 试试： --camera 1\n"
            "          4) Linux 下确认当前用户在 video 组里： groups" % int(index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise CameraError("摄像头 index=%d 已打开但读不出第一帧（被占用或被驱动拒绝）。" % int(index))
    return cap


class HandTracker:
    """One mediapipe Hands graph: video mode, single hand."""

    def __init__(self, max_num_hands=1, min_detection_confidence=0.6,
                 min_tracking_confidence=0.5, model_complexity=1):
        self._mp = import_mediapipe()
        self._hands = None
        try:
            self._hands = self._mp.solutions.hands.Hands(
                static_image_mode=False,        # frame-to-frame tracking is what keeps it smooth
                max_num_hands=int(max_num_hands),
                model_complexity=int(model_complexity),
                min_detection_confidence=float(min_detection_confidence),
                min_tracking_confidence=float(min_tracking_confidence))
        except Exception as exc:                # noqa: BLE001
            raise MissingDependency(
                "mediapipe 已安装，但 Hands 图构建失败（%s）。\n"
                "  常见原因：numpy 2.x ABI 不匹配 / wheel 与当前平台不符。\n"
                "  安装：  %s" % (exc, pip_hint())) from exc

    @property
    def version(self):
        return getattr(self._mp, "__version__", "unknown")

    def detect(self, rgb):
        # mediapipe 0.10.x legacy solutions derive the frame timestamp themselves; the
        # older process(image, timestamp_ms) form was removed from this signature.
        result = self._hands.process(np.ascontiguousarray(rgb))
        if not result.multi_hand_landmarks:
            return None, None
        landmarks = [(lm.x, lm.y, lm.z) for lm in result.multi_hand_landmarks[0].landmark]
        world = None
        if result.multi_hand_world_landmarks:
            world = [(lm.x, lm.y, lm.z) for lm in result.multi_hand_world_landmarks[0].landmark]
        return landmarks, world

    def close(self):
        if self._hands is not None:
            try:
                self._hands.close()
            finally:
                self._hands = None


class CameraHandSource:
    """Camera + tracker on a reader thread; latest always holds the newest sample."""

    def __init__(self, camera_index=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                 keep_frame=True, tracker_factory=HandTracker):
        self.camera_index = int(camera_index)
        self.requested_size = (int(width), int(height))
        self.keep_frame = bool(keep_frame)
        self._tracker_factory = tracker_factory
        self._cv2 = None
        self._cap = None
        self._tracker = None
        self._thread = None
        self._cv2_factory = import_cv2
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._sample = None
        self._seq = 0
        self.frames = 0
        self.detected = 0
        self.error = ""

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._cv2 = self._cv2_factory()
        self._cap = open_camera(self._cv2, self.camera_index, *self.requested_size)
        try:
            self._tracker = self._tracker_factory()
        except BaseException:
            self._cap.release()
            self._cap = None
            raise
        self._thread = threading.Thread(target=self._run, name="camera-hand", daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()             # release anyone blocked in wait_for_sample
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)      # the reader checks the event between frames
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.close()
        return False

    # -- data --------------------------------------------------------------
    @property
    def latest(self):
        with self._cond:
            return self._sample

    @property
    def version(self):
        if self._tracker is not None:
            return self._tracker.version
        return "unknown"

    @property
    def cv2_version(self):
        return getattr(self._cv2, "__version__", "unknown") if self._cv2 is not None else "unknown"

    @property
    def size(self):
        sample = self.latest
        return sample.image_size if sample is not None else self.requested_size

    def wait_for_sample(self, seq, timeout=0.5):
        """Block until a sample newer than seq exists (or the timeout / shutdown)."""
        with self._cond:
            ready = self._cond.wait_for(
                lambda: self._stop.is_set() or (self._sample is not None and self._sample.seq != seq),
                float(timeout))
            return self._sample if ready and not self._stop.is_set() else None

    def _publish(self, sample):
        with self._cond:
            self._sample = sample
            self._cond.notify_all()

    # -- reader thread -----------------------------------------------------
    def _run(self):
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    self.error = "摄像头连续 %d 次读帧失败，读线程退出。" % failures
                    with self._cond:
                        self._cond.notify_all()
                    return
                self._stop.wait(READ_BACKOFF_S)     # genuine device backoff, not a polling loop
                continue
            failures = 0
            mono = time.monotonic()
            rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            try:
                landmarks, world = self._tracker.detect(rgb)
            except Exception as exc:                # noqa: BLE001
                self.error = "mediapipe 推理失败：%s" % exc
                with self._cond:
                    self._cond.notify_all()
                return
            self.frames += 1
            if landmarks is not None:
                self.detected += 1
            self._seq += 1
            self._publish(HandSample(
                landmarks=landmarks, world_landmarks=world,
                frame=(frame if self.keep_frame else None),
                width=int(frame.shape[1]), height=int(frame.shape[0]),
                mono=mono, seq=self._seq))


def build_frame(sample, smoother=None):
    """HandSample -> WristFrame; an absent hand becomes an invalid frame, never None."""
    if sample is None:
        return wrist_frame.invalid_frame("no sample")
    if not sample.found:
        return wrist_frame.invalid_frame("no hand in frame")
    frame = wrist_frame.build(sample.landmarks, sample.image_size, sample.world_landmarks)
    return smoother.update(frame) if smoother is not None else frame


# --------------------------------------------------------------------------- #
# connection probe: prove camera -> mediapipe -> landmarks before touching the robot
# --------------------------------------------------------------------------- #
def probe(camera_index=0, seconds=3.0, preview=False, width=DEFAULT_WIDTH,
          height=DEFAULT_HEIGHT, source_factory=CameraHandSource):
    """Watch the camera for the given seconds and report what the hand pipeline sees."""
    print("摄像头链路诊断：index=%d, %.1f s" % (camera_index, float(seconds)))
    source = source_factory(camera_index, width, height, keep_frame=bool(preview))
    try:
        source.start()
    except (MissingDependency, CameraError) as exc:
        print("  [FAIL] %s" % exc)
        return 1
    cv2 = source._cv2
    print("  mediapipe %s / opencv %s" % (source.version, cv2.__version__))

    xs, ys, zs, spans, scales = [], [], [], [], []
    smoothed = wrist_frame.FrameSmoother(alpha=0.35)
    deadline = time.monotonic() + float(seconds)
    seq = -1
    try:
        while time.monotonic() < deadline:
            sample = source.wait_for_sample(seq, timeout=0.5)
            if sample is None or sample.seq == seq:
                if source.error:
                    break
                continue
            seq = sample.seq
            if sample.found:
                arr = np.asarray(sample.landmarks, dtype=np.float64)
                xs.append(arr[:, 0]); ys.append(arr[:, 1]); zs.append(arr[:, 2])
                px = wrist_frame.image_points(arr, sample.image_size)
                spans.append(float(np.linalg.norm(px[wrist_frame.INDEX_MCP_IDX]
                                                  - px[wrist_frame.PINKY_MCP_IDX])))
                frame = build_frame(sample, smoothed)
                if frame.valid:
                    scales.append(frame.scale)
            if preview and sample.frame is not None:
                cv2.imshow("hand_teleop probe", sample.frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        print("  (中断)")
    finally:
        frames, detected, error = source.frames, source.detected, source.error
        source.close()
        if preview:
            cv2.destroyAllWindows()

    print("  帧数 %d，检测到手 %d 帧 (%.0f%%)"
          % (frames, detected, (100.0 * detected / frames) if frames else 0.0))
    if error:
        print("  [FAIL] %s" % error)
    if not xs:
        print("  [FAIL] 没有检测到手：把手掌正对镜头、光线充足，再试一次。")
        return 1
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    print("  landmark 范围  x [%.3f, %.3f]  y [%.3f, %.3f]  z [%.4f, %.4f]"
          % (x.min(), x.max(), y.min(), y.max(), z.min(), z.max()))
    scale_med = float(np.median(scales)) if scales else float("nan")
    print("  手掌跨度 %.0f px，估算尺度 %.5f m/px (~%.1f mm/px)"
          % (float(np.median(spans)), scale_med, 1000.0 * scale_med))
    print("  RESULT: CAMERA CHAIN OK")
    return 0

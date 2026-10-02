"""Webcam capture with live face tracking."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .face import FaceObs, FaceTracker
from .paths import LINUX, WHO, WINDOWS


@dataclass
class Recording:
    started: float
    ts: list[float] = field(default_factory=list)
    grays: list[np.ndarray] = field(default_factory=list)
    anchors: list["np.ndarray | None"] = field(default_factory=list)
    mouth_open: list[float] = field(default_factory=list)

    def snapshot(self):
        n = min(len(self.ts), len(self.grays), len(self.anchors))
        return self.ts[:n], self.grays[:n], self.anchors[:n]

    @property
    def face_ratio(self) -> float:
        return sum(a is not None for a in self.anchors) / max(len(self.anchors), 1)

    @property
    def duration(self) -> float:
        return self.ts[-1] - self.ts[0] if len(self.ts) > 1 else 0.0


def list_cameras() -> list[dict]:
    if WINDOWS or LINUX:
        return []
    from AVFoundation import AVCaptureDevice, AVMediaTypeMuxed, AVMediaTypeVideo
    devs = list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)) + \
        list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeMuxed))
    devs.sort(key=lambda d: d.uniqueID())
    return [{"index": i, "name": str(d.localizedName()), "id": str(d.uniqueID()),
             "builtin": "BuiltIn" in str(d.deviceType())} for i, d in enumerate(devs)]


def resolve_camera(pref) -> "int | str":
    if isinstance(pref, int) or (isinstance(pref, str) and os.path.exists(pref)):
        return pref
    cams = list_cameras()
    if not cams:
        return 0
    if pref and pref != "auto":
        for c in cams:
            if pref == c["id"] or pref.lower() in c["name"].lower():
                return c["index"]
    for c in cams:
        if c["builtin"]:
            return c["index"]
    non_phone = [c for c in cams if "iphone" not in c["name"].lower()]
    return (non_phone or cams)[0]["index"]


class Camera:
    def __init__(self, index: "int | str" = "auto", width: int = 1280, height: int = 720, idle_close: float = 45.0,
                 on_frame=None):
        self.index, self.width, self.height = index, width, height
        self.idle_close = idle_close
        self.on_frame = on_frame
        self._cap = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._rec: Recording | None = None
        self._last_used = time.time()
        self._tracker: FaceTracker | None = None
        self.error: str | None = None
        self.ready = threading.Event()
        self.track_always = False

    def ensure_open(self):
        self._last_used = time.time()
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.ready.clear()
        self._thread = threading.Thread(target=self._run, name="lipflow-camera", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def is_open(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start_recording(self) -> Recording:
        self.ensure_open()
        with self._lock:
            self._rec = Recording(started=time.time())
            return self._rec

    def stop_recording(self) -> "Recording | None":
        with self._lock:
            rec, self._rec = self._rec, None
        self._last_used = time.time()
        return rec

    @property
    def recording(self) -> "Recording | None":
        return self._rec

    def set_source(self, pref):
        self.index = pref
        self.close()

    def _open(self):
        if isinstance(self.index, str) and os.path.exists(self.index):
            cap = cv2.VideoCapture(self.index)
            self._file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {self.index}")
            return cap
        self._file_fps = None
        idx = resolve_camera(self.index)
        if WINDOWS:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        elif LINUX:
            cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            if WINDOWS:
                raise RuntimeError("Could not open the camera. Turn on Settings → Privacy & security → Camera → "
                                   "Let desktop apps access your camera, and close other apps using it")
            if LINUX:
                raise RuntimeError("Could not open the camera. Check PipeWire/v4l2 and close other apps using it")
            raise RuntimeError(f"Could not open the camera. Allow {WHO} in "
                               "Settings → Privacy & Security → Camera")
        return cap

    def _run(self):
        try:
            self._cap = self._open()
            if self._tracker is None:
                self._tracker = FaceTracker()
            self.error = None
        except Exception as e:
            self.error = str(e)
            print(f"[camera] {e}")
            return
        t0 = time.time()
        warm = n_read = 0
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if self._file_fps:
                    n_read += 1
                    time.sleep(max(0.0, t0 + n_read / self._file_fps - time.time()))
                now = time.time()
                if not ok:
                    time.sleep(0.01)
                    continue
                warm += 1
                if warm == 3:
                    self.ready.set()
                rec = self._rec
                obs: FaceObs | None = None
                if rec is not None or self.track_always:
                    obs = self._tracker.detect(frame, int((now - t0) * 1000))
                if rec is not None and warm >= 3:
                    with self._lock:
                        if self._rec is rec:
                            rec.ts.append(now)
                            rec.grays.append(face_crop(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), obs, rec))
                            rec.anchors.append(obs.anchors if obs else None)
                            rec.mouth_open.append(obs.mouth_open if obs else 0.0)
                if self.on_frame is not None:
                    try:
                        self.on_frame(frame, obs, rec is not None)
                    except Exception as e:
                        print(f"[camera] on_frame: {e}")
                if rec is None and not self.track_always and now - self._last_used > self.idle_close:
                    break
        finally:
            self._cap.release()
            self._cap = None
            self.ready.clear()


def face_crop(gray: np.ndarray, obs: "FaceObs | None", rec: "Recording") -> tuple:
    h, w = gray.shape
    if obs is not None:
        x0, y0 = obs.pts.min(0)
        x1, y1 = obs.pts.max(0)
        cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.5
        rec._box = (int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)),
                    int(min(cx + side / 2, w)), int(min(cy + side / 2, h)))
    box = getattr(rec, "_box", None) or (0, 0, w, h)
    bx0, by0, bx1, by1 = box
    return np.ascontiguousarray(gray[by0:by1, bx0:bx1]), (bx0, by0)


def mouth_thumbnail(frame_bgr: np.ndarray, obs: "FaceObs | None", size: int = 112) -> "np.ndarray | None":
    if obs is None:
        return None
    lips = obs.outer_lips
    cx, cy = lips.mean(0)
    half = max(np.ptp(lips[:, 0]), 1) * 0.95
    h, w = frame_bgr.shape[:2]
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, w))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = cv2.resize(frame_bgr[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    return cv2.flip(crop, 1)


PINK = (115, 92, 250)
PINK_SOFT = (170, 150, 255)


def mouth_view(frame_bgr: np.ndarray, obs: "FaceObs | None", w: int = 240, h: int = 150) -> np.ndarray:
    fh, fw = frame_bgr.shape[:2]
    if obs is None:
        view = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        view = cv2.flip((view * 0.45).astype(np.uint8), 1)
        cv2.putText(view, "looking for your face...", (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
        return view
    lips = obs.outer_lips
    cx, cy = lips.mean(0)
    half_w = max(np.ptp(lips[:, 0]), 10) * 0.9
    half_h = half_w * h / w
    x0, y0 = cx - half_w, cy - half_h
    scale = w / (2 * half_w)
    M = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    view = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    view = cv2.flip(view, 1)

    def to_view(pts):
        p = (pts - (x0, y0)) * scale
        p[:, 0] = w - 1 - p[:, 0]
        return p

    overlay = view.copy()
    for contour in (obs.outer_lips, obs.inner_lips):
        cv2.polylines(overlay, [np.round(to_view(contour) * 4).astype(np.int32)], True, PINK_SOFT, 1,
                      cv2.LINE_AA, shift=2)
    view = cv2.addWeighted(overlay, 0.7, view, 0.3, 0)
    for x, y in to_view(obs.lip_points):
        cv2.circle(view, (int(round(x * 4)), int(round(y * 4))), 8, PINK, -1, cv2.LINE_AA, shift=2)
    return view

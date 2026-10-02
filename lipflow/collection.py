"""Human-operated, image-only collection of independently labelled silent speech.

No VSR model, microphone, audio stream, cloud service or model prediction is used.
Unconfirmed video is temporary and deleted on discard/normal exit. Only a human
confirmation publishes a video and its manifest row. UI overlays never enter it.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np

from .confidence import Quality
from .dictation import clip_problem
from .evaluation import characters

FPS = 25
WINDOW = 'Lipflow silent speech collection'


def _dataset(manifest: Path, split: str) -> dict:
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('split') != split or not isinstance(data.get('samples'), list):
            raise ValueError('Use separate output directories for train, dev and test; existing manifest is incompatible')
        return data
    return {'schema_version': 1, 'split': split, 'training_overlap_checked': False,
            'description': 'Operator-confirmed silent webcam recordings; original image frames, no audio.',
            'development_samples': [], 'samples': []}


def _atomic_json(path: Path, data: dict):
    descriptor, temporary = tempfile.mkstemp(prefix='.manifest-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def directory_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob('*') if path.is_file() and not path.is_symlink())


class Recording:
    """Bounded streaming nearest-neighbour25fps recording, with one image buffer."""
    def __init__(self, directory: Path, max_bytes: int, writer_factory=None, tracking_available=True):
        self.directory, self.max_bytes = directory, max_bytes
        self.base_bytes = directory_bytes(directory)
        descriptor, name = tempfile.mkstemp(prefix='.unconfirmed-', suffix='.mp4', dir=directory)
        os.close(descriptor)
        self.path = Path(name)
        self.factory = writer_factory or cv2.VideoWriter
        self.writer = None
        self.previous = None
        self.shape = None
        self.ts, self.mouth_open, self.mouth_pixels, self.face_found = [], [], [], []
        self.frames_written = self.roi_pixels = 0
        self.roi_sum = self.roi_square_sum = 0.0
        self.finished = False
        self.tracking_available = tracking_available

    @property
    def duration(self):
        return self.ts[-1] - self.ts[0] if len(self.ts) > 1 else 0.0

    @property
    def face_ratio(self):
        return sum(self.face_found) / max(len(self.face_found), 1)

    def _write(self):
        self.writer.write(self.previous)
        self.frames_written += 1

    def push(self, frame: np.ndarray, timestamp: float, observation=None):
        if self.finished or not math.isfinite(timestamp) or self.ts and timestamp <= self.ts[-1]:
            raise ValueError('Recording timestamps must be finite and strictly increasing')
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Capture must contain original uint8 BGR images')
        if self.writer is None:
            self.shape = frame.shape
            self.writer = self.factory(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), FPS,
                                       (frame.shape[1], frame.shape[0]))
            if not self.writer.isOpened():
                raise RuntimeError('Cannot create MP4 video writer')
        elif frame.shape != self.shape:
            raise ValueError('Camera changed image dimensions during recording')
        if self.previous is not None:
            midpoint = (self.ts[-1] + timestamp) / 2
            while self.ts[0] + self.frames_written / FPS <= midpoint + 1e-9:
                self._write()
        self.previous = frame.copy()  # camera buffers can be reused after read()
        self.ts.append(timestamp)
        self.face_found.append(observation is not None)
        self.mouth_open.append(float(observation.mouth_open) if observation is not None else 0.0)
        self.mouth_pixels.append(float(np.ptp(observation.outer_lips[:, 0])) if observation is not None else 0.0)
        if observation is not None:
            # Small per-frame crops provide approximate live warnings; never save them.
            from .face import mouth_rois
            try:
                roi = mouth_rois([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)], [observation.anchors], window_margin=0)
                if roi is not None:
                    values = roi.astype(np.float64)
                    self.roi_sum += float(values.sum())
                    self.roi_square_sum += float(np.square(values).sum())
                    self.roi_pixels += values.size
            except (ValueError, cv2.error, TypeError):
                pass
        if self.base_bytes + self.path.stat().st_size > self.max_bytes:
            raise ValueError('Collection output quota exceeded')

    def finish(self):
        if self.finished:
            return
        if self.previous is None:
            raise ValueError('No frames recorded')
        while self.ts[0] + self.frames_written / FPS <= self.ts[-1] + 1e-9:
            self._write()
        self.writer.release()
        self.finished = True
        self.previous = None
        if directory_bytes(self.directory) > self.max_bytes:
            raise ValueError('Collection output quota exceeded')

    def discard(self):
        if self.writer is not None and not self.finished:
            self.writer.release()
        self.finished = True
        self.previous = None
        self.path.unlink(missing_ok=True)

    def evidence(self):
        mean = self.roi_sum / self.roi_pixels if self.roi_pixels else None
        std = math.sqrt(max(self.roi_square_sum / self.roi_pixels - mean * mean, 0)) if self.roi_pixels else None
        quality = Quality(self.face_ratio, mean if mean is not None else math.nan,
                          std if std is not None else math.nan,
                          float(np.median(self.mouth_pixels)) if self.mouth_pixels else 0)
        warnings = []
        if self.tracking_available:
            problem = clip_problem(self)
            if problem:
                warnings.append(': '.join(problem))
            if quality.problem():
                warnings.append(quality.problem())
        else:
            warnings.append('Face tracking unavailable; face/motion/lighting quality was not verified')
        return {'quality': {'face_ratio': self.face_ratio if self.tracking_available else None,
                            'brightness': mean, 'contrast': std,
                            'mouth_pixels': quality.mouth_pixels if self.tracking_available else None},
                'face_tracking_available': self.tracking_available,
                'quality_method': 'Approximate per-frame aligned mouth preview; no model inference',
                'mouth_motion_std': float(np.std([m for m in self.mouth_open if m > 0] or [0])),
                'warnings': warnings, 'capture_frames': len(self.ts), 'saved_frames': self.frames_written,
                'saved_fps': FPS, 'duration_seconds': self.duration,
                'timestamp_clock': 'time.monotonic (relative to first captured frame)',
                'capture_timestamps_seconds': [timestamp - self.ts[0] for timestamp in self.ts]}


def commit(recording: Recording, reference: str, speaker: str, session: str, split: str,
           *, label_verified=False, silent_verified=False) -> dict:
    """Publish only after both explicit human confirmations; no overwrite/label prediction."""
    if label_verified is not True or silent_verified is not True:
        raise ValueError('Human confirmation of the reference and silent articulation is required')
    if not characters(reference) or not speaker.strip() or not session.strip() or split not in ('train', 'dev', 'test'):
        raise ValueError('Reference, speaker, session and an explicit data split are required')
    recording.finish()
    output, final, published = recording.directory, None, False
    lock = output / '.collection.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError('Another collection update is active; a stale .collection.lock requires operator inspection') from None
    os.close(descriptor)
    try:
        manifest = output / 'manifest.json'
        data = _dataset(manifest, split)
        sample_id = uuid.uuid4().hex
        ids = {sample['id'] for sample in data['samples']}
        videos = output / 'videos'
        videos.mkdir(exist_ok=True)
        while sample_id in ids or (videos / f'{sample_id}.mp4').exists():
            sample_id = uuid.uuid4().hex
        final = videos / f'{sample_id}.mp4'
        digest = hashlib.sha256()
        with recording.path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        row = {'id': sample_id, 'video': final.relative_to(output).as_posix(), 'reference': reference,
               'speaker': speaker, 'session': session, 'split': split, 'label_verified': True,
               'label_source': 'Operator confirmed pre-recording reference against their own silent articulation',
               'domain': 'webcam', 'mouth_roi': False, 'articulation': 'silent', 'articulation_verified': True,
               'sha256': digest.hexdigest(), 'collection': recording.evidence()}
        data['samples'].append(row)
        data['training_overlap_checked'] = False  # a new sample invalidates any previous manual audit
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if directory_bytes(output) + len(encoded) + 4096 > recording.max_bytes:
            raise ValueError('Collection output quota exceeded')
        os.link(recording.path, final)  # atomic, fails if destination exists; never overwrites
        published = True
        recording.path.unlink()
        _atomic_json(manifest, data)
        return row
    except BaseException:
        if published:
            final.unlink(missing_ok=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


def _camera_permission():
    if sys.platform != 'darwin':
        return
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    if status == 0:
        completed, granted = threading.Event(), []
        def result(ok):
            granted.append(bool(ok))
            completed.set()
        AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, result)
        from Foundation import NSDate, NSRunLoop
        deadline = time.monotonic() + 30
        while not completed.is_set() and time.monotonic() < deadline:
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(.05))
        status = 3 if granted and granted[0] else 2
    if status != 3:
        raise PermissionError('Allow this terminal/Python app in macOS System Settings → Privacy & Security → Camera, then retry')


def _prompts(prompts):
    if prompts is None:
        return None
    lines = Path(prompts).read_text(encoding='utf-8').splitlines() if isinstance(prompts, (str, Path)) else list(prompts)
    references = [line.strip() for line in lines if line.strip()]
    if not references or any(not characters(line) for line in references):
        raise ValueError('Prompts must contain independently authored, nonempty reference sentences')
    return iter(references)


def _review_instructions(reference, recording):
    print('画面检查：', recording.evidence()['warnings'] or '未发现明显问题')
    print(f'按 Enter 确认“{reference}”与嘴型逐字一致，且未发声或耳语；否则 R 重录，Esc 丢弃。')


def collect(output, speaker, session, split='train', prompts=None, camera='auto', max_seconds=15, max_megabytes=512):
    """Collect only when a human runs this function, never implicitly as a test.

    SPACE start/stop; R discard/retake the same reference; Esc discard/quit;
    ENTER after stopping confirms exact reference fidelity AND no voiced/whispered
    speech. Loaded/entered references appear in the terminal, never recorded pixels.
    """
    if not isinstance(speaker, str) or not speaker.strip() or not isinstance(session, str) or not session.strip():
        raise ValueError('Supply explicit speaker and session identifiers; reuse the same session across its clips')
    if split not in ('train', 'dev', 'test'):
        raise ValueError('split must be train, dev or test')
    if not math.isfinite(max_seconds) or not .6 <= max_seconds <= 60 or not math.isfinite(max_megabytes) or max_megabytes <= 0:
        raise ValueError('max_seconds must be .6..60 and max_megabytes must be positive')
    if isinstance(camera, (str, Path)) and Path(camera).is_file():
        raise ValueError('Collection requires a physical webcam device; video files cannot be labelled as webcam data')
    references = _prompts(prompts)
    speaker, session = speaker.strip(), session.strip()
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'manifest.json'
    _dataset(manifest, split)
    max_bytes = int(max_megabytes * 1024 * 1024)
    if directory_bytes(directory) >= max_bytes:
        raise ValueError('Collection output quota exceeded')
    _camera_permission()  # native permission requested on this CLI's main thread
    from .camera import resolve_camera
    from .face import FaceTracker
    source = resolve_camera(int(camera) if isinstance(camera, str) and camera.isdecimal() else camera)
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
    tracker, recording = None, None
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam; check camera permissions and close other camera applications')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            tracker = FaceTracker()
        except Exception as exc:
            print(f'Face-quality checks unavailable: {exc}. Saved data will record unknown quality.', file=sys.stderr)
        while True:
            if references is None:
                reference = input('录制前输入准确文本（空行结束）：').strip()
                if not reference:
                    break
                if not characters(reference):
                    print('文本需要至少一个文字或数字。')
                    continue
            else:
                reference = next(references, None)
                if reference is None:
                    break
            print(f'准备无声说出：{reference}\nSPACE 开始/停止；R 丢弃重录；Esc 丢弃并退出。')
            review, last_frame = False, None
            while True:
                if not review:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError('Webcam stopped returning images')
                    timestamp = time.monotonic()
                    observation = tracker.detect(frame, int(timestamp * 1000)) if tracker else None
                    last_frame = frame
                    if recording is not None:
                        if recording.ts and timestamp - recording.ts[0] >= max_seconds:
                            review = True
                        else:
                            recording.push(frame, timestamp, observation)
                    if recording is not None and review:
                        recording.finish()
                        _review_instructions(reference, recording)
                preview = last_frame.copy()
                state = 'ENTER confirms exact text AND SILENT speech; R retake; ESC discard' if review else 'SPACE start / stop; R retake; ESC quit'
                cv2.putText(preview, state, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
                if recording is not None:
                    cv2.putText(preview, f'{recording.duration:.1f}s / {max_seconds}s', (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                cv2.imshow(WINDOW, preview)  # only this copy receives UI pixels
                key = cv2.waitKey(20) & 0xff
                if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return manifest
                if key in (ord('r'), ord('R')):
                    if recording is not None:
                        recording.discard()
                        recording = None
                    review = False
                elif key == 32 and not review:
                    if recording is None:
                        recording = Recording(directory, max_bytes, tracking_available=tracker is not None)
                    elif recording.ts:
                        recording.finish()
                        review = True
                        _review_instructions(reference, recording)
                elif key in (10, 13) and review:
                    row = commit(recording, reference, speaker, session, split, label_verified=True, silent_verified=True)
                    print(f"已保存 {row['id']}（{split}，同一会话 {session}）。")
                    recording = None
                    break
    finally:
        if recording is not None:
            recording.discard()
        cap.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
    return manifest

"""No test opens a physical webcam or microphone; all writers/cameras are synthetic."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lipflow import collection


class Writer:
    instances = []

    def __init__(self, path, codec, fps, shape):
        self.path, self.frames, self.released = Path(path), [], False
        self.instances.append(self)

    def isOpened(self):
        return True

    def write(self, image):
        self.frames.append(image.copy())
        with self.path.open('ab') as stream:
            stream.write(image.tobytes())

    def release(self):
        self.released = True


def recording(tmp_path, timestamps=(0, .1, .2), max_bytes=8 * 1024 * 1024):
    rec = collection.Recording(tmp_path, max_bytes, writer_factory=Writer)
    for i, timestamp in enumerate(timestamps):
        rec.push(np.full((32, 32, 3), i + 10, np.uint8), timestamp)
    return rec


def test_nearest_capture_resampling_streams_raw_pixels_and_bounded_buffer(tmp_path):
    rec = recording(tmp_path)
    rec.finish()
    assert [int(frame[0, 0, 0]) for frame in rec.writer.frames] == [10, 10, 11, 11, 12, 12]
    assert rec.frames_written == 6 and rec.previous is None
    assert rec.evidence()['capture_timestamps_seconds'] == [0, .1, .2]
    assert rec.evidence()['timestamp_clock'].startswith('time.monotonic')


def test_already_25fps_keeps_every_synthetic_frame(tmp_path):
    rec = recording(tmp_path, [10000 + i / 25 for i in range(150)])
    rec.finish()
    assert len(rec.writer.frames) == 150
    assert [int(frame[0, 0, 0]) for frame in rec.writer.frames] == list(range(10, 160))


def test_confirmation_is_required_and_discard_removes_unconfirmed_video(tmp_path):
    rec = recording(tmp_path)
    with pytest.raises(ValueError, match='Human confirmation'):
        collection.commit(rec, '请明天发送文件', 'speaker1', 'session1', 'train', label_verified=True)
    assert not (tmp_path / 'manifest.json').exists()
    assert not (tmp_path / 'videos').exists()
    rec.discard()
    assert not rec.path.exists() and rec.writer.released


def test_append_preserves_explicit_session_labels_and_old_video_and_resets_audit(tmp_path):
    first = collection.commit(recording(tmp_path), '不要明天发送', 'speaker1', 'real-session1', 'train',
                              label_verified=True, silent_verified=True)
    manifest = tmp_path / 'manifest.json'
    old = json.loads(manifest.read_text(encoding='utf-8'))
    old['training_overlap_checked'] = True
    manifest.write_text(json.dumps(old), encoding='utf-8')
    first_bytes = (tmp_path / first['video']).read_bytes()
    second = collection.commit(recording(tmp_path), '后天再发送', 'speaker1', 'real-session1', 'train',
                               label_verified=True, silent_verified=True)
    data = json.loads(manifest.read_text(encoding='utf-8'))
    assert len(data['samples']) == 2 and first['id'] != second['id']
    assert {row['session'] for row in data['samples']} == {'real-session1'}
    assert {row['reference'] for row in data['samples']} == {'不要明天发送', '后天再发送'}
    assert not data['training_overlap_checked']
    for row in data['samples']:
        assert not Path(row['video']).is_absolute()
        assert row['articulation'] == 'silent' and row['articulation_verified'] and row['label_verified']
        assert row['sha256'] == hashlib.sha256((tmp_path / row['video']).read_bytes()).hexdigest()
        assert row['domain'] == 'webcam' and not row['mouth_roi']
    assert (tmp_path / first['video']).read_bytes() == first_bytes
    assert not list(tmp_path.glob('.unconfirmed-*'))


def test_quota_rejects_video_without_publishing_manifest(tmp_path):
    rec = collection.Recording(tmp_path, 20, writer_factory=Writer)
    rec.push(np.zeros((32, 32, 3), np.uint8), 0)
    with pytest.raises(ValueError, match='quota'):
        rec.finish()
    rec.discard()
    assert not rec.path.exists() and not (tmp_path / 'manifest.json').exists()


def test_manifest_failure_rolls_back_only_new_video(tmp_path, monkeypatch):
    row = collection.commit(recording(tmp_path), '请发文件', 's1', 'session1', 'test',
                            label_verified=True, silent_verified=True)
    previous = (tmp_path / 'manifest.json').read_bytes()
    def fail(*args):
        raise OSError('simulated manifest write failure')
    monkeypatch.setattr(collection, '_atomic_json', fail)
    with pytest.raises(OSError):
        collection.commit(recording(tmp_path), '请勿发文件', 's1', 'session1', 'test',
                          label_verified=True, silent_verified=True)
    assert (tmp_path / 'manifest.json').read_bytes() == previous
    assert len(list((tmp_path / 'videos').glob('*.mp4'))) == 1
    assert (tmp_path / row['video']).exists()
    assert not (tmp_path / '.collection.lock').exists()


def test_failed_destination_link_cannot_remove_someone_elses_file(tmp_path, monkeypatch):
    external = []
    def raced_link(source, destination):
        Path(destination).write_bytes(b'external file')
        external.append(Path(destination))
        raise FileExistsError(destination)
    monkeypatch.setattr(collection.os, 'link', raced_link)
    rec = recording(tmp_path)
    with pytest.raises(FileExistsError):
        collection.commit(rec, '请发文件', 's1', 'session1', 'train', label_verified=True, silent_verified=True)
    assert external[0].read_bytes() == b'external file'
    rec.discard()


def test_mixed_split_is_rejected_without_overwriting_existing_manifest(tmp_path):
    collection.commit(recording(tmp_path), '请发文件', 's1', 'session1', 'train',
                      label_verified=True, silent_verified=True)
    original = (tmp_path / 'manifest.json').read_bytes()
    rec = recording(tmp_path)
    with pytest.raises(ValueError, match='separate output'):
        collection.commit(rec, '不要发文件', 's1', 'session1', 'test', label_verified=True, silent_verified=True)
    assert (tmp_path / 'manifest.json').read_bytes() == original
    rec.discard()


def test_input_validation_rejects_media_files_before_camera_permissions(tmp_path, monkeypatch):
    video = tmp_path / 'existing.mp4'
    video.write_bytes(b'synthetic file')
    def forbidden():
        pytest.fail('Camera permission/capture must not start for file input')
    monkeypatch.setattr(collection, '_camera_permission', forbidden)
    with pytest.raises(ValueError, match='physical webcam'):
        collection.collect(tmp_path / 'output', 's1', 'session1', camera=video)


def test_prompt_file_loaded_before_capture_and_blank_references_rejected(tmp_path):
    path = tmp_path / 'prompts.txt'
    path.write_text('请明天发送\n\n不要发给张三\n', encoding='utf-8')
    assert list(collection._prompts(path)) == ['请明天发送', '不要发给张三']
    with pytest.raises(ValueError):
        collection._prompts(['，。'])


def test_ui_overlay_never_enters_mock_video_and_retakes_never_publish(tmp_path, monkeypatch):
    from lipflow import camera, face
    Writer.instances = []
    class Camera:
        def isOpened(self): return True
        def set(self, *args): pass
        def read(self): return True, np.full((32, 32, 3), 17, np.uint8)
        def release(self): self.released = True
    cap = Camera()
    def no_tracker(): raise FileNotFoundError('synthetic missing quality model')
    keys = iter([32, 32, ord('r'), 32, 32, 13])
    monkeypatch.setattr(collection, '_camera_permission', lambda: None)
    monkeypatch.setattr(camera, 'resolve_camera', lambda value: 0)
    monkeypatch.setattr(face, 'FaceTracker', no_tracker)
    monkeypatch.setattr(collection.cv2, 'VideoCapture', lambda *args: cap)
    monkeypatch.setattr(collection.cv2, 'VideoWriter', Writer)
    monkeypatch.setattr(collection.cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(collection.cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(collection.cv2, 'destroyAllWindows', lambda: None)
    monkeypatch.setattr(collection.cv2, 'waitKey', lambda delay: next(keys))
    monkeypatch.setattr(collection.cv2, 'putText', lambda image, *args: image.fill(255))
    clock = iter(i * .04 for i in range(50))
    monkeypatch.setattr(collection.time, 'monotonic', lambda: next(clock))
    manifest = collection.collect(tmp_path, 's1', 'same-session', prompts=['不要明天发送'])
    data = json.loads(manifest.read_text(encoding='utf-8'))
    assert len(data['samples']) == 1 and data['samples'][0]['session'] == 'same-session'
    assert len(Writer.instances) == 2 and not Writer.instances[0].path.exists()
    assert all(np.all(frame == 17) for writer in Writer.instances for frame in writer.frames)
    assert data['samples'][0]['collection']['face_tracking_available'] is False
    assert data['samples'][0]['collection']['quality']['face_ratio'] is None
    assert cap.released and not list(tmp_path.glob('.unconfirmed-*'))

"""Geometry and whole-transform scoring, without optional model weights/GPU."""
import cv2
import numpy as np
import pytest

from scripts.diagnose_chinese_crops import (CropTransform, diagnostic_grid,
                                          summarize, transform_frames)


def asymmetric_images():
    images = np.zeros((2, 96, 96, 3), dtype=np.uint8)
    images[:, 20, 30] = 255
    images[:, 40, 70] = (0, 255, 0)
    return images


def test_identity_and_mirror_preserve_asymmetric_pixel_positions():
    images = asymmetric_images()
    original = images.copy()
    identity = transform_frames(images, CropTransform('identity'))
    expected = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in images])
    np.testing.assert_array_equal(identity, expected)
    mirrored = transform_frames(images, CropTransform('mirror', mirror=True))
    np.testing.assert_array_equal(mirrored, expected[:, :, ::-1])
    np.testing.assert_array_equal(images, original)


def test_positive_affine_displacement_moves_image_content_right_and_down():
    shifted = transform_frames(asymmetric_images(), CropTransform('shift', dx=6, dy=12))
    assert shifted[0, 32, 36] == 255
    assert shifted[0, 20, 30] == 0


def test_official_float_grayscale_preserves_fractional_values_instead_of_rounding():
    images = np.zeros((1, 96, 96, 3), dtype=np.uint8)
    images[..., 2] = 255  # pure red in BGR
    gray = transform_frames(images, CropTransform('float', grayscale='torchvision_float'))
    assert gray.dtype == np.float32
    assert float(gray[0, 0, 0]) == pytest.approx(.2989 * 255)
    assert float(gray[0, 0, 0]) != float(cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)[0, 0])


def test_fixed_grid_contains_one_identity_and_distinct_effective_transforms():
    grid = diagnostic_grid()
    assert len(grid) == 12
    assert len({transform.name for transform in grid}) == len(grid)
    assert len({(t.scale, t.dx, t.dy, t.mirror, t.grayscale) for t in grid}) == len(grid)
    assert grid[0] == CropTransform('baseline')
    assert next(t for t in grid if t.name == 'mirror').mirror


def test_failed_samples_stay_in_whole_transform_denominator():
    rows = [dict(character_errors=0, reference_characters=10, exact=True, error=None),
            dict(character_errors=20, reference_characters=20, exact=False, error='decode failed')]
    metrics = summarize(rows)
    assert metrics['samples'] == 2
    assert metrics['cer'] == pytest.approx(20 / 30)
    assert metrics['exact_sentences'] == 1
    assert metrics['inference_failures'] == 1
    assert summarize([])['cer'] is None


@pytest.mark.parametrize('transform', [CropTransform('bad', scale=0), CropTransform('bad', dx=float('nan')),
                                      CropTransform('bad', grayscale='invalid')])
def test_invalid_transforms_cannot_silently_produce_results(transform):
    with pytest.raises(ValueError):
        transform_frames(asymmetric_images(), transform)

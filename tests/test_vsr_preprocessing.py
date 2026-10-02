"""Regression checks for the published VSR input and decoding contracts."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from lipflow.vsr import LipReader


@pytest.mark.parametrize("start", [0.0, 1720000000.0])
def test_already_25fps_preserves_every_frame(start):
    timestamps = [start + i / 25 for i in range(150)]
    assert LipReader.resample(timestamps, len(timestamps)) == list(range(150))


def test_variable_rate_chooses_nearest_capture_without_forward_bias():
    timestamps = [0.0, 0.03, 0.081, 0.13, 0.159, 0.201]
    assert LipReader.resample(timestamps, len(timestamps)) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("timestamps, frames", [([0.0], 2), ([0.1, 0.0], 2), ([0.0, float("nan")], 2)])
def test_resample_rejects_invalid_capture_timeline(timestamps, frames):
    with pytest.raises(ValueError, match="finite and sorted"):
        LipReader.resample(timestamps, frames)


def test_published_center_crop_and_normalization_preserve_pixel_positions():
    # Deliberately asymmetric intensities expose transposed axes or shifted crops.
    rois = np.arange(3 * 96 * 96, dtype=np.uint32).reshape(3, 96, 96).astype(np.uint8)
    expected = (rois[:, 4:92, 4:92].astype(np.float32) / 255.0 - 0.421) / 0.165
    tensor = LipReader.to_tensor(rois)
    assert tuple(tensor.shape) == (1, 3, 88, 88)
    np.testing.assert_array_equal(tensor.numpy()[0], expected)


@pytest.mark.skipif(not Path("models/zh/vsr/model.pth").exists(), reason="optional research weights not installed")
def test_chinese_decoder_uses_published_cmlr_length_bonus():
    reader = LipReader(language="zh", beam_size=2, use_lm=False, personal=False)
    assert reader.beam.weights["length_bonus"] == 0.3
    assert "length_bonus" in reader.beam.full_scorers
    assert reader.beam.weights["ctc"] == 0.1


@pytest.mark.parametrize("av_decoder", [False, True])
def test_new_utterance_cannot_reuse_previous_memory_projection(av_decoder):
    torch.manual_seed(9)
    attention = MultiHeadedAttention(1, 4, 0.0).eval()
    query = torch.randn(2, 1, 4)
    memory = torch.randn(1, 5, 4)
    observed = []

    def project_in_search(encoded):
        key = encoded.expand(2, -1, -1)
        observed.append(attention.forward_qkv(query, key, key)[1].clone())
        return []

    project_in_search.full_scorers = {"decoder": attention}
    reader = LipReader.__new__(LipReader)
    # AVReader's original visual decoder is unused after the AV scorer replaces it.
    unused = MultiHeadedAttention(1, 4, 0.0).eval() if av_decoder else attention
    reader.model = SimpleNamespace(decoder=unused)
    reader.beam = project_in_search
    reader.hypotheses(memory)
    # Reused storage represents a second utterance with the same pointer/length.
    memory.add_(20)
    reader.hypotheses(memory)
    expected = attention.linear_k(memory).view(1, -1, 1, 4).transpose(1, 2).expand(2, -1, -1, -1)
    assert not torch.equal(observed[0], observed[1])
    torch.testing.assert_close(observed[1], expected)

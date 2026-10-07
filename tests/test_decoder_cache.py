import copy
from types import SimpleNamespace

import pytest
import torch

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from lipflow.vsr import LipReader


@pytest.mark.parametrize("replacement_decoder", [False, True])
@pytest.mark.parametrize("nbest", [1, 2])
def test_each_dictation_matches_a_fresh_decoder(replacement_decoder, nbest):
    with torch.random.fork_rng():
        torch.manual_seed(7)
        decoder = Decoder(5, attention_dim=4, attention_heads=1, linear_units=8,
                          num_blocks=1, dropout_rate=0, positional_dropout_rate=0).eval()
        memory = torch.randn(6, 4)
        next_memory = torch.randn_like(memory)
    reader = LipReader.__new__(LipReader)
    reader.token_list = ["<blank>", "▁A", "▁B", "▁C", "<eos>"]
    reader.model = SimpleNamespace(decoder=copy.deepcopy(decoder) if replacement_decoder else decoder)
    reader.beam = BatchBeamSearch(beam_size=2, vocab_size=5, weights={"decoder": 1.0},
                                 scorers={"decoder": decoder}, sos=4, eos=4,
                                 token_list=reader.token_list).eval()
    fresh_beam = copy.deepcopy(reader.beam)
    attention = decoder.decoders[0].src_attn
    projections = []
    searches = []
    attention.linear_k.register_forward_hook(
        lambda module, args, output: projections.append(args[0].stride(0)))
    reader.beam.register_forward_hook(lambda module, args, output: searches.append(output))

    first = reader.beam_search(memory, nbest=nbest)
    assert isinstance(first, str if nbest == 1 else list)
    assert projections.count(0) == 1
    old_tag = attention._mem_kv[0]
    memory.copy_(next_memory)
    assert old_tag == (memory.data_ptr(), memory.shape[0], memory.device)

    second = reader.beam_search(memory, nbest=nbest)
    assert isinstance(second, str if nbest == 1 else list)
    with torch.inference_mode():
        expected = fresh_beam(memory)
    assert len(searches[-1]) == len(expected)
    for actual, fresh in zip(searches[-1], expected):
        assert actual.yseq.tolist() == fresh.yseq.tolist()
        torch.testing.assert_close(actual.score, fresh.score)
    assert projections.count(0) == 2

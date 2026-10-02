"""Development-only Mandarin decoding comparison; no downloads or GUI activation.

Uses the strictly verified CNVSRC reader and optional bound research adapter.
Encoded visual frames are cached in memory once per clip. References reach only
CER evaluation after decoding. The default CTC grid is fixed before inference;
reverse rescoring runs only on the base configuration selected by dev CER.
Never use this script on held-out/test data or call its result readiness evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import sys
import time

from lipflow.evaluation import Dataset, Prediction, character_error, characters, evaluate, load_manifest
from lipflow.confidence import Hypothesis, assess

CTC_WEIGHTS = (.1, .3, .5, .7, 1.0)
DEV_SPLITS = frozenset(('dev', 'val', 'validation'))
PURE_CTC_WORKSPACE_BUDGET = 1024 * 1024 * 1024


@dataclass(frozen=True)
class Candidate:
    text: str
    score: float
    token_ids: tuple[int, ...]
    forward_score: float
    ctc_score: float
    reverse_score: float | None = None
    beam_score: float | None = None
    forced_eos: bool = False
    trailing_eos_count: int = 1
    beam_token_ids: tuple[int, ...] = ()
    reverse_rerank_skipped_reason: str | None = None


def validate_protocol(dataset: Dataset, weights, beam_size: int, nbest: int, reverse_weight: float):
    if dataset.split not in DEV_SPLITS or any(s.split not in DEV_SPLITS for s in dataset.samples):
        raise ValueError('Only dev/val/validation samples are accepted; held-out/test data is prohibited')
    if not dataset.samples or len({s.id for s in dataset.samples}) != len(dataset.samples):
        raise ValueError('Development samples must be nonempty and have unique IDs')
    if any(not s.label_verified or not s.label_source for s in dataset.samples):
        raise ValueError('Every development reference requires verified label provenance')
    if (not weights or any(isinstance(w, bool) or not isinstance(w, (int, float))
                           or not math.isfinite(w) or not 0 < w <= 1 for w in weights)
            or len(set(weights)) != len(weights)):
        raise ValueError('CTC weights must be unique finite values in (0, 1]')
    if (type(beam_size) is not int or type(nbest) is not int or beam_size < 1
            or not 1 <= nbest <= beam_size):
        raise ValueError('Require integer beam_size >= nbest >= 1')
    if (isinstance(reverse_weight, bool) or not isinstance(reverse_weight, (int, float))
            or not math.isfinite(reverse_weight) or not 0 <= reverse_weight <= 1):
        raise ValueError('Reverse weight must be finite in [0, 1]')


def _clear_memory(decoder):
    for module in decoder.modules():
        if hasattr(module, '_mem_kv'):
            del module._mem_kv


def pure_ctc_workspace_bytes(encoded, beam_size: int, vocabulary_size: int) -> int:
    """Conservative six-buffer estimate for full-vocabulary CTC prefix scoring.

    ESPnet CTCPrefixScoreTH materializes r/x/log_phi/log_phi_x on frame x
    hypothesis x vocabulary axes when prebeam is disabled at CTC weight 1.
    This is a resource gate, not a claim that total process memory is bounded.
    """
    if len(encoded.shape) != 2 or encoded.shape[0] < 1:
        raise ValueError('Expected nonempty T,D encoded visual frames')
    return 6 * int(encoded.shape[0]) * 2 * beam_size * vocabulary_size * encoded.element_size()


def decode_candidates(reader, encoded, ctc_weight: float, beam_size: int, nbest: int) -> list[Candidate]:
    """Build fresh beam and prefix scorers for each serial search; no reader mutation."""
    import torch
    from espnet.asr.asr_utils import add_results_to_json
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.scorers.length_bonus import LengthBonus

    scorers = reader.model.scorers()
    scorers['length_bonus'] = LengthBonus(len(reader.token_list))
    beam = BatchBeamSearch(
        beam_size=beam_size, vocab_size=len(reader.token_list),
        weights={'decoder': 1 - ctc_weight, 'ctc': ctc_weight, 'length_bonus': 0.0},
        scorers=scorers, sos=len(reader.token_list) - 1, eos=len(reader.token_list) - 1,
        token_list=reader.token_list, pre_beam_score_key=None if ctc_weight == 1 else 'decoder',
    ).to(reader.device).eval()
    _clear_memory(reader.model.decoder)
    seen, result, forced_in_beam = set(), [], False
    try:
        with torch.inference_mode():
            for hypothesis in beam(encoded):
                ids = tuple(int(i) for i in hypothesis.yseq.tolist())
                eos = len(reader.token_list) - 1
                if len(ids) < 2 or ids[0] != eos or ids[-1] != eos:
                    raise ValueError('Beam hypothesis must contain the original SOS/EOS boundaries')
                # With default maxlenratio=0, ESPnet's final iteration appends
                # one unscored EOS to every surviving hypothesis. This may make
                # two trailing EOS tokens when the final search already chose it.
                forced_eos = len(ids) == int(encoded.shape[0]) + 2
                forced_in_beam |= forced_eos
                end = len(ids)
                while end > 1 and ids[end - 1] == eos:
                    end -= 1
                trailing_eos_count = len(ids) - end
                body = ids[1:end]
                if any(i <= 0 or i >= eos for i in body):
                    raise ValueError('Beam hypothesis contains an invalid body token')
                # Only normalize token boundaries for rendering; score evidence
                # remains exactly the beam's. Never add a missing EOS score.
                rendered = {**hypothesis.asdict(), 'yseq': [eos, *body, eos]}
                text = reader._clean(add_results_to_json([rendered], reader.token_list))
                if not text or text in seen:
                    continue
                score = float(hypothesis.score)
                forward = float(hypothesis.scores.get('decoder', 0.0))
                ctc = float(hypothesis.scores.get('ctc', 0.0))
                if not all(math.isfinite(x) for x in (score, forward, ctc)):
                    raise ValueError('Non-finite beam evidence')
                seen.add(text)
                if len(result) < nbest:
                    result.append(Candidate(text, score, body, forward, ctc, beam_score=score,
                                            forced_eos=forced_eos,
                                            trailing_eos_count=trailing_eos_count,
                                            beam_token_ids=ids))
    finally:
        _clear_memory(reader.model.decoder)
    # Inspect the full returned beam, including duplicates and candidates beyond
    # nbest. Mixing teacher-forced EOS scores with unscored forced endings would
    # alter its score scale. Skip the whole utterance rather than selected rows.
    if forced_in_beam:
        result = [replace(c, reverse_rerank_skipped_reason='forced_eos_in_beam') for c in result]
    return result


def reverse_log_probability(decoder, encoded, token_ids, eos: int) -> float:
    """Teacher-force candidate IDs backwards, including EOS, without any reference."""
    import torch

    if type(eos) is not int or eos <= 1 or any(type(i) is not int or not 0 < i < eos for i in token_ids):
        raise ValueError('Candidate IDs must be ordinary vocabulary tokens')
    reversed_ids = list(reversed(token_ids))
    inputs = torch.tensor([[eos] + reversed_ids], dtype=torch.long, device=encoded.device)
    targets = torch.tensor([reversed_ids + [eos]], dtype=torch.long, device=encoded.device)
    causal = torch.ones((1, inputs.size(1), inputs.size(1)), dtype=torch.bool,
                        device=encoded.device).tril()
    _clear_memory(decoder)
    try:
        with torch.inference_mode():
            logits, _ = decoder(inputs, causal, encoded.unsqueeze(0), None)
            if logits.shape != (1, inputs.size(1), eos + 1):
                raise ValueError('Unexpected reverse-decoder output shape')
            score = float(logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).sum())
    finally:
        _clear_memory(decoder)
    if not math.isfinite(score):
        raise ValueError('Non-finite reverse-decoder score')
    return score


def rerank_candidates(candidates, ctc_weight, reverse_weight, reverse_scorer):
    """Preserve the beam scale: S' = S + (1-CTC)*r*(reverse-forward)."""
    if not 0 <= ctc_weight <= 1 or not 0 <= reverse_weight <= 1:
        raise ValueError('Rescoring weights must be in [0, 1]')
    if reverse_weight == 0 or ctc_weight == 1:
        return list(candidates)
    if any(c.forced_eos or c.reverse_rerank_skipped_reason for c in candidates):
        return [replace(c, reverse_rerank_skipped_reason='forced_eos_in_beam') for c in candidates]
    scored = []
    for candidate in candidates:
        reverse = reverse_scorer(candidate.token_ids)
        score = candidate.score + (1 - ctc_weight) * reverse_weight * (reverse - candidate.forward_score)
        if not math.isfinite(reverse) or not math.isfinite(score):
            raise ValueError('Non-finite rescoring evidence')
        scored.append(replace(candidate, score=score, reverse_score=reverse,
                              beam_score=candidate.score if candidate.beam_score is None else candidate.beam_score))
    return sorted(scored, key=lambda c: c.score, reverse=True)


def _assessment(candidates, greedy, quality, error):
    if error:
        return 'retry', 'Development inference failed; retained raw output is diagnostic', None
    decision = assess([Hypothesis(c.text, c.score, max(len(c.token_ids), 1)) for c in candidates],
                      greedy, quality, policy='review', language='zh')
    return decision.action, decision.reason, decision.margin


def _configuration_report(dataset, predictions, evidence):
    report = evaluate(dataset, predictions)
    # Development data cannot establish readiness, even if every sentence is correct.
    report['readiness'] = {'ready': False, 'status': 'development_comparison_only'}
    for row in report['samples']:
        row.update(evidence[row['sample_id']])
    report['resource_limited_samples'] = sum(row['resource_limited'] for row in report['samples'])
    oracle_errors = sum(min((character_error(h['text'], row['reference'])[0]
                            for h in row['hypotheses']), default=row['reference_characters'])
                        for row in report['samples'])
    denominator = report['metrics']['reference_characters']
    report['diagnostic'] = {'oracle_nbest_character_errors': oracle_errors,
                            'oracle_nbest_cer': oracle_errors / denominator if denominator else None,
                            'oracle_used_for_selection': False}
    return report


def compare(dataset: Dataset, reader, visual_input, *, weights=CTC_WEIGHTS, beam_size=40, nbest=10,
            reverse_weight=.3, decoder=decode_candidates, reverse_scorer=None,
            cache_max_bytes=256 * 1024 * 1024,
            pure_ctc_max_workspace_bytes=PURE_CTC_WORKSPACE_BUDGET, clock=time.monotonic) -> dict:
    """Cache images' encoder outputs in RAM, then decode serially; never send references."""
    validate_protocol(dataset, weights, beam_size, nbest, reverse_weight)
    if type(cache_max_bytes) is not int or cache_max_bytes < 1:
        raise ValueError('cache_max_bytes must be a positive integer')
    if type(pure_ctc_max_workspace_bytes) is not int or pure_ctc_max_workspace_bytes < 1:
        raise ValueError('pure_ctc_max_workspace_bytes must be a positive integer')
    predictions = {w: [] for w in weights}
    evidence = {w: {} for w in weights}
    cache, cache_bytes, encode_total = {}, 0, 0.0
    started = clock()
    for sample in dataset.samples:
        begin, duration, encoded, preparation_error = clock(), None, None, None
        quality, greedy = None, ''
        try:
            # The frame reader also receives no reference text, enforcing the boundary.
            rois, duration, quality = visual_input(replace(sample, reference=''), reader)
            encoded = reader.encode(rois)
            greedy = reader.greedy(encoded)
            size = encoded.numel() * encoded.element_size()
            if size < 1 or cache_bytes + size > cache_max_bytes:
                raise ValueError('Encoded-frame cache quota exceeded; supply a smaller dev manifest')
            cache_bytes += size
        except Exception as exc:
            preparation_error = f'{type(exc).__name__}: {exc}'
            duration = getattr(exc, 'duration', duration)
            encoded = None
        encode_seconds = clock() - begin
        encode_total += encode_seconds
        cache[sample.id] = (encoded, duration, encode_seconds, preparation_error, quality, greedy)
        for weight in weights:
            decode_started, candidates, error = clock(), [], preparation_error
            resource_limited, estimated_workspace = False, None
            if encoded is not None:
                try:
                    if weight == 1:
                        estimated_workspace = pure_ctc_workspace_bytes(encoded, beam_size, len(reader.token_list))
                        resource_limited = estimated_workspace > pure_ctc_max_workspace_bytes
                    if resource_limited:
                        error = (f'ResourceLimited: pure CTC prefix workspace estimate {estimated_workspace} '
                                 f'exceeds budget {pure_ctc_max_workspace_bytes}; decoder was not invoked')
                    else:
                        candidates = decoder(reader, encoded, weight, beam_size, nbest)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
            decode_seconds = clock() - decode_started if encoded is not None else 0.0
            raw = candidates[0].text if candidates else ''
            action, reason, margin = _assessment(candidates, greedy, quality, error)
            predictions[weight].append(Prediction(
                sample.id, raw, duration, encode_seconds + decode_seconds,
                action, reason, error=error, margin=margin))
            evidence[weight][sample.id] = {
                'hypotheses': [asdict(h) for h in candidates],
                'preparation_and_encode_seconds': encode_seconds, 'decode_seconds': decode_seconds,
                'resource_limited': resource_limited,
                'estimated_pure_ctc_workspace_bytes': estimated_workspace,
                'quality': asdict(quality) if quality is not None else None,
                'greedy': greedy,
                'reference_oov_characters': sorted(set(characters(sample.reference)) - set(reader.token_list)),
            }
        print(f'[{len(cache)}/{len(dataset.samples)}] {sample.id}: encoded once', file=sys.stderr)
    configurations = []
    for weight in weights:
        report = _configuration_report(dataset, predictions[weight], evidence[weight])
        report['parameters'] = {'ctc_weight': weight, 'reverse_weight': 0.0,
                                'beam_size': beam_size, 'nbest': nbest, 'length_bonus': 0.0, 'external_lm': False}
        configurations.append(report)
    best = min(configurations, key=lambda r: (r['metrics']['cer'], r['parameters']['ctc_weight']))
    best_weight = best['parameters']['ctc_weight']
    if reverse_weight and best_weight < 1:
        reranked_predictions, reranked_evidence = [], {}
        original_rows = {p.sample_id: p for p in predictions[best_weight]}
        for sample in dataset.samples:
            encoded, duration, encode_seconds, preparation_error, quality, greedy = cache[sample.id]
            original = original_rows[sample.id]
            candidates = [Candidate(**{**h, 'token_ids': tuple(h['token_ids'])})
                          for h in evidence[best_weight][sample.id]['hypotheses']]
            begin, error = clock(), original.error
            if encoded is not None and candidates and not error:
                try:
                    score = (lambda ids: reverse_scorer(reader, encoded, ids)) if reverse_scorer else (
                        lambda ids: reverse_log_probability(reader.model.r_decoder, encoded, ids,
                                                            len(reader.token_list) - 1))
                    candidates = rerank_candidates(candidates, best_weight, reverse_weight, score)
                except Exception as exc:
                    # A rescoring failure cannot erase the original diagnostic
                    # prediction or rewrite its beam evidence.
                    error = f'{type(exc).__name__}: {exc}'
            reverse_seconds = clock() - begin
            raw = candidates[0].text if candidates else ''
            action, reason, margin = _assessment(candidates, greedy, quality, error)
            reranked_predictions.append(replace(original, raw=raw, error=error,
                action=action, reason=reason, margin=margin,
                processing_seconds=original.processing_seconds + reverse_seconds))
            reranked_evidence[sample.id] = {**evidence[best_weight][sample.id],
                'hypotheses': [asdict(h) for h in candidates], 'reverse_rescoring_seconds': reverse_seconds,
                'reverse_reranking_skipped': any(c.reverse_rerank_skipped_reason for c in candidates),
                'reverse_reranking_skip_reason': next((c.reverse_rerank_skipped_reason for c in candidates
                                                       if c.reverse_rerank_skipped_reason), None)}
        report = _configuration_report(dataset, reranked_predictions, reranked_evidence)
        report['parameters'] = {**best['parameters'], 'reverse_weight': reverse_weight}
        configurations.append(report)
    return {
        'schema_version': 1, 'scope': 'development_decoder_comparison', 'readiness': False,
        'protocol': {'ctc_weights': list(weights), 'beam_size': beam_size, 'nbest': nbest,
                     'length_bonus': 0.0, 'external_lm': False,
                     'reverse_weight': reverse_weight, 'reverse_base_ctc_weight': best_weight,
                     'base_selection': 'lowest dev top-1 raw CER; ties by ascending CTC weight',
                     'reverse_formula': 'S_new = S_beam + (1 - ctc_weight) * reverse_weight * (S_reverse - S_forward)',
                     'reverse_target': 'reverse(candidate body token IDs), then original EOS; input starts original SOS',
                     'forced_eos_policy': 'detect len(yseq)=encoded_frames+2 at default maxlen; strip all trailing EOS for body; if any beam hypothesis is force-ended, skip reverse reranking for the entire utterance and preserve original scores/order',
                     'routing': 'shared confidence.assess(policy=review, language=zh); quality and greedy cached once per clip; raw retry outputs retained',
                     'references_reach_decoder': False, 'cache_bytes': cache_bytes,
                     'cache_max_bytes': cache_max_bytes, 'cache_persistence': 'in-memory only',
                     'pure_ctc_max_workspace_bytes': pure_ctc_max_workspace_bytes,
                     'pure_ctc_workspace_formula': '6 * encoded_frames * 2 * beam_size * vocabulary_size * dtype_bytes'},
        'timing': {'comparison_wall_seconds': clock() - started,
                   'preparation_and_encode_seconds_total': encode_total,
                   'reported_configuration_latency': 'full preparation+encode cost plus that configuration decoding; reverse adds reranking',
                   'caveat': 'Encoding is shared across configurations in this serial comparison. Actual wall time is not independent end-to-end inference latency; model startup/warmup excluded. No amortized speed claim.'},
        'configurations': configurations,
        'limitations': ['Development results select parameters and cannot establish test/generalization readiness.',
                        'All failures and unrepresentable reference characters remain in CER denominators.',
                        'Oracle n-best CER is diagnostic only and cannot be achieved using unavailable references.',
                        'Pure-CTC resource-limited samples are recorded with blank outputs and kept in all denominators; workspace estimates do not bound total process RAM.',
                        'Ordinary voiced public videos do not establish deliberate silent-articulation usability.'],
    }


def main() -> int:
    from evaluate_cnvsrc import (_reader, _load_adapter, CHECKPOINT_SHA256, SOURCE_REVISION,
                                 WEIGHT_REVISION, VOCABULARY_SHA256, CONFIG_SHA256)
    from evaluate_chinese import _visual_input
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-research-license', action='store_true')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--ctc-weights', type=float, nargs='+', default=list(CTC_WEIGHTS))
    parser.add_argument('--beam-size', type=int, default=40)
    parser.add_argument('--nbest', type=int, default=10)
    parser.add_argument('--reverse-weight', type=float, default=.3)
    parser.add_argument('--cache-max-mib', type=int, default=256)
    parser.add_argument('--pure-ctc-max-workspace-mib', type=int, default=1024)
    args = parser.parse_args()
    if not args.accept_research_license:
        parser.error('Read the author VSR/LICENSE and pass --accept-research-license for research comparison')
    try:
        dataset = load_manifest(args.manifest)
        validate_protocol(dataset, args.ctc_weights, args.beam_size, args.nbest, args.reverse_weight)
        if args.cache_max_mib < 1:
            raise ValueError('--cache-max-mib must be positive')
        if args.pure_ctc_max_workspace_mib < 1:
            raise ValueError('--pure-ctc-max-workspace-mib must be positive')
        started = time.monotonic()
        reader = _reader(Path(args.checkpoint), Path(args.source_dir), args.beam_size,
                         args.ctc_weights[0], args.device)
        adapter = _load_adapter(reader, Path(args.adapter)) if args.adapter else None
        reader.warmup()
        startup = time.monotonic() - started
        report = compare(dataset, reader, _visual_input, weights=tuple(args.ctc_weights),
                         beam_size=args.beam_size, nbest=args.nbest, reverse_weight=args.reverse_weight,
                         cache_max_bytes=args.cache_max_mib * 1024 * 1024,
                         pure_ctc_max_workspace_bytes=args.pure_ctc_max_workspace_mib * 1024 * 1024)
        report['model'] = {'name': 'CNVSRC2025 strictly verified research reader',
                           'checkpoint_sha256': CHECKPOINT_SHA256, 'source_revision': SOURCE_REVISION,
                           'weight_revision': WEIGHT_REVISION, 'vocabulary_sha256': VOCABULARY_SHA256,
                           'configuration_sha256': CONFIG_SHA256, 'strict_load': True,
                           'adapter': adapter, 'encoder_device': str(reader.enc_device),
                           'decoder_device': str(reader.device), 'startup_and_warmup_seconds': startup,
                           'automatic_activation': False}
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Development comparison saved: {output}; no readiness claim')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

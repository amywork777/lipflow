r"""Optional pure-visual CNVSRC2025 research comparison, separate from the GUI.

Supply local author sources and checkpoint; this script never downloads files or
reads audio. --adapter loads a strictly bound local encoder-only research
adaptation; it never activates GUI or personal weights. Example from the Lipflow repository:
    uv run python scripts/evaluate_cnvsrc.py --accept-research-license \
        --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth \
        --source-dir /local/CNVSRC2025 --manifest test.json --output report.json

Known source: https://github.com/liu12366262626/CNVSRC2025
Source revision: e5c4454016ba4eef9e586e77dd58e8981bb5c3e1
Weights: https://huggingface.co/ReflectionL/CNVSRC2025Baseline
Weight revision: b16f238d0df860da7e3b9834f959780b1d388f44
The author limits use to non-commercial comparative/benchmarking research.
Exit 2 means the declared readiness criteria were not met, not a model crash.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

from lipflow.evaluation import Prediction, characters, evaluate, load_manifest

SOURCE_REVISION = 'e5c4454016ba4eef9e586e77dd58e8981bb5c3e1'
WEIGHT_REVISION = 'b16f238d0df860da7e3b9834f959780b1d388f44'
CHECKPOINT_SHA256 = '577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c'
CONFIG_SHA256 = 'b0464bcac797a2bafd98102d8ddaf041c6c9957b52566e5019c2a43de6860a04'
VOCABULARY_SHA256 = '635e12ebb5f7dcd60637a4f3c329cd543f1e0e34aa4a6d62ba87185c3666aae0'

# Functional parameters from the pinned visual configuration. The original code
# is not imported; the compatible encoder and decoder are already in Lipflow.
ARCHITECTURE = dict(
    adim=768, aheads=12, eunits=3072, elayers=12,
    transformer_input_layer='conv3d', dropout_rate=.1,
    transformer_attn_dropout_rate=.1, transformer_encoder_attn_layer_type='rel_mha',
    macaron_style=True, use_cnn_module=True, cnn_module_kernel=31,
    zero_triu=False, a_upsample_ratio=1, relu_type='swish',
    ddim=768, dheads=12, dunits=3072, dlayers=6, r_dlayers=3,
    lsm_weight=.1, transformer_length_normalized_loss=False,
    mtlalpha=.1, mtlbeta=.3, ctc_type='builtin', rel_pos_type='latest',
    report_cer=False, report_wer=False,
)


def _verified_file(path: Path, expected: str) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f'Unexpected source/checkpoint SHA256 for {path}: {actual}')
    return actual


def _reader(checkpoint: Path, source_dir: Path, beam_size: int, ctc_weight: float, device: str):
    """Strictly load all tensors, including the training-only reverse decoder."""
    import torch
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from espnet.nets.pytorch_backend.transformer.decoder import Decoder
    from espnet.nets.scorers.length_bonus import LengthBonus
    from lipflow.vsr import LipReader, pick_encoder_device

    source = source_dir / 'VSR' if (source_dir / 'VSR').is_dir() else source_dir
    _verified_file(source / 'conf/train_cncvs_2_3_cnvsrc.yaml', CONFIG_SHA256)
    vocabulary = source / 'datamodule/char_units.txt'
    _verified_file(vocabulary, VOCABULARY_SHA256)
    _verified_file(checkpoint, CHECKPOINT_SHA256)
    tokens = ['<blank>'] + [line.split()[0] for line in vocabulary.read_text(encoding='utf-8').splitlines()] + ['<eos>']
    if len(tokens) != 4470 or len(set(tokens)) != len(tokens):
        raise ValueError('Expected the original 4470-token vocabulary')

    class BidirectionalModel(E2E):
        def __init__(self):
            args = argparse.Namespace(**ARCHITECTURE, char_list=tokens)
            super().__init__(len(tokens), args)
            self.r_decoder = Decoder(
                odim=len(tokens), attention_dim=args.ddim, attention_heads=args.dheads,
                linear_units=args.dunits, num_blocks=args.r_dlayers,
                dropout_rate=args.dropout_rate, positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )

    class ResearchReader(LipReader):
        def __init__(self):
            # Deliberately bypass the GUI's language/model/personal-weight loader.
            self.language = 'zh'
            self.personal_vsr = self.personal_lm = False
            self.token_list = tokens
            self.enc_device = pick_encoder_device(device)
            self.device = torch.device('cuda') if self.enc_device.type == 'cuda' else torch.device('cpu')
            self.model = BidirectionalModel()
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state, strict=True)
            self.tensor_count = len(state)
            self.tensor_values = sum(tensor.numel() for tensor in state.values())
            del state
            self.model.to(self.device).eval()
            self.model.encoder.to(self.enc_device)
            scorers = self.model.scorers()
            scorers['length_bonus'] = LengthBonus(len(tokens))
            self.beam = BatchBeamSearch(
                beam_size=beam_size, vocab_size=len(tokens),
                weights=dict(decoder=1 - ctc_weight, ctc=ctc_weight, length_bonus=0.0),
                scorers=scorers, sos=len(tokens) - 1, eos=len(tokens) - 1,
                token_list=tokens, pre_beam_score_key=None if ctc_weight == 1 else 'decoder',
            ).to(self.device).eval()

    return ResearchReader()



def _load_adapter(reader, path: Path) -> dict:
    """Validate a complete final-encoder research artifact before any mutation.

    The main reader has already strictly verified its pinned base checkpoint.
    Loading only selected parameters keeps inherited BN statistics, frontend,
    earlier encoder layers, CTC and both decoders intact.
    """
    import torch

    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        artifact = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError(f'Cannot load research adapter with weights_only=True: {type(exc).__name__}') from exc
    if not isinstance(artifact, dict) or type(artifact.get('schema_version')) is not int or artifact['schema_version'] != 1:
        raise ValueError('Expected research adapter schema_version=1')
    if artifact.get('language') != 'zh' or reader.language != 'zh':
        raise ValueError('Research adapter language must be zh and match its reader')
    if artifact.get('base_checkpoint_sha256') != CHECKPOINT_SHA256:
        raise ValueError('Research adapter is bound to a different base checkpoint SHA256')
    metadata = artifact.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError('Research adapter metadata must be an object')
    base = metadata.get('base')
    if not isinstance(base, dict) or any(base.get(key) != value for key, value in (
            ('checkpoint_sha256', CHECKPOINT_SHA256), ('configuration_sha256', CONFIG_SHA256),
            ('vocabulary_sha256', VOCABULARY_SHA256), ('source_revision', SOURCE_REVISION))):
        raise ValueError('Research adapter metadata has inconsistent base provenance')
    layers = artifact.get('last_encoder_layers')
    blocks = reader.model.encoder.encoders
    if type(layers) is not int or not 1 <= layers <= len(blocks):
        raise ValueError('Research adapter last_encoder_layers must fit the encoder')
    protocol = metadata.get('protocol')
    if not isinstance(protocol, dict) or type(protocol.get('last_encoder_layers')) is not int or protocol['last_encoder_layers'] != layers:
        raise ValueError('Research adapter layer count differs from training metadata')
    parameters = artifact.get('adapter_parameters')
    if not isinstance(parameters, dict) or not parameters or any(not isinstance(key, str) for key in parameters):
        raise ValueError('Research adapter_parameters must be a nonempty named tensor map')
    prefixes = [f'encoder.encoders.{index}.' for index in range(len(blocks) - layers, len(blocks))]
    if getattr(reader.model.encoder, 'after_norm', None) is not None:
        prefixes.append('encoder.after_norm.')
    expected = {name: parameter for name, parameter in reader.model.named_parameters()
                if name.startswith(tuple(prefixes))}
    if not expected or set(parameters) != set(expected):
        missing, unexpected = sorted(set(expected) - set(parameters)), sorted(set(parameters) - set(expected))
        raise ValueError(f'Research adapter must contain the complete selected encoder parameters; '
                         f'missing={missing}, unexpected={unexpected}')
    if type(metadata.get('adapter_tensor_count')) is not int or metadata['adapter_tensor_count'] != len(expected):
        raise ValueError('Research adapter parameter count differs from training metadata')
    changed = metadata.get('changed_tensor_count')
    if type(changed) is not int or not 0 <= changed <= len(expected):
        raise ValueError('Research adapter changed_tensor_count is invalid')
    for name, parameter in expected.items():
        tensor = parameters[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f'Research adapter tensor must be dense: {name}')
        if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
            raise ValueError(f'Research adapter shape/dtype differs from the base: {name}')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Research adapter contains non-finite values: {name}')
    # All checks and device allocations precede copies. Invalid files cannot
    # partly overwrite a reader and then be accepted with the remaining base.
    prepared = {name: tensor.to(device=expected[name].device) for name, tensor in parameters.items()}
    with torch.no_grad():
        for name, tensor in prepared.items():
            expected[name].copy_(tensor)
    reader.model.eval()
    return {'file_sha256': digest, 'schema_version': 1, 'language': 'zh',
            'base_checkpoint_sha256': CHECKPOINT_SHA256, 'last_encoder_layers': layers,
            'loaded_parameter_count': len(expected), 'changed_tensor_count_reported': changed,
            'strict_validation': True, 'automatic_activation': False,
            'usage': 'Local research evaluation only; no GUI or personal model installation'}


def configured_hypotheses(reader, encoded, *, beam_size, ctc_weight, nbest=5, reverse_weight=0.0):
    """Apply one frozen configuration; this function never sees labels or selects settings."""
    if reverse_weight == 0:
        hypotheses = reader.hypotheses(encoded, nbest=nbest)
        return hypotheses, [asdict(hypothesis) for hypothesis in hypotheses]
    from compare_chinese_decoders import decode_candidates, rerank_candidates, reverse_log_probability
    from lipflow.confidence import Hypothesis

    candidates = decode_candidates(reader, encoded, ctc_weight, beam_size, nbest)
    candidates = rerank_candidates(
        candidates, ctc_weight, reverse_weight,
        lambda ids: reverse_log_probability(reader.model.r_decoder, encoded, ids,
                                            len(reader.token_list) - 1),
    )
    hypotheses = [Hypothesis(row.text, row.score, max(len(row.token_ids), 1)) for row in candidates]
    return hypotheses, [asdict(row) for row in candidates]


def run(args) -> dict:
    from evaluate_chinese import ClipRejected, _visual_input
    from lipflow.confidence import assess

    dataset = load_manifest(args.manifest)
    started = time.monotonic()
    reader = _reader(Path(args.checkpoint), Path(args.source_dir), args.beam_size, args.ctc_weight, args.device)
    adapter = _load_adapter(reader, Path(args.adapter)) if getattr(args, 'adapter', None) else None
    reader.warmup()
    startup = time.monotonic() - started
    predictions, candidates = [], {}
    for sample in dataset.samples:
        started, duration = time.monotonic(), None
        try:
            # The model receives images only. References stay in evaluate().
            rois, duration, quality = _visual_input(sample, reader)
            encoded = reader.encode(rois)
            hypotheses, candidate_evidence = configured_hypotheses(
                reader, encoded, beam_size=args.beam_size, ctc_weight=args.ctc_weight,
                nbest=getattr(args, 'nbest', 5), reverse_weight=getattr(args, 'reverse_weight', 0.0),
            )
            decision = assess(hypotheses, reader.greedy(encoded), quality, policy='review', language='zh')
            prediction = Prediction(
                sample.id, hypotheses[0].text if hypotheses else '', duration,
                time.monotonic() - started, decision.action, decision.reason, margin=decision.margin,
            )
            candidates[sample.id] = candidate_evidence
        except ClipRejected as exc:
            prediction = Prediction(sample.id, '', exc.duration, time.monotonic() - started,
                                    'retry', reason=str(exc))
        except Exception as exc:
            # Failed samples remain in the CER denominator.
            prediction = Prediction(sample.id, '', duration, time.monotonic() - started,
                                    'retry', error=f'{type(exc).__name__}: {exc}')
        predictions.append(prediction)
        print(f'[{len(predictions)}/{len(dataset.samples)}] {sample.id}: {prediction.action}', file=sys.stderr)

    report = evaluate(dataset, predictions)
    vocabulary = set(reader.token_list)
    report['model'] = {
        'name': 'CNVSRC2025 research comparison', 'language': 'zh',
        'source_url': 'https://github.com/liu12366262626/CNVSRC2025',
        'source_revision': SOURCE_REVISION, 'configuration_sha256': CONFIG_SHA256,
        'weight_url': 'https://huggingface.co/ReflectionL/CNVSRC2025Baseline',
        'weight_revision': WEIGHT_REVISION, 'checkpoint_sha256': CHECKPOINT_SHA256,
        'vocabulary_sha256': VOCABULARY_SHA256, 'vocabulary_size': len(reader.token_list),
        'strict_load': True, 'state_tensor_count': reader.tensor_count,
        'state_tensor_values_including_buffers': reader.tensor_values,
        'beam_size': args.beam_size, 'ctc_weight': args.ctc_weight,
        'nbest': getattr(args, 'nbest', 5), 'reverse_weight': getattr(args, 'reverse_weight', 0.0),
        'decoder_configuration_selection_performed': False,
        'length_bonus': 0.0, 'external_lm': False, 'personal': False,
        'encoder_device': str(reader.enc_device), 'decoder_device': str(reader.device),
        'startup_seconds_including_verification_and_warmup': startup,
        'adapter': adapter,
    }
    for row in report['samples']:
        row['hypotheses'] = candidates.get(row['sample_id'], [])
        # Vocabulary diagnostics do not change hypotheses, sample inclusion or CER.
        row['reference_oov_characters'] = sorted(set(characters(row['reference'])) - vocabulary)
    report['limitations'].extend([
        'This optional research model is not integrated into the GUI or used for automatic input.',
        'The author restricts code/model usage to non-commercial comparative/benchmarking research.',
        'Published CNVSRC scores describe that benchmark, not unvoiced webcam dictation.',
    ])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--accept-research-license', action='store_true')
    parser.add_argument('--checkpoint', required=True, help='Local model_avg_cncvs_2_3_cnvsrc.pth')
    parser.add_argument('--source-dir', required=True, help='Local pinned CNVSRC2025 checkout, or its VSR directory')
    parser.add_argument('--manifest', required=True, help='Local schema_version=1 evaluation manifest')
    parser.add_argument('--output')
    parser.add_argument('--adapter', help='Local complete encoder_adapter.pth from the research training script')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--beam-size', type=int, default=40)
    parser.add_argument('--ctc-weight', type=float, default=.5)
    parser.add_argument('--nbest', type=int, default=None,
                        help='Frozen candidate count (default min(5, beam-size)); do not select on test data')
    parser.add_argument('--reverse-weight', type=float, default=0.0,
                        help='Frozen reverse rescoring weight selected on dev only; 0 preserves original decoding')
    args = parser.parse_args()
    if not args.accept_research_license:
        parser.error('Read the author VSR/LICENSE and pass --accept-research-license for non-commercial research')
    if args.beam_size < 1 or not math.isfinite(args.ctc_weight) or not 0 <= args.ctc_weight <= 1:
        parser.error('--beam-size must be positive and --ctc-weight finite within [0,1]')
    if args.nbest is None:
        args.nbest = min(5, args.beam_size)
    if not 1 <= args.nbest <= args.beam_size:
        parser.error('--nbest must be within [1, beam-size]')
    if not math.isfinite(args.reverse_weight) or not 0 <= args.reverse_weight <= 1:
        parser.error('--reverse-weight must be finite within [0,1]')
    try:
        report = run(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    return 0 if report['readiness']['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

"""Fetch a small, reproducible VISUAL subset from official Chinese-LiPS.

No authentication, gate bypass, whole ZIP, WAV, slide, or OCR download. Media
stay local and must not be committed or redistributed with the application.
The source is normal speech recorded with video, not intentional mute speech.

    python scripts/fetch_chinese_lips.py --accept-noncommercial-license \
        --output-dir /tmp/lipflow-chinese-lips
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib

REPOSITORY = 'BAAI/Chinese-LiPS'
REVISION = 'db96948538811029011eee44602438a26710ecd9'
SOURCE = 'https://huggingface.co/datasets/' + REPOSITORY
LICENSE = 'CC-BY-NC-SA-4.0'
FILES = {
    'train': ('meta_train.csv', 8766725, 'bc0150e082657698019a976b3a280c23f0640df38603f658870f548fe4268bac',
              'processed_train.zip', 8730189641, '1b153f840d8f5bd29bcc89cd46d602fe89066a6ae2420af51fcf5dbbb9a5cefe'),
    'dev': ('meta_valid.csv', 555046, '29b5ed52e4ec3a44de3284a81362165c4088e18973fe301d286282a471edcec9',
            'processed_val.zip', 546449545, 'cafd3aa2b53749dbb7026867873ea4db988162694fb8705aaa8024ec94fd2e9f'),
    'test': ('meta_test.csv', 1102537, 'f2ad37b496446043d5182dddaa3d4114a403292c154416f12be06b9f94cdf095',
             'processed_test.zip', 1044092064, '77726bcec79f6cc9ad09a3c814edd35f268308e5f4a4762c18b5797cf644deba'),
}
ID_PATTERN = re.compile(r'^\d+_\d+_[MF]_[A-Z]+_\d+$')


class DownloadBudget:
    def __init__(self, maximum: int):
        if maximum <= 0:
            raise ValueError('download budget must be positive')
        self.maximum, self.requested, self.lock = maximum, 0, threading.Lock()

    def reserve(self, count: int):
        with self.lock:
            if count < 0 or self.requested + count > self.maximum:
                raise ValueError('download budget exceeded; no full archive fallback is allowed')
            self.requested += count


def url_for(filename: str) -> str:
    return f'{SOURCE}/resolve/{REVISION}/{filename}'


def fetch_range(url: str, size: int, start: int, count: int, budget: DownloadBudget,
                opener=urllib.request.urlopen) -> bytes:
    if start < 0 or count < 0 or start + count > size:
        raise ValueError('invalid byte range')
    if count == 0:
        return b''
    end = start + count - 1
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    for attempt in range(3):
        budget.reserve(count)  # Every retry is charged before opening the connection.
        try:
            with opener(request, timeout=45) as response:
                expected = f'bytes {start}-{end}/{size}'
                # Reject before reading a server that ignores Range; never fetch its
                # entire multi-GB archive, even if HTTP reports success.
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError(f'Server did not honor bounded HTTP Range: {response.status}')
                data = response.read(count + 1)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == 2:
                raise
        time.sleep(.5 * (2 ** attempt))
    if len(data) != count:
        raise ValueError('truncated or oversized range response')
    return data


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int, budget: DownloadBudget):
        self.url, self.size, self.budget, self.pos = url, size, budget, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('invalid seek mode')
        target = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if target < 0:
            raise ValueError('negative seek')
        self.pos = target
        return target

    def read(self, count=-1):
        if self.pos >= self.size:
            return b''
        count = self.size - self.pos if count < 0 else min(count, self.size - self.pos)
        result = fetch_range(self.url, self.size, self.pos, count, self.budget)
        self.pos += count
        return result


def select_rows(rows: list[dict], count: int, speakers: int, seed: str) -> list[dict]:
    """Hash-ranked speakers and clips, round-robin, independent of transcript."""
    if count < speakers or speakers < 1:
        raise ValueError('clip count must be >= positive speaker count')
    groups = {}
    for row in rows:
        identifier = row.get('ID', '')
        if not ID_PATTERN.fullmatch(identifier) or not row.get('TEXT', '').strip():
            raise ValueError('invalid source ID or empty author label')
        groups.setdefault(identifier.split('_')[0], []).append(row)
    rank = lambda value: hashlib.sha256((seed + ':' + value).encode()).digest()
    identities = sorted(groups, key=rank)[:speakers]
    if len(identities) != speakers:
        raise ValueError('not enough source speakers')
    ranked = {speaker: sorted(groups[speaker], key=lambda row: rank(row['ID'])) for speaker in identities}
    selected, round_number = [], 0
    while len(selected) < count:
        before = len(selected)
        for speaker in identities:
            if round_number < len(ranked[speaker]):
                selected.append(ranked[speaker][round_number])
                if len(selected) == count:
                    break
        if len(selected) == before:
            raise ValueError('not enough source clips for requested speakers')
        round_number += 1
    return selected


def read_member(url: str, size: int, info: zipfile.ZipInfo, budget: DownloadBudget) -> bytes:
    """Read exactly one known ZIP member; do not extract paths from the ZIP."""
    if info.file_size > 2_000_000 or info.compress_size > 2_000_000:
        raise ValueError('unexpectedly large mouth video member')
    header = fetch_range(url, size, info.header_offset, 30, budget)
    signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack('<4s5H3I2H', header)
    if signature != b'PK\x03\x04' or flags & 1 or method != info.compress_type:
        raise ValueError('invalid or encrypted local ZIP header')
    block = fetch_range(url, size, info.header_offset + 30,
                        name_len + extra_len + info.compress_size, budget)
    encoding = 'utf-8' if flags & 0x800 else 'cp437'
    if block[:name_len].decode(encoding) != info.filename:
        raise ValueError('local ZIP name differs from central directory')
    compressed = block[name_len + extra_len:]
    if method == zipfile.ZIP_STORED:
        data = compressed
    elif method == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-15)
        data = inflater.decompress(compressed, info.file_size + 1)
        if not inflater.eof or inflater.unconsumed_tail or inflater.unused_data:
            raise ValueError('invalid or oversized compressed member')
    else:
        raise ValueError('unsupported archive compression method')
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError('member size/CRC verification failed')
    return data


def _metadata(filename, size, digest, output, budget):
    path = output / 'metadata' / filename
    data = path.read_bytes() if path.exists() else fetch_range(url_for(filename), size, 0, size, budget)
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('source metadata SHA256 verification failed')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def test_exclusions(path: Path | None) -> tuple[frozenset[str], dict | None]:
    """Reserve new speakers without reading prior recognition or ranking labels."""
    if path is None:
        return frozenset(), None
    encoded = path.read_bytes()
    manifest = json.loads(encoded)
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1 or
            manifest.get('split') != 'test' or not isinstance(manifest.get('samples'), list) or
            not manifest['samples']):
        raise ValueError('Exclusion manifest must be a nonempty schema-1 test partition')
    source = manifest.get('source', {})
    if source.get('repository') != REPOSITORY or source.get('revision') != REVISION:
        raise ValueError('Exclusion manifest must use the same pinned Chinese-LiPS source')
    excluded = set()
    for sample in manifest['samples']:
        if not isinstance(sample, dict):
            raise ValueError('Exclusion manifest has an invalid sample')
        speaker = sample.get('speaker', '')
        identifier = sample.get('source_id', '')
        if (not isinstance(speaker, str) or not re.fullmatch(r'chinese-lips:\d+', speaker) or
                not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier) or
                speaker.removeprefix('chinese-lips:') != identifier.split('_')[0]):
            raise ValueError('Exclusion speaker must match the pinned source ID')
        excluded.add(speaker.removeprefix('chinese-lips:'))
    return frozenset(excluded), {
        'manifest_sha256': hashlib.sha256(encoded).hexdigest(),
        'excluded_speaker_ids': sorted(excluded), 'prior_sample_count': len(manifest['samples']),
        'purpose': 'Fresh final test speakers; exclusion uses identities only, not prediction quality',
    }


def prepare_subset(output: Path, counts: dict, speakers: dict, seed: str, maximum: int,
                   exclude_test_manifest: Path | None = None, download_workers: int = 4) -> dict:
    if type(download_workers) is not int or not 1 <= download_workers <= 16:
        raise ValueError('download workers must be an integer within [1,16]')
    excluded, exclusion_evidence = test_exclusions(exclude_test_manifest)
    output.mkdir(parents=True, exist_ok=True)
    budget, manifests, identity_sets = DownloadBudget(maximum), {}, {}
    for split in ('dev', 'train', 'test'):
        meta, meta_size, meta_hash, archive, archive_size, archive_hash = FILES[split]
        rows = _metadata(meta, meta_size, meta_hash, output, budget)
        identity_sets[split] = {row['ID'].split('_')[0] for row in rows}
        selection_rows = rows
        if split == 'test' and excluded:
            if not excluded.issubset(identity_sets[split]):
                raise ValueError('Excluded speaker does not belong to the pinned official test split')
            selection_rows = [row for row in rows if row['ID'].split('_')[0] not in excluded]
        selected = select_rows(selection_rows, counts[split], speakers[split], seed + ':' + split)
        remote = RangeFile(url_for(archive), archive_size, budget)
        with zipfile.ZipFile(remote) as zipped:
            index = {entry.filename: entry for entry in zipped.infolist()}
        folder = archive.removesuffix('.zip')
        manifest = {
            'schema_version': 1, 'split': split, 'training_overlap_checked': False,
            'description': ('Official Chinese-LiPS speaker-disjoint subset; original normal speech, '
                            'already cropped mouth videos. Base model training overlap is not fully audited; '
                            'this does not establish intentional mute or live webcam readiness.'),
            'source': {'repository': REPOSITORY, 'revision': REVISION, 'license': LICENSE,
                       'metadata_file': meta, 'metadata_sha256': meta_hash, 'archive': archive,
                       'archive_bytes': archive_size, 'archive_sha256_from_publisher': archive_hash,
                       'whole_archive_hash_verified': False, 'member_crc_verified': True,
                       'seed': seed, 'selection': 'hash-ranked speakers and IDs, round-robin; no label filtering',
                       'prior_test_exclusions': exclusion_evidence if split == 'test' else None,
                       'audio_downloaded': False, 'slides_downloaded': False,
                       'label_provenance': 'Author-provided manual transcript; no model-generated labels'},
            'samples': [],
        }
        def fetch(row):
            identifier = row['ID']
            member = f'{folder}/{identifier}.mp4'
            if member not in index:
                raise ValueError(f'author metadata has no matching video: {identifier}')
            info = index[member]
            path = output / 'videos' / split / f'{identifier}.mp4'
            cached = path.read_bytes() if path.exists() else b''
            if len(cached) == info.file_size and zlib.crc32(cached) == info.CRC:
                data = cached
            else:
                data = read_member(url_for(archive), archive_size, info, budget)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.part')
                temporary.write_bytes(data)
                temporary.replace(path)
            return {'id': f'chinese-lips:{split}:{identifier}',
                    'video': str(path.relative_to(output)), 'reference': row['TEXT'],
                    'speaker': f'chinese-lips:{identifier.split("_")[0]}',
                    # A topic recording identity, not invented per-clip sessions.
                    'session': identifier.rsplit('_', 1)[0],
                    'split': split, 'domain': 'mouth_roi', 'mouth_roi': True,
                    'articulation': 'voiced', 'articulation_verified': True,
                    'label_source': f'{SOURCE}/blob/{REVISION}/{meta}#ID={identifier}',
                    'label_verified': True, 'sha256': hashlib.sha256(data).hexdigest(),
                    'archive_member': member, 'archive_member_crc32': f'{info.CRC:08x}',
                    'source_id': identifier, 'topic': row['TOPIC']}
        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            manifest['samples'] = list(pool.map(fetch, selected))
        manifests[split] = manifest
        (output / f'{split}.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'{split}: {len(selected)} clips, {speakers[split]} speakers; {budget.requested:,} bytes requested', flush=True)
    for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        if identity_sets[a] & identity_sets[b]:
            raise ValueError('official metadata has overlapping speaker identities across splits')
    manifests['test']['development_samples'] = [
        {key: row[key] for key in ('id', 'speaker', 'session', 'video')}
        for split in ('train', 'dev') for row in manifests[split]['samples']]
    manifests['test']['source']['official_speaker_split_overlap_checked'] = True
    (output / 'test.json').write_text(json.dumps(manifests['test'], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {'source': SOURCE, 'revision': REVISION, 'license': LICENSE,
               'seed': seed, 'prior_test_exclusions': exclusion_evidence,
               'download_bytes_requested': budget.requested, 'download_budget_bytes': maximum,
               'download_workers': download_workers,
               'download_accounting': ('Requested ranges in this invocation, including retries; '
                                       'previous runs and locally cached members are not included.'),
               'splits': {key: {'clips': len(value['samples']), 'speakers': len({s['speaker'] for s in value['samples']}),
                               'manifest': str(output / f'{key}.json')} for key, value in manifests.items()},
               'limitations': ['Only author mouth crops; not live webcam or intentional mute recordings.',
                               'Unknown base model pretraining identity/content overlap.',
                               'Author labels are not independently re-transcribed by this fetcher.']}
    (output / 'provenance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--accept-noncommercial-license', action='store_true',
                        help='Accept source CC-BY-NC-SA-4.0 attribution/noncommercial/share-alike terms')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-download-mb', type=int, default=500)
    parser.add_argument('--seed', default='lipflow-chinese-lips-v1')
    parser.add_argument('--download-workers', type=int, default=4,
                        help='Bounded parallel range downloads (1-16); cached videos are reused')
    parser.add_argument('--exclude-test-manifest', type=Path,
                        help='Prior inspected test manifest; reserve wholly different final test speakers')
    for split, count, speakers in [('train', 180, 12), ('dev', 30, 6), ('test', 50, 10)]:
        parser.add_argument(f'--{split}-count', type=int, default=count)
        parser.add_argument(f'--{split}-speakers', type=int, default=speakers)
    args = parser.parse_args(argv)
    if not args.accept_noncommercial_license:
        parser.error('Read the official dataset license, then pass --accept-noncommercial-license')
    try:
        counts = {split: getattr(args, f'{split}_count') for split in FILES}
        speakers = {split: getattr(args, f'{split}_speakers') for split in FILES}
        for split in FILES:
            if counts[split] < speakers[split] or speakers[split] < 1:
                raise ValueError(f'{split}: clip count must be >= positive speaker count')
        summary = prepare_subset(args.output_dir.resolve(), counts, speakers, args.seed,
                                 args.max_download_mb * 1_000_000, args.exclude_test_manifest,
                                 args.download_workers)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

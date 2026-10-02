"""No-network tests for bounded official dataset extraction and split selection."""
import importlib.util
import io
import json
import urllib.error
from pathlib import Path
import zipfile

import pytest

_spec = importlib.util.spec_from_file_location('fetch_chinese_lips', Path(__file__).resolve().parents[1] / 'scripts' / 'fetch_chinese_lips.py')
fetcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetcher)


class Response(io.BytesIO):
    def __init__(self, data, status, content_range):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': content_range}
        self.read_called = False

    def read(self, size=-1):
        self.read_called = True
        return super().read(size)


def test_server_ignoring_range_is_rejected_before_body_read():
    response = Response(b'large archive', 200, None)
    with pytest.raises(ValueError, match='honor bounded'):
        fetcher.fetch_range('https://example.test/archive.zip', 1_000_000, 0, 4,
                            fetcher.DownloadBudget(20), opener=lambda *a, **k: response)
    assert not response.read_called


def test_wrong_content_range_is_rejected_before_body_read():
    response = Response(b'data', 206, 'bytes 0-3/100')
    with pytest.raises(ValueError, match='honor bounded'):
        fetcher.fetch_range('https://example.test/archive.zip', 100, 4, 4,
                            fetcher.DownloadBudget(20), opener=lambda *a, **k: response)
    assert not response.read_called


def test_budget_stops_request_before_network():
    calls = []
    with pytest.raises(ValueError, match='budget exceeded'):
        fetcher.fetch_range('https://example.test/archive.zip', 100, 0, 11,
                            fetcher.DownloadBudget(10), opener=lambda *a, **k: calls.append(a))
    assert not calls


def test_truncated_response_is_not_accepted():
    with pytest.raises(ValueError, match='truncated'):
        fetcher.fetch_range('https://example.test/archive.zip', 10, 0, 4,
                            fetcher.DownloadBudget(20),
                            opener=lambda *a, **k: Response(b'abc', 206, 'bytes 0-3/10'))


def test_transient_connection_retry_uses_same_bounded_range_and_budget(monkeypatch):
    monkeypatch.setattr(fetcher.time, 'sleep', lambda seconds: None)
    calls, budget = [], fetcher.DownloadBudget(12)
    def opener(request, **kwargs):
        calls.append(request.headers['Range'])
        if len(calls) == 1:
            raise urllib.error.URLError('temporary connection loss')
        return Response(b'data', 206, 'bytes 0-3/10')
    assert fetcher.fetch_range('https://example.test', 10, 0, 4, budget, opener) == b'data'
    assert calls == ['bytes=0-3', 'bytes=0-3']
    assert budget.requested == 8


def test_retries_exhaust_budget_before_another_network_request(monkeypatch):
    monkeypatch.setattr(fetcher.time, 'sleep', lambda seconds: None)
    calls = []
    def opener(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError('temporary loss')
    with pytest.raises(ValueError, match='budget exceeded'):
        fetcher.fetch_range('https://example.test', 10, 0, 4, fetcher.DownloadBudget(4), opener)
    assert len(calls) == 1


def test_missing_resource_is_not_retried(monkeypatch):
    monkeypatch.setattr(fetcher.time, 'sleep', lambda seconds: pytest.fail('must not retry404'))
    calls = []
    def opener(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://example.test', 404, 'Not found', {}, None)
    with pytest.raises(urllib.error.HTTPError):
        fetcher.fetch_range('https://example.test', 10, 0, 4, fetcher.DownloadBudget(100), opener)
    assert len(calls) == 1


def rows():
    return [{'ID': f'{speaker:03}_25_M_KJ_{clip:03}', 'TEXT': '作者提供的标注'}
            for speaker in range(1, 6) for clip in range(1, 8)]


def test_selection_is_deterministic_balanced_and_does_not_rank_labels():
    source = rows()
    selected = fetcher.select_rows(source, 12, 3, 'seed')
    modified = [dict(row, TEXT='另一份非空标注') for row in reversed(source)]
    alternative = fetcher.select_rows(modified, 12, 3, 'seed')
    assert [row['ID'] for row in selected] == [row['ID'] for row in alternative]
    speakers = {row['ID'].split('_')[0] for row in selected}
    assert len(speakers) == 3
    assert all(sum(row['ID'].startswith(s + '_') for row in selected) == 4 for s in speakers)


def test_malicious_source_identifier_rejected():
    with pytest.raises(ValueError, match='invalid source'):
        fetcher.select_rows([{'ID': '../../other-file', 'TEXT': '测试'}], 1, 1, 'seed')


@pytest.mark.parametrize('workers', [0, 17, True, 1.5])
def test_invalid_download_parallelism_rejected_before_output(tmp_path, workers):
    with pytest.raises(ValueError, match='download workers'):
        fetcher.prepare_subset(tmp_path/'new', {}, {}, 'seed', 100, download_workers=workers)
    assert not (tmp_path/'new').exists()


def test_new_test_speakers_exclude_all_prior_speakers_without_ranking_references(tmp_path):
    prior = {'schema_version': 1, 'split': 'test',
             'source': {'repository': fetcher.REPOSITORY, 'revision': fetcher.REVISION},
             'samples': [{'speaker': 'chinese-lips:001', 'source_id': '001_25_M_KJ_001',
                          'reference': '以前的标签'}]}
    path = tmp_path / 'prior-test.json'
    path.write_text(json.dumps(prior))
    excluded, evidence = fetcher.test_exclusions(path)
    assert excluded == {'001'} and evidence['prior_sample_count'] == 1
    remaining = [row for row in rows() if row['ID'].split('_')[0] not in excluded]
    selected = fetcher.select_rows(remaining, 12, 3, 'new-seed')
    assert all(not row['ID'].startswith('001_') for row in selected)
    prior['samples'][0]['reference'] = '另一份内容完全不同的标签'
    path.write_text(json.dumps(prior))
    assert fetcher.test_exclusions(path)[0] == excluded
    assert [row['ID'] for row in selected] == [row['ID'] for row in fetcher.select_rows(
        [dict(row, TEXT='更改标签') for row in remaining], 12, 3, 'new-seed')]


@pytest.mark.parametrize('changes', [
    {'split': 'train'}, {'samples': []},
    {'source': {'repository': fetcher.REPOSITORY, 'revision': 'wrong'}},
    {'samples': [{'speaker': 'chinese-lips:001', 'source_id': '002_25_M_KJ_001'}]},
])
def test_invalid_test_exclusion_refuses_before_download(tmp_path, monkeypatch, changes):
    prior = {'schema_version': 1, 'split': 'test',
             'source': {'repository': fetcher.REPOSITORY, 'revision': fetcher.REVISION},
             'samples': [{'speaker': 'chinese-lips:001', 'source_id': '001_25_M_KJ_001'}]}
    prior.update(changes)
    path = tmp_path / 'prior.json'
    path.write_text(json.dumps(prior))
    calls = []
    monkeypatch.setattr(fetcher, '_metadata', lambda *a: calls.append(a))
    with pytest.raises(ValueError):
        fetcher.prepare_subset(tmp_path / 'new', {}, {}, 'seed', 100, path)
    assert not calls and not (tmp_path / 'new').exists()


def archive_bytes(data=b'mouth video bytes'):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr('processed_test/001_25_M_KJ_001.mp4', data)
    encoded = output.getvalue()
    with zipfile.ZipFile(io.BytesIO(encoded)) as zipped:
        info = zipped.infolist()[0]
    return encoded, info


def test_member_extraction_reads_only_requested_ranges_and_verifies_crc(monkeypatch):
    encoded, info = archive_bytes()
    requests = []
    def range_read(url, size, start, count, budget):
        requests.append((start, count))
        budget.reserve(count)
        return encoded[start:start + count]
    monkeypatch.setattr(fetcher, 'fetch_range', range_read)
    assert fetcher.read_member('https://example.test', len(encoded), info, fetcher.DownloadBudget(1000)) == b'mouth video bytes'
    assert len(requests) == 2
    assert sum(count for _, count in requests) < len(encoded)
    info.CRC ^= 1
    with pytest.raises(ValueError, match='CRC verification'):
        fetcher.read_member('https://example.test', len(encoded), info, fetcher.DownloadBudget(1000))


def test_declared_oversized_member_is_rejected_without_network(monkeypatch):
    _, info = archive_bytes()
    info.file_size = 2_000_001
    calls = []
    monkeypatch.setattr(fetcher, 'fetch_range', lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='large mouth'):
        fetcher.read_member('https://example.test', 10, info, fetcher.DownloadBudget(1000))
    assert not calls


def test_installer_requires_explicit_noncommercial_license(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fetcher, 'prepare_subset', lambda *args: calls.append(args))
    with pytest.raises(SystemExit) as raised:
        fetcher.main(['--output-dir', str(tmp_path)])
    assert raised.value.code == 2
    assert not calls

import hashlib
import os
from types import SimpleNamespace

from src.modeling import _bundle_matches_data


def test_git_checkout_timestamp_does_not_invalidate_identical_data(tmp_path):
    source = tmp_path / 'data.csv'
    source.write_bytes(b'x,y\n1,2\n')
    bundle = SimpleNamespace(data_file=source.name, data_mtime=source.stat().st_mtime,
                             data_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    os.utime(source, (1, 1))
    assert _bundle_matches_data(bundle, source)
    source.write_bytes(b'x,y\n1,3\n')
    os.utime(source, (1, 1))
    assert not _bundle_matches_data(bundle, source)


def test_legacy_bundle_still_checks_filename_and_timestamp(tmp_path):
    source = tmp_path / 'data.csv'
    source.write_bytes(b'x,y\n1,2\n')
    bundle = SimpleNamespace(data_file=source.name, data_mtime=source.stat().st_mtime)
    assert _bundle_matches_data(bundle, source)
    os.utime(source, (1, 1))
    assert not _bundle_matches_data(bundle, source)

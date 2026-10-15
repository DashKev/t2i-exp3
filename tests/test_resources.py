import hashlib
import json
from pathlib import Path

import pytest

from t2i_exp3.resources import download


def test_existing_resource_requires_checksum(tmp_path):
    p = tmp_path / 'asset'
    p.write_bytes(b'known resource')
    download('https://example.invalid/never-contacted', p, hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='Checksum mismatch'):
        download('https://example.invalid/never-contacted', p, '0' * 64)


def test_public_source_tree_has_no_private_references():
    root = Path(__file__).resolve().parents[1]
    prohibited = ['/' + 'home/', '/' + 'srv/', '10.' + '176.']
    for folder in ['src', 'configs', 'data', 'docs']:
        for p in (root / folder).rglob('*'):
            if p.is_file() and p.suffix in {'.py', '.json', '.md'} and '__pycache__' not in str(p):
                if p.name.endswith('.local.json'):
                    continue
                if 'raw' in p.parts or 'prepared' in p.parts:
                    continue
                text = p.read_text()
                assert not any(word in text for word in prohibited), p

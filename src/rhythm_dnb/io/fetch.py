"""Explicit acquisition of a pinned public artifact; inference never calls this."""

from pathlib import Path
from urllib.request import urlopen
from ..provenance import file_hash


def fetch_source(url, destination, expected_sha256):
    # PSEUDOCODE: require HTTPS/hash -> stream to exclusive partial file -> verify before publishing.
    if not url.startswith('https://') or len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('A pinned HTTPS artifact is required.')
    destination = Path(destination)
    if destination.exists():
        if file_hash(destination) != expected_sha256:
            raise ValueError('Existing source differs from the pinned digest.')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(destination.suffix + '.partial')
    with urlopen(url, timeout=60) as response, pending.open('xb') as stream:
        for block in iter(lambda: response.read(1024 * 1024), b''):
            stream.write(block)
    if file_hash(pending) != expected_sha256:
        raise ValueError('Download checksum mismatch; partial retained for inspection.')
    # Hard-link creation is exclusive and cannot replace a concurrently created destination.
    destination.hardlink_to(pending)
    pending.unlink()
    return destination

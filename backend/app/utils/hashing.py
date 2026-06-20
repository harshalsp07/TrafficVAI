"""SHA-256 hashing utilities for evidence files.

Used to produce tamper-evident digests that can be referenced in
BSA Section 63 certificates and audit logs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1 << 16  # 64 KiB


def sha256_hex(file_path: str | Path) -> str:
    """Return the lowercase hex SHA-256 digest of *file_path*.

    Reads the file in 64 KiB chunks to keep memory usage constant regardless
    of file size.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of raw *data* bytes."""
    return hashlib.sha256(data).hexdigest()

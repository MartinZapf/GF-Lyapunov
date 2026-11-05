"""Small, solver-independent helpers for campaign artifact publication."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def atomic_target(path):
    """Yield a temporary sibling for binary writers, then publish after success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".",
                                         suffix=path.suffix)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_text(path, text):
    """Publish UTF-8 text through a unique sibling, preserving an old file on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path, data):
    atomic_text(path, json.dumps(data, indent=2, ensure_ascii=True) + "\n")


def result_receipt(path, artifact, systems):
    artifact = Path(artifact).resolve()
    atomic_json(path, {"schema": "gf.result_receipt/1", "artifact": str(artifact),
                       "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                       "requested_systems": list(systems)})

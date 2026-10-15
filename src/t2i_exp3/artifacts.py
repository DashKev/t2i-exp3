from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: str | Path, value: Any) -> None:
    path = Path(path)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        _write_all(fd, payload)
        _fsync_best_effort(fd)
    finally:
        os.close(fd)


def append_jsonl(path: str | Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    fd = os.open(Path(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        _write_all(fd, payload)
        _fsync_best_effort(fd)
    finally:
        os.close(fd)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _fsync_best_effort(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError as exc:
        unsupported = {errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL)}
        if exc.errno not in unsupported:
            raise


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

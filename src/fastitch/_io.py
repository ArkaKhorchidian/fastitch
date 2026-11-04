"""Input handling: paths (plain or gzip), raw bytes, or file-like objects."""
from __future__ import annotations

import io
import os
import re
from datetime import date
from typing import BinaryIO, Iterator, Union

Source = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview, BinaryIO]

DEFAULT_CHUNK = 16 << 20  # 16 MiB


def _open_gzip(path):
    try:  # Intel ISA-L: ~3-4x faster gzip decompression when installed
        from isal import igzip

        return igzip.open(path, "rb")
    except ImportError:
        import gzip

        return gzip.open(path, "rb")


def iter_chunks(source: Source, chunk_size: int = DEFAULT_CHUNK) -> Iterator[bytes]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        mv = memoryview(source)
        for i in range(0, len(mv), chunk_size):
            yield bytes(mv[i : i + chunk_size])
        return
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        with open(path, "rb") as probe:
            gz = probe.read(2) == b"\x1f\x8b"
        fh = _open_gzip(path) if gz else open(path, "rb", buffering=0)
        with fh:
            yield from _read_all(fh, chunk_size)
        return
    if hasattr(source, "read"):
        yield from _read_all(source, chunk_size)
        return
    raise TypeError(f"Unsupported source type: {type(source).__name__}")


def _read_all(fh: io.RawIOBase, chunk_size: int) -> Iterator[bytes]:
    while True:
        chunk = fh.read(chunk_size)
        if not chunk:
            return
        yield chunk


_DATE_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)")


def infer_date(source: Source) -> date | None:
    """NASDAQ names files like ``01302019.NASDAQ_ITCH50.gz`` (MMDDYYYY)."""
    if not isinstance(source, (str, os.PathLike)):
        return None
    m = _DATE_RE.search(os.path.basename(os.fspath(source)))
    if not m:
        return None
    mm, dd, yyyy = (int(x) for x in m.groups())
    try:
        return date(yyyy, mm, dd)
    except ValueError:
        return None

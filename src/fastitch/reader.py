"""High-level reading API."""
from __future__ import annotations

import os
from datetime import date
from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa

from . import _core
from ._convert import (
    default_backend,
    parse_duration,
    parse_time_of_day,
    timestamp_array,
    to_backend,
    to_table,
)
from ._io import DEFAULT_CHUNK, Source, infer_date, iter_chunks
from ._spec import SPECS, resolve_types


def _layouts(codes):
    return {c: [(f.offset, f.size, f.kind) for f in SPECS[c].fields] for c in codes}


def _symbols(symbols) -> list[str] | None:
    if symbols is None:
        return None
    if isinstance(symbols, str):
        symbols = [symbols]
    return [s.upper() for s in symbols]


def _resolve_date(source, day):
    if day is False:
        return None
    if day is None:
        return infer_date(source)
    if isinstance(day, str):
        return date.fromisoformat(day)
    return day


def iter_batches(
    source: Source,
    types=None,
    *,
    symbols: Sequence[str] | str | None = None,
    start=None,
    end=None,
    date=None,
    raw_prices: bool = False,
    backend: str | None = None,
    batch_rows: int = 5_000_000,
    chunk_size: int = DEFAULT_CHUNK,
) -> Iterator[dict]:
    """Stream an ITCH file as batches of ``{message_name: frame}`` with bounded memory.

    See :func:`read` for argument descriptions. ``batch_rows`` caps how many decoded
    rows (across all types) are buffered before a batch is yielded.
    """
    codes = resolve_types(types)
    backend = backend or default_backend()
    day = _resolve_date(source, date)
    parser = _core.Parser(_layouts(codes), _symbols(symbols), parse_time_of_day(start), parse_time_of_day(end))
    for chunk in iter_chunks(source, chunk_size):
        parser.feed(chunk)
        if parser.pending_rows >= batch_rows:
            yield _emit(parser, day, raw_prices, backend)
    if parser.pending_rows:
        yield _emit(parser, day, raw_prices, backend)
    if parser.has_partial:
        raise ValueError("Input ended in the middle of a message: the file looks truncated")


def _emit(parser, day, raw_prices, backend):
    out = {}
    for code, cols in parser.take().items():
        spec = SPECS[code]
        out[spec.name] = to_backend(to_table(spec, cols, day, raw_prices), backend)
    return out


def read(
    source: Source,
    types=None,
    *,
    symbols: Sequence[str] | str | None = None,
    start=None,
    end=None,
    date=None,
    raw_prices: bool = False,
    backend: str | None = None,
    chunk_size: int = DEFAULT_CHUNK,
) -> dict:
    """Decode an ITCH 5.0 file into one DataFrame per message type.

    Parameters
    ----------
    source
        Path to a NASDAQ ITCH 5.0 file (plain or ``.gz``), raw bytes, or a binary file object.
    types
        Message types to decode: codes (``"A"``), names (``"add_order"``) or groups
        (``"orders"``, ``"trades"``, ``"admin"``, ``"auction"``, ``"all"``). Default: all.
    symbols
        Only keep messages for these tickers (market-wide messages are always kept).
    start, end
        Time-of-day window, e.g. ``start="09:30", end="16:00"`` (inclusive).
    date
        Trading date for the ``timestamp`` column. Inferred from NASDAQ-style file names
        (``01302019.NASDAQ_ITCH50.gz``) when omitted; pass ``False`` to keep timestamps
        as durations since midnight.
    raw_prices
        Keep prices as integers (1/10000 dollars) instead of floats.
    backend
        ``"polars"`` (default if installed), ``"pandas"`` or ``"arrow"``.

    Returns
    -------
    dict
        ``{message_name: DataFrame}``, e.g. ``frames["add_order"]``. Every requested
        type is present, possibly as an empty frame.
    """
    codes = resolve_types(types)
    backend = backend or default_backend()
    day = _resolve_date(source, date)
    parser = _core.Parser(_layouts(codes), _symbols(symbols), parse_time_of_day(start), parse_time_of_day(end))
    for chunk in iter_chunks(source, chunk_size):
        parser.feed(chunk)
    if parser.has_partial:
        raise ValueError("Input ended in the middle of a message: the file looks truncated")
    taken = parser.take()
    out = {}
    for code in codes:
        spec = SPECS[code]
        out[spec.name] = to_backend(to_table(spec, taken.get(code, []), day, raw_prices), backend)
    return out


def read_frame(source: Source, type: str, **kwargs):
    """Decode a single message type and return just that DataFrame."""
    codes = resolve_types(type)
    if len(codes) != 1:
        raise ValueError("read_frame takes exactly one message type; use read() for several")
    return read(source, codes, **kwargs)[SPECS[codes[0]].name]


def count(source: Source, chunk_size: int = DEFAULT_CHUNK) -> dict[str, int]:
    """Count messages per type without decoding any fields. Returns {name: count}."""
    parser = _core.Parser({})
    for chunk in iter_chunks(source, chunk_size):
        parser.feed(chunk)
    counts = parser.counts()
    return {SPECS[c].name if c in SPECS else f"unknown_{c!r}": n for c, n in sorted(counts.items(), key=lambda kv: -kv[1])}


def to_parquet(
    source: Source,
    out_dir: str | os.PathLike,
    types=None,
    *,
    symbols=None,
    start=None,
    end=None,
    date=None,
    raw_prices: bool = False,
    compression: str = "zstd",
    batch_rows: int = 5_000_000,
    chunk_size: int = DEFAULT_CHUNK,
) -> dict[str, int]:
    """Stream an ITCH file to one Parquet file per message type, in constant memory.

    Returns ``{message_name: rows_written}``.
    """
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    rows: dict[str, int] = {}
    try:
        for batch in iter_batches(
            source, types, symbols=symbols, start=start, end=end, date=date,
            raw_prices=raw_prices, backend="arrow", batch_rows=batch_rows, chunk_size=chunk_size,
        ):
            for name, table in batch.items():
                if name not in writers:
                    path = os.path.join(out_dir, f"{name}.parquet")
                    writers[name] = pq.ParquetWriter(path, table.schema, compression=compression)
                writers[name].write_table(table)
                rows[name] = rows.get(name, 0) + table.num_rows
    finally:
        for w in writers.values():
            w.close()
    return rows


def book(
    source: Source,
    symbols: Sequence[str] | str | None,
    *,
    depth: int = 5,
    every=None,
    start=None,
    end=None,
    date=None,
    raw_prices: bool = False,
    backend: str | None = None,
    chunk_size: int = DEFAULT_CHUNK,
):
    """Reconstruct the limit order book and return top-of-book snapshots.

    Parameters
    ----------
    symbols
        Tickers to rebuild (``None`` rebuilds every symbol, which needs much more memory).
    depth
        Number of price levels per side (1-50).
    every
        ``None`` emits a row whenever the top ``depth`` levels change; a duration such as
        ``"1s"`` or ``"100ms"`` samples the book on a regular time grid instead.
    start, end
        Time-of-day window for emitted rows. The book is still built from the open,
        so snapshots inside the window are exact.

    Returns
    -------
    DataFrame with ``timestamp``, ``symbol`` and ``bid_price_1, bid_size_1, ask_price_1,
    ask_size_1, ...`` columns. Empty levels have price 0 and size 0 (or null prices
    when ``raw_prices`` is False).
    """
    backend = backend or default_backend()
    day = _resolve_date(source, date)
    interval = 0 if every is None else parse_duration(every)
    if interval < 0:
        raise ValueError("every must be positive")
    eng = _core.Book(_symbols(symbols), depth, interval, parse_time_of_day(start), parse_time_of_day(end))
    for chunk in iter_chunks(source, chunk_size):
        eng.feed(chunk)
    ts_b, loc_b, cols = eng.take()
    ts = np.frombuffer(ts_b, dtype=np.uint64)
    loc = np.frombuffer(loc_b, dtype=np.uint16)
    sym_map = {k: v.rstrip() for k, v in eng.symbols().items()}

    uniq, inv = np.unique(loc, return_inverse=True)
    dictionary = pa.array([sym_map.get(int(u), str(int(u))) for u in uniq], type=pa.string())
    symbol = pa.DictionaryArray.from_arrays(pa.array(inv.astype(np.int32)), dictionary)

    arrays = [timestamp_array(ts, day), symbol]
    names = ["timestamp", "symbol"]
    for level in range(depth):
        for j, label in enumerate(("bid_price", "bid_size", "ask_price", "ask_size")):
            v = np.frombuffer(cols[4 * level + j], dtype=np.uint32)
            if label.endswith("price") and not raw_prices:
                arr = pa.array(v / 1e4, mask=(v == 0))
            elif label.endswith("size"):
                arr = pa.array(v.astype(np.int64))  # signed, so bid - ask can't wrap around
            else:
                arr = pa.array(v)
            arrays.append(arr)
            names.append(f"{label}_{level + 1}")
    return to_backend(pa.Table.from_arrays(arrays, names=names), backend)

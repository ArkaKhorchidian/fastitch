"""Turn raw column buffers from the native core into Arrow arrays/tables."""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ._spec import MessageSpec

_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
NS_PER_DAY = 86_400 * 10**9


def _fixed_strings(buf: bytes, width: int, n: int, strip: bool) -> pa.Array:
    data = pa.py_buffer(buf)
    offsets = pa.py_buffer(np.arange(0, (n + 1) * width, width, dtype=np.int64)[: n + 1].tobytes())
    arr = pa.LargeStringArray.from_buffers(n, offsets, data)
    if strip:
        arr = pc.utf8_rtrim(arr, characters=" ")
    return arr


def timestamp_array(ns: np.ndarray, day: date | None) -> pa.Array:
    if day is None:
        return pa.array(ns.astype(np.int64), type=pa.duration("ns"))
    base = np.datetime64(day.isoformat(), "ns").astype(np.int64)
    return pa.array(ns.astype(np.int64) + base, type=pa.timestamp("ns"))


def to_table(spec: MessageSpec, cols: list[bytes], day: date | None, raw_prices: bool) -> pa.Table:
    if not cols:
        return empty_table(spec, day, raw_prices)
    first = spec.fields[0]
    n = len(cols[0]) // _out_width(first)
    arrays, names = [], []
    for field, buf in zip(spec.fields, cols):
        if field.kind in ("c", "a"):
            arr = _fixed_strings(buf, field.size, n, strip=field.kind == "a")
        else:
            width = _out_width(field)
            values = np.frombuffer(buf, dtype=_UINT[width], count=n)
            if field.kind == "ts":
                arr = timestamp_array(values, day)
            elif field.kind in ("p4", "p8") and not raw_prices:
                scale = 1e4 if field.kind == "p4" else 1e8
                arr = pa.array(values / scale)
            else:
                arr = pa.array(values)
        arrays.append(arr)
        names.append(field.name)
    return pa.Table.from_arrays(arrays, names=names)


def _out_width(field) -> int:
    if field.kind in ("c", "a"):
        return field.size
    return 8 if field.size == 6 else field.size


def empty_table(spec: MessageSpec, day: date | None, raw_prices: bool) -> pa.Table:
    schema = []
    for f in spec.fields:
        if f.kind in ("c", "a"):
            t = pa.large_string()
        elif f.kind == "ts":
            t = pa.timestamp("ns") if day else pa.duration("ns")
        elif f.kind in ("p4", "p8") and not raw_prices:
            t = pa.float64()
        else:
            t = pa.from_numpy_dtype(_UINT[_out_width(f)])
        schema.append(pa.field(f.name, t))
    return pa.schema(schema).empty_table()


def to_backend(table: pa.Table, backend: str):
    if backend == "arrow":
        return table
    if backend == "polars":
        import polars as pl

        return pl.from_arrow(table)
    if backend == "pandas":
        return table.to_pandas()
    raise ValueError(f"backend must be 'polars', 'pandas' or 'arrow', got {backend!r}")


def default_backend() -> str:
    try:
        import polars  # noqa: F401

        return "polars"
    except ImportError:
        try:
            import pandas  # noqa: F401

            return "pandas"
        except ImportError:
            return "arrow"


# ---------------------------------------------------------------------------
# Time parsing helpers
# ---------------------------------------------------------------------------
_UNITS = {"ns": 1, "us": 10**3, "ms": 10**6, "s": 10**9, "min": 60 * 10**9, "m": 60 * 10**9, "h": 3600 * 10**9}
_DUR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ns|us|ms|s|min|m|h)\s*$")


def parse_duration(value) -> int:
    """'1s', '250ms', '5min', timedelta or int nanoseconds -> int nanoseconds."""
    if isinstance(value, bool):
        raise TypeError("duration cannot be a bool")
    if isinstance(value, int):
        return value
    if isinstance(value, timedelta):
        return int(round(value.total_seconds() * 1e9))
    if isinstance(value, str):
        m = _DUR_RE.match(value)
        if m:
            return int(round(float(m.group(1)) * _UNITS[m.group(2)]))
    raise ValueError(f"Cannot parse duration {value!r}; use e.g. '1s', '100ms', '5min'")


def parse_time_of_day(value) -> int | None:
    """'09:30', '09:30:00.25', datetime.time, timedelta, or int ns -> ns since midnight."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, timedelta):
        return parse_duration(value)
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return ((value.hour * 60 + value.minute) * 60 + value.second) * 10**9 + value.microsecond * 1000
    if isinstance(value, str):
        parts = value.strip().split(":")
        if 2 <= len(parts) <= 3:
            h, m = int(parts[0]), int(parts[1])
            s = float(parts[2]) if len(parts) == 3 else 0.0
            return (h * 3600 + m * 60) * 10**9 + int(round(s * 1e9))
    raise ValueError(f"Cannot parse time of day {value!r}; use e.g. '09:30' or '15:59:59.5'")

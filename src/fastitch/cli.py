"""Command-line interface: ``fastitch info | convert | book``."""
from __future__ import annotations

import argparse
import sys
import time

from . import __version__, book, count, to_parquet
from ._io import iter_chunks


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _cmd_info(a):
    t0 = time.perf_counter()
    counts = count(a.file)
    dt = time.perf_counter() - t0
    total = sum(counts.values())
    width = max(len(k) for k in counts) if counts else 10
    for name, n in counts.items():
        print(f"  {name:<{width}}  {n:>14,}  {100 * n / total:5.1f}%")
    print(f"  {'total':<{width}}  {total:>14,}")
    print(f"\n{total / dt / 1e6:,.1f}M msgs/s ({dt:.2f}s)", file=sys.stderr)


def _cmd_convert(a):
    t0 = time.perf_counter()
    rows = to_parquet(a.file, a.out, a.types or None, symbols=a.symbols, start=a.start, end=a.end,
                      compression=a.compression)
    dt = time.perf_counter() - t0
    for name, n in sorted(rows.items(), key=lambda kv: -kv[1]):
        print(f"  {a.out}/{name}.parquet  {n:,} rows")
    print(f"\nwrote {sum(rows.values()):,} rows in {dt:.2f}s", file=sys.stderr)


def _cmd_book(a):
    df = book(a.file, a.symbols, depth=a.depth, every=a.every, start=a.start, end=a.end, backend="arrow")
    if a.out:
        if a.out.endswith(".csv"):
            import pyarrow.csv as pcsv

            pcsv.write_csv(df, a.out)
        else:
            import pyarrow.parquet as pq

            pq.write_table(df, a.out, compression="zstd")
        print(f"wrote {df.num_rows:,} snapshots to {a.out}", file=sys.stderr)
    else:
        print(df.slice(0, a.head).to_pandas().to_string() if _has_pandas() else df.slice(0, a.head))


def _has_pandas() -> bool:
    try:
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False


def _cmd_sample(a):
    from .synthetic import generate

    data = generate(n_messages=a.messages, seed=a.seed)
    with open(a.out, "wb") as fh:
        fh.write(data)
    print(f"wrote {a.messages:,} synthetic messages ({_size(len(data))}) to {a.out}", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fastitch", description="Fast NASDAQ TotalView-ITCH 5.0 toolkit")
    p.add_argument("--version", action="version", version=f"fastitch {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="count messages by type")
    s.add_argument("file")
    s.set_defaults(fn=_cmd_info)

    s = sub.add_parser("convert", help="convert to one Parquet file per message type")
    s.add_argument("file")
    s.add_argument("-o", "--out", required=True, help="output directory")
    s.add_argument("-t", "--types", nargs="*", help="codes, names or groups (orders, trades, ...)")
    s.add_argument("-s", "--symbols", nargs="*")
    s.add_argument("--start", help="e.g. 09:30")
    s.add_argument("--end", help="e.g. 16:00")
    s.add_argument("--compression", default="zstd")
    s.set_defaults(fn=_cmd_convert)

    s = sub.add_parser("book", help="reconstruct the order book")
    s.add_argument("file")
    s.add_argument("-s", "--symbols", nargs="+", required=True)
    s.add_argument("-d", "--depth", type=int, default=5)
    s.add_argument("-e", "--every", help="sampling interval, e.g. 1s or 100ms (default: on change)")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("-o", "--out", help="write .parquet or .csv instead of printing")
    s.add_argument("--head", type=int, default=20)
    s.set_defaults(fn=_cmd_book)

    s = sub.add_parser("sample", help="write a synthetic ITCH file to play with")
    s.add_argument("out")
    s.add_argument("-n", "--messages", type=int, default=1_000_000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_cmd_sample)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except BrokenPipeError:  # e.g. `fastitch info file | head`
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)


if __name__ == "__main__":
    main()

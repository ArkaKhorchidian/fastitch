import gzip
import random
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fastitch
from fastitch._spec import EXPECTED_LENGTHS, SPECS
from fastitch.synthetic import encode, frame, generate

DATA = generate(60_000, seed=42)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def _random_values(spec, rng):
    vals = {}
    for f in spec.fields:
        if f.kind == "c":
            vals[f.name] = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        elif f.kind == "a":
            vals[f.name] = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(1, f.size)))
        else:
            vals[f.name] = rng.randrange(0, 2 ** (8 * f.size))
    return vals


def test_spec_lengths_match_itch_5_0():
    assert {c: s.length for c, s in SPECS.items()} == EXPECTED_LENGTHS
    assert len(SPECS) == 23


@pytest.mark.parametrize("code", sorted(SPECS))
def test_every_field_of_every_message_type_round_trips(code):
    rng = random.Random(code)
    spec = SPECS[code]
    rows = [_random_values(spec, rng) for _ in range(50)]
    data = frame(encode(code, **r) for r in rows)
    table = fastitch.read(data, code, backend="arrow", raw_prices=True, date=False)[spec.name]
    assert table.num_rows == 50
    for f in spec.fields:
        col = table.column(f.name)
        if f.kind == "ts":
            col = col.cast(pa.int64())
        for row, got in zip(rows, col.to_pylist()):
            assert got == row[f.name], (f.name, got, row[f.name])


def test_timestamps_are_exact_to_the_nanosecond():
    ts = [0, 1, 34_200_000_000_001, 57_599_999_999_999, 2**48 - 1]
    data = frame(encode("S", timestamp=t, event_code="O") for t in ts)
    raw = fastitch.read(data, "S", backend="arrow", date=False)["system_event"]
    got = raw.column("timestamp").cast(pa.int64()).to_pylist()
    assert got == ts


def test_prices_are_scaled_and_dates_applied():
    data = frame([encode("A", timestamp=34_200 * 10**9, order_reference_number=1, side="B",
                         shares=100, stock="AAPL", price=1_234_567)])
    df = fastitch.read_frame(data, "add_order", backend="pandas", date="2019-01-30")
    assert df.price.iloc[0] == pytest.approx(123.4567)
    assert str(df.timestamp.iloc[0]) == "2019-01-30 09:30:00"
    assert df.stock.iloc[0] == "AAPL"


def test_date_is_inferred_from_nasdaq_file_names(tmp_path):
    p = tmp_path / "01302019.NASDAQ_ITCH50"
    p.write_bytes(frame([encode("S", timestamp=10**9, event_code="O")]))
    df = fastitch.read_frame(p, "S", backend="pandas")
    assert str(df.timestamp.iloc[0]) == "2019-01-30 00:00:01"


@pytest.mark.parametrize("chunk", [1, 2, 3, 37, 4096])
def test_results_do_not_depend_on_chunk_boundaries(chunk):
    small = generate(4_000, seed=3)
    a = fastitch.read(small, backend="arrow", date=False)
    b = fastitch.read(small, backend="arrow", date=False, chunk_size=chunk)
    for name in a:
        assert a[name].equals(b[name]), name


def test_count_matches_read():
    counts = fastitch.count(DATA)
    frames = fastitch.read(DATA, backend="arrow")
    for name, n in counts.items():
        assert frames[name].num_rows == n
    assert sum(counts.values()) == 60_000


def test_symbol_filter_keeps_market_wide_messages():
    f = fastitch.read(DATA, ["S", "R", "A", "E"], symbols="aapl", backend="pandas")
    assert set(f["add_order"].stock) == {"AAPL"}
    assert list(f["stock_directory"].stock) == ["AAPL"]
    locate = int(f["stock_directory"].stock_locate.iloc[0])
    assert set(f["order_executed"].stock_locate) == {locate}
    assert len(f["system_event"]) == 6  # locate-0 messages always pass


def test_time_window():
    f = fastitch.read(DATA, "A", start="10:00", end="10:30", backend="pandas", date=False)["add_order"]
    secs = f.timestamp.dt.total_seconds()  # microsecond precision is plenty here
    assert len(f) > 0 and secs.min() >= 36_000 and secs.max() <= 37_800


def test_type_groups_and_names():
    f = fastitch.read(DATA, ["orders", "trade"], backend="arrow")
    assert set(f) == {"add_order", "add_order_mpid", "order_executed", "order_executed_with_price",
                      "order_cancel", "order_delete", "order_replace", "trade"}
    with pytest.raises(ValueError):
        fastitch.read(DATA, "nope")


def test_requested_types_with_no_rows_are_empty_frames():
    f = fastitch.read(DATA, ["noii"], backend="polars")
    assert f["noii"].height == 0 and "far_price" in f["noii"].columns


def test_backends():
    import pandas as pd
    import polars as pl

    assert isinstance(fastitch.read_frame(DATA, "A", backend="polars"), pl.DataFrame)
    assert isinstance(fastitch.read_frame(DATA, "A", backend="pandas"), pd.DataFrame)
    assert isinstance(fastitch.read_frame(DATA, "A", backend="arrow"), pa.Table)


def test_gzip_and_file_objects(tmp_path):
    p = tmp_path / "x.gz"
    p.write_bytes(gzip.compress(DATA))
    a = fastitch.read_frame(DATA, "A", backend="arrow")
    assert fastitch.read_frame(p, "A", backend="arrow").equals(a)
    with open(p, "rb") as fh:
        assert fastitch.read_frame(gzip.GzipFile(fileobj=fh), "A", backend="arrow").equals(a)


def test_truncated_input_raises():
    with pytest.raises(ValueError, match="truncated"):
        fastitch.read(DATA[:-5])


def test_iter_batches_and_parquet(tmp_path):
    full = fastitch.read(DATA, "orders", backend="arrow")
    batches = list(fastitch.iter_batches(DATA, "orders", backend="arrow", batch_rows=5_000, chunk_size=50_000))
    assert len(batches) > 3
    for name, table in full.items():
        parts = [b[name] for b in batches if name in b]
        assert pa.concat_tables(parts).equals(table), name

    rows = fastitch.to_parquet(DATA, tmp_path, "orders", batch_rows=5_000, chunk_size=50_000)
    for name, table in full.items():
        assert rows[name] == table.num_rows
        assert pq.read_table(tmp_path / f"{name}.parquet").equals(table)


# ---------------------------------------------------------------------------
# Order book: compare against an independent pure-Python implementation
# ---------------------------------------------------------------------------
def _reference_book(data, symbol, depth, every_ns=0):
    from fastitch._spec import SPECS as S

    frames = fastitch.read(data, ["R", "A", "F", "E", "C", "X", "D", "U"], backend="arrow",
                           raw_prices=True, date=False)
    locate = [r["stock_locate"] for r in frames["stock_directory"].to_pylist() if r["stock"] == symbol][0]
    events = []
    for name, table in frames.items():
        if name == "stock_directory":
            continue
        code = [c for c, s in S.items() if s.name == name][0]
        table = table.set_column(2, "timestamp", table.column("timestamp").cast(pa.int64()))
        for r in table.to_pylist():
            if r["stock_locate"] == locate:
                events.append((r["timestamp"], r["tracking_number"], code, r))
    # Messages are already in file order per type; restore global order by timestamp,
    # which the synthetic generator makes strictly increasing.
    events.sort(key=lambda e: e[0])

    orders, bids, asks = {}, {}, {}

    def lvl(side, px, d):
        book = bids if side == "B" else asks
        book[px] = book.get(px, 0) + d
        if book[px] <= 0:
            del book[px]

    def top():
        row = []
        b = sorted(bids.items(), key=lambda kv: -kv[0])[:depth]
        a = sorted(asks.items())[:depth]
        for i in range(depth):
            row += list(b[i]) if i < len(b) else [0, 0]
            row += list(a[i]) if i < len(a) else [0, 0]
        return row

    rows, last, next_emit = [], None, None
    for ts, _, code, r in events:
        if every_ns:
            if next_emit is None:
                next_emit = (ts // every_ns + 1) * every_ns
            while ts >= next_emit:
                rows.append([next_emit] + top())
                next_emit += every_ns
        if code in "AF":
            orders[r["order_reference_number"]] = [r["side"], r["price"], r["shares"]]
            lvl(r["side"], r["price"], r["shares"])
        elif code in "ECX":
            o = orders[r["order_reference_number"]]
            q = r.get("executed_shares", r.get("cancelled_shares"))
            o[2] -= q
            lvl(o[0], o[1], -q)
            if o[2] == 0:
                del orders[r["order_reference_number"]]
        elif code == "D":
            o = orders.pop(r["order_reference_number"])
            lvl(o[0], o[1], -o[2])
        elif code == "U":
            o = orders.pop(r["original_order_reference_number"])
            lvl(o[0], o[1], -o[2])
            orders[r["new_order_reference_number"]] = [o[0], r["price"], r["shares"]]
            lvl(o[0], r["price"], r["shares"])
        if not every_ns:
            t = top()
            if t != last:
                rows.append([ts] + t)
                last = t
    return rows


def _book_rows(df, depth):
    cols = ["timestamp"]
    for i in range(1, depth + 1):
        cols += [f"bid_price_{i}", f"bid_size_{i}", f"ask_price_{i}", f"ask_size_{i}"]
    import polars as pl

    t = df.select(cols).with_columns(pl.col("timestamp").cast(pl.Int64))
    return [list(r) for r in t.iter_rows()]


@pytest.mark.parametrize("depth", [1, 5])
def test_book_on_change_matches_reference(depth):
    df = fastitch.book(DATA, "MSFT", depth=depth, raw_prices=True, date=False, backend="polars")
    assert _book_rows(df, depth) == _reference_book(DATA, "MSFT", depth)


def test_book_time_grid_matches_reference():
    df = fastitch.book(DATA, ["NVDA"], depth=3, every="1s", raw_prices=True, date=False, backend="polars")
    assert _book_rows(df, 3) == _reference_book(DATA, "NVDA", 3, every_ns=10**9)


def test_book_is_never_crossed_and_handles_many_symbols():
    df = fastitch.book(DATA, None, depth=1, backend="pandas")
    assert set(df.symbol.astype(str)) == {"AAPL", "MSFT", "NVDA", "AMZN", "TSLA"}
    both = df.dropna(subset=["bid_price_1", "ask_price_1"])
    assert (both.bid_price_1 < both.ask_price_1).all()


def test_book_window_does_not_desync():
    full = fastitch.book(DATA, "AAPL", depth=2, backend="pandas", date=False)
    win = fastitch.book(DATA, "AAPL", depth=2, start="11:00", end="12:00", backend="pandas", date=False)
    secs = full.timestamp.dt.total_seconds()
    expected = full[(secs >= 39_600) & (secs <= 43_200)].reset_index(drop=True)
    assert win.reset_index(drop=True).equals(expected)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli(tmp_path):
    f = tmp_path / "s.itch"
    run = lambda *a: subprocess.run([sys.executable, "-m", "fastitch.cli", *a], capture_output=True, text=True, check=True)
    run("sample", str(f), "-n", "20000")
    assert "add_order" in run("info", str(f)).stdout
    run("convert", str(f), "-o", str(tmp_path / "pq"), "-t", "orders")
    assert (tmp_path / "pq" / "add_order.parquet").exists()
    run("book", str(f), "-s", "AAPL", "-e", "1s", "-o", str(tmp_path / "b.parquet"))
    assert pq.read_table(tmp_path / "b.parquet").num_rows > 0

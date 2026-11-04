<h1 align="center">fastitch</h1>

<p align="center">
  <b>NASDAQ TotalView-ITCH 5.0 → Polars, pandas, Arrow or Parquet. At native speed, in one line.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/fastitch/"><img src="https://img.shields.io/pypi/v/fastitch.svg" alt="PyPI"></a>
  <a href="https://github.com/ArkaKhorchidian/fastitch/actions/workflows/ci.yml"><img src="https://github.com/ArkaKhorchidian/fastitch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/pypi/pyversions/fastitch.svg" alt="Python">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
</p>

```python
import fastitch

frames = fastitch.read("01302019.NASDAQ_ITCH50.gz", types="orders", symbols=["AAPL", "NVDA"])
frames["add_order"]          # a Polars DataFrame, one row per Add Order message
```

NASDAQ publishes full-depth, message-by-message order book data as TotalView-ITCH 5.0 files:
a few gigabytes per day, hundreds of millions of binary messages. Existing Python parsers hand
you **one Python object per message**, which is fine for a demo and painful for a trading day.

fastitch decodes straight into **columnar Arrow buffers in C++**, so a whole day becomes ordinary
DataFrames you can filter, group and join, and it never builds a Python object per message.

| Task (5M messages, 164 MB) | fastitch | itchcpp (C++) | itchfeed (Python) | meatpy (Python) |
|---|---:|---:|---:|---:|
| Add orders → pandas DataFrame | **0.59 s** | 21.8 s (37x slower) | 30.0 s (51x slower) | n/a |
| Decode every message | **0.75 s** | 1.9 s (2.5x slower) | 7.2 s (9.6x slower) | 10.4 s (14x slower) |

<sub>Synthetic session from `fastitch.synthetic`, 2-vCPU Linux VM, Python 3.13, best of 3, timings include imports.
fastitch's "decode" also builds Arrow tables for all 23 types; the others only create message objects.
Reproduce with <code>python benchmarks/bench.py</code>; numbers on real NASDAQ days welcome in an issue.</sub>

## Install

```bash
pip install fastitch            # prebuilt wheels for Linux, macOS, Windows · Python 3.10–3.13
pip install "fastitch[all]"     # + polars, pandas, and ISA-L for faster .gz reading
```

## What you get

- **All 23 ITCH 5.0 message types**, every field, named and typed (`add_order`, `order_executed`, `noii`, …)
- **DataFrames out**: Polars (default), pandas or a zero-copy Arrow table
- **Filters pushed into C++**: by symbol, by time of day (`start="09:30", end="16:00"`), by message type or group (`"orders"`, `"trades"`, `"auction"`, `"admin"`)
- **Order-book reconstruction**: top-N levels, emitted on every change or sampled on a time grid
- **Constant-memory streaming**: `iter_batches()` and `to_parquet()` handle full days on a laptop
- **Real timestamps**: the trading date is read from NASDAQ file names, so you get `datetime[ns]`, not "nanoseconds since midnight"
- **Reads `.gz` directly**, plus raw bytes and file objects
- **A CLI** for when you don't want to open Python at all
- **Synthetic sessions** built in, so you can try everything without a 5 GB download

## Usage

### Decode messages

```python
import fastitch

# Everything (one DataFrame per message type)
frames = fastitch.read("01302019.NASDAQ_ITCH50.gz")

# Only what you need: filtering happens in C++ before anything reaches Python
fills = fastitch.read(
    "01302019.NASDAQ_ITCH50.gz",
    types=["order_executed", "order_executed_with_price", "trade"],
    symbols=["AAPL", "MSFT"],
    start="09:30", end="16:00",
)

# A single type, as pandas
adds = fastitch.read_frame("01302019.NASDAQ_ITCH50.gz", "add_order", backend="pandas")
```

`types` accepts codes (`"A"`), names (`"add_order"`) or groups: `orders`, `trades`, `auction`, `admin`, `all`.
Prices are floats in dollars (pass `raw_prices=True` for the exact integer ticks).

### Rebuild the order book

```python
book = fastitch.book("01302019.NASDAQ_ITCH50.gz", "AAPL", depth=5, every="100ms")
```

`every=None` (the default) emits a row each time the top `depth` levels change; a duration such as
`"1s"` or `"100ms"` samples on a regular grid. The book is always built from the open, so a
`start`/`end` window never desyncs it. Correctness is tested row-for-row against an independent
pure-Python reference implementation.

### Full days, constant memory

```python
# Stream to one Parquet file per message type
fastitch.to_parquet("01302019.NASDAQ_ITCH50.gz", "parquet/", types="orders")

# Or process batch by batch
for batch in fastitch.iter_batches("01302019.NASDAQ_ITCH50.gz", "trades", batch_rows=5_000_000):
    handle(batch["order_executed"])
```

### Command line

```bash
fastitch info     01302019.NASDAQ_ITCH50.gz                 # message counts by type
fastitch convert  01302019.NASDAQ_ITCH50.gz -o parquet/ -t orders -s AAPL MSFT
fastitch book     01302019.NASDAQ_ITCH50.gz -s AAPL -d 10 -e 1s -o aapl_book.parquet
fastitch sample   demo.itch -n 1000000                      # synthetic data to play with
```

### No data yet?

```python
from fastitch.synthetic import generate

data = generate(1_000_000, symbols=["AAPL", "MSFT"])   # a self-consistent ITCH session, as bytes
fastitch.read(data)
```

Real sample days are published by NASDAQ at <https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/>.
See [`examples/quickstart.py`](examples/quickstart.py) for spreads, imbalance and volume bars in a dozen lines.

## How it works

- `_spec.py` holds the byte layout of every ITCH 5.0 message: the single source of truth, checked against the spec's message lengths at import.
- The C++ core is layout-driven: it frames the length-prefixed stream (messages may straddle read chunks), applies symbol/time filters, and appends each field to a contiguous native-endian column. The GIL is released while parsing.
- Python wraps those columns as Arrow arrays without copying them again, then hands them to Polars or pandas.
- The order book keeps live orders in a hash map and price levels in sorted maps per symbol, applying adds, executions, cancels, deletes and replaces exactly as the spec defines.

## Benchmarks

Run them yourself: `python benchmarks/bench.py` (synthetic) or `python benchmarks/bench.py <ITCH file>`.
Each tool runs in its own process; best of three.

## Contributing

Issues and PRs welcome. Good first contributions: MoldUDP64 packet captures as input, more book
analytics (queue position, order flow imbalance), and a Rust or DuckDB front end.

```bash
pip install -e ".[test]" && pytest
```

## License

MIT. ITCH is a trademark of Nasdaq, Inc.; this project is not affiliated with or endorsed by Nasdaq.

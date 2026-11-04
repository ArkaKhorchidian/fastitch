"""Five-minute tour of fastitch on a synthetic session (no download needed).

For real data, NASDAQ publishes sample TotalView-ITCH 5.0 files at
https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/ ; pass the downloaded .gz path instead.
"""
import polars as pl

import fastitch
from fastitch.synthetic import generate

data = generate(1_000_000, symbols=["AAPL", "MSFT", "NVDA"], seed=7)

# 1. What's in the file?
print(fastitch.count(data))

# 2. Every add order for AAPL in the first hour, as a Polars DataFrame
adds = fastitch.read_frame(data, "add_order", symbols="AAPL", start="09:30", end="10:30", date="2026-11-04")
print(adds.head())

# 3. Executions -> one-minute traded volume
fills = fastitch.read(data, ["E", "C"], symbols="AAPL", date="2026-11-04")
volume = (
    pl.concat([fills["order_executed"], fills["order_executed_with_price"]], how="diagonal")
    .sort("timestamp")
    .group_by_dynamic("timestamp", every="1m")
    .agg(pl.col("executed_shares").sum().alias("volume"))
)
print(volume.head())

# 4. Rebuild the order book: 5 levels, sampled every second
book = fastitch.book(data, "AAPL", depth=5, every="1s", date="2026-11-04")
book = book.with_columns(
    mid=(pl.col("bid_price_1") + pl.col("ask_price_1")) / 2,
    spread_bps=(pl.col("ask_price_1") - pl.col("bid_price_1")) / pl.col("ask_price_1") * 1e4,
    imbalance=(pl.col("bid_size_1") - pl.col("ask_size_1")) / (pl.col("bid_size_1") + pl.col("ask_size_1")),
)
print(book.select("timestamp", "mid", "spread_bps", "imbalance").head())

# 5. Convert a whole day to Parquet in constant memory
print(fastitch.to_parquet(data, "itch_parquet", types="orders"))

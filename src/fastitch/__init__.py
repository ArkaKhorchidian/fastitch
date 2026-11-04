"""fastitch: fast NASDAQ TotalView-ITCH 5.0 decoding for Python.

>>> import fastitch
>>> frames = fastitch.read("01302019.NASDAQ_ITCH50.gz", types="orders", symbols=["AAPL"])
>>> frames["add_order"].head()
"""
from ._spec import GROUPS, SPECS
from .reader import book, count, iter_batches, read, read_frame, to_parquet

__version__ = "0.1.0"

MESSAGE_TYPES = {code: spec.name for code, spec in SPECS.items()}

__all__ = [
    "read", "read_frame", "iter_batches", "to_parquet", "book", "count",
    "MESSAGE_TYPES", "GROUPS", "__version__",
]

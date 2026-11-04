"""Encode ITCH 5.0 messages and generate realistic synthetic sessions.

Useful for tests, demos and benchmarks when you don't have a multi-GB NASDAQ file
at hand::

    import fastitch
    from fastitch.synthetic import generate

    data = generate(n_messages=1_000_000, symbols=["AAPL", "MSFT"], seed=7)
    frames = fastitch.read(data)
"""
from __future__ import annotations

import random
import struct
from typing import Iterable

from ._spec import SPECS


def encode(code: str, **values) -> bytes:
    """Encode one message (without the 2-byte length prefix).

    Prices are given as integers in the wire unit (1/10000 dollars for 4-decimal prices).
    Unspecified numeric fields default to 0 and character fields to a space.
    """
    spec = SPECS[code]
    out = bytearray(spec.length)
    out[0] = ord(code)
    for f in spec.fields:
        v = values.get(f.name)
        if f.kind in ("c", "a"):
            s = (v if v is not None else " ").encode("ascii")[: f.size].ljust(f.size, b" ")
            out[f.offset : f.offset + f.size] = s
        else:
            v = int(v or 0)
            out[f.offset : f.offset + f.size] = v.to_bytes(f.size, "big")
    return bytes(out)


def frame(messages: Iterable[bytes]) -> bytes:
    """Prefix each message with its big-endian u16 length, as in NASDAQ files."""
    return b"".join(struct.pack(">H", len(m)) + m for m in messages)


def generate(
    n_messages: int = 100_000,
    symbols: Iterable[str] = ("AAPL", "MSFT", "NVDA", "AMZN", "TSLA"),
    seed: int = 0,
    start_ns: int = 9 * 3600 * 10**9 + 30 * 60 * 10**9,
    duration_ns: int = int(6.5 * 3600 * 10**9),
) -> bytes:
    """Generate a self-consistent synthetic ITCH session.

    Includes system events, a stock directory, and a stream of add / execute / cancel /
    delete / replace / non-displayed trade messages whose resulting order book is
    always valid (no executions against unknown orders, no crossed adds).
    """
    rng = random.Random(seed)
    symbols = list(symbols)
    msgs: list[bytes] = []
    t = start_ns - 10**9

    msgs.append(encode("S", timestamp=t, event_code="O"))
    msgs.append(encode("S", timestamp=t, event_code="S"))
    for locate, sym in enumerate(symbols, start=1):
        msgs.append(encode("R", stock_locate=locate, timestamp=t, stock=sym, market_category="Q",
                           financial_status="N", round_lot_size=100, round_lots_only="N",
                           issue_classification="C", authenticity="P"))
        msgs.append(encode("H", stock_locate=locate, timestamp=t, stock=sym, trading_state="T"))
    msgs.append(encode("S", timestamp=start_ns, event_code="Q"))

    mid = {loc: rng.randint(50, 500) * 10_000 for loc in range(1, len(symbols) + 1)}
    live: dict[int, list] = {}  # ref -> [locate, side, price, shares]
    live_keys: list[int] = []
    next_ref = 1
    match = 1
    step = max(1, duration_ns // max(1, n_messages))
    t = start_ns

    def pick_live():
        while live_keys:
            i = rng.randrange(len(live_keys))
            ref = live_keys[i]
            if ref in live:
                return ref
            live_keys[i] = live_keys[-1]
            live_keys.pop()
        return None

    while len(msgs) < n_messages - 3:
        t += rng.randint(1, 2 * step)
        locate = rng.randint(1, len(symbols))
        r = rng.random()
        ref = pick_live() if r >= 0.45 else None
        if ref is None or r < 0.45:
            side = rng.choice("BS")
            tick = rng.randint(1, 20) * 100
            price = mid[locate] - tick if side == "B" else mid[locate] + tick
            shares = rng.choice((100, 100, 200, 300, 500, 1000))
            code = "F" if rng.random() < 0.05 else "A"
            msgs.append(encode(code, stock_locate=locate, timestamp=t, order_reference_number=next_ref,
                               side=side, shares=shares, stock=symbols[locate - 1], price=price,
                               attribution="ABCD"))
            live[next_ref] = [locate, side, price, shares]
            live_keys.append(next_ref)
            next_ref += 1
            continue
        loc, side, price, shares = live[ref]
        if r < 0.60:  # partial or full execution
            q = min(shares, rng.choice((100, 100, 200, shares)))
            if rng.random() < 0.1:
                msgs.append(encode("C", stock_locate=loc, timestamp=t, order_reference_number=ref,
                                   executed_shares=q, match_number=match, printable="Y", execution_price=price))
            else:
                msgs.append(encode("E", stock_locate=loc, timestamp=t, order_reference_number=ref,
                                   executed_shares=q, match_number=match))
            match += 1
            live[ref][3] -= q
            if live[ref][3] == 0:
                del live[ref]
        elif r < 0.70:
            q = min(shares - 1, 100) if shares > 100 else 0
            if q:
                msgs.append(encode("X", stock_locate=loc, timestamp=t, order_reference_number=ref, cancelled_shares=q))
                live[ref][3] -= q
            else:
                msgs.append(encode("D", stock_locate=loc, timestamp=t, order_reference_number=ref))
                del live[ref]
        elif r < 0.90:
            msgs.append(encode("D", stock_locate=loc, timestamp=t, order_reference_number=ref))
            del live[ref]
        elif r < 0.97:
            new_price = price + rng.choice((-100, 100))
            # Keep bids below and asks above the mid so the book never crosses.
            new_price = min(new_price, mid[loc] - 100) if side == "B" else max(new_price, mid[loc] + 100)
            new_shares = rng.choice((100, 200, 300))
            msgs.append(encode("U", stock_locate=loc, timestamp=t, original_order_reference_number=ref,
                               new_order_reference_number=next_ref, shares=new_shares, price=new_price))
            del live[ref]
            live[next_ref] = [loc, side, new_price, new_shares]
            live_keys.append(next_ref)
            next_ref += 1
        else:
            msgs.append(encode("P", stock_locate=loc, timestamp=t, order_reference_number=0, side=side,
                               shares=rng.choice((100, 200)), stock=symbols[loc - 1], price=mid[loc],
                               match_number=match))
            match += 1

    msgs.append(encode("S", timestamp=t + 1, event_code="M"))
    msgs.append(encode("S", timestamp=t + 2, event_code="E"))
    msgs.append(encode("S", timestamp=t + 3, event_code="C"))
    return frame(msgs)

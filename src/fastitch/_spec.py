"""NASDAQ TotalView-ITCH 5.0 message layouts.

This table is the single source of truth: the C++ core is generic and parses
whatever layout it is handed from here, and the Python side uses the same
table to name columns and convert prices/strings.

Field kinds
-----------
``u``      unsigned big-endian integer (1, 2, 4 or 8 bytes)
``ts``     6-byte nanoseconds-since-midnight timestamp
``p4``     4-byte price, 4 implied decimals
``p8``     8-byte price, 8 implied decimals
``c``      single ASCII character
``a``      fixed-width, right-space-padded ASCII string
"""
from __future__ import annotations

from dataclasses import dataclass

# Every ITCH 5.0 message starts with the same 11-byte header.
_HEADER = [
    ("stock_locate", "u", 2),
    ("tracking_number", "u", 2),
    ("timestamp", "ts", 6),
]


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    size: int
    offset: int


@dataclass(frozen=True)
class MessageSpec:
    code: str
    name: str
    length: int
    fields: tuple[Field, ...]


def _build(code: str, name: str, body: list[tuple[str, str, int]]) -> MessageSpec:
    fields = []
    offset = 1  # byte 0 is the message type
    for fname, kind, size in _HEADER + body:
        fields.append(Field(fname, kind, size, offset))
        offset += size
    return MessageSpec(code, name, offset, tuple(fields))


_RAW: list[tuple[str, str, list[tuple[str, str, int]]]] = [
    ("S", "system_event", [("event_code", "c", 1)]),
    ("R", "stock_directory", [
        ("stock", "a", 8),
        ("market_category", "c", 1),
        ("financial_status", "c", 1),
        ("round_lot_size", "u", 4),
        ("round_lots_only", "c", 1),
        ("issue_classification", "c", 1),
        ("issue_subtype", "a", 2),
        ("authenticity", "c", 1),
        ("short_sale_threshold", "c", 1),
        ("ipo_flag", "c", 1),
        ("luld_ref_price_tier", "c", 1),
        ("etp_flag", "c", 1),
        ("etp_leverage_factor", "u", 4),
        ("inverse_indicator", "c", 1),
    ]),
    ("H", "stock_trading_action", [
        ("stock", "a", 8),
        ("trading_state", "c", 1),
        ("reserved", "c", 1),
        ("reason", "a", 4),
    ]),
    ("Y", "reg_sho_restriction", [("stock", "a", 8), ("reg_sho_action", "c", 1)]),
    ("L", "market_participant_position", [
        ("mpid", "a", 4),
        ("stock", "a", 8),
        ("primary_market_maker", "c", 1),
        ("market_maker_mode", "c", 1),
        ("market_participant_state", "c", 1),
    ]),
    ("V", "mwcb_decline_level", [
        ("level_1", "p8", 8),
        ("level_2", "p8", 8),
        ("level_3", "p8", 8),
    ]),
    ("W", "mwcb_status", [("breached_level", "c", 1)]),
    ("K", "ipo_quoting_period_update", [
        ("stock", "a", 8),
        ("ipo_quotation_release_time", "u", 4),
        ("ipo_quotation_release_qualifier", "c", 1),
        ("ipo_price", "p4", 4),
    ]),
    ("J", "luld_auction_collar", [
        ("stock", "a", 8),
        ("auction_collar_reference_price", "p4", 4),
        ("upper_auction_collar_price", "p4", 4),
        ("lower_auction_collar_price", "p4", 4),
        ("auction_collar_extension", "u", 4),
    ]),
    ("h", "operational_halt", [
        ("stock", "a", 8),
        ("market_code", "c", 1),
        ("operational_halt_action", "c", 1),
    ]),
    ("A", "add_order", [
        ("order_reference_number", "u", 8),
        ("side", "c", 1),
        ("shares", "u", 4),
        ("stock", "a", 8),
        ("price", "p4", 4),
    ]),
    ("F", "add_order_mpid", [
        ("order_reference_number", "u", 8),
        ("side", "c", 1),
        ("shares", "u", 4),
        ("stock", "a", 8),
        ("price", "p4", 4),
        ("attribution", "a", 4),
    ]),
    ("E", "order_executed", [
        ("order_reference_number", "u", 8),
        ("executed_shares", "u", 4),
        ("match_number", "u", 8),
    ]),
    ("C", "order_executed_with_price", [
        ("order_reference_number", "u", 8),
        ("executed_shares", "u", 4),
        ("match_number", "u", 8),
        ("printable", "c", 1),
        ("execution_price", "p4", 4),
    ]),
    ("X", "order_cancel", [
        ("order_reference_number", "u", 8),
        ("cancelled_shares", "u", 4),
    ]),
    ("D", "order_delete", [("order_reference_number", "u", 8)]),
    ("U", "order_replace", [
        ("original_order_reference_number", "u", 8),
        ("new_order_reference_number", "u", 8),
        ("shares", "u", 4),
        ("price", "p4", 4),
    ]),
    ("P", "trade", [
        ("order_reference_number", "u", 8),
        ("side", "c", 1),
        ("shares", "u", 4),
        ("stock", "a", 8),
        ("price", "p4", 4),
        ("match_number", "u", 8),
    ]),
    ("Q", "cross_trade", [
        ("shares", "u", 8),
        ("stock", "a", 8),
        ("cross_price", "p4", 4),
        ("match_number", "u", 8),
        ("cross_type", "c", 1),
    ]),
    ("B", "broken_trade", [("match_number", "u", 8)]),
    ("I", "noii", [
        ("paired_shares", "u", 8),
        ("imbalance_shares", "u", 8),
        ("imbalance_direction", "c", 1),
        ("stock", "a", 8),
        ("far_price", "p4", 4),
        ("near_price", "p4", 4),
        ("current_reference_price", "p4", 4),
        ("cross_type", "c", 1),
        ("price_variation_indicator", "c", 1),
    ]),
    ("N", "rpii", [("stock", "a", 8), ("interest_flag", "c", 1)]),
    ("O", "dlcr_price_discovery", [
        ("stock", "a", 8),
        ("open_eligibility_status", "c", 1),
        ("minimum_allowable_price", "p4", 4),
        ("maximum_allowable_price", "p4", 4),
        ("near_execution_price", "p4", 4),
        ("near_execution_time", "u", 8),
        ("lower_price_range_collar", "p4", 4),
        ("upper_price_range_collar", "p4", 4),
    ]),
]

SPECS: dict[str, MessageSpec] = {code: _build(code, name, body) for code, name, body in _RAW}

# Expected on-the-wire lengths from the ITCH 5.0 specification; checked at import
# time so a typo in the table above can never ship silently.
EXPECTED_LENGTHS = {
    "S": 12, "R": 39, "H": 25, "Y": 20, "L": 26, "V": 35, "W": 12, "K": 28,
    "J": 35, "h": 21, "A": 36, "F": 40, "E": 31, "C": 36, "X": 23, "D": 19,
    "U": 35, "P": 44, "Q": 40, "B": 19, "I": 50, "N": 20, "O": 48,
}
for _code, _spec in SPECS.items():
    if _spec.length != EXPECTED_LENGTHS[_code]:  # pragma: no cover
        raise AssertionError(f"ITCH spec for {_code!r} is {_spec.length} bytes, expected {EXPECTED_LENGTHS[_code]}")

NAME_TO_CODE = {s.name: s.code for s in SPECS.values()}

# Friendly groupings for the `types=` argument.
GROUPS = {
    "orders": ["A", "F", "E", "C", "X", "D", "U"],
    "trades": ["P", "Q", "E", "C", "B"],
    "admin": ["S", "R", "H", "Y", "L", "V", "W", "K", "J", "h"],
    "auction": ["I", "N", "O", "Q"],
    "all": list(SPECS),
}


def resolve_types(types) -> list[str]:
    """Accept codes ('A'), names ('add_order'), groups ('orders') or a mix."""
    if types is None:
        return list(SPECS)
    if isinstance(types, str):
        types = [types]
    out: list[str] = []
    for t in types:
        if t in SPECS:
            codes = [t]
        elif t in NAME_TO_CODE:
            codes = [NAME_TO_CODE[t]]
        elif t in GROUPS:
            codes = GROUPS[t]
        else:
            raise ValueError(
                f"Unknown ITCH message type {t!r}. Use a code like 'A', a name like "
                f"'add_order', or a group: {', '.join(GROUPS)}"
            )
        for c in codes:
            if c not in out:
                out.append(c)
    return out

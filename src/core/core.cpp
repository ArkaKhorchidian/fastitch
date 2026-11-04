// fastitch core: streaming NASDAQ TotalView-ITCH 5.0 decoding into columnar buffers,
// plus limit-order-book reconstruction.
//
// The parser is deliberately layout-agnostic: Python hands it the byte layout of
// every message type (see fastitch/_spec.py), and it decodes each field into a
// contiguous native-endian column. Python then wraps those columns zero-copy as
// NumPy / Arrow buffers.

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

namespace {

// ---------------------------------------------------------------------------
// Big-endian helpers (compilers lower these to a single bswap)
// ---------------------------------------------------------------------------
inline uint16_t be16(const uint8_t* p) { return uint16_t((uint16_t(p[0]) << 8) | p[1]); }
inline uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}
inline uint64_t be48(const uint8_t* p) {
    return (uint64_t(p[0]) << 40) | (uint64_t(p[1]) << 32) | (uint64_t(p[2]) << 24) |
           (uint64_t(p[3]) << 16) | (uint64_t(p[4]) << 8) | uint64_t(p[5]);
}
inline uint64_t be64(const uint8_t* p) { return (uint64_t(be32(p)) << 32) | be32(p + 4); }

constexpr size_t kHeaderLen = 11;  // type + locate + tracking + 6-byte timestamp

// ---------------------------------------------------------------------------
// Length-prefixed framing shared by Parser and Book.
// NASDAQ's binary files are a sequence of [u16 big-endian length][message].
// A message may straddle two chunks; the incomplete tail is kept in `carry`.
// ---------------------------------------------------------------------------
template <class OnMsg>
void frame(const uint8_t* data, size_t n, std::vector<uint8_t>& carry, OnMsg&& on_msg) {
    size_t pos = 0;
    if (!carry.empty()) {
        // Complete the carried length prefix first.
        while (carry.size() < 2 && pos < n) carry.push_back(data[pos++]);
        if (carry.size() < 2) return;
        size_t need = 2 + size_t(be16(carry.data()));
        size_t take = std::min(need - carry.size(), n - pos);
        carry.insert(carry.end(), data + pos, data + pos + take);
        pos += take;
        if (carry.size() < need) return;
        on_msg(carry.data() + 2, need - 2);
        carry.clear();
    }
    while (pos + 2 <= n) {
        size_t len = be16(data + pos);
        if (pos + 2 + len > n) break;
        on_msg(data + pos + 2, len);
        pos += 2 + len;
    }
    if (pos < n) carry.assign(data + pos, data + n);
}

std::string pad8(const std::string& s) {
    std::string out = s.substr(0, 8);
    out.resize(8, ' ');
    return out;
}

// Locate-code -> symbol bookkeeping, learned from Stock Directory ('R') messages,
// and the optional symbol / time filters shared by Parser and Book.
struct Filter {
    bool by_symbol = false;
    std::unordered_set<std::string> wanted;  // 8-byte padded symbols
    std::vector<uint8_t> locate_ok = std::vector<uint8_t>(65536, 0);
    std::vector<std::string> locate_symbol = std::vector<std::string>(65536);
    uint64_t start_ns = 0;
    uint64_t end_ns = UINT64_MAX;

    void set_symbols(const std::optional<std::vector<std::string>>& syms) {
        if (!syms) return;
        by_symbol = true;
        for (auto& s : *syms) wanted.insert(pad8(s));
    }

    // Observe a message; returns whether it passes the filters.
    bool accept(const uint8_t* m, size_t len) {
        uint16_t locate = be16(m + 1);
        if (m[0] == 'R' && len >= 19) {
            std::string sym(reinterpret_cast<const char*>(m + 11), 8);
            locate_symbol[locate] = sym;
            locate_ok[locate] = !by_symbol || wanted.count(sym);
        }
        uint64_t ts = be48(m + 5);
        if (ts < start_ns || ts > end_ns) return false;
        // Market-wide messages (locate 0: system events, MWCB) are always kept.
        if (by_symbol && locate != 0 && !locate_ok[locate]) return false;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Columnar parser
// ---------------------------------------------------------------------------
enum Kind : uint8_t { K_U1, K_U2, K_U4, K_U6, K_U8, K_RAW };

struct Column {
    uint16_t offset;
    uint16_t size;   // bytes on the wire
    uint16_t width;  // bytes per decoded value
    Kind kind;
    // Raw malloc'd storage: realloc can grow large blocks in place (mremap on Linux)
    // without copying or zero-filling, unlike std::vector::resize.
    uint8_t* buf = nullptr;

    void grow(size_t bytes) {
        void* p = std::realloc(buf, bytes);
        if (!p) throw std::bad_alloc();
        buf = static_cast<uint8_t*>(p);
    }
    void release() { std::free(buf); buf = nullptr; }
};

struct Table {
    bool enabled = false;
    uint16_t length = 0;
    std::vector<Column> cols;
    uint64_t rows = 0;
    uint64_t cap = 0;
};

class Parser {
public:
    // layouts: {type_code: [(offset, size, kind), ...]} where kind is one of
    // the strings in _spec.py ("u", "ts", "p4", "p8", "c", "a").
    Parser(const std::map<std::string, std::vector<std::tuple<int, int, std::string>>>& layouts,
           std::optional<std::vector<std::string>> symbols, std::optional<uint64_t> start_ns,
           std::optional<uint64_t> end_ns) {
        filter_.set_symbols(symbols);
        if (start_ns) filter_.start_ns = *start_ns;
        if (end_ns) filter_.end_ns = *end_ns;
        for (auto& [code, fields] : layouts) {
            if (code.size() != 1) throw std::invalid_argument("type codes must be one character");
            Table& t = tables_[uint8_t(code[0])];
            t.enabled = true;
            int end = 0;
            for (auto& [off, size, kind] : fields) {
                Column c{uint16_t(off), uint16_t(size), uint16_t(size), K_RAW, nullptr};
                if (kind == "c" || kind == "a") c.kind = K_RAW;
                else if (size == 1) c.kind = K_U1;
                else if (size == 2) c.kind = K_U2;
                else if (size == 4) c.kind = K_U4;
                else if (size == 6) { c.kind = K_U6; c.width = 8; }
                else if (size == 8) c.kind = K_U8;
                else throw std::invalid_argument("unsupported integer width");
                t.cols.push_back(std::move(c));
                end = std::max(end, off + size);
            }
            t.length = uint16_t(end);
        }
    }

    ~Parser() {
        for (auto& t : tables_)
            for (auto& c : t.cols) c.release();
    }
    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;

    void feed(nb::bytes chunk) {
        const auto* p = reinterpret_cast<const uint8_t*>(chunk.c_str());
        size_t n = chunk.size();
        nb::gil_scoped_release release;
        bytes_ += n;
        frame(p, n, carry_, [this](const uint8_t* m, size_t len) { on_message(m, len); });
    }

    // Hand back all decoded rows and reset the buffers (for batched reading).
    // Returns {type_code: [column_bytes, ...]}.
    std::map<std::string, std::vector<nb::bytes>> take() {
        std::map<std::string, std::vector<nb::bytes>> out;
        for (int code = 0; code < 256; ++code) {
            Table& t = tables_[code];
            if (!t.enabled || t.rows == 0) continue;
            std::vector<nb::bytes> cols;
            cols.reserve(t.cols.size());
            for (auto& c : t.cols) {
                cols.emplace_back(reinterpret_cast<const char*>(c.buf), size_t(t.rows) * c.width);
                c.release();
            }
            out.emplace(std::string(1, char(code)), std::move(cols));
            t.rows = 0;
            t.cap = 0;
        }
        return out;
    }

    size_t pending_rows() const {
        size_t r = 0;
        for (auto& t : tables_) r += t.rows;
        return r;
    }

    std::map<std::string, uint64_t> counts() const {
        std::map<std::string, uint64_t> out;
        for (int code = 0; code < 256; ++code)
            if (seen_[code]) out.emplace(std::string(1, char(code)), seen_[code]);
        return out;
    }

    std::map<int, std::string> symbols() const {
        std::map<int, std::string> out;
        for (int i = 0; i < 65536; ++i)
            if (!filter_.locate_symbol[i].empty()) out.emplace(i, filter_.locate_symbol[i]);
        return out;
    }

    uint64_t bytes_read() const { return bytes_; }
    uint64_t messages() const { return messages_; }
    bool has_partial() const { return !carry_.empty(); }
    uint64_t malformed() const { return malformed_; }

private:
    void on_message(const uint8_t* m, size_t len) {
        if (len < kHeaderLen) { ++malformed_; return; }
        ++messages_;
        ++seen_[m[0]];
        bool pass = filter_.accept(m, len);
        Table& t = tables_[m[0]];
        if (!t.enabled || !pass) return;
        if (len < t.length) { ++malformed_; return; }
        if (t.rows == t.cap) {  // amortized doubling across all columns at once
            t.cap = t.cap ? t.cap * 2 : 4096;
            for (auto& c : t.cols) c.grow(size_t(t.cap) * c.width);
        }
        for (auto& c : t.cols) {
            const uint8_t* f = m + c.offset;
            uint8_t* dst = c.buf + size_t(t.rows) * c.width;
            switch (c.kind) {
                case K_U1: *dst = f[0]; break;
                case K_U2: store(dst, be16(f)); break;
                case K_U4: store(dst, be32(f)); break;
                case K_U6: store(dst, be48(f)); break;
                case K_U8: store(dst, be64(f)); break;
                case K_RAW: std::memcpy(dst, f, c.size); break;
            }
        }
        ++t.rows;
    }

    template <class T>
    static void store(uint8_t* dst, T v) {
        std::memcpy(dst, &v, sizeof(T));  // native (little-endian) order
    }

    std::array<Table, 256> tables_{};
    std::array<uint64_t, 256> seen_{};
    Filter filter_;
    std::vector<uint8_t> carry_;
    uint64_t bytes_ = 0, messages_ = 0, malformed_ = 0;
};

// ---------------------------------------------------------------------------
// Order book reconstruction
// ---------------------------------------------------------------------------
class Book {
public:
    // interval_ns == 0 -> emit a row every time the top `depth` levels change.
    // interval_ns  > 0 -> emit a row per symbol on a regular time grid.
    Book(std::optional<std::vector<std::string>> symbols, int depth, uint64_t interval_ns,
         std::optional<uint64_t> start_ns, std::optional<uint64_t> end_ns)
        : depth_(depth), interval_(interval_ns) {
        if (depth < 1 || depth > 50) throw std::invalid_argument("depth must be between 1 and 50");
        filter_.set_symbols(symbols);
        if (start_ns) emit_from_ = *start_ns;
        if (end_ns) emit_until_ = *end_ns;
        cols_.resize(4 * size_t(depth));
    }

    void feed(nb::bytes chunk) {
        const auto* p = reinterpret_cast<const uint8_t*>(chunk.c_str());
        size_t n = chunk.size();
        nb::gil_scoped_release release;
        frame(p, n, carry_, [this](const uint8_t* m, size_t len) { on_message(m, len); });
    }

    // Returns (timestamps, locates, [bid_px_1, bid_sz_1, ask_px_1, ask_sz_1, ...]) as raw bytes.
    std::tuple<nb::bytes, nb::bytes, std::vector<nb::bytes>> take() {
        auto ts = to_bytes(out_ts_);
        auto loc = to_bytes(out_locate_);
        std::vector<nb::bytes> cols;
        for (auto& c : cols_) cols.push_back(to_bytes(c));
        return {std::move(ts), std::move(loc), std::move(cols)};
    }

    size_t pending_rows() const { return out_ts_.size(); }
    size_t live_orders() const { return orders_.size(); }

    std::map<int, std::string> symbols() const {
        std::map<int, std::string> out;
        for (int i = 0; i < 65536; ++i)
            if (!filter_.locate_symbol[i].empty()) out.emplace(i, filter_.locate_symbol[i]);
        return out;
    }

private:
    struct Order {
        uint16_t locate;
        bool buy;
        uint32_t price;
        uint32_t shares;
    };
    struct Sides {
        std::map<uint32_t, uint64_t, std::greater<uint32_t>> bids;
        std::map<uint32_t, uint64_t> asks;
        std::vector<uint32_t> last;  // last emitted top-of-book (change detection)
        uint64_t next_emit = 0;
    };

    template <class T>
    static nb::bytes to_bytes(std::vector<T>& v) {
        nb::bytes b(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
        std::vector<T>().swap(v);
        return b;
    }

    template <class M>
    static void level_add(M& levels, uint32_t px, int64_t delta) {
        auto it = levels.find(px);
        if (it == levels.end()) {
            if (delta > 0) levels.emplace(px, uint64_t(delta));
            return;
        }
        int64_t q = int64_t(it->second) + delta;
        if (q <= 0) levels.erase(it);
        else it->second = uint64_t(q);
    }

    void change(const Order& o, int64_t delta) {
        Sides& s = books_[o.locate];
        if (o.buy) level_add(s.bids, o.price, delta);
        else level_add(s.asks, o.price, delta);
    }

    void reduce(uint64_t ref, uint32_t shares) {
        auto it = orders_.find(ref);
        if (it == orders_.end()) return;
        uint32_t q = std::min(shares, it->second.shares);
        change(it->second, -int64_t(q));
        it->second.shares -= q;
        if (it->second.shares == 0) orders_.erase(it);
    }

    void remove(uint64_t ref) {
        auto it = orders_.find(ref);
        if (it == orders_.end()) return;
        change(it->second, -int64_t(it->second.shares));
        orders_.erase(it);
    }

    void top(const Sides& s, std::vector<uint32_t>& out) const {
        out.assign(4 * size_t(depth_), 0);
        int i = 0;
        for (auto it = s.bids.begin(); it != s.bids.end() && i < depth_; ++it, ++i) {
            out[4 * i] = it->first;
            out[4 * i + 1] = uint32_t(std::min<uint64_t>(it->second, UINT32_MAX));
        }
        i = 0;
        for (auto it = s.asks.begin(); it != s.asks.end() && i < depth_; ++it, ++i) {
            out[4 * i + 2] = it->first;
            out[4 * i + 3] = uint32_t(std::min<uint64_t>(it->second, UINT32_MAX));
        }
    }

    void emit(uint64_t ts, uint16_t locate, const std::vector<uint32_t>& row) {
        if (ts < emit_from_ || ts > emit_until_) return;
        out_ts_.push_back(ts);
        out_locate_.push_back(locate);
        for (size_t k = 0; k < row.size(); ++k) cols_[k].push_back(row[k]);
    }

    void on_message(const uint8_t* m, size_t len) {
        if (len < kHeaderLen) return;
        if (m[0] == 'R') {  // learn locate -> symbol
            filter_.accept(m, len);
            return;
        }
        // Only the symbol filter may skip messages here: dropping updates by time
        // would desync the book, so the time window is applied when emitting.
        uint16_t locate = be16(m + 1);
        if (filter_.by_symbol && locate != 0 && !filter_.locate_ok[locate]) return;
        uint64_t ts = be48(m + 5);

        // Regular-grid sampling: before applying an event at time ts, emit the
        // state as of every grid point strictly before ts.
        bool book_msg = true;
        switch (m[0]) {
            case 'A': case 'F': case 'E': case 'C': case 'X': case 'D': case 'U': break;
            default: book_msg = false;
        }
        if (!book_msg) return;

        if (interval_ > 0) {
            Sides& s = books_[locate];
            if (s.next_emit == 0) s.next_emit = (ts / interval_ + 1) * interval_;
            if (ts >= s.next_emit) {
                top(s, scratch_);
                while (ts >= s.next_emit) {
                    emit(s.next_emit, locate, scratch_);
                    s.next_emit += interval_;
                }
            }
        }

        switch (m[0]) {
            case 'A':
            case 'F': {
                if (len < 36) return;
                Order o{locate, m[19] == 'B', be32(m + 32), be32(m + 20)};
                auto [it, inserted] = orders_.emplace(be64(m + 11), o);
                if (!inserted) {  // duplicate reference: replace defensively
                    change(it->second, -int64_t(it->second.shares));
                    it->second = o;
                }
                change(o, int64_t(o.shares));
                break;
            }
            case 'E': if (len >= 31) reduce(be64(m + 11), be32(m + 19)); break;
            case 'C': if (len >= 36) reduce(be64(m + 11), be32(m + 19)); break;
            case 'X': if (len >= 23) reduce(be64(m + 11), be32(m + 19)); break;
            case 'D': if (len >= 19) remove(be64(m + 11)); break;
            case 'U': {
                if (len < 35) return;
                auto it = orders_.find(be64(m + 11));
                if (it == orders_.end()) return;
                Order o{locate, it->second.buy, be32(m + 31), be32(m + 27)};
                change(it->second, -int64_t(it->second.shares));
                orders_.erase(it);
                orders_[be64(m + 19)] = o;
                change(o, int64_t(o.shares));
                break;
            }
        }

        if (interval_ == 0) {
            Sides& s = books_[locate];
            top(s, scratch_);
            if (scratch_ != s.last) {
                s.last = scratch_;
                emit(ts, locate, scratch_);
            }
        }
    }

    int depth_;
    uint64_t interval_;
    uint64_t emit_from_ = 0, emit_until_ = UINT64_MAX;
    Filter filter_;
    std::vector<uint8_t> carry_;
    std::unordered_map<uint64_t, Order> orders_;
    std::unordered_map<uint16_t, Sides> books_;
    std::vector<uint32_t> scratch_;
    std::vector<uint64_t> out_ts_;
    std::vector<uint16_t> out_locate_;
    std::vector<std::vector<uint32_t>> cols_;
};

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "fastitch native core";

    nb::class_<Parser>(m, "Parser")
        .def(nb::init<const std::map<std::string, std::vector<std::tuple<int, int, std::string>>>&,
                      std::optional<std::vector<std::string>>, std::optional<uint64_t>,
                      std::optional<uint64_t>>(),
             nb::arg("layouts"), nb::arg("symbols") = nb::none(), nb::arg("start_ns") = nb::none(),
             nb::arg("end_ns") = nb::none())
        .def("feed", &Parser::feed)
        .def("take", &Parser::take)
        .def("counts", &Parser::counts)
        .def("symbols", &Parser::symbols)
        .def_prop_ro("pending_rows", &Parser::pending_rows)
        .def_prop_ro("bytes_read", &Parser::bytes_read)
        .def_prop_ro("messages", &Parser::messages)
        .def_prop_ro("malformed", &Parser::malformed)
        .def_prop_ro("has_partial", &Parser::has_partial);

    nb::class_<Book>(m, "Book")
        .def(nb::init<std::optional<std::vector<std::string>>, int, uint64_t, std::optional<uint64_t>,
                      std::optional<uint64_t>>(),
             nb::arg("symbols") = nb::none(), nb::arg("depth") = 5, nb::arg("interval_ns") = 0,
             nb::arg("start_ns") = nb::none(), nb::arg("end_ns") = nb::none())
        .def("feed", &Book::feed)
        .def("take", &Book::take)
        .def("symbols", &Book::symbols)
        .def_prop_ro("pending_rows", &Book::pending_rows)
        .def_prop_ro("live_orders", &Book::live_orders);
}

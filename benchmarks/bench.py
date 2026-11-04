"""Benchmark fastitch against other Python ITCH 5.0 parsers.

    pip install fastitch itchfeed itchcpp meatpy pandas
    python benchmarks/bench.py                 # synthetic session (default 5M messages)
    python benchmarks/bench.py 01302019.NASDAQ_ITCH50   # or a real, uncompressed NASDAQ file

Each tool runs in its own subprocess. Two tasks are timed:

  decode   decode every message in the file (fastitch also builds Arrow tables)
  frame    get every Add Order into a pandas DataFrame
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

TASKS = {
    "fastitch": {
        "decode": "import fastitch; f = fastitch.read(PATH, backend='arrow'); n = sum(t.num_rows for t in f.values())",
        "frame": "import fastitch; df = fastitch.read_frame(PATH, 'A', backend='pandas'); n = len(df)",
    },
    "itchcpp (C++)": {
        "decode": "import itchcpp; n = sum(1 for _ in itchcpp.MessageParser().parse_file(open(PATH, 'rb')))",
        "frame": (
            "import itchcpp, pandas as pd\n"
            "rows = [m.get_attributes() for m in itchcpp.MessageParser(b'A').parse_file(open(PATH, 'rb'))]\n"
            "df = pd.DataFrame(rows); n = len(df)"
        ),
    },
    "itchfeed (Python)": {
        "env": {"ITCH_NO_CPP": "1"},
        "decode": "import itch.parser as p; n = sum(1 for _ in p.MessageParser().parse_file(open(PATH, 'rb')))",
        "frame": (
            "import itch.parser as p, pandas as pd\n"
            "rows = [m.get_attributes() for m in p.MessageParser(b'A').parse_file(open(PATH, 'rb'))]\n"
            "df = pd.DataFrame(rows); n = len(df)"
        ),
    },
    "meatpy (Python)": {
        "decode": "from meatpy.itch50.itch50_message_reader import ITCH50MessageReader as R; n = sum(1 for _ in R().read_file(PATH))",
    },
}


def run(code: str, path: str, env: dict | None) -> tuple[float, int] | None:
    prog = f"import time\nPATH = {path!r}\nt0 = time.perf_counter()\n{code}\nprint(time.perf_counter() - t0, n)"
    e = dict(os.environ, **(env or {}))
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env=e)
    if r.returncode:
        print(r.stderr.strip().splitlines()[-1], file=sys.stderr)
        return None
    t, n = r.stdout.split()
    return float(t), int(n)


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        from fastitch.synthetic import generate

        n = int(os.environ.get("BENCH_MESSAGES", 5_000_000))
        path = os.path.join(tempfile.gettempdir(), f"fastitch_bench_{n}.itch")
        if not os.path.exists(path):
            print(f"generating {n:,} synthetic messages...", file=sys.stderr)
            with open(path, "wb") as fh:
                fh.write(generate(n, seed=1))
    size = os.path.getsize(path)
    print(f"file: {path} ({size / 1e6:,.0f} MB)\n")

    results = {}
    for tool, spec in TASKS.items():
        for task in ("decode", "frame"):
            if task not in spec:
                continue
            best = None
            for _ in range(int(os.environ.get("BENCH_REPEAT", 3))):
                r = run(spec[task], path, spec.get("env"))
                if r and (best is None or r[0] < best[0]):
                    best = r
            if best:
                results.setdefault(tool, {})[task] = best
                t, n = best
                print(f"{tool:<20} {task:<7} {t:8.2f}s  {n / t / 1e6:8.2f}M rows/s  {size / t / 1e6:8.0f} MB/s")
    base = results.get("fastitch", {})
    print()
    for tool, res in results.items():
        if tool == "fastitch":
            continue
        for task, (t, _) in res.items():
            if task in base:
                print(f"fastitch is {t / base[task][0]:5.1f}x faster than {tool} at {task}")
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as fh:
        json.dump({k: {t: {"seconds": v[0], "rows": v[1]} for t, v in r.items()} for k, r in results.items()} | {"bytes": size}, fh, indent=2)


if __name__ == "__main__":
    main()

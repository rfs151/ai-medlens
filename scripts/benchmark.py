"""Compare /api/detect latency between backends (Python vs C++, CPU vs GPU machine).

Usage:
    python scripts/benchmark.py samples/polyp.jpg --model polyp \
        --url py=http://localhost:8000 --url cpp=http://localhost:8081 -n 100 -c 4
"""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def bench(name: str, base: str, image: bytes, model: str | None, n: int, concurrency: int) -> dict:
    params = {"model": model} if model else {}
    with httpx.Client(base_url=base, timeout=60) as client:
        def one(_):
            t0 = time.perf_counter()
            r = client.post("/api/detect", params=params, files={"file": ("img.jpg", image, "image/jpeg")})
            r.raise_for_status()
            return (time.perf_counter() - t0) * 1000, r.json()["timing_ms"]

        for _ in range(5):  # warm-up
            one(0)
        t_start = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:
            results = list(pool.map(one, range(n)))
        wall = time.perf_counter() - t_start

    rtt = sorted(r[0] for r in results)
    infer = [r[1].get("inference", 0) for r in results]
    return {
        "backend": name,
        "p50_ms": rtt[len(rtt) // 2],
        "p95_ms": rtt[int(len(rtt) * 0.95) - 1],
        "infer_mean_ms": statistics.mean(infer),
        "throughput_rps": n / wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--url", action="append", required=True, help="name=http://host:port, repeatable")
    ap.add_argument("--model")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("-c", "--concurrency", type=int, default=1)
    args = ap.parse_args()

    image = args.image.read_bytes()
    rows = [bench(*u.split("=", 1), image, args.model, args.n, args.concurrency) for u in args.url]
    print(f"\n{'backend':<10}{'p50 ms':>10}{'p95 ms':>10}{'infer ms':>10}{'req/s':>10}")
    for r in rows:
        print(f"{r['backend']:<10}{r['p50_ms']:>10.1f}{r['p95_ms']:>10.1f}{r['infer_mean_ms']:>10.1f}{r['throughput_rps']:>10.1f}")


if __name__ == "__main__":
    main()

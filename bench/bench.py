"""Benchmark the covered GeoPandas operations. Run only with ``pixi run bench``."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import geopandas as gpd
import numpy as np
from shapely import box

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import mojogeopandas as mgpd  # noqa: E402


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def grids(n=1_600, offset=0.45):
    side = int(np.sqrt(n))
    x, y = np.meshgrid(np.arange(side, dtype=float), np.arange(side, dtype=float))
    first = gpd.GeoDataFrame({"v": np.arange(side * side)}, geometry=box(x.ravel(), y.ravel(), x.ravel() + 0.9, y.ravel() + 0.9))
    second = gpd.GeoDataFrame({"w": np.arange(side * side)}, geometry=box(x.ravel() + offset, y.ravel() + offset, x.ravel() + offset + 0.9, y.ravel() + offset + 0.9))
    return first, second


def main():
    left, right = grids()
    cases = [
        ("sjoin intersects (1,600 x 1,600 boxes)", lambda: mgpd.sjoin(left, right), lambda: gpd.sjoin(left, right)),
        ("overlay intersection (1,600 x 1,600 boxes)", lambda: mgpd.overlay(left, right), lambda: gpd.overlay(left, right)),
        ("overlay difference (1,600 x 1,600 boxes)", lambda: mgpd.overlay(left, right, how="difference"), lambda: gpd.overlay(left, right, how="difference")),
    ]
    print(f"machine: {platform.platform()} | Python {platform.python_version()}")
    print(f"{'case':<47}{'mojo-geopandas':>17}{'geopandas':>14}{'ratio':>9}")
    print("-" * 89)
    for name, ours, theirs in cases:
        ours()
        a, b = timeit(ours), timeit(theirs)
        verdict = "faster" if a < b else "slower"
        print(f"{name:<47}{a * 1e3:>15.1f}ms{b * 1e3:>12.1f}ms{b / a:>8.2f}x  {verdict}")


if __name__ == "__main__":
    main()

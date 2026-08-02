# mojo-geopandas

`mojo-geopandas` is a Mojo-accelerated, drop-in subset of
[GeoPandas](https://geopandas.org/) focused on spatial joins and planar overlay.
It keeps GeoPandas and Shapely as the geometry and tabular interface, while
moving broad-phase bounding-box work into a small compiled Mojo library.

## Covered API

| GeoPandas API | Status |
| --- | --- |
| `sjoin` | `inner`, `left`, and `right`; all named predicates exposed by the installed GeoPandas spatial index, `dwithin`, and `on_attribute` |
| `sjoin_nearest` | exact nearest neighbours, ties, `max_distance`, `distance_col`, `exclusive`, and all join modes |
| `overlay` | `intersection`, `difference`, `symmetric_difference`, `union`, and `identity`; `keep_geom_type` and `make_valid` |

This is deliberately not a replacement for all of GeoPandas. File I/O,
coordinate transforms, plotting, GeoSeries/GeoDataFrame methods, dissolve,
clip, spatial indexing APIs, and non-covered tools remain upstream GeoPandas.
All topology is still GEOS/Shapely topology, so the results have GeoPandas'
correctness rather than an approximate bounding-box interpretation.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi` supplies the pinned Mojo nightly, GeoPandas, Shapely, and the test
environment. The package is available to `pixi run` commands through the
project `PYTHONPATH`.

## Usage

```python
import geopandas as gpd
from shapely import box
import mojogeopandas as mgpd

parcels = gpd.GeoDataFrame({"parcel": [1]}, geometry=[box(0, 0, 2, 2)])
zones = gpd.GeoDataFrame({"zone": ["R"]}, geometry=[box(1, 1, 3, 3)])

print(mgpd.sjoin(parcels, zones, predicate="intersects"))
print(mgpd.overlay(parcels, zones, how="intersection"))
```

The public signatures are the upstream signatures, so the covered calls can
usually replace `geopandas.sjoin`, `geopandas.sjoin_nearest`, or
`geopandas.overlay` directly.

## How it works

```
GeoDataFrame.geometry.bounds  ->  contiguous float64 (minx, miny, maxx, maxy)
                                      |
                                      v
src/capi.mojo                 ->  AABB intersects / contains / distance bounds
                                      |
                                      v
Shapely                       ->  exact predicate, distance, intersection, difference
                                      |
                                      v
GeoPandas helpers             ->  upstream-compatible indexes, suffixes, and output frame
```

The shared library exports C ABI functions from one Mojo compilation unit.
`ctypes` passes validated, contiguous float64 NumPy buffers as pointers, and Mojo reconstructs
`UnsafePointer[Float64, AnyOrigin[mut=True]]` internally. No ownership crosses
the boundary and the kernel allocates nothing. Empty arrays do not enter the C ABI.
AABBs are inclusive, so touching
geometries remain candidates; Shapely always decides the final relation.
For nearest joins, squared AABB distance is a lower bound that lets the exact
distance scan stop once no remaining box can improve the current result.

## Benchmarks

Measured by `pixi run bench` on `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`,
Python 3.13.14, with GeoPandas 1.1.4 and Shapely 2.1.2. Times are the best of
three runs and include the Python/geometry work but exclude Mojo library load.

| case | mojo-geopandas | geopandas | result |
| --- | ---: | ---: | --- |
| `sjoin intersects` (1,600 x 1,600 boxes) | 20.1 ms | 7.3 ms | 0.37x slower |
| `overlay intersection` (1,600 x 1,600 boxes) | 143.4 ms | 130.4 ms | 0.91x slower |
| `overlay difference` (1,600 x 1,600 boxes) | 652.9 ms | 754.7 ms | 1.16x faster |

The honest trade-off is visible here: GeoPandas' STRtree is exceptionally good
for query-only joins, while the Mojo broad phase pays off when an overlay must
visit many candidates and perform repeated difference operations. Benchmark
your geometry distribution; sparse and highly skewed datasets can move this
balance substantially.

## Verification

The test suite compares every covered operation directly with the installed
upstream GeoPandas release, including predicates, join modes, indexes,
suffixes, nearest distances, all overlay modes, line overlays, and invalid
polygon repair.

```bash
pixi run build && pixi run test && pixi run bench
```

## License

MIT

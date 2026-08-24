"""Spatial joins and overlays with Mojo AABB broad-phase filtering."""

from __future__ import annotations

import importlib
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries

from ._lib import bbox_distance2, bbox_pairs

_sjoin_mod = importlib.import_module("geopandas.tools.sjoin")
_overlay_mod = importlib.import_module("geopandas.tools.overlay")


def _pairs(left: GeoDataFrame, right: GeoDataFrame, predicate: str, distance=None):
    """Return position pairs after a lossless Mojo AABB candidate pass."""
    lb = np.asarray(left.geometry.bounds, dtype=np.float64)
    rb = np.asarray(right.geometry.bounds, dtype=np.float64)
    if len(left) == 0 or len(right) == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    relation = "contains" if predicate == "contains" else "intersects"
    if predicate == "within":
        ri, li = bbox_pairs(rb, lb, "contains")
        order = np.lexsort((ri, li))
        li, ri = li[order], ri[order]
    else:
        if predicate == "dwithin":
            if distance is None:
                raise ValueError("'distance' parameter is required for 'dwithin' predicate")
            d = np.broadcast_to(np.asarray(distance, dtype=np.float64), (len(left),))
            if not np.isfinite(d).all() or (d < 0).any():
                raise ValueError("'distance' must contain only finite, non-negative values")
            lb = lb.copy()
            lb[:, :2] -= d[:, None]
            lb[:, 2:] += d[:, None]
        li, ri = bbox_pairs(lb, rb, relation)
    if not len(li):
        return li.astype(np.intp), ri.astype(np.intp)
    a = left.geometry.array.take(li)
    b = right.geometry.array.take(ri)
    if predicate == "dwithin":
        good = shapely.dwithin(a, b, d[li])
    else:
        good = getattr(shapely, predicate)(a, b)
    return li[good].astype(np.intp), ri[good].astype(np.intp)


def sjoin(
    left_df,
    right_df,
    how="inner",
    predicate="intersects",
    lsuffix="left",
    rsuffix="right",
    distance=None,
    on_attribute=None,
    **kwargs,
):
    """Spatial join with the same public signature as :func:`geopandas.sjoin`."""
    if kwargs:
        first = next(iter(kwargs))
        raise TypeError(f"sjoin() got an unexpected keyword argument '{first}'")
    attrs = _sjoin_mod._maybe_make_list(on_attribute)
    _sjoin_mod._basic_checks(left_df, right_df, how, lsuffix, rsuffix, on_attribute=attrs)
    valid_predicates = left_df.geometry.array.sindex.valid_query_predicates
    if predicate not in valid_predicates:
        names = sorted(item for item in valid_predicates if item is not None)
        raise ValueError(f"Got predicate={predicate!r}; valid predicates are {names}")
    li, ri = _pairs(left_df, right_df, predicate, distance)
    if attrs:
        for attr in attrs:
            (li, ri), _ = _sjoin_mod._filter_shared_attribute(left_df, right_df, li, ri, attr)
    joined, _ = _sjoin_mod._frame_join(
        left_df, right_df, (li, ri), None, how, lsuffix, rsuffix, predicate, on_attribute=attrs
    )
    return joined


def _nearest_pairs(query, target, max_distance, exclusive):
    bounds = bbox_distance2(np.asarray(query.geometry.bounds), np.asarray(target.geometry.bounds))
    qi, ti, distances = [], [], []
    for i, geometry in enumerate(query.geometry):
        if geometry is None or geometry.is_empty:
            continue
        best = np.inf
        matches = []
        for j in np.argsort(bounds[i], kind="stable"):
            lower = bounds[i, j]
            if not np.isfinite(lower) or lower > best * best:
                break
            candidate = target.geometry.iloc[j]
            if candidate is None or candidate.is_empty or (exclusive and geometry.equals(candidate)):
                continue
            value = geometry.distance(candidate)
            if max_distance is not None and value > max_distance:
                continue
            if value < best:
                best, matches = value, [j]
            elif value == best:
                matches.append(j)
        for j in matches:
            qi.append(i)
            ti.append(j)
            distances.append(best)
    return np.asarray(qi, dtype=np.intp), np.asarray(ti, dtype=np.intp), np.asarray(distances, dtype=float)


def sjoin_nearest(
    left_df,
    right_df,
    how="inner",
    max_distance=None,
    lsuffix="left",
    rsuffix="right",
    distance_col=None,
    exclusive=False,
):
    """Nearest-neighbour spatial join with GeoPandas' public signature."""
    _sjoin_mod._basic_checks(left_df, right_df, how, lsuffix, rsuffix)
    left_df.geometry.values.check_geographic_crs(stacklevel=1)
    right_df.geometry.values.check_geographic_crs(stacklevel=1)
    if max_distance is not None and max_distance <= 0:
        raise ValueError("max_distance must be greater than 0")
    if how == "right":
        ri, li, distances = _nearest_pairs(right_df, left_df, max_distance, exclusive)
    else:
        li, ri, distances = _nearest_pairs(left_df, right_df, max_distance, exclusive)
    joined, result_distances = _sjoin_mod._frame_join(
        left_df, right_df, (li, ri), distances if distance_col is not None else None,
        how, lsuffix, rsuffix, None,
    )
    if distance_col is not None:
        joined[distance_col] = result_distances
    return joined


def _overlay_intersection(df1, df2):
    idx1, idx2 = _pairs(df1, df2, "intersects")
    if len(idx1):
        left = df1.geometry.take(idx1).reset_index(drop=True)
        right = df2.geometry.take(idx2).reset_index(drop=True)
        intersections = left.intersection(right)
        poly = intersections.geom_type.isin(_overlay_mod.POLYGON_GEOM_TYPES)
        intersections.loc[poly] = intersections[poly].make_valid()
        pairs = pd.DataFrame({"__idx1": idx1, "__idx2": idx2})
        first = df1.reset_index(drop=True)
        second = df2.reset_index(drop=True)
        result = pairs.merge(first.drop(first._geometry_column_name, axis=1), left_on="__idx1", right_index=True)
        result = result.merge(
            second.drop(second._geometry_column_name, axis=1), left_on="__idx2", right_index=True,
            suffixes=("_1", "_2"),
        )
        return GeoDataFrame(result, geometry=intersections, crs=df1.crs)
    result = df1.iloc[:0].merge(
        df2.iloc[:0].drop(df2.geometry.name, axis=1), left_index=True, right_index=True,
        suffixes=("_1", "_2"),
    )
    result["__idx1"] = np.nan
    result["__idx2"] = np.nan
    return result[result.columns.drop(df1.geometry.name).tolist() + [df1.geometry.name]]


def _overlay_difference(df1, df2):
    idx1, idx2 = _pairs(df1, df2, "intersects")
    neighbours = [[] for _ in range(len(df1))]
    for i, j in zip(idx1, idx2):
        neighbours[i].append(j)
    new = [
        reduce(lambda x, y: x.difference(y), [geom] + [df2.geometry.iloc[j] for j in js])
        for geom, js in zip(df1.geometry, neighbours)
    ]
    differences = GeoSeries(new, index=df1.index, crs=df1.crs)
    poly = differences.geom_type.isin(_overlay_mod.POLYGON_GEOM_TYPES)
    differences.loc[poly] = differences[poly].make_valid()
    result = df1[~differences.is_empty].copy()
    result[result._geometry_column_name] = differences[~differences.is_empty]
    return result


def _overlay_symmetric_difference(df1, df2):
    first, second = _overlay_difference(df1, df2), _overlay_difference(df2, df1)
    first["__idx1"], first["__idx2"] = range(len(first)), np.nan
    second["__idx2"], second["__idx1"] = range(len(second)), np.nan
    first, second = _overlay_mod._ensure_geometry_column(first), _overlay_mod._ensure_geometry_column(second)
    result = first.merge(second, on=["__idx1", "__idx2"], how="outer", suffixes=("_1", "_2"))
    geometry = result.geometry_1.copy()
    geometry.name = "geometry"
    geometry.loc[result.geometry_1.isnull()] = result.loc[result.geometry_1.isnull(), "geometry_2"]
    result.drop(["geometry_1", "geometry_2"], axis=1, inplace=True)
    return GeoDataFrame(result, geometry=geometry, crs=df1.crs)


def overlay(df1, df2, how="intersection", keep_geom_type=None, make_valid=True):
    """Overlay GeoDataFrames using Mojo to find exact-operation candidates."""
    allowed = ["intersection", "union", "identity", "symmetric_difference", "difference"]
    if how not in allowed:
        raise ValueError(f"`how` was '{how}' but is expected to be in {allowed}")
    if isinstance(df1, GeoSeries) or isinstance(df2, GeoSeries):
        raise NotImplementedError("overlay currently only implemented for GeoDataFrames")
    if not _overlay_mod._check_crs(df1, df2):
        _overlay_mod._crs_mismatch_warn(df1, df2, stacklevel=3)
    warn_keep = keep_geom_type is None
    keep_geom_type = True if keep_geom_type is None else keep_geom_type
    for i, df in enumerate((df1, df2)):
        kinds = [df.geom_type.isin(t).any() for t in (
            _overlay_mod.POLYGON_GEOM_TYPES, _overlay_mod.LINE_GEOM_TYPES, _overlay_mod.POINT_GEOM_TYPES
        )]
        if sum(kinds) > 1:
            raise NotImplementedError(f"df{i + 1} contains mixed geometry types.")
    geom_type = df1.geom_type.iloc[0] if keep_geom_type and len(df1) else None
    if keep_geom_type and not len(df1):
        warnings.warn("`keep_geom_type=True` is invalid when df1 is empty as there is no geometry type to keep. Setting `keep_geom_type=False`", stacklevel=2)
        keep_geom_type = False

    def valid(df):
        result = df.copy()
        if result.geom_type.isin(_overlay_mod.POLYGON_GEOM_TYPES).all():
            bad = ~result.geometry.is_valid
            if bad.any() and not make_valid:
                raise ValueError("You have passed make_valid=False along with invalid input geometries. Use make_valid=True or make sure that all geometries are valid before using overlay.")
            if bad.any():
                result.loc[bad, result._geometry_column_name] = result.loc[bad].geometry.make_valid()
                result = _overlay_mod._collection_extract(result, "Polygon", False)
        return result

    df1, df2 = valid(df1), valid(df2)
    if how == "intersection":
        result = _overlay_intersection(df1, df2)
    elif how == "difference":
        result = _overlay_difference(df1, df2)
    elif how == "symmetric_difference":
        result = _overlay_symmetric_difference(df1, df2)
    elif how == "union":
        inter = _overlay_intersection(df1, df2)
        sym = _overlay_symmetric_difference(df1, df2)
        result = pd.concat([inter, sym], ignore_index=True, sort=False)
        cols = [c for c in result.columns if c != "geometry"] + ["geometry"]
        result = result.reindex(columns=cols)
    else:
        inter = _overlay_intersection(df1, df2)
        diff = _overlay_difference(df1, df2)
        diff = _overlay_mod._ensure_geometry_column(diff)
        diff.columns = [c if c in inter.columns else f"{c}_1" for c in diff.columns]
        result = pd.concat([inter, diff], ignore_index=True, sort=False)
        cols = [c for c in inter.columns if c != "geometry"] + ["geometry"]
        result = result.reindex(columns=cols)
    if how != "difference":
        result = result.drop(["__idx1", "__idx2"], axis=1)
    if keep_geom_type:
        result = _overlay_mod._collection_extract(result, geom_type, warn_keep)
    result.reset_index(drop=True, inplace=True)
    return result

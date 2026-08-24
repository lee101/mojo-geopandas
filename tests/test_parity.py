"""Behavioural parity checks against the installed GeoPandas release."""

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import LineString, Point, Polygon, box

import mojogeopandas as mgpd
from mojogeopandas._lib import bbox_distance2, bbox_mask, bbox_pairs


@pytest.fixture
def frames():
    left = gpd.GeoDataFrame(
        {"value": [10, 20, 30], "group": ["a", "b", "a"]},
        geometry=[box(0, 0, 2, 2), box(3, 3, 5, 5), box(8, 8, 9, 9)],
        index=["one", "two", "three"], crs="EPSG:3857",
    )
    right = gpd.GeoDataFrame(
        {"value": [100, 200, 300], "group": ["a", "a", "b"]},
        geometry=[box(1, 1, 4, 4), box(4, 4, 6, 6), box(20, 20, 21, 21)],
        index=["r1", "r2", "r3"], crs="EPSG:3857",
    )
    return left, right


def test_aabb_masks_match_numpy():
    left = np.array([[0, 0, 2, 2], [5, 5, 6, 6]], dtype=float)
    right = np.array([[1, 1, 4, 4], [6, 6, 7, 7]], dtype=float)
    assert np.array_equal(bbox_mask(left, right, "intersects"), [[True, False], [False, True]])
    assert np.array_equal(bbox_mask(left, right, "contains"), [[False, False], [False, False]])


def test_aabb_simd_tail_matches_numpy():
    left = np.array([[i, i + 0.1, i + 1.2, i + 1.3] for i in range(9)], dtype=float)
    right = np.array([[i + 0.5, i - 0.1, i + 1.5, i + 0.8] for i in range(9)], dtype=float)
    intersects = (
        (left[:, None, 0] <= right[None, :, 2])
        & (right[None, :, 0] <= left[:, None, 2])
        & (left[:, None, 1] <= right[None, :, 3])
        & (right[None, :, 1] <= left[:, None, 3])
    )
    contains = (
        (left[:, None, 0] <= right[None, :, 0])
        & (left[:, None, 1] <= right[None, :, 1])
        & (left[:, None, 2] >= right[None, :, 2])
        & (left[:, None, 3] >= right[None, :, 3])
    )
    dx = np.maximum(
        0, np.maximum(right[None, :, 0] - left[:, None, 2], left[:, None, 0] - right[None, :, 2])
    )
    dy = np.maximum(
        0, np.maximum(right[None, :, 1] - left[:, None, 3], left[:, None, 1] - right[None, :, 3])
    )
    assert np.array_equal(bbox_mask(left, right, "intersects"), intersects)
    assert np.array_equal(bbox_mask(left, right, "contains"), contains)
    assert np.array_equal(bbox_distance2(left, right), dx * dx + dy * dy)
    for relation, expected in (("intersects", intersects), ("contains", contains)):
        li, ri = bbox_pairs(left, right, relation)
        expected_li, expected_ri = np.nonzero(expected)
        assert np.array_equal(li, expected_li)
        assert np.array_equal(ri, expected_ri)


def test_aabb_parallel_pairs_match_numpy():
    values = np.arange(1001, dtype=np.float64)
    left = np.column_stack((values, values, values + 1.25, values + 1.25))
    right = np.column_stack((values + 0.5, values - 0.5, values + 1.5, values + 0.5))
    expected = (
        (left[:, None, 0] <= right[None, :, 2])
        & (right[None, :, 0] <= left[:, None, 2])
        & (left[:, None, 1] <= right[None, :, 3])
        & (right[None, :, 1] <= left[:, None, 3])
    )
    li, ri = bbox_pairs(left, right, "intersects")
    expected_li, expected_ri = np.nonzero(expected)
    assert np.array_equal(li, expected_li)
    assert np.array_equal(ri, expected_ri)


def test_aabb_boundary_validation_and_empty_inputs():
    empty = np.empty((0, 4), dtype=np.float64)
    right = np.array([[0, 0, 1, 1]], dtype=np.float64)
    assert bbox_mask(empty, right, "intersects").shape == (0, 1)
    assert bbox_distance2(right, empty).shape == (1, 0)
    with pytest.raises(TypeError, match="float64"):
        bbox_mask(right.astype(np.float32), right, "intersects")
    with pytest.raises(ValueError, match="shape"):
        bbox_mask(np.empty((1, 3), dtype=np.float64), right, "intersects")


@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize("predicate", ["intersects", "contains", "within", "touches", "overlaps", "covers", "covered_by", "crosses"])
def test_sjoin_matches_geopandas(frames, how, predicate):
    left, right = frames
    expected = gpd.sjoin(left, right, how=how, predicate=predicate, lsuffix="l", rsuffix="r")
    actual = mgpd.sjoin(left, right, how=how, predicate=predicate, lsuffix="l", rsuffix="r")
    assert_geodataframe_equal(actual, expected, check_like=False)


def test_sjoin_dwithin_and_attribute_constraint(frames):
    left, right = frames
    points = gpd.GeoDataFrame({"group": ["a", "b"]}, geometry=[Point(0, 0), Point(5.5, 5.5)], crs=left.crs)
    expected = gpd.sjoin(points, right, predicate="dwithin", distance=np.array([1.5, 2.2]))
    actual = mgpd.sjoin(points, right, predicate="dwithin", distance=np.array([1.5, 2.2]))
    assert_geodataframe_equal(actual, expected)
    expected = gpd.sjoin(left, right, predicate="intersects", on_attribute="group")
    actual = mgpd.sjoin(left, right, predicate="intersects", on_attribute="group")
    assert_geodataframe_equal(actual, expected)


def test_sjoin_empty_and_invalid_arguments(frames):
    left, right = frames
    empty = left.iloc[:0]
    assert_geodataframe_equal(mgpd.sjoin(empty, right, how="left"), gpd.sjoin(empty, right, how="left"))
    with pytest.raises(ValueError):
        mgpd.sjoin(left, right, how="outer")
    with pytest.raises(ValueError):
        mgpd.sjoin(left, right, predicate="dwithin")
    with pytest.raises(ValueError):
        mgpd.sjoin(left, right, predicate="not-a-predicate")


@pytest.mark.parametrize("how", ["inner", "left", "right"])
def test_sjoin_nearest_matches_geopandas(frames, how):
    left, right = frames
    expected = gpd.sjoin_nearest(left, right, how=how, max_distance=20, distance_col="distance")
    actual = mgpd.sjoin_nearest(left, right, how=how, max_distance=20, distance_col="distance")
    assert_geodataframe_equal(actual, expected)


def test_sjoin_nearest_ties_and_exclusive():
    left = gpd.GeoDataFrame({"a": [1, 2]}, geometry=[Point(0, 0), Point(5, 0)])
    right = gpd.GeoDataFrame({"b": [1, 2, 3]}, geometry=[Point(-1, 0), Point(1, 0), Point(5, 0)])
    for exclusive in (False, True):
        expected = gpd.sjoin_nearest(left, right, max_distance=2, distance_col="distance", exclusive=exclusive)
        actual = mgpd.sjoin_nearest(left, right, max_distance=2, distance_col="distance", exclusive=exclusive)
        assert_geodataframe_equal(actual, expected)


@pytest.mark.parametrize("how", ["intersection", "difference", "symmetric_difference", "union", "identity"])
@pytest.mark.parametrize("keep_geom_type", [False, True])
def test_overlay_matches_geopandas(frames, how, keep_geom_type):
    left, right = frames
    expected = gpd.overlay(left, right, how=how, keep_geom_type=keep_geom_type)
    actual = mgpd.overlay(left, right, how=how, keep_geom_type=keep_geom_type)
    assert_geodataframe_equal(actual, expected, check_like=False)


def test_overlay_non_polygon_and_make_valid():
    lines = gpd.GeoDataFrame({"a": [1]}, geometry=[LineString([(0, 0), (4, 0)])])
    cutters = gpd.GeoDataFrame({"b": [2]}, geometry=[LineString([(2, -1), (2, 1)])])
    assert_geodataframe_equal(
        mgpd.overlay(lines, cutters, how="intersection", keep_geom_type=False),
        gpd.overlay(lines, cutters, how="intersection", keep_geom_type=False),
    )
    invalid = gpd.GeoDataFrame({"a": [1]}, geometry=[Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])])
    clip = gpd.GeoDataFrame({"b": [1]}, geometry=[box(0, 0, 1.5, 2)])
    assert_geodataframe_equal(
        mgpd.overlay(invalid, clip, how="intersection", make_valid=True),
        gpd.overlay(invalid, clip, how="intersection", make_valid=True),
    )
    with pytest.raises(ValueError):
        mgpd.overlay(invalid, clip, make_valid=False)

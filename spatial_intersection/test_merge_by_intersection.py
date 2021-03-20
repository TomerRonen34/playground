from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas import Series
from pandas.testing import assert_series_equal
from shapely.geometry import Polygon, GeometryCollection, Point
from geopandas.testing import assert_geodataframe_equal
from geopandas import GeoDataFrame

from spatial_intersection.merge_by_intersection import merge_gdfs_by_biggest_intersection
from spatial_intersection.utils import plot_geometry


class TestMergeByIntersection:
    def test_merge_gdfs_by_biggest_intersection(self):
        geometries_gdf, regions_gdf, expected_index_matches, expected_gdf = self._generate_test_params()
        merged_by_biggest_intersection = merge_gdfs_by_biggest_intersection(geometries_gdf, regions_gdf)

        index_matches = merged_by_biggest_intersection["index_right"]
        assert_series_equal(index_matches, expected_index_matches, check_names=False)
        assert_geodataframe_equal(merged_by_biggest_intersection, expected_gdf)

    @classmethod
    def _generate_test_params(cls):
        geometries_gdf = cls._generate_geometries_gdf()
        regions_gdf = cls._generate_regions_gdf()
        expected_index_matches, expected_gdf = cls._generate_expected_results(geometries_gdf, regions_gdf)
        cls._try_plot_geometries(geometries_gdf, regions_gdf)
        return geometries_gdf, regions_gdf, expected_index_matches, expected_gdf

    @staticmethod
    def _generate_regions_gdf() -> GeoDataFrame:
        s1_coords = np.array([(-1, 0), (-1, 1), (0, 1), (0, 0)])
        s2_coords = s1_coords + np.array([(1, 0)])
        s3_coords = s1_coords + np.array([(1, -1)])
        s4_coords = s1_coords + np.array([(0, -1)])
        s1, s2, s3, s4 = map(Polygon, [s1_coords, s2_coords, s3_coords, s4_coords])
        regions_gdf = gpd.GeoDataFrame({"s_name": ["s1", "s2", "s3", "s4"],
                                        "s_geom": [s1, s2, s3, s4]},
                                       geometry="s_geom", index=range(10, 14), crs=4326)
        return regions_gdf

    @staticmethod
    def _generate_geometries_gdf() -> GeoDataFrame:
        g1_coords = np.array([(-0.5, 0.5), (0.25, 0.5), (0.25, 0.25), (-0.5, 0.25)])  # intersect 10 & 11, mostly 10
        g2_coords = np.array([(-0.5, 0.8), (0.5, 0.8), (0.5, 0.6), (-0.5, 0.6)])  # intersects 10 & 11 equally
        g3_coords = np.array([(0.25, -0.25), (0.5, -0.25), (0.5, -0.5), (0.25, -0.5)])  # intersects 12 only
        g4_coords = np.array([(-1, 0.25), (0.25, -1), (-1, -1)])  # intersects 10 & 12 & 13, mostly 13
        g5_coords = np.array([(0.75, 0.5), (1.25, 0.5), (1.25, -0.2), (0.75, -0.2)])  # intersects 12 & 11, mostly 11
        g6_coords = np.array([(1.25, -0.5), (1.5, -0.5), (1.5, -0.75), (1.25, -0.75)])  # doesn't intersect anything
        g1, g2, g3, g4, g5, g6 = map(Polygon, [g1_coords, g2_coords, g3_coords,
                                               g4_coords, g5_coords, g6_coords])
        geometries_gdf = gpd.GeoDataFrame({"g_name": ["g1", "g2", "g3", "g4", "g5", "g6"],
                                           "g_geom": [g1, g2, g3, g4, g5, g6]},
                                          geometry="g_geom", index=range(20, 26), crs=4326)
        return geometries_gdf

    @staticmethod
    def _generate_expected_results(geometries_gdf: GeoDataFrame, regions_gdf: GeoDataFrame
                                   ) -> Tuple[Series, GeoDataFrame]:
        expected_regions_index = [10, 10, 12, 13, 11, None]
        expected_index_matches = pd.Series(expected_regions_index, index=geometries_gdf.index.copy())

        expected_gdf = geometries_gdf.copy()
        expected_gdf["index_right"] = expected_regions_index
        expected_regions_gdf = regions_gdf.loc[expected_regions_index]
        for col in regions_gdf.columns:
            expected_gdf[col] = expected_regions_gdf[col].values

        return expected_index_matches, expected_gdf

    @staticmethod
    def _try_plot_geometries(geometries_gdf: GeoDataFrame, regions_gdf: GeoDataFrame) -> None:
        to_vis = GeometryCollection([Point(0, 0), *regions_gdf.geometry.tolist(),
                                     *geometries_gdf.geometry.tolist()])
        try:
            plot_geometry(to_vis)
        except:
            pass

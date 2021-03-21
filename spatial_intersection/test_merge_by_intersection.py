import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from pandas import Series
from pandas.testing import assert_series_equal

from spatial_intersection.merge_by_intersection import merge_gdfs_by_biggest_intersection
from spatial_intersection.merge_by_intersection_test_params import TestParams


class TestMergeByIntersection:
    """
    check out the file 'vis_test_geometries.html', saved locally and created by generate_test_params_general_case()
    """

    @pytest.mark.parametrize("geometries_gdf,regions_gdf,expected_index_matches,expected_gdf",
                             [TestParams.generate_test_params_general_case(),
                              TestParams.generate_test_params_no_intersections()])
    def test_merge_gdfs_by_biggest_intersection(self, geometries_gdf: GeoDataFrame, regions_gdf: GeoDataFrame,
                                                expected_index_matches: Series, expected_gdf: GeoDataFrame):
        merged_by_biggest_intersection = merge_gdfs_by_biggest_intersection(geometries_gdf, regions_gdf)

        index_matches = merged_by_biggest_intersection["index_right"]
        assert_series_equal(index_matches, expected_index_matches, check_names=False)
        assert_geodataframe_equal(merged_by_biggest_intersection, expected_gdf, check_dtype=False)

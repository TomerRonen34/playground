from typing import Union

from geopandas import GeoDataFrame
from pandas import Series, DataFrame
import geopandas as gpd
from shapely.geometry import GeometryCollection


def merge_gdfs_by_biggest_intersection(left_gdf: GeoDataFrame, right_gdf: GeoDataFrame) -> GeoDataFrame:
    """
    For every row in left_gdf, concatenates the row of right_gdf which has the biggest intersection with it.
    Also adds an "index_right" column to document which row of right_gdf was chosen.

    Geometries in left_gdf that have no intersections in right_gdf are concatenated with NaN values,
    as in pd.merge(left_df, right_df, how="left").
    """
    sjoined_geometries = _sjoin_geometries(left_gdf, right_gdf)
    sjoined_geometries["intersection_area"] = _calculate_intersection_area(sjoined_geometries)
    index_matches_by_highest_intersection = _get_index_matches_by_highest_intersection(sjoined_geometries)
    merged_by_biggest_intersection = _merge_by_indexes(left_gdf, right_gdf, index_matches_by_highest_intersection)
    return merged_by_biggest_intersection


def _sjoin_geometries(left_gdf: GeoDataFrame, right_gdf: GeoDataFrame) -> GeoDataFrame:
    left_gdf = gpd.GeoDataFrame(left_gdf.geometry.rename("geometry"))
    right_gdf = gpd.GeoDataFrame(right_gdf.geometry.rename("geometry"))

    sjoined = gpd.sjoin(left_gdf, right_gdf, how="left", op='intersects')
    sjoined_geometries = sjoined.merge(right_gdf, left_on="index_right", right_index=True,
                                       suffixes=['_left', '_right'], how="left")
    return sjoined_geometries


def _calculate_intersection_area(sjoined_geometries: GeoDataFrame) -> Series:
    intersection = sjoined_geometries.apply(lambda row:
                                            row["geometry_left"].intersection(row["geometry_right"])
                                            if row["geometry_right"] is not None else GeometryCollection()
                                            , axis=1)
    intersection_area = intersection.apply(lambda geom: geom.area)
    return intersection_area


def _get_index_matches_by_highest_intersection(sjoined_geometries: GeoDataFrame) -> DataFrame:
    sjoined_sorted = sjoined_geometries.sort_values(by=["intersection_area", "index_right"], ascending=[False, True])
    sjoined_dedup = sjoined_sorted[~sjoined_sorted.index.duplicated(keep="first")]
    index_matches_by_highest_intersection = (sjoined_dedup["index_right"]
                                             .reset_index().rename(columns={"index": "index_left"}))
    return index_matches_by_highest_intersection


def _merge_by_indexes(left_df: Union[DataFrame, GeoDataFrame],
                      right_df: Union[DataFrame, GeoDataFrame],
                      index_matches: DataFrame
                      ) -> Union[DataFrame, GeoDataFrame]:
    left_df = left_df.copy()
    index_matches = index_matches.set_index("index_left")
    left_df["index_right"] = index_matches
    merged = left_df.merge(right_df, left_on="index_right", right_index=True,
                           suffixes=['_left', '_right'], how="left")
    return merged

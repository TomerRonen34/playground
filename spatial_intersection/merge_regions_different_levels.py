from typing import Iterable
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, GeometryCollection, Point

from spatial_intersection.merge_by_intersection import merge_gdfs_by_biggest_intersection
from spatial_intersection.utils import vis_geometries_as_html

u_gdf = GeoDataFrame({"u_name": ["u1", "u2"],
                      "u_geom": [Polygon([(0, 0), (4, 0), (1, 4), (0, 4)]),
                                 Polygon([(0, 4), (4, 4), (0, 8)])]},
                     geometry="u_geom", crs=4326)

g_gdf = GeoDataFrame({"g_name": ["g1", "g2"],
                      "u_name": ["u1", "u2"],
                      "g_geom": [Polygon([(0, 0), (2, 0), (0, 2)]),
                                 Polygon([(0, 4), (2, 4), (0, 6)])]},
                     geometry="g_geom", crs=4326)

m_gdf = GeoDataFrame({"m_name": ["m1", "m2"],
                      "g_name": ["g1", "g2"],
                      "u_name": ["u1", "u2"],
                      "m_geom": [Polygon([(0, 0), (1, 0), (0, 1)]),
                                 Polygon([(0, 4), (1, 4), (0, 5)])]},
                     geometry="m_geom", crs=4326)

merhavs_geoms = GeoDataFrame(
    {"intersects": ["", "u2", "u2 g2 m2", "u1 g1 m1", "u1 g1", "u1", "u2 g2 m2"],
     "merhav_geom": [Point(3, 3), Point(3, 4), Point(0.5, 4.5), Point(0.5, 0.2), Point(1, 0.8), Point(1, 2),
                     Point(0, 3.9)  # Point(0, 3.9) is a tricky one - intersects u1 (allegedly takes precedence over u2), but also m2
                     ]},
    geometry="merhav_geom", crs=4326)
merhavs_geoms.geometry = merhavs_geoms.geometry.buffer(0.2)

to_vis = GeometryCollection(u_gdf.geometry.tolist() + g_gdf.geometry.tolist()
                            + m_gdf.geometry.tolist() + merhavs_geoms.geometry.tolist())
vis_geometries_as_html(to_vis, "u_g_m_merhavs.html")


def construct_merhav_names_by_intersecting_regions():
    pass


def _choose_first_nonempty_string(strings: Iterable[str]) -> str:
    for string in strings:
        if pd.notna(string) and string != '':
            return string
    return ''


merhavs_us = merge_gdfs_by_biggest_intersection(merhavs_geoms, u_gdf)
merhavs_gs = merge_gdfs_by_biggest_intersection(merhavs_geoms, g_gdf)
merhavs_ms = merge_gdfs_by_biggest_intersection(merhavs_geoms, m_gdf)

possible_u_names = pd.Series(zip(merhavs_ms["u_name"], merhavs_gs["u_name"], merhavs_us["u_name"]))
possible_g_names = pd.Series(zip(merhavs_ms["g_name"], merhavs_gs["g_name"]))
u_names = possible_u_names.apply(_choose_first_nonempty_string)
g_names = possible_g_names.apply(_choose_first_nonempty_string)
m_names = merhavs_ms["m_name"].fillna('')

merhav_names = [f"{u} {g} {m}".strip() for u, g, m in zip(u_names, g_names, m_names)]

assert (merhavs_geoms["intersects"].values == merhav_names).all()
a = 0

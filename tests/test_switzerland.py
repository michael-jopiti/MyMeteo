# tests/test_switzerland_unittest.py
# High-rigor stdlib unittest for `switzerland.py`.
# Creates a tiny synthetic Switzerland-like GeoJSON dataset on the fly and
# exercises:
# - File I/O and CRS alignment
# - District attribute synthesis (dissolve municipalities when districts lack IDs)
# - District adjacency graph (rook/queen) and node attribute schema
# - Level validation and exit paths
# - Plotting helpers produce files (no GUI; Agg backend)
# - Station↔district selection/edge-drawing utilities
#
# Run:
#   export PYTHONPATH=$PWD/src
#   pytest -q tests/test_switzerland_unittest.py
#   # or
#   python -m unittest -v tests.test_switzerland_unittest

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # headless, no GUI
import networkx as nx
from shapely.geometry import Polygon, Point


# Import the module under test (expects src-layout with PYTHONPATH=$PWD/src)
try:
    from utils.switzerland import Switzerland
except Exception as e:
    Switzerland = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@unittest.skipIf(Switzerland is None, f"Import failed: {_IMPORT_ERROR}")
class TestSwitzerland(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp(prefix="ch_ut_"))
        self.geo_dir = self.td / "geo"
        self.geo_dir.mkdir(parents=True, exist_ok=True)

        # --- Build a minimal synthetic dataset in EPSG:4326 ---
        # Two adjacent districts (share a vertical border at x=1.0)
        # District 10: square [0,1]x[0,1]
        # District 11: square [1,2]x[0,1]
        d10 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        d11 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        # Switzerland outline: a bit larger around both districts
        ch_poly = Polygon([(-0.2, -0.2), (2.2, -0.2), (2.2, 1.2), (-0.2, 1.2)])

        # Lakes: one small polygon inside district 10
        lake = Polygon([(0.2, 0.2), (0.35, 0.2), (0.35, 0.35), (0.2, 0.35)])

        # Municipalities: split each district into two municipalities, each carrying BEZIRKSNUM
        mA = Polygon([(0, 0), (0.5, 0), (0.5, 1), (0, 1)])         # in dist 10
        mB = Polygon([(0.5, 0), (1, 0), (1, 1), (0.5, 1)])          # in dist 10
        mC = Polygon([(1, 0), (1.5, 0), (1.5, 1), (1, 1)])          # in dist 11
        mD = Polygon([(1.5, 0), (2, 0), (2, 1), (1.5, 1)])          # in dist 11

        # Save GeoJSON files with attributes
        gpd.GeoDataFrame(
            {"KANTONSNUM": [0], "NAME": ["CH"], "geometry": [ch_poly]},
            crs="EPSG:4326",
        ).to_file(self.geo_dir / "ch.geojson", driver="GeoJSON")

        # Primary districts file WITH attributes (normal path)
        gpd.GeoDataFrame(
            {"BEZIRKSNUM": [10, 11], "NAME": ["D10", "D11"], "geometry": [d10, d11]},
            crs="EPSG:4326",
        ).to_file(self.geo_dir / "ch-districts.geojson", driver="GeoJSON")

        # Municipalities with BFS + BEZIRKSNUM (used also for dissolve path)
        gpd.GeoDataFrame(
            {
                "BFS_NUMMER": [1001, 1002, 1101, 1102],
                "BEZIRKSNUM": [10, 10, 11, 11],
                "geometry": [mA, mB, mC, mD],
            },
            crs="EPSG:4326",
        ).to_file(self.geo_dir / "ch-municipalities.geojson", driver="GeoJSON")

        # A "rich" municipalities file expected by Switzerland.__init__
        gpd.GeoDataFrame(
            {
                "BFS_NUMMER": [1001, 1002, 1101, 1102],
                "geometry": [mA, mB, mC, mD],
            },
            crs="EPSG:4326",
        ).to_file(self.geo_dir / "municipalities.geojson", driver="GeoJSON")

        # Lakes
        gpd.GeoDataFrame(
            {"NAME": ["LakeX"], "geometry": [lake]},
            crs="EPSG:4326",
        ).to_file(self.geo_dir / "ch-lakes.geojson", driver="GeoJSON")

        # Instance under test
        self.sw = Switzerland(geojson_dir_path=str(self.geo_dir))

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    # ---------------------- Initialization & CRS ----------------------

    def test_crs_alignment_and_validity(self):
        self.assertIsNotNone(self.sw.geo_switzerland.crs)
        self.assertEqual(self.sw.geo_districts.crs, self.sw.geo_switzerland.crs)
        self.assertEqual(self.sw.geo_municipalities.crs, self.sw.geo_switzerland.crs)
        self.assertEqual(self.sw.geo_lakes.crs, self.sw.geo_switzerland.crs)
        # Valid geometries after buffer(0) fix in __init__
        self.assertTrue(self.sw.geo_districts.geometry.is_valid.all())

    # ---------------------- Graph building (district) ----------------------

    def test_get_graph_rook_and_schema(self):
        G = self.sw.get_graph(level="district", rook_adjacency=True)
        # Two districts, one shared border -> exactly one edge
        self.assertEqual(G.number_of_nodes(), 2)
        self.assertEqual(G.number_of_edges(), 1)

        # Node attribute schema
        for _, a in G.nodes(data=True):
            for key in ("name", "x", "y", "lon", "lat", "area", "geometry"):
                self.assertIn(key, a)
        # Graph metadata
        self.assertEqual(G.graph["level"], "district")
        self.assertTrue(G.graph["rook"])

    def test_get_graph_queen(self):
        # Queen vs rook should be equivalent here (they touch along a segment)
        Gq = self.sw.get_graph(level="district", rook_adjacency=False, rebuild=True)
        self.assertEqual(Gq.number_of_nodes(), 2)
        self.assertEqual(Gq.number_of_edges(), 1)
        self.assertFalse(Gq.graph["rook"])

    def test_get_graph_invalid_and_wip_levels(self):
        # Invalid level -> ValueError
        with self.assertRaises(ValueError):
            self.sw.get_graph(level="not-a-level")
        # WIP levels print and sys.exit(0)
        with self.assertRaises(SystemExit) as ctx_c:
            self.sw.get_graph(level="cantonal")
        self.assertEqual(ctx_c.exception.code, 0)
        with self.assertRaises(SystemExit) as ctx_m:
            self.sw.get_graph(level="municipality")
        self.assertEqual(ctx_m.exception.code, 0)

    # ---------------------- District dissolve path ----------------------

    def test_dissolve_municipalities_when_districts_lack_ids(self):
        # Overwrite districts file to have only geometry -> triggers _ensure_district_attributes -> dissolve
        only_geom = gpd.GeoDataFrame({"geometry": self.sw.geo_districts.geometry}, crs=self.sw.geo_districts.crs)
        only_geom.to_file(self.geo_dir / "ch-districts.geojson", driver="GeoJSON")

        sw2 = Switzerland(geojson_dir_path=str(self.geo_dir))
        # Now districts must have synthesized attributes from municipalities
        self.assertIn("BEZIRKSNUM", sw2.geo_districts.columns)
        self.assertIn("NAME", sw2.geo_districts.columns)
        # Dissolve should still yield 2 districts
        self.assertEqual(len(sw2.geo_districts), 2)
        # Graph should still work
        G = sw2.get_graph(level="district", rook_adjacency=True)
        self.assertEqual(G.number_of_nodes(), 2)
        self.assertEqual(G.number_of_edges(), 1)

    # ---------------------- Plots / output artifacts ----------------------

    def test_draw_districts_map_writes_file(self):
        out = self.td / "swiss_districts.png"
        self.sw.draw_districts_map(output_path=str(out), dpi=150)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    def test_draw_graph_writes_file(self):
        # Ensure graph cached/available
        self.sw.get_graph(level="district", rook_adjacency=True, rebuild=True)
        out = self.td / "swiss_districts_graph.pdf"
        self.sw.draw_graph(output_path=str(out), dpi=150)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    # ---------------------- Station ↔ district utilities ----------------------

    def _make_station_graph(self):
        # Build a tiny station graph with two stations:
        # S1 inside district 10 (0.3,0.3); S2 near district 11 (1.7,0.5)
        Gs = nx.Graph()
        Gs.add_node("S1", lon=0.30, lat=0.30)
        Gs.add_node("S2", lon=1.70, lat=0.50)
        return Gs

    def test_draw_all_stations_districts(self):
        out = self.td / "stations_vs_districts.png"
        Gs = self._make_station_graph()
        # radius 20 km over our tiny unit square world is arbitrary; treat as degrees-based buffer in LV95 (meters),
        # but our data are synthetic and projected internally, so any positive radius should select something.
        self.sw.draw_all_stations_districts(
            station_graph=Gs,
            radius_km=50.0,
            output_path=str(out),
            dpi=120,
        )
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    def test_draw_stations_to_district_centroids(self):
        out = self.td / "stations_to_centroids.png"
        Gs = self._make_station_graph()
        self.sw.draw_stations_to_district_centroids(
            station_graph=Gs,
            radius_km=500.0,
            output_path=str(out),
            dpi=120,
            station_size=10.0,
            centroid_size=12.0,
        )
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

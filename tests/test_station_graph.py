# tests/test_meteo_stations_extensive_unittest.py
# High-rigor stdlib unittest suite for utils.meteo_stations, designed for DL graph QA.
# Each test runs in its own isolated temp tree to avoid cross-test contamination.

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    # src-layout expected: run tests with PYTHONPATH=$PWD/src
    import utils.meteo_stations as ms
except Exception as e:
    ms = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
    R = 6371.0088
    p1 = math.radians(a_lat)
    p2 = math.radians(b_lat)
    dphi = math.radians(b_lat - a_lat)
    dlmb = math.radians(b_lon - a_lon)
    x = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


@unittest.skipIf(ms is None, f"Import failed: {_IMPORT_ERROR}")
class TestMeteoStationsExtensive(unittest.TestCase):
    # Canonical two-station setup used by most tests
    A = ("ABO", 46.0000, 7.0000, 600.0)
    B = ("ZUR", 46.5000, 7.6000, 430.0)
    AB_DIST = _haversine_km(A[1], A[2], B[1], B[2])

    def setUp(self):
        self.td = Path(tempfile.mkdtemp(prefix="mstations_ut_"))
        self.root = self.td / "parquets"
        (self.root / "2000").mkdir(parents=True)
        (self.root / "2001").mkdir(parents=True)
        self._write_base_tree()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    # -------------------------- Helpers --------------------------

    def _write_base_tree(self):
        # 2000
        pd.DataFrame(
            {
                "time_utc": pd.date_range("2000-01-01", periods=3, freq="h", tz="UTC"),
                "lat": [self.A[1]] * 3,
                "lon": [self.A[2]] * 3,
                "elev": [self.A[3]] * 3,
                "temp": [1.0, 2.0, 3.0],
            }
        ).to_parquet(self.root / "2000" / f"{self.A[0]}.parquet", engine=ms.ENGINE, index=False)
        pd.DataFrame(
            {
                "time_utc": pd.date_range("2000-01-01", periods=2, freq="h", tz="UTC"),
                "lat": [self.B[1]] * 2,
                "lon": [self.B[2]] * 2,
                "elev": [self.B[3]] * 2,
                "wind": [5.0, 6.0],
            }
        ).to_parquet(self.root / "2000" / f"{self.B[0]}.parquet", engine=ms.ENGINE, index=False)
        # 2001
        pd.DataFrame(
            {
                "time_utc": pd.date_range("2001-01-01", periods=1, freq="h", tz="UTC"),
                "lat": [self.A[1]],
                "lon": [self.A[2]],
                "elev": [self.A[3]],
                "humidity": [40.0],
            }
        ).to_parquet(self.root / "2001" / f"{self.A[0]}.parquet", engine=ms.ENGINE, index=False)
        pd.DataFrame(
            {
                "time_utc": pd.date_range("2001-01-01", periods=1, freq="h", tz="UTC"),
                "lat": [self.B[1]],
                "lon": [self.B[2]],
                "elev": [self.B[3]],
                "pressure": [1013.2],
            }
        ).to_parquet(self.root / "2001" / f"{self.B[0]}.parquet", engine=ms.ENGINE, index=False)

    # -------------------- Core graph construction --------------------

    def test_schema_and_determinism(self):
        G1 = ms.get_station_graph(str(self.root), auto_coords=False, d_km=None, show_progress=False)
        G2 = ms.get_station_graph(str(self.root), auto_coords=False, d_km=None, show_progress=False)

        self.assertEqual(set(G1.nodes), set(G2.nodes))
        self.assertEqual(set(G1.edges), set(G2.edges))
        self.assertEqual(G1.number_of_nodes(), 2)
        self.assertEqual(G1.number_of_edges(), 0)

        for n, a in G1.nodes(data=True):
            self.assertIn("station_abbr", a)
            self.assertIn("years", a)
            self.assertIn("columns_union", a)
            self.assertIn("first_year", a)
            self.assertIn("last_year", a)
            self.assertIsInstance(a["columns_union"], list)
            self.assertIsInstance(a["years"], dict)
            self.assertIsInstance(a["first_year"], int)
            self.assertIsInstance(a["last_year"], int)

            # per-year JSON records are serializable
            y2000 = a["years"]["2000"]
            self.assertIn("n_rows", y2000)
            self.assertIn("columns", y2000)
            self.assertIn("data", y2000)
            rec0 = y2000["data"][0]
            self.assertIn("time_utc", rec0)
            self.assertIsInstance(rec0["time_utc"], (str, type(None)))

        self.assertEqual(G1.nodes["ABO"]["first_year"], 2000)
        self.assertEqual(G1.nodes["ABO"]["last_year"], 2001)
        self.assertEqual(G1.nodes["ZUR"]["first_year"], 2000)
        self.assertEqual(G1.nodes["ZUR"]["last_year"], 2001)

    def test_drop_columns_respected(self):
        G = ms.get_station_graph(
            str(self.root),
            auto_coords=False,
            d_km=None,
            show_progress=False,
            drop_columns=["temp", "wind"],
        )
        cols = set(G.nodes["ABO"]["columns_union"]) | set(G.nodes["ZUR"]["columns_union"])
        self.assertNotIn("temp", cols)
        self.assertNotIn("wind", cols)
        self.assertIn("time_utc", cols)

    def test_year_filtering(self):
        G = ms.get_station_graph(str(self.root), years=[2000], auto_coords=False, d_km=None, show_progress=False)
        self.assertEqual(G.number_of_nodes(), 2)
        for _, a in G.nodes(data=True):
            self.assertEqual(a["first_year"], 2000)
            self.assertEqual(a["last_year"], 2000)
            self.assertEqual(set(a["years"].keys()), {"2000"})

    # ------------------------ Geometry / edges ------------------------

    def test_radius_edges_correctness_and_symmetry(self):
        # Below threshold -> no edges
        G0 = ms.get_station_graph(str(self.root), auto_coords=False, d_km=self.AB_DIST - 2.0, show_progress=False)
        self.assertEqual(G0.number_of_edges(), 0)

        # Above threshold -> one undirected edge
        G1 = ms.get_station_graph(str(self.root), auto_coords=False, d_km=self.AB_DIST + 2.0, show_progress=False)
        self.assertEqual(G1.number_of_edges(), 1)
        (u, v, d) = next(iter(G1.edges(data=True)))
        self.assertIn("distance_km", d)
        self.assertAlmostEqual(d["distance_km"], self.AB_DIST, delta=5.0)
        self.assertTrue(G1.has_edge(u, v) and G1.has_edge(v, u))

    def test_edge_count_monotone_in_radius(self):
        radii = [0.0, self.AB_DIST - 2, self.AB_DIST + 2, self.AB_DIST + 100]
        counts = [ms.get_station_graph(str(self.root), auto_coords=False, d_km=r, show_progress=False).number_of_edges()
                  for r in radii]
        self.assertEqual(counts, sorted(counts))

    def test_partial_coords_no_edges(self):
        G = ms.get_station_graph(str(self.root), auto_coords=False, d_km=None, show_progress=False)
        G.nodes["ABO"]["lat"] = None
        G.nodes["ABO"]["lon"] = None
        ms._maybe_add_distance_edges(G, d_km=self.AB_DIST + 10)
        self.assertEqual(G.number_of_edges(), 0)

    # ----------------- Coordinate attachment paths ------------------

    def test_attach_coords_from_dataframe_enables_edges(self):
        G = ms.get_station_graph(str(self.root), auto_coords=False, d_km=None, show_progress=False)
        for n in G.nodes:
            for k in ("lat", "lon", "elev_m"):
                G.nodes[n][k] = None
        df = pd.DataFrame(
            {
                "station_abbr": ["ABO", "ZUR"],
                "lat": [self.A[1], self.B[1]],
                "lon": [self.A[2], self.B[2]],
                "elev_m": [self.A[3], self.B[3]],
                "station_name": ["Abo", "Zurich"],
            }
        )
        updated = ms.attach_coords_from_dataframe(G, df)
        self.assertEqual(updated, 2)
        ms._maybe_add_distance_edges(G, d_km=self.AB_DIST + 2.0)
        self.assertEqual(G.number_of_edges(), 1)

    def test_auto_coords_download_merge_via_file_urls(self):
        coords_csv = self.td / "coords.csv"
        pd.DataFrame(
            {
                "station_abbr": ["ABO", "ZUR"],
                "latitude": [self.A[1], self.B[1]],
                "longitude": [self.A[2], self.B[2]],
                "elevation_m": [self.A[3], self.B[3]],
                "station_name": ["Abo Name", "Zur Name"],
            }
        ).to_csv(coords_csv, index=False)

        G = ms.get_station_graph(
            str(self.root),
            auto_coords=True,
            coord_urls=[f"file://{coords_csv.resolve()}"],
            geocode_missing=False,
            d_km=None,
            show_progress=False,
            coords_cache_dir=self.td / "cache",
        )
        have = sum(1 for _, a in G.nodes(data=True) if a.get("lat") is not None and a.get("lon") is not None)
        self.assertEqual(have, 2)

    def test_auto_coords_with_mocked_geocode_path(self):
        coords_csv = self.td / "coords_partial.csv"
        pd.DataFrame(
            {
                "station_abbr": ["ABO", "ZUR"],
                "latitude": [np.nan, np.nan],
                "longitude": [np.nan, np.nan],
                "elevation_m": [self.A[3], self.B[3]],
                "station_name": ["AboName", "ZurName"],
            }
        ).to_csv(coords_csv, index=False)

        with patch.object(ms, "_swisstopo_geocode_one", return_value=(46.2, 7.3)):
            G = ms.get_station_graph(
                str(self.root),
                auto_coords=True,
                coord_urls=[f"file://{coords_csv.resolve()}"],
                geocode_missing=True,
                geocode_rate_sleep_s=0.0,
                d_km=None,
                show_progress=False,
                coords_cache_dir=self.td / "cache_gc",
            )
        have = sum(1 for _, a in G.nodes(data=True) if a.get("lat") is not None and a.get("lon") is not None)
        self.assertEqual(have, 2)

    # -------------------- Save graph (IO sanitization) --------------------

    def test_save_graph_gexf_and_graphml_compact_and_full(self):
        G = ms.get_station_graph(str(self.root), auto_coords=False, d_km=self.AB_DIST + 2, show_progress=False)
        # add values that previously broke writers
        G.nodes["ABO"]["misc_none"] = None
        G.nodes["ABO"]["misc_list"] = [1, 2, 3]
        G.nodes["ZUR"]["misc_dict"] = {"a": 1}

        out1 = ms.save_graph(G, self.td / "stations_compact.gexf", compact=True)
        self.assertTrue(out1.exists() and out1.suffix.lower() == ".gexf")

        out2 = ms.save_graph(G, self.td / "stations_full.graphml", compact=False)
        self.assertTrue(out2.exists() and out2.suffix.lower() == ".graphml")

    # --------- Alternate coord columns (robust schema inference) ---------

    def test_infer_coords_from_alt_column_names_isolated(self):
        # build a separate root to avoid contaminating other tests
        root2 = self.td / "alt_root"
        (root2 / "2001").mkdir(parents=True)
        pd.DataFrame(
            {
                "time_utc": pd.date_range("2001-06-01", periods=1, freq="h", tz="UTC"),
                "latitude": [45.75],
                "longitude": [7.25],
                "elevation_m": [777],
                "other": [0.1],
            }
        ).to_parquet(root2 / "2001" / "ALT.parquet", engine=ms.ENGINE, index=False)

        G = ms.get_station_graph(str(root2), years=[2001], auto_coords=False, d_km=None, show_progress=False)
        self.assertEqual(G.number_of_nodes(), 1)
        self.assertIn("ALT", G)
        self.assertAlmostEqual(G.nodes["ALT"]["lat"], 45.75, places=6)
        self.assertAlmostEqual(G.nodes["ALT"]["lon"], 7.25, places=6)
        self.assertAlmostEqual(G.nodes["ALT"]["elev_m"], 777.0, places=6)

    # ------------------------- Error / edge cases -------------------------

    def test_missing_root_and_empty_dir(self):
        with self.assertRaises(FileNotFoundError):
            ms.get_station_graph(str(self.td / "no_such_dir"), auto_coords=False, show_progress=False)
        empty = self.td / "empty_root"
        empty.mkdir(exist_ok=True)
        with self.assertRaises(SystemExit):
            ms.get_station_graph(str(empty), auto_coords=False, show_progress=False)

    def test_single_station_single_year(self):
        single_root = self.td / "one"
        (single_root / "1999").mkdir(parents=True)
        pd.DataFrame(
            {
                "time_utc": pd.date_range("1999-01-01", periods=2, freq="h", tz="UTC"),
                "lat": [46.1, 46.1],
                "lon": [7.1, 7.1],
            }
        ).to_parquet(single_root / "1999" / "ONE.parquet", engine=ms.ENGINE, index=False)
        G = ms.get_station_graph(str(single_root), auto_coords=False, d_km=100.0, show_progress=False)
        self.assertEqual(G.number_of_nodes(), 1)
        self.assertEqual(G.number_of_edges(), 0)
        out = ms.save_graph(G, self.td / "one_st.gexf", compact=True)
        self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()

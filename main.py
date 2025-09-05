#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import datetime
import argparse
from pathlib import Path

import pyfiglet

from src.utils.switzerland import Switzerland
from src.utils.nx_to_pyG import nx_to_pyg
from src.utils.meteo_stations import (
    get_station_graph,   # builds NetworkX graph by scanning parquet directories ONLY
    save_graph,          # saves the station graph if requested
)


def banner():
    print(pyfiglet.figlet_format("MyMeteo", font="slant"))
    print(f"System: {platform.system()}")
    print(f"Release: {platform.release()}")
    print(f"Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python running on: {os.name}")
    print(f"Working directory: {os.getcwd()}")
    print("Launched at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 30, "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MyMeteo: Weather predictions from Switzerland, trying to beat MeteoSwiss (yeah, sure...)"
    )
    parser.add_argument("-v", "--version", action="version", version="MyMeteo")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")

    # Project data root
    parser.add_argument(
        "-d", "--data", type=str, default="data",
        help="Path to project data directory (default: ./data)",
    )

    # Switzerland administrative graph level
    parser.add_argument(
        "-l", "--level", type=str, default="district",
        choices=["municipal", "district", "cantonal", "plz"],
        help="Geographical level for the Switzerland graph",
    )

    # Station graph inputs — ONLY parquet dir is needed now
    parser.add_argument(
        "-pr", "--parquet-root", type=str, default=None,
        help="Root of MeteoSwiss parquet dataset (structure: <root>/<YEAR>/<STATION>.parquet). "
             "If not provided, defaults to <data>/MeteoSwiss/train",
    )
    parser.add_argument(
        "--years", type=str, default="",
        help="Comma-separated list of years to consider (e.g., 2005,2006). If empty, all <YEAR> subfolders are used.",
    )
    parser.add_argument(
        "--station-d-km", type=float, default=None,
        help="If set, connect stations within this distance (km). Requires lat/lon present in parquet.",
    )
    parser.add_argument(
        "--drop-station-cols", type=str, default="",
        help="Comma-separated list of columns to drop from each parquet before storing in node attributes.",
    )
    parser.add_argument(
        "--save-station-graph", type=str, default="",
        help="Optional path to save the station graph (.gexf/.graphml/.gpickle/.edgelist). If empty, not saved.",
    )

    # PyG conversion knobs
    parser.add_argument(
        "--sigma-m", type=float, default=25_000.0,
        help="Gaussian sigma in meters for inverse-distance weighting (nx_to_pyg)",
    )
    parser.add_argument(
        "--normalize-x", action="store_true", default=True,
        help="Normalize node features when converting to PyG (default: True)",
    )
    parser.add_argument(
        "--no-normalize-x", dest="normalize_x", action="store_false",
        help="Disable normalization of node features for PyG",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    banner()

    # ---------------------------------------------------------------------
    # Switzerland administrative graph
    # ---------------------------------------------------------------------
    print(f"[INFO]  Predictions at '{args.level}' level\n")

    switzerland = Switzerland()
    g_swiss = switzerland.get_graph(level=args.level)

    data_swiss = nx_to_pyg(
        g_swiss,
        node_feature_keys=("lon", "lat", "area"),
        edge_weight_mode="inv_dist",
        gauss_sigma_m=float(args.sigma_m),
        normalize_x=bool(args.normalize_x),
    )

    print("[SWISS-GRAPH]  Node feature matrix:", data_swiss.x.shape)         # [N, F]
    print("[SWISS-GRAPH]  Edge index matrix:", data_swiss.edge_index.shape)  # [2, E]
    print("[SWISS-GRAPH]  Edge weights:", data_swiss.edge_weight.shape)      # [E]
    print("[SWISS-GRAPH]  Positions:", data_swiss.pos.shape, "\n")           # [N, 2]

    # ---------------------------------------------------------------------
    # MeteoSwiss station graph (built ONLY from parquet directory)
    # ---------------------------------------------------------------------
    parquet_root = Path(args.parquet_root) if args.parquet_root else (Path(args.data) / "MeteoSwiss" / "train")
    years = [y.strip() for y in args.years.split(",") if y.strip()] if args.years else None
    drop_cols = [c.strip() for c in args.drop_station_cols.split(",") if c.strip()]

    print(f"[STATIONS]     Scanning parquet dir: {parquet_root}")
    G_st = get_station_graph(
        parquet_root=parquet_root,
        years=years,
        d_km=(float(args.station_d_km) if args.station_d_km is not None else None),
        drop_columns=drop_cols or None,
    )

    # Basic stats
    print(f"[STATION-GRAPH] Nodes: {G_st.number_of_nodes()} | "
          f"Edges: {G_st.number_of_edges()} "
          f"{'(distance <= %.1f km)' % args.station_d_km if args.station_d_km else '(no distance threshold)'}")

    # Save station graph if requested
    if args.save_station_graph:
        out_path = save_graph(G_st, Path(args.save_station_graph))
        print(f"[STATION-GRAPH] Saved to: {out_path}")

    # Print a small summary of the aggregated info per node
    # (first 3 nodes: abbreviation, years covered, total rows across years)
    if args.verbose:
        import itertools as it
        print("[STATION-GRAPH] Sample node summaries:")
        for node, attrs in it.islice(G_st.nodes(data=True), 3):
            years_cov = sorted(map(int, attrs.get("years", {}).keys()))
            total_rows = sum(yinfo.get("n_rows", 0) for yinfo in attrs.get("years", {}).values())
            print(f"  - {node}: years={years_cov[:6]}{'...' if len(years_cov) > 6 else ''} "
                  f"rows_total={total_rows} "
                  f"coords=({attrs.get('lat')}, {attrs.get('lon')})")

    # ---------------------------------------------------------------------
    # Convert station graph to PyG (only if lon/lat exist on ALL nodes)
    # ---------------------------------------------------------------------
    # Check coordinate availability
    all_have_coords = all(
        ("lat" in G_st.nodes[n] and "lon" in G_st.nodes[n]
         and G_st.nodes[n]["lat"] is not None and G_st.nodes[n]["lon"] is not None)
        for n in G_st.nodes
    )

    if all_have_coords:
        node_feats = ("lon", "lat")  # elev_m is optional and often missing in parquet
        data_st = nx_to_pyg(
            G_st,
            node_feature_keys=node_feats,
            edge_weight_mode="inv_dist",
            gauss_sigma_m=float(args.sigma_m),
            normalize_x=bool(args.normalize_x),
        )
        print("[STATION-GRAPH] Node feature matrix:", data_st.x.shape)           # [N, F]
        print("[STATION-GRAPH] Edge index matrix:", data_st.edge_index.shape)    # [2, E]
        print("[STATION-GRAPH] Edge weights:", data_st.edge_weight.shape)        # [E]
        print("[STATION-GRAPH] Positions:", data_st.pos.shape, "\n")             # [N, 2]
    else:
        print("[STATION-GRAPH] Skipping PyG conversion (lat/lon not available for all stations).")
        print("                Provide coordinates in the parquet columns (e.g., 'lat'/'lon') "
              "or drop nodes without coords before conversion.\n")
        
    switzerland.draw_stations_to_district_centroids(
        station_graph=G_st,
        radius_km=args.station_d_km,
        output_path=f"stations_to_district_centroids_{int(args.station_d_km)}km.png",
        dpi=240,
    )


if __name__ == "__main__":
    main()

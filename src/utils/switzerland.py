#!/usr/bin/env python3
# Retrieve different types of Swiss geographical data from data/Switzerland 
# For the moment only geojson no topological data
#
# Summary of join logic between Swiss administrative/spatial layers:
#
# Municipalities ↔ Districts  →  BEZIRKSNUM
# Municipalities ↔ Cantons    →  KANTONSNUM
# Municipalities ↔ Postal codes → via spatial join (geometry intersection), not a shared ID
# Municipalities ↔ Lakes      →  via spatial join (geometry intersection)
#
# Switzerland object needs to have switzerland accessible at all levels, binding canton, districts, municipalities
# For the moment I want predictions only at district level

import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point
from shapely.ops import unary_union
from math import radians, sin, cos, asin, sqrt


import sys

# Swiss cantons with official BFS KANTONSNUM codes
CANTONS = {
    "AG": 1,   # Aargau
    "AI": 2,   # Appenzell Innerrhoden
    "AR": 3,   # Appenzell Ausserrhoden
    "BE": 4,   # Bern
    "BL": 5,   # Basel-Landschaft
    "BS": 6,   # Basel-Stadt
    "FR": 7,   # Fribourg
    "GE": 8,   # Genève
    "GL": 9,   # Glarus
    "GR": 10,  # Graubünden
    "JU": 11,  # Jura
    "LU": 12,  # Luzern
    "NE": 13,  # Neuchâtel
    "NW": 14,  # Nidwalden
    "OW": 15,  # Obwalden
    "SG": 16,  # St. Gallen
    "SH": 17,  # Schaffhausen
    "SO": 18,  # Solothurn
    "SZ": 19,  # Schwyz
    "TG": 20,  # Thurgau
    "TI": 21,  # Ticino
    "UR": 22,  # Uri
    "VD": 23,  # Vaud
    "VS": 24,  # Valais / Wallis
    "ZG": 25,  # Zug
    "ZH": 26,  # Zürich
}

def _ensure_crs(gdf: gpd.GeoDataFrame, default_epsg: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Set a default CRS if missing (GeoJSON is usually WGS84/EPSG:4326)."""
    if gdf.crs is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf = gdf.set_crs(default_epsg, allow_override=True)
    return gdf

def _pick_col(gdf: gpd.GeoDataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    """Pick the first existing column (case-insensitive) from candidates; None if not required."""
    low = {c.lower(): c for c in gdf.columns}
    for c in candidates:
        if c in gdf.columns:
            return c
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise KeyError(f"None of the expected columns found. Tried: {candidates}. Available: {list(gdf.columns)}")
    return None

def _valid_poly(g: BaseGeometry) -> BaseGeometry:
    if g is None:
        return g
    if not g.is_valid:
        return g.buffer(0)
    return g

def _rep_point_xy(g: BaseGeometry) -> Tuple[float, float]:
    # representative_point lies inside; centroid might lie outside for concave shapes
    p = g.representative_point()
    return float(p.x), float(p.y)

class Switzerland:
    def __init__(self, geojson_dir_path: str = "data/Switzerland/geo"):
        self.geojson_dir_path = Path(geojson_dir_path)

        self.geo_switzerland = gpd.read_file(self.geojson_dir_path / "ch.geojson")
        self.geo_districts = gpd.read_file(self.geojson_dir_path / "ch-districts.geojson")
        self.geo_municipalities = gpd.read_file(self.geojson_dir_path / "ch-municipalities.geojson")
        self.geo_lakes = gpd.read_file(self.geojson_dir_path / "ch-lakes.geojson")

        # Optional, if you have a municipalities file with richer attributes
        self.municipalities = gpd.read_file(self.geojson_dir_path / "municipalities.geojson")

        # Ensure all have a CRS
        self.geo_switzerland = _ensure_crs(self.geo_switzerland)
        self.geo_districts = _ensure_crs(self.geo_districts)
        self.geo_municipalities = _ensure_crs(self.geo_municipalities)
        self.geo_lakes = _ensure_crs(self.geo_lakes)
        self.municipalities = _ensure_crs(self.municipalities)

        # Align all to the Switzerland (canton) layer's CRS
        target_crs = self.geo_switzerland.crs
        if self.geo_districts.crs != target_crs:
            self.geo_districts = self.geo_districts.to_crs(target_crs) 
        if self.geo_municipalities.crs != target_crs:
            self.geo_municipalities = self.geo_municipalities.to_crs(target_crs)
        if self.geo_lakes.crs != target_crs:
            self.geo_lakes = self.geo_lakes.to_crs(target_crs) 
        if self.municipalities.crs != target_crs:
            self.municipalities = self.municipalities.to_crs(target_crs) 

        # Fix invalid geometries to avoid plotting errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.geo_districts["geometry"] = self.geo_districts.buffer(0)
            self.geo_switzerland["geometry"] = self.geo_switzerland.buffer(0)
            self.geo_lakes["geometry"] = self.geo_lakes.buffer(0)

        # If districts layer has no attributes (only 'geometry'), synthesize from municipalities
        self._ensure_district_attributes()

        # cache
        self._graph: Optional[nx.Graph] = None

    def _ensure_district_attributes(self):
        """Ensure district GeoDataFrame has an ID column; if not, build from municipalities."""
        gdf = self.geo_districts
        # If only geometry present, try to rebuild by dissolving municipalities by BEZIRKSNUM
        if list(gdf.columns) == ["geometry"] or _pick_col(gdf, ("BEZIRKSNUM","BEZIRK_ID","DISTRICT_ID","ID"), required=False) is None:
            muni = self.geo_municipalities.copy()
            dist_fk = _pick_col(muni, ("BEZIRKSNUM","BEZIRK_NUM","BEZIRK_ID","DISTRICT_ID"), required=False)
            if dist_fk is not None:
                # dissolve to districts, keep id
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    muni["geometry"] = muni.geometry.apply(_valid_poly) 
                dis = muni[[dist_fk, "geometry"]].dissolve(by=dist_fk, as_index=False, aggfunc="first")
                dis = dis.rename(columns={dist_fk: "BEZIRKSNUM"})
                # Add a placeholder name column
                dis["NAME"] = dis["BEZIRKSNUM"].astype(str)
                self.geo_districts = dis.set_crs(muni.crs, allow_override=True) # type: ignore
            else:
                # fallback: create sequential IDs
                tmp = gdf.copy().reset_index(drop=False).rename(columns={"index":"DIST_ID"})
                tmp["NAME"] = tmp["DIST_ID"].astype(str)
                self.geo_districts = tmp

    def get_cantons(self):
        """Retrieve Swiss cantons geojson data."""
        print(len(self.geo_switzerland))

    def get_districts(self):
        """Retrieve Swiss districts geojson data."""
        print(len(self.geo_districts))

    def get_municipalities(self):
        """Retrieve Swiss municipalities geojson data."""
        uniques = self.municipalities["BFS_NUMMER"].unique()
        print(len(uniques))

    def draw_districts_map(self, output_path: str = "swiss_districts.png", dpi: int = 220):
        """
        Draw Switzerland with all districts. Saves to output_path.
        - District polygons filled lightly with black edges
        - Canton boundaries overlaid thicker
        - Lakes overlaid (optional aesthetic)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect("equal", adjustable="box")
        ax.set_axisbelow(True)
        ax.grid(True, linewidth=0.3)

        # Districts
        self.geo_districts.plot(ax=ax, facecolor="#d9e3f0", edgecolor="black", linewidth=0.4)

        # Canton boundaries (overlay only borders)
        self.geo_switzerland.boundary.plot(ax=ax, color="black", linewidth=1.0)

        # Lakes (optional, if present)
        if not self.geo_lakes.empty:
            self.geo_lakes.plot(ax=ax, facecolor="#a7c7ff", edgecolor="#5f8dd3", linewidth=0.3, alpha=0.9)

        ax.set_title("Switzerland — Districts")
        ax.set_xlabel("Longitude / X")
        ax.set_ylabel("Latitude / Y")

        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        print(f"Saved districts map to: {output_path}")

    # ===================== Graph API =====================

    def get_graph(
        self,
        level: str = "district",  # NEW: 'cantonal' | 'district' | 'municipality'
        rook_adjacency: bool = True,
        projected_epsg_for_metrics: Optional[str] = "EPSG:2056",  # Swiss LV95 (meters)
        id_col_candidates=("BEZIRKSNUM", "BEZIRK_ID", "DISTRICT_ID", "ID", "DIST_ID"),
        name_col_candidates=("NAME", "BEZIRKSNAME", "DISTRICT", "NAME_DE", "Name"),
        rebuild: bool = False,
    ) -> nx.Graph:
        """
        Build (or return cached) NetworkX graph over an administrative level.

        Parameters
        ----------
        level : {'cantonal','district','municipality'}
            Graph granularity. For now, only 'district' is implemented.
            If 'cantonal' or 'municipality' is requested, prints "work in progress" and exits.
        rook_adjacency : bool
            rook=True → share a boundary segment; rook=False → queen contiguity (any touching).
        """
        lvl = str(level).strip().lower()
        allowed = {"cantonal", "district", "municipality"}
        if lvl not in allowed:
            raise ValueError(f"level must be one of {allowed}, got: {level}")

        if lvl != "district":
            print(f"[info] {lvl} graph: work in progress")
            sys.exit(0)

        if (self._graph is not None) and (not rebuild):
            return self._graph

        gdf = self.geo_districts.copy()
        gdf = _ensure_crs(gdf)

        # choose id and name; if missing, synthesize
        id_col = _pick_col(gdf, id_col_candidates, required=False)
        if id_col is None:
            gdf = gdf.reset_index(drop=False).rename(columns={"index": "DIST_ID"})
            id_col = "DIST_ID"

        name_col = _pick_col(gdf, name_col_candidates, required=False)
        if name_col is None:
            gdf["__NAME__"] = gdf[id_col].astype(str)
            name_col = "__NAME__"

        # clean geometries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf["geometry"] = gdf.geometry.apply(_valid_poly)

        # metric projection for robust area/lengths
        gdf_m = gdf.to_crs(projected_epsg_for_metrics) if projected_epsg_for_metrics else gdf
        # geographical projection (lon/lat WGS84)
        gdf_geo = gdf.to_crs("EPSG:4326")

        G = nx.Graph()
        for (_, row_m), (_, row_geo) in zip(gdf_m.iterrows(), gdf_geo.iterrows()):
            geom_m = row_m.geometry
            geom_geo = row_geo.geometry
            if geom_m is None or geom_m.is_empty:
                continue
            x, y = _rep_point_xy(geom_m)       # metric coords
            lon, lat = _rep_point_xy(geom_geo) # lon/lat

            G.add_node(
                row_m[id_col],
                name=str(row_m.get(name_col, row_m[id_col])),
                x=x,
                y=y,
                lon=lon,
                lat=lat,
                area=float(geom_m.area),
                geometry=geom_m,
            )

        # spatial index neighbor search
        sindex = gdf_m.sindex
        geoms = gdf_m.geometry.values
        ids = gdf_m[id_col].values

        def _candidates(i: int):
            try:
                return list(sindex.query(geoms[i], predicate="intersects"))
            except Exception:
                return list(sindex.intersection(geoms[i].bounds))

        for i in range(len(geoms)):
            gi = geoms[i]
            if gi is None or gi.is_empty:
                continue
            for j in _candidates(i):
                if j <= i:
                    continue
                gj = geoms[j]
                if gj is None or gj.is_empty:
                    continue
                if not gi.intersects(gj):
                    continue

                if rook_adjacency:
                    inter = gi.boundary.intersection(gj.boundary)
                    if inter.is_empty or inter.length <= 0:
                        continue
                else:
                    if not (gi.touches(gj) or gi.overlaps(gj) or gi.intersects(gj)):
                        continue

                G.add_edge(ids[i], ids[j])

        G.graph["crs_nodes"] = str(gdf_m.crs)
        G.graph["id_col"] = id_col
        G.graph["name_col"] = name_col
        G.graph["rook"] = rook_adjacency
        G.graph["level"] = lvl

        self._graph = G
        return G


    def draw_graph(
        self,
        output_path: str = "swiss_districts_graph.pdf",  # vector default for publications
        dpi: int = 300,
        node_size: float = 10.0,
        edge_width: float = 0.5,
        face_alpha: float = 0.85,
        lake_alpha: float = 0.9,
    ):
        """
        Draw districts (projected to EPSG:4326) and overlay the district adjacency graph
        using node geographical coordinates (lon/lat). Styled for LaTeX-ready figures.

        Notes:
        - Saves vector formats (PDF/SVG) cleanly. For PNG, dpi is used.
        - Requires get_graph() to have been run (auto-runs if needed).
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        G = self.get_graph()  # build/cached, ensures 'lon','lat' in node attrs

        # ---------- Publication (LaTeX-like) style ----------
        # If you have a LaTeX installation, uncomment the next two lines to use full TeX rendering.
        # mpl.rcParams["text.usetex"] = True
        # mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb}"
        mpl.rcParams.update({
            "font.family": "serif",
            "font.size": 9.5,                # adjust to your journal template
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.3,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        })

        # ---------- Prepare layers in EPSG:4326 ----------
        gdf = self.geo_districts.to_crs("EPSG:4326")
        lakes = self.geo_lakes.to_crs("EPSG:4326") if not self.geo_lakes.empty else self.geo_lakes
        # light geometry fix (already done at init, but keep safe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf["geometry"] = gdf.buffer(0)
            if not lakes.empty:
                lakes["geometry"] = lakes.buffer(0)

        # ---------- Node positions from lon/lat ----------
        pos = {n: (d["lon"], d["lat"]) for n, d in G.nodes(data=True)}

        # ---------- Figure ----------
        fig_w, fig_h = 6.0, 4.5  # inches; tweak for your column width
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

        # Base polygons (muted fill, hairline borders)
        gdf.plot(ax=ax, facecolor="#e6eef6", edgecolor="#4f5b66", linewidth=0.35, alpha=face_alpha)

        # Lakes (optional)
        if not lakes.empty:
            lakes.plot(ax=ax, facecolor="#b7cbe9", edgecolor="#7a8da6", linewidth=0.25, alpha=lake_alpha)

        # Edges
        # Draw lines between node lon/lat pairs
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], linewidth=edge_width)

        # Nodes
        xs = [d["lon"] for _, d in G.nodes(data=True)]
        ys = [d["lat"] for _, d in G.nodes(data=True)]
        ax.scatter(xs, ys, s=node_size)

        # Axes cosmetics for publication
        ax.set_title("District adjacency graph - Switzerland")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Tight layout and save
        fig.tight_layout()
        ext = Path(output_path).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".tif"}:
            fig.savefig(output_path, dpi=dpi)
        else:
            fig.savefig(output_path)  # vector formats (pdf/svg) ignore dpi
        plt.close(fig)
        print(f"Saved districts graph map to: {output_path}")

    def draw_all_stations_districts(
            self,
            station_graph,
            radius_km: float,
            output_path: str = "stations_vs_districts.png",
            dpi: int = 240,
            face_alpha: float = 0.95,
            lake_alpha: float = 0.9,
        ):
            """
            Plot all stations (red) and shade in white every district whose polygon
            is within `radius_km` of at least one station (geodesic radius handled
            via metric buffering in EPSG:2056).

            Parameters
            ----------
            station_graph : networkx.Graph with node attrs lat, lon (WGS84)
            radius_km     : radius in kilometers
            output_path   : file to save (png/pdf/svg, etc.)
            """
            import geopandas as gpd
            import matplotlib.pyplot as plt
            import warnings

            # --- Collect station points with coordinates ---
            pts_rec = [
                (n, float(d["lon"]), float(d["lat"]))
                for n, d in station_graph.nodes(data=True)
                if d.get("lon") is not None and d.get("lat") is not None
            ]
            if not pts_rec:
                raise ValueError("No stations with lat/lon found on the station_graph.")

            # GeoDataFrames in WGS84 for plotting and LV95 for metric buffers
            gdf_dist_wgs = self.geo_districts.to_crs("EPSG:4326").copy()
            gdf_dist_lv95 = self.geo_districts.to_crs("EPSG:2056").copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gdf_dist_lv95["geometry"] = gdf_dist_lv95.buffer(0)  # validity fix

            # Determine a district ID column
            id_col_candidates = ("BEZIRKSNUM", "BEZIRK_ID", "DISTRICT_ID", "ID", "DIST_ID")
            id_col = next((c for c in id_col_candidates if c in gdf_dist_wgs.columns), None)
            if id_col is None:
                gdf_dist_wgs = gdf_dist_wgs.reset_index(drop=False).rename(columns={"index": "DIST_ID"})
                gdf_dist_lv95 = gdf_dist_lv95.reset_index(drop=False).rename(columns={"index": "DIST_ID"})
                id_col = "DIST_ID"

            # Build station points
            gdf_pts_wgs = gpd.GeoDataFrame(
                {"station_abbr": [r[0] for r in pts_rec]},
                geometry=[Point(r[1], r[2]) for r in pts_rec],  # lon, lat
                crs="EPSG:4326",
            )
            gdf_pts_lv95 = gdf_pts_wgs.to_crs("EPSG:2056")

            # --- Buffer all stations (metric, meters) and union them ---
            radius_m = float(radius_km) * 1000.0
            buffers = gdf_pts_lv95.buffer(radius_m)
            buf_union = unary_union(list(buffers.values))  # shapely geometry (Multi/Poly)

            # --- Select districts intersecting any buffer ---
            if buf_union.is_empty:
                target_ids = set()
            else:
                # fast boolean mask via vectorized intersects
                mask = gdf_dist_lv95.geometry.intersects(buf_union)
                target_ids = set(gdf_dist_lv95.loc[mask, id_col].tolist())

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.3, alpha=0.25)

            # Base districts (light grey)
            gdf_dist_wgs.plot(ax=ax, facecolor="#d0d5db", edgecolor="#7b8794", linewidth=0.4, alpha=face_alpha)

            # Lakes (optional)
            try:
                lakes = self.geo_lakes.to_crs("EPSG:4326")
                if not lakes.empty:
                    lakes.plot(ax=ax, facecolor="#a7c7ff", edgecolor="#5f8dd3", linewidth=0.3, alpha=lake_alpha)
            except Exception:
                pass

            # Highlight target districts in white with darker edges
            if target_ids:
                sel = gdf_dist_wgs[gdf_dist_wgs[id_col].isin(list(target_ids))]
                sel.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=0.9, alpha=1.0)

            # Plot all stations in red
            ax.scatter(gdf_pts_wgs.geometry.x, gdf_pts_wgs.geometry.y, s=16, c="red", zorder=5)

            ax.set_title(f"Districts within {radius_km:.0f} km of any MeteoSwiss station")
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")

            fig.tight_layout()
            # save (vector formats ignore dpi)
            if str(output_path).lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                fig.savefig(output_path, dpi=dpi)
            else:
                fig.savefig(output_path)
            plt.close(fig)

            # Small report
            print(f"[PLOT] Stations plotted: {len(pts_rec)}")
            print(f"[PLOT] Districts highlighted: {len(target_ids)}")
            print(f"[PLOT] Saved: {output_path}")

    def draw_stations_to_district_centroids(
        self,
        station_graph,
        radius_km: float,
        output_path: str = "stations_to_district_centroids.png",
        dpi: int = 240,
        station_size: float = 22.0,
        centroid_size: float = 28.0,
        edge_width: float = 0.5,
    ):
        """
        Plot ALL stations (red) that appear in your parquet-derived station_graph,
        plot ALL district centroids as white points with black outline, and draw
        edges from each station to every district centroid within `radius_km`.

        Parameters
        ----------
        station_graph : networkx.Graph
            Nodes must include 'lat' and 'lon' in WGS84 (EPSG:4326) to be plotted.
        radius_km : float
            Great-circle distance threshold to connect a station to a district centroid.
        output_path : str
            Output figure path.
        """
        import warnings
        import numpy as np
        import geopandas as gpd
        import matplotlib.pyplot as plt

        # ---------- Collect stations with coordinates ----------
        sta_records = [
            {"station_abbr": n, "lon": float(d["lon"]), "lat": float(d["lat"])}
            for n, d in station_graph.nodes(data=True)
            if d.get("lon") is not None and d.get("lat") is not None
        ]
        if not sta_records:
            raise ValueError("Station graph has no nodes with lat/lon. Ensure auto_coords=True when building it.")

        gdf_sta = gpd.GeoDataFrame(
            sta_records,
            geometry=[Point(r["lon"], r["lat"]) for r in sta_records],
            crs="EPSG:4326",
        )

        # ---------- District centroids (true geometric centroid in metric CRS, then back to WGS84) ----------
        dist_wgs = self.geo_districts.to_crs("EPSG:4326").copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dist_m = self.geo_districts.to_crs("EPSG:2056").copy()
            dist_m["geometry"] = dist_m.buffer(0)

        # Determine district ID column
        id_col = None
        for c in ("BEZIRKSNUM", "BEZIRK_ID", "DISTRICT_ID", "ID", "DIST_ID"):
            if c in dist_wgs.columns:
                id_col = c
                break
        if id_col is None:
            dist_wgs = dist_wgs.reset_index(drop=False).rename(columns={"index": "DIST_ID"})
            dist_m   = dist_m.reset_index(drop=False).rename(columns={"index": "DIST_ID"})
            id_col = "DIST_ID"

        # Compute centroids in metric CRS, then transform to WGS84 for plotting and distance calc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cent_m = dist_m.geometry.centroid
        cent_wgs = gpd.GeoSeries(cent_m, crs=dist_m.crs).to_crs("EPSG:4326")
        gdf_cent = gpd.GeoDataFrame(
            {
                id_col: dist_wgs[id_col].values,
                "lon": cent_wgs.x.values,
                "lat": cent_wgs.y.values,
            },
            geometry=cent_wgs,
            crs="EPSG:4326",
        )

        # ---------- Vectorized haversine distances (station ↔ centroid) ----------
        R = 6371.0088  # km

        def haversine_pairwise(lat1, lon1, lat2, lon2):
            """
            lat1, lon1: shape (N, 1)
            lat2, lon2: shape (1, M)
            returns: (N, M) distances in km
            """
            lat1r = np.radians(lat1); lon1r = np.radians(lon1)
            lat2r = np.radians(lat2); lon2r = np.radians(lon2)
            dlat = lat2r - lat1r
            dlon = lon2r - lon1r
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
            return 2.0 * R * np.arcsin(np.sqrt(a))

        sta_lat = gdf_sta["lat"].to_numpy()[:, None]  # (N,1)
        sta_lon = gdf_sta["lon"].to_numpy()[:, None]  # (N,1)
        cen_lat = gdf_cent["lat"].to_numpy()[None, :] # (1,M)
        cen_lon = gdf_cent["lon"].to_numpy()[None, :] # (1,M)

        D = haversine_pairwise(sta_lat, sta_lon, cen_lat, cen_lon)  # (N, M)
        within = D <= float(radius_km)

        # Build edge list for reporting and plotting
        edges = []
        for i_sta in range(within.shape[0]):
            if not within[i_sta].any():
                continue
            sta_id = gdf_sta["station_abbr"].iloc[i_sta]
            for j_c in np.where(within[i_sta])[0]:
                dist_km = float(D[i_sta, j_c])
                dist_id = gdf_cent[id_col].iloc[j_c]
                edges.append((sta_id, dist_id, dist_km))

        # ---------- Plot ----------
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.25)

        # Base districts (very light background for context)
        dist_wgs.plot(ax=ax, facecolor="#e6eef6", edgecolor="#4f5b66", linewidth=0.35, alpha=0.95)

        # District centroids: white with black outline
        ax.scatter(
            gdf_cent["lon"].to_numpy(),
            gdf_cent["lat"].to_numpy(),
            s=centroid_size,
            facecolors="white",
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
        )

        # Stations: red points
        ax.scatter(
            gdf_sta["lon"].to_numpy(),
            gdf_sta["lat"].to_numpy(),
            s=station_size,
            c="red",
            zorder=5,
        )

        # Edges from station → centroid when within radius
        for sta_id, dist_id, dist_km in edges:
            srow = gdf_sta.loc[gdf_sta["station_abbr"] == sta_id].iloc[0]
            crow = gdf_cent.loc[gdf_cent[id_col] == dist_id].iloc[0]
            ax.plot(
                [srow["lon"], crow["lon"]],
                [srow["lat"], crow["lat"]],
                linewidth=edge_width,
                color="#444444",
                alpha=0.8,
                zorder=3,
            )

        ax.set_title(f"Stations (red) and District Centroids (white/black) — edges ≤ {radius_km:.0f} km")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

        fig.tight_layout()
        if str(output_path).lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            fig.savefig(output_path, dpi=dpi)
        else:
            fig.savefig(output_path)
        plt.close(fig)

        # ---------- Report ----------
        print(f"[PLOT] Stations plotted: {len(gdf_sta)}")
        print(f"[PLOT] District centroids plotted: {len(gdf_cent)}")
        print(f"[PLOT] Station→District edges (≤ {radius_km:.1f} km): {len(edges)}")
        # Print edges (truncate if huge)
        # max_show = 500
        # if len(edges) <= max_show:
        #     for sta_id, dist_id, dist_km in edges:
        #         print(f"   {sta_id} -> {dist_id} ({dist_km:.1f} km)")
        # else:
        #     for sta_id, dist_id, dist_km in edges[:max_show]:
        #         print(f"   {sta_id} -> {dist_id} ({dist_km:.1f} km)")
        #     print(f"   ... ({len(edges)-max_show} more)")
        print(f"[PLOT] Saved: {output_path}")




if __name__ == "__main__":
    switzerland = Switzerland(geojson_dir_path="/Users/michaeljopiti/MyMeteo/data/Switzerland/geo")
    switzerland.get_districts()
    switzerland.draw_districts_map(output_path="swiss_districts.png", dpi=220)
    # Build and draw the graph overlay
    # Example: pass level explicitly (only 'district' implemented; others will print WIP and exit)
    switzerland.get_graph(level="district", rook_adjacency=True)  # or False for queen contiguity
    switzerland.draw_graph(output_path="swiss_districts_graph.png", dpi=220)

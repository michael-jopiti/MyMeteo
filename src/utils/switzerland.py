#!/usr/bin/env python3
# Retrieve different types of Swiss geographical data from src/data/Switzerland 
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
    def __init__(self, geojson_dir_path: str = "src/data/Switzerland/geo"):
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
                self.geo_districts = dis.set_crs(muni.crs, allow_override=True)
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
        rook_adjacency: bool = True,
        projected_epsg_for_metrics: Optional[str] = "EPSG:2056",  # Swiss LV95 (meters)
        id_col_candidates=("BEZIRKSNUM", "BEZIRK_ID", "DISTRICT_ID", "ID", "DIST_ID"),
        name_col_candidates=("NAME", "BEZIRKSNAME", "DISTRICT", "NAME_DE", "Name"),
        rebuild: bool = False,
    ) -> nx.Graph:
        """
        Build (or return cached) NetworkX graph over districts.
        Nodes: districts at representative point (inside polygon).
          attributes:
            - name: district name
            - x, y: coords in metric CRS (e.g. LV95 EPSG:2056)
            - lon, lat: geographical coords (EPSG:4326)
            - area: polygon area in metric CRS
            - geometry: projected geometry
        Edges: neighboring districts.
          rook_adjacency=True → share a boundary segment (non-zero length).
          rook_adjacency=False → queen contiguity (any touching).
        """
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



if __name__ == "__main__":
    switzerland = Switzerland(geojson_dir_path="/Users/michaeljopiti/MyMeteo/src/data/Switzerland/geo")
    switzerland.get_districts()
    switzerland.draw_districts_map(output_path="swiss_districts.png", dpi=220)
    # Build and draw the graph overlay
    switzerland.get_graph(rook_adjacency=True)  # or False for queen contiguity
    switzerland.draw_graph(output_path="swiss_districts_graph.png", dpi=220)

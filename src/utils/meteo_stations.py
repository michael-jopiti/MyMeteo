#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/utils/meteostations.py

Build a NetworkX graph from a directory of MeteoSwiss parquet files.
Now with: automatic download & MERGE of station coordinates from MeteoSwiss Open Data
(SMN -> OBS -> SMN-precip priority), plus fallback geocoding of station_name
via swisstopo SearchServer to fill missing lat/lon, and injection of
(lat, lon, elev, station_name) into nodes.

Folder layout (required)
------------------------
<parquet_root>/<YEAR>/<STATION_ABBR>.parquet
e.g.
data/MeteoSwiss/train/2005/ABO.parquet
data/MeteoSwiss/train/2024/ABO.parquet
...

Node payload
------------
{
  "station_abbr": "ABO",
  "years": {...},               # per-year parquet snapshot (n_rows, columns, records, source_path)
  "columns_union": [...],
  "first_year": 2005,
  "last_year":  2024,
  "lat": <float | None>,
  "lon": <float | None>,
  "elev_m": <float | None>,
  "station_name": <str | None>
}

Edges (optional)
----------------
If `d_km` is provided AND lat/lon exist, connect stations <= d_km (great-circle),
edge attribute: distance_km.

Dependencies
------------
- pandas (required)
- numpy & networkx (required if d_km is provided)
- tqdm (optional) for progress bar
- requests (optional) for HTTP (falls back to urllib)
- pyproj (optional) to convert LV95 -> WGS84 if needed

Example
-------
from src.utils.meteostations import get_station_graph, save_graph

G = get_station_graph(
    parquet_root="data/MeteoSwiss/train",
    d_km=50.0,
    auto_coords=True,
    coords_cache_dir="data/cache",
    geocode_missing=True,  # use swisstopo to fill missing lat/lon from station_name
)

print(G.number_of_nodes(), G.number_of_edges())
save_graph(G, "data/stations_all_years.gexf")
"""

from __future__ import annotations

import sys
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, List, Set, Tuple
from urllib.parse import quote_plus

# --- pandas (required) ---
try:
    import pandas as pd
except Exception:
    print("ERROR: pandas is required. Install with: pip install pandas", file=sys.stderr)
    raise

# --- tqdm (optional) ---
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # fallback to no-op progress bar

__all__ = [
    "get_station_graph",
    "save_graph",
    "attach_coords_from_csv",
    "download_meteoswiss_station_coords",
]

# ---------------------------------------------------------------------
# Parquet engine selection
# ---------------------------------------------------------------------
def _choose_engine() -> str:
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception:
            print("ERROR: neither 'pyarrow' nor 'fastparquet' is available.", file=sys.stderr)
            print("Install one of them, e.g.: pip install pyarrow   OR   pip install fastparquet", file=sys.stderr)
            raise

ENGINE = _choose_engine()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
_COORD_CANDIDATES = {
    "lat": {"lat", "latitude", "station_lat", "y", "lat_deg", "wgs84_latitude"},
    "lon": {"lon", "longitude", "station_lon", "x", "lon_deg", "wgs84_longitude"},
    "elev": {"elev", "elevation", "elevation_m", "alt", "alt_m", "altitude", "altitude_m"},
}

def _is_year_dir(p: Path) -> bool:
    return p.is_dir() and re.fullmatch(r"\d{4}", p.name) is not None

def _scan_year_dirs(root: Path, years: Optional[Iterable[int | str]]) -> List[Path]:
    ydirs = [p for p in root.iterdir() if _is_year_dir(p)]
    if years:
        allow = {str(y) for y in years}
        ydirs = [p for p in ydirs if p.name in allow]
    return sorted(ydirs, key=lambda p: p.name)

def _safe_float(x) -> Optional[float]:
    if pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None

def _safe_int(x) -> Optional[int]:
    if pd.isna(x):
        return None
    try:
        fx = float(x)
        ix = int(fx)
        return ix if ix == fx else None
    except Exception:
        return None

def _to_jsonable_scalar(v):
    """Convert pandas/object scalars to JSON-safe values."""
    if pd.isna(v):
        return None
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            pass
    iv = _safe_int(v)
    if iv is not None:
        return iv
    fv = _safe_float(v)
    if fv is not None:
        return fv
    return str(v)

def _df_to_records_jsonable(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {}
        for k, v in row.items():
            rec[str(k)] = _to_jsonable_scalar(v)
        out.append(rec)
    return out

def _infer_coord(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce")
    s = s[~s.isna()]
    if s.empty:
        return None
    return float(s.iloc[0])

def _infer_station_coords(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    cols = {str(c).lower(): c for c in df.columns}
    lat_col = next((cols[c] for c in _COORD_CANDIDATES["lat"] if c in cols), None)
    lon_col = next((cols[c] for c in _COORD_CANDIDATES["lon"] if c in cols), None)
    elev_col = next((cols[c] for c in _COORD_CANDIDATES["elev"] if c in cols), None)

    lat = _infer_coord(df[lat_col]) if lat_col else None
    lon = _infer_coord(df[lon_col]) if lon_col else None
    elev = _infer_coord(df[elev_col]) if elev_col else None
    return lat, lon, elev

def _maybe_parse_time(df: pd.DataFrame) -> pd.DataFrame:
    if "time_utc" in df.columns:
        try:
            df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
        except Exception:
            pass
    return df

# ---------------------------------------------------------------------
# Download & attach station coordinates
# ---------------------------------------------------------------------
# Prioritize SMN (automatic network), then OBS (visual), then SMN-precip.
_DEFAULT_COORD_URLS: List[str] = [
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_stations.csv",
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-obs/ogd-obs_meta_stations.csv",
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn-precip/ogd-smn-precip_meta_stations.csv",
]

def _http_get(url: str, timeout: int = 30) -> bytes:
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        try:
            from urllib.request import urlopen  # type: ignore
            with urlopen(url, timeout=timeout) as resp:  # nosec B310
                return resp.read()
        except Exception as e:
            raise RuntimeError(f"Failed to download: {url} ({e})") from e

def download_meteoswiss_station_coords(
    out_csv: str | Path | None = None,
    urls: Optional[Iterable[str]] = None,
    timeout: int = 30,
) -> Path:
    """Download the FIRST available stations CSV (kept for backwards-compat)."""
    urls = list(urls) if urls else list(_DEFAULT_COORD_URLS)
    last_err = None
    for url in urls:
        try:
            blob = _http_get(url, timeout=timeout)
            out = Path(out_csv) if out_csv else Path("meteoswiss_stations_meta.csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(blob)
            return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All coordinate URLs failed: {urls}\nLast error: {last_err}")

def _normalize_meta_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cmap = {c.lower(): c for c in df.columns}
    return {
        "abbr": cmap.get("station_abbr") or cmap.get("station") or cmap.get("abbr") or cmap.get("stn"),
        "name": (cmap.get("station_name") or cmap.get("name") or cmap.get("stationname") or
                 cmap.get("station_name_de") or cmap.get("station_name_en") or
                 cmap.get("station_name_fr") or cmap.get("station_name_it")),
        "lat":  cmap.get("latitude") or cmap.get("lat") or cmap.get("wgs84_latitude"),
        "lon":  cmap.get("longitude") or cmap.get("lon") or cmap.get("wgs84_longitude"),
        "elev": cmap.get("elevation_m") or cmap.get("elevation") or cmap.get("altitude") or
                cmap.get("altitude_m") or cmap.get("elev"),
        "lv95_e": cmap.get("e_lv95") or cmap.get("e") or cmap.get("east") or cmap.get("easting"),
        "lv95_n": cmap.get("n_lv95") or cmap.get("n") or cmap.get("north") or cmap.get("northing"),
    }

def _lv95_to_wgs84(e: pd.Series, n: pd.Series) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    try:
        from pyproj import Transformer  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None, None
    e = pd.to_numeric(e, errors="coerce")
    n = pd.to_numeric(n, errors="coerce")
    mask = (~e.isna()) & (~n.isna())
    if not mask.any():
        return None, None
    transformer = Transformer.from_crs(2056, 4326, always_xy=True)  # (lon, lat)
    lon, lat = transformer.transform(e[mask].to_numpy(), n[mask].to_numpy())
    lat_s = pd.Series(index=e.index, dtype=float); lat_s.loc[mask] = lat
    lon_s = pd.Series(index=e.index, dtype=float); lon_s.loc[mask] = lon
    return lat_s, lon_s

def _read_coords_csv(path: Path) -> pd.DataFrame:
    """Read + normalize a MeteoSwiss stations CSV to columns: station_abbr, lat, lon, elev_m, station_name."""
    try:
        df = pd.read_csv(path, sep=";", encoding="cp1252")
    except Exception:
        try:
            df = pd.read_csv(path)  # comma-separated fallback
        except Exception:
            # ultimate fallback: try utf-8 + semicolon
            df = pd.read_csv(path, sep=";", encoding="utf-8")

    cols = _normalize_meta_columns(df)
    abbr_c = cols["abbr"]; lat_c = cols["lat"]; lon_c = cols["lon"]
    elev_c = cols["elev"]; name_c = cols["name"]

    # If no WGS84 but LV95 present, convert (if pyproj available).
    if (lat_c is None or lon_c is None) and cols["lv95_e"] and cols["lv95_n"]:
        lat_s, lon_s = _lv95_to_wgs84(df[cols["lv95_e"]], df[cols["lv95_n"]])
        if lat_s is not None and lon_s is not None:
            df["__lat_from_lv95"] = lat_s
            df["__lon_from_lv95"] = lon_s
            lat_c = "__lat_from_lv95"; lon_c = "__lon_from_lv95"

    if abbr_c is None:
        raise ValueError(f"Stations CSV missing station id column: {list(df.columns)}")

    out = pd.DataFrame({
        "station_abbr": df[abbr_c].astype(str).str.upper().str.strip(),
        "lat": pd.to_numeric(df[lat_c], errors="coerce") if lat_c else pd.Series([None]*len(df)),
        "lon": pd.to_numeric(df[lon_c], errors="coerce") if lon_c else pd.Series([None]*len(df)),
        "elev_m": pd.to_numeric(df[elev_c], errors="coerce") if elev_c else pd.Series([None]*len(df)),
        "station_name": (df[name_c] if name_c else pd.Series([None]*len(df))),
    })
    return out

def _merge_coords_priority(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Priority merge: earlier dfs have higher priority. For duplicates, take first non-null from earlier."""
    combined: Dict[str, Dict[str, Any]] = {}
    for df in dfs:
        for _, r in df.iterrows():
            abbr = str(r["station_abbr"]).upper()
            if abbr not in combined:
                combined[abbr] = {
                    "station_abbr": abbr, "lat": None, "lon": None,
                    "elev_m": None, "station_name": None
                }
            target = combined[abbr]
            for k in ("lat", "lon", "elev_m", "station_name"):
                val = r[k]
                if k == "station_name":
                    if target[k] in (None, "") and pd.notna(val):
                        target[k] = str(val)
                else:
                    if target[k] is None or (isinstance(target[k], float) and pd.isna(target[k])):
                        if pd.notna(val):
                            target[k] = float(val)
    return pd.DataFrame(list(combined.values()))

def _download_and_merge_coords(cache_dir: Path, urls: List[str]) -> pd.DataFrame:
    """Download all provided URLs (cached) and merge into a single normalized DataFrame."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dfs: List[pd.DataFrame] = []
    for i, url in enumerate(urls):
        fname = cache_dir / f"stations_meta_{i}.csv"
        if not fname.exists():
            try:
                blob = _http_get(url)
                fname.write_bytes(blob)
            except Exception as e:
                print(f"[warn] Skipping coords source (download failed): {url} ({e})", file=sys.stderr)
                continue
        try:
            dfs.append(_read_coords_csv(fname))
        except Exception as e:
            print(f"[warn] Skipping coords source (parse failed): {fname} ({e})", file=sys.stderr)
    if not dfs:
        raise RuntimeError("No coordinate sources could be downloaded or parsed.")
    return _merge_coords_priority(dfs)

def attach_coords_from_dataframe(G, coords_df: pd.DataFrame) -> int:
    """Attach lat/lon/elev_m/station_name to nodes from a normalized DataFrame."""
    updated = 0
    coords_df = coords_df.dropna(subset=["station_abbr"]).copy()
    coords_df["station_abbr"] = coords_df["station_abbr"].astype(str).str.upper().str.strip()
    lut = coords_df.set_index("station_abbr").to_dict(orient="index")
    for n in list(G.nodes):
        if n in lut:
            entry = lut[n]
            changed = False
            for k in ("lat", "lon", "elev_m", "station_name"):
                val = entry.get(k)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                if G.nodes[n].get(k) is None:
                    G.nodes[n][k] = float(val) if k in ("lat", "lon", "elev_m") else str(val)
                    changed = True
            if changed:
                updated += 1
    return updated

def attach_coords_from_csv(G, meta_csv: str | Path) -> int:
    """Backward-compatible CSV attach (single file)."""
    df = _read_coords_csv(Path(meta_csv))
    return attach_coords_from_dataframe(G, df)

# ---------------------------------------------------------------------
# swisstopo geocoding (SearchServer)
# ---------------------------------------------------------------------
_SWISSTOPO_SEARCH_URL = (
    "https://api3.geo.admin.ch/rest/services/api/SearchServer?"
    "type=locations&origins=gg25,gazetteer&geometryFormat=geojson&sr=4326&limit=1&searchText="
)

def _http_json(url: str, timeout: int = 15) -> Optional[dict]:
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        try:
            from urllib.request import urlopen  # type: ignore
            import json as _json
            with urlopen(url, timeout=timeout) as resp:  # nosec B310
                return _json.loads(resp.read().decode("utf-8", errors="ignore"))
        except Exception as e:
            print(f"[warn] swisstopo http failed: {e}", file=sys.stderr)
            return None

def _clean_name_for_query(name: str) -> List[str]:
    """
    Create a list of query candidates from a station name, e.g.
    'Zürich / Kloten' -> ['Zürich / Kloten', 'Kloten', 'Zürich', 'Zurich', 'Klosters-Serneus'...]
    Keep it simple: try full, then split on '/', ',', '(', ')', take non-empty trims, de-dup.
    Also try basic ASCII fold for the main tokens.
    """
    if not name or not isinstance(name, str):
        return []
    base = [name.strip()]
    # split tokens
    parts = re.split(r"[\/,()]+", name)
    parts = [p.strip() for p in parts if p.strip()]
    # de-dup while preserving order
    seen = set()
    out = []
    for cand in base + parts:
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    # naive ASCII fold fallback versions
    def _fold(s: str) -> str:
        try:
            import unicodedata as ud
            return "".join(c for c in ud.normalize("NFKD", s) if not ud.combining(c))
        except Exception:
            return s
    folded = []
    for s in out:
        f = _fold(s)
        if f != s and f not in seen:
            folded.append(f)
            seen.add(f)
    return out + folded

def _swisstopo_geocode_one(query: str, timeout: int = 15) -> Optional[Tuple[float, float]]:
    url = _SWISSTOPO_SEARCH_URL + quote_plus(query)
    data = _http_json(url, timeout=timeout)
    if not data:
        return None
    # New API returns a list or an object with "results"/"features"
    features = None
    if isinstance(data, dict):
        features = data.get("features") or data.get("results")
    elif isinstance(data, list):
        features = data
    if not features:
        return None
    feat = features[0]
    # GeoJSON feature
    geom = feat.get("geometry") if isinstance(feat, dict) else None
    if geom and geom.get("type") == "Point":
        coords = geom.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            try:
                return float(lat), float(lon)
            except Exception:
                return None
    # Some responses embed 'attrs' with y/x (if not geojson)
    attrs = feat.get("attrs") if isinstance(feat, dict) else None
    if attrs and ("lat" in attrs and "lon" in attrs):
        try:
            return float(attrs["lat"]), float(attrs["lon"])
        except Exception:
            return None
    if attrs and ("y" in attrs and "x" in attrs):
        try:
            # Often LV95 if sr not 4326, but we ask 4326; still handle if provided
            return float(attrs["y"]), float(attrs["x"])
        except Exception:
            return None
    return None

def _load_geocode_cache(path: Path) -> Dict[str, Tuple[float, float]]:
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            # ensure tuple
            return {k: (float(v[0]), float(v[1])) for k, v in obj.items()}
        except Exception:
            return {}
    return {}

def _save_geocode_cache(path: Path, cache: Dict[str, Tuple[float, float]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _geocode_missing_coords(
    df: pd.DataFrame,
    cache_dir: Path,
    rate_sleep_s: float = 0.15,
    max_per_run: Optional[int] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    For rows with missing lat/lon but a station_name, query swisstopo and fill lat/lon.
    Returns (updated_df, num_geocoded).
    Caches results by exact station_name string.
    """
    cache_json = cache_dir / "swisstopo_geocode_cache.json"
    cache = _load_geocode_cache(cache_json)

    need = df[(df["lat"].isna() | df["lon"].isna()) & df["station_name"].notna()].copy()
    if need.empty:
        return df, 0

    # progress bar
    iterable = list(need.index)
    if tqdm is not None:
        pbar = tqdm(total=len(iterable), desc="[SWISSTOPO] Geocoding station_name", unit="stn")
    else:
        pbar = None

    geocoded = 0
    n_done = 0
    for idx in iterable:
        name = str(df.at[idx, "station_name"]).strip()
        if not name:
            if pbar: pbar.update(1)
            continue

        # Use cache first
        if name in cache:
            lat, lon = cache[name]
            if pd.isna(df.at[idx, "lat"]): df.at[idx, "lat"] = lat
            if pd.isna(df.at[idx, "lon"]): df.at[idx, "lon"] = lon
            geocoded += 1
            if pbar: pbar.update(1)
            n_done += 1
            if max_per_run is not None and n_done >= max_per_run: break
            continue

        # Try multiple candidates derived from the name
        candidates = _clean_name_for_query(name)
        found = None
        for q in candidates:
            res = _swisstopo_geocode_one(q)
            if res is not None:
                found = res  # (lat, lon)
                break

        if found:
            lat, lon = found
            cache[name] = (lat, lon)
            if pd.isna(df.at[idx, "lat"]): df.at[idx, "lat"] = lat
            if pd.isna(df.at[idx, "lon"]): df.at[idx, "lon"] = lon
            geocoded += 1

        if pbar: pbar.update(1)
        n_done += 1
        if max_per_run is not None and n_done >= max_per_run:
            break

        # be polite to the API
        time.sleep(rate_sleep_s)

    if pbar:
        try: pbar.close()
        except Exception: pass

    # Persist cache and return
    _save_geocode_cache(cache_json, cache)
    return df, geocoded

def _auto_attach_coords(
    G,
    cache_dir: Optional[str | Path],
    urls: Optional[Iterable[str]],
    *,
    geocode_missing: bool = True,
    geocode_rate_sleep_s: float = 0.15,
    geocode_max_per_run: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Auto-attach coordinates by downloading & merging multiple sources
    and optionally geocoding station_name via swisstopo to fill missing lat/lon.
    Returns (num_updated_nodes_from_tables_and_geocode, total_with_coords_after).
    Also writes a combined CSV with any geocoded coordinates into the cache_dir.
    """
    cache_dir = Path(cache_dir) if cache_dir else Path(".")
    url_list = list(urls) if urls else list(_DEFAULT_COORD_URLS)

    # 1) Download + merge official tables
    try:
        merged = _download_and_merge_coords(cache_dir, url_list)
    except Exception as e:
        print(f"[warn] Could not auto-attach coordinates (download/merge): {e}", file=sys.stderr)
        # even if this fails, we can still try geocoding from names in the graph directly
        merged = pd.DataFrame(columns=["station_abbr", "lat", "lon", "elev_m", "station_name"])

    # 2) If missing lat/lon, try geocoding by station_name
    merged_before = merged.copy()
    n_missing_before = int(((merged["lat"].isna()) | (merged["lon"].isna())).sum())
    geocoded_n = 0
    if geocode_missing and "station_name" in merged.columns and not merged.empty and n_missing_before > 0:
        merged, geocoded_n = _geocode_missing_coords(
            merged, cache_dir, rate_sleep_s=geocode_rate_sleep_s, max_per_run=geocode_max_per_run
        )

    # 3) Attach to graph (prefer not to overwrite existing)
    updated_from_tables = attach_coords_from_dataframe(G, merged)

    # 4) Write combined CSV (UTF-8) with any geocoded fills for transparency
    try:
        out_csv = cache_dir / "meteoswiss_stations_meta_combined.csv"
        out_csv.write_text(merged.to_csv(index=False), encoding="utf-8")
    except Exception:
        pass

    # If the merged list did not cover some graph nodes (e.g., entirely missing),
    # we can try geocoding directly from node station_name as a last resort
    # to attach coords for those specific nodes.
    extra_direct_geocoded = 0
    if geocode_missing:
        # collect nodes lacking coords but having a station_name
        to_fill = []
        for n, a in G.nodes(data=True):
            if (a.get("lat") is None or a.get("lon") is None) and a.get("station_name"):
                to_fill.append((n, a.get("station_name")))
        if to_fill:
            # build small df and geocode
            tmp_df = pd.DataFrame({
                "station_abbr": [n for n, _ in to_fill],
                "station_name": [str(s) for _, s in to_fill],
                "lat": [None]*len(to_fill),
                "lon": [None]*len(to_fill),
                "elev_m": [None]*len(to_fill),
            })
            tmp_df, extra_direct_geocoded = _geocode_missing_coords(
                tmp_df, cache_dir, rate_sleep_s=geocode_rate_sleep_s, max_per_run=None
            )
            # attach
            updated_from_geo = attach_coords_from_dataframe(G, tmp_df)
            updated_from_tables += updated_from_geo

    have_coords = sum(1 for _, a in G.nodes(data=True) if a.get("lat") is not None and a.get("lon") is not None)
    if geocoded_n or extra_direct_geocoded:
        print(f"[STATIONS] swisstopo geocoded {geocoded_n + extra_direct_geocoded} station(s).", file=sys.stderr)
    return updated_from_tables, have_coords

# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------
def _haversine_km_vec(lat1, lon1, lat2, lon2):
    try:
        import numpy as np
    except Exception:
        raise ImportError("numpy is required for distance computation. Install with: pip install numpy") from None
    R = 6371.0088
    lat1r, lon1r, lat2r, lon2r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def _maybe_add_distance_edges(G, d_km: Optional[float]) -> None:
    if d_km is None:
        return
    try:
        import numpy as np
        import networkx as nx  # noqa: F401
    except Exception:
        raise ImportError("networkx and numpy are required for building edges. Install with: pip install networkx numpy") from None

    nodes, lats, lons = [], [], []
    for n, attrs in G.nodes(data=True):
        lat, lon = attrs.get("lat"), attrs.get("lon")
        if lat is not None and lon is not None:
            nodes.append(n); lats.append(float(lat)); lons.append(float(lon))
    if len(nodes) < 2:
        return

    lats = pd.Series(lats, dtype=float).to_numpy()
    lons = pd.Series(lons, dtype=float).to_numpy()
    dmat = _haversine_km_vec(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
    iu, ju = np.triu_indices(len(nodes), k=1)
    mask = dmat[iu, ju] <= float(d_km)
    for i_idx, j_idx, dist in zip(iu[mask], ju[mask], dmat[iu[mask], ju[mask]]):
        G.add_edge(nodes[i_idx], nodes[j_idx], distance_km=float(dist))

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def get_station_graph(
    parquet_root: str | Path,
    years: Optional[Iterable[int | str]] = None,
    d_km: Optional[float] = None,
    drop_columns: Optional[Iterable[str]] = None,
    show_progress: bool = True,
    *,
    auto_coords: bool = True,
    coords_cache_dir: str | Path | None = None,
    coord_urls: Optional[Iterable[str]] = None,
    geocode_missing: bool = True,
    geocode_rate_sleep_s: float = 0.15,
    geocode_max_per_run: Optional[int] = None,
):
    """
    Build a NetworkX graph from <parquet_root>/<YEAR>/<STATION>.parquet only,
    then (optionally) auto-download & merge coordinates and add them to nodes.
    If some stations still lack lat/lon after merging, optionally geocode
    their station_name with swisstopo and fill lat/lon; write the combined
    stations CSV (with any geocoded coordinates) into coords_cache_dir.

    Parameters
    ----------
    parquet_root : path to directory containing <YEAR> subfolders with *.parquet
    years        : optional iterable to restrict which <YEAR> folders to load
    d_km         : connect stations within d_km (km) if coords available
    drop_columns : optional iterable of column names to remove from each parquet before storing
    show_progress: show a progress bar while reading parquet files (requires tqdm)
    auto_coords  : if True, try to download and attach coordinates from multiple sources
    coords_cache_dir : where to cache the downloaded/combined stations CSVs and geocode cache
    coord_urls   : override list of URLs to try (priority order)
    geocode_missing : if True, use swisstopo SearchServer to fill missing lat/lon from station_name
    geocode_rate_sleep_s : polite delay between geocoding requests
    geocode_max_per_run  : optional cap on the number of geocoding requests per run

    Returns
    -------
    networkx.Graph
    """
    try:
        import networkx as nx
    except Exception:
        raise ImportError("networkx is required to build the graph. Install with: pip install networkx") from None

    root = Path(parquet_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Parquet root not found or not a directory: {root}")

    ydirs = _scan_year_dirs(root, years)
    if not ydirs:
        raise SystemExit(f"No <YEAR> subfolders found under: {root}")

    drop_columns = list(drop_columns) if drop_columns else []

    # Count total parquet files for progress bar
    total_files = sum(len(list(ydir.glob("*.parquet"))) for ydir in ydirs)

    class _NoBar:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, n=1): pass
        def close(self): pass

    pbar = _NoBar()
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total_files, desc="[STATIONS] Analyzing parquet files", unit="file")

    G = nx.Graph()
    try:
        for ydir in ydirs:
            year = ydir.name
            for pq in sorted(ydir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(pq, engine=ENGINE)
                except Exception as e:
                    print(f"[warn] Failed to read {pq}: {e}", file=sys.stderr)
                    pbar.update(1)
                    continue

                df = _maybe_parse_time(df)

                if drop_columns:
                    keep_cols = [c for c in df.columns if c not in drop_columns]
                    df = df[keep_cols]

                lat, lon, elev = _infer_station_coords(df)

                abbr = pq.stem.upper()
                if abbr not in G:
                    G.add_node(
                        abbr,
                        station_abbr=abbr,
                        years={},
                        columns_union=set(),
                        first_year=int(year),
                        last_year=int(year),
                        lat=lat,
                        lon=lon,
                        elev_m=elev,
                        station_name=None,
                    )

                node = G.nodes[abbr]
                node["first_year"] = min(node["first_year"], int(year))
                node["last_year"]  = max(node["last_year"], int(year))
                if node.get("lat") is None and lat is not None: node["lat"] = lat
                if node.get("lon") is None and lon is not None: node["lon"] = lon
                if node.get("elev_m") is None and elev is not None: node["elev_m"] = elev

                n_rows = int(df.shape[0])
                cols = list(map(str, df.columns.tolist()))
                node["columns_union"].update(cols)
                node["years"][year] = {
                    "n_rows": n_rows,
                    "columns": cols,
                    "data": _df_to_records_jsonable(df),
                    "source_path": str(pq.resolve()),
                }

                pbar.update(1)
    finally:
        for _, attrs in G.nodes(data=True):
            if isinstance(attrs.get("columns_union"), set):
                attrs["columns_union"] = sorted(attrs["columns_union"])
        try: pbar.close()
        except Exception: pass

    # Auto-download & MERGE coordinates across networks (+ geocode missing)
    if auto_coords:
        updated, have_coords = _auto_attach_coords(
            G,
            coords_cache_dir,
            coord_urls,
            geocode_missing=geocode_missing,
            geocode_rate_sleep_s=geocode_rate_sleep_s,
            geocode_max_per_run=geocode_max_per_run,
        )
        if updated:
            print(f"[STATIONS] Attached coordinates to {updated} station(s) (tables + geocode).", file=sys.stderr)
        else:
            print(f"[STATIONS] Coordinate attach: 0 stations updated.", file=sys.stderr)
        print(f"[STATIONS] Coordinate coverage after attach: {have_coords}/{G.number_of_nodes()} nodes.", file=sys.stderr)

    # Distance edges if possible
    _maybe_add_distance_edges(G, d_km)

    return G

# ---------------------------------------------------------------------
# Save graph
# ---------------------------------------------------------------------
def save_graph(G, out_path: str | Path) -> Path:
    try:
        import networkx as nx
    except Exception:
        raise ImportError("networkx is required to save graphs. Install with: pip install networkx") from None

    out_path = Path(out_path)
    if not out_path.suffix:
        out_path = out_path.with_suffix(".gexf")

    suf = out_path.suffix.lower()
    if suf == ".gexf":
        nx.write_gexf(G, out_path)
    elif suf in (".graphml", ".xml"):
        nx.write_graphml(G, out_path)
    elif suf in (".gpickle", ".pickle"):
        nx.write_gpickle(G, out_path)
    elif suf in (".edgelist", ".edges", ".txt"):
        nx.write_edgelist(G, out_path, data=[("distance_km", float)])
    else:
        out_path = out_path.with_suffix(".gexf")
        nx.write_gexf(G, out_path)
    return out_path

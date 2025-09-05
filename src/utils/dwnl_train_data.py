#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download LAST N YEARS of HOURLY measurements from ALL MeteoSwiss ground stations (SwissMetNet)
via the official FSDI STAC API and save per-station Parquet/CSV.

Sources:
- STAC docs & collection: https://opendatadocs.meteoswiss.ch/general/download  (FSDI/STAC)
- Automatic weather stations collection: ch.meteoschweiz.ogd-smn
- Update scheme: "historical" contains data up to Dec 31 of last year (UTC).

Examples:
  # last 3 complete years ending last year (e.g., 2022–2024 if today is 2025)
  python dwnl_train_data.py --out ./meteoswiss --years 3 --format parquet --workers 8 --combine

  # explicit start year to last year (inclusive)
  python dwnl_train_data.py --out ./meteoswiss --start-year 2015 --format csv

Notes:
- We pull STAC items for collection ch.meteoschweiz.ogd-smn (one item ~ one station),
  pick the *historical* CSV/Parquet asset per station (contains hourly data across many years),
  then filter rows per requested year and write outputs.
- CSVs are semicolon-separated and often encoded Windows-1252; we parse robustly.
- Timestamps are UTC in "dd.mm.yyyy HH:MM" → converted to pandas datetime (UTC).
- Output layout (default): <out>/<YEAR>/<STATION>.(parquet|csv).
- If a station has no usable historical asset or no rows after filtering, it is skipped.
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import io
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

STAC_BASE = "https://data.geo.admin.ch/api/stac/v1"
COLLECTION_ID = "ch.meteoschweiz.ogd-smn"
USER_AGENT = "meteoswiss-hourly-downloader/1.1 (academic use)"

# ---- Helpers ----------------------------------------------------------------

def this_and_last_year(reference_utc: Optional[dt.datetime] = None) -> Tuple[int, int]:
    if reference_utc is None:
        reference_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    return reference_utc.year, reference_utc.year - 1

def stac_get(url: str, params: Optional[dict] = None, etag: Optional[str] = None) -> requests.Response:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if etag:
        headers["If-None-Match"] = etag
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r

def list_collection_items(collection_id: str) -> List[Dict]:
    """Retrieve ALL STAC items for the collection (paged)."""
    items = []
    url = f"{STAC_BASE}/collections/{collection_id}/items"
    params = {"limit": 1000}
    while True:
        resp = stac_get(url, params=params)
        payload = resp.json()
        feats = payload.get("features", [])
        items.extend(feats)
        nxt = None
        for link in payload.get("links", []):
            if link.get("rel") == "next":
                nxt = link.get("href")
                break
        if not nxt:
            break
        url = nxt
        params = None
    return items

def pick_hourly_historical_asset(assets: Dict):
    """Relaxed: pick any historical CSV/Parquet asset (these contain hourly series)."""
    for k, a in assets.items():
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if ("historical" in href) and (href.endswith(".csv") or href.endswith(".parquet")):
            return (k, a)
    # fallback: any CSV/Parquet if no "historical" mention
    for k, a in assets.items():
        href = (a.get("href") or "").strip()
        if href.endswith(".csv") or href.endswith(".parquet"):
            return (k, a)
    return None

def read_station_code(item: Dict) -> str:
    props = item.get("properties", {}) or {}
    station = (
        props.get("station_code")
        or props.get("station")
        or props.get("name")
        or item.get("id")
        or "UNKNOWN"
    )
    station = re.sub(r"[^A-Za-z0-9_\-\.]", "_", str(station))
    return station.upper()

def parse_meteoswiss_csv_to_df(content_bytes: bytes) -> pd.DataFrame:
    """Parse MeteoSwiss semicolon CSV into DataFrame with a 'time_utc' column."""
    try:
        txt = content_bytes.decode("cp1252")
    except UnicodeDecodeError:
        txt = content_bytes.decode("utf-8", errors="replace")

    df = pd.read_csv(
        io.StringIO(txt),
        sep=";",
        engine="python",
        dtype=str,
    )
    df.columns = [c.strip() for c in df.columns]

    time_cols = [c for c in df.columns if re.search(r"(time|datum|date)", c, re.I)]
    if not time_cols:
        time_cols = [df.columns[0]]
    ts_col = time_cols[0]

    df[ts_col] = df[ts_col].astype(str).str.strip()
    try:
        dt_series = pd.to_datetime(df[ts_col], format="%d.%m.%Y %H:%M", utc=True)
    except Exception:
        dt_series = pd.to_datetime(df[ts_col], dayfirst=True, utc=True, errors="coerce")

    df.insert(0, "time_utc", dt_series)
    df = df.dropna(subset=["time_utc"]).copy()
    return df

def filter_df_to_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["time_utc"].dt.year == year].copy()

def download_bytes(url: str, *, max_tries: int = 3) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    last_exc = None
    for _ in range(max_tries):
        try:
            r = requests.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
    if last_exc:
        raise last_exc

def parquet_supported() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False

def save_df(df: pd.DataFrame, path: str, fmt: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "parquet":
        if parquet_supported():
            df.to_parquet(path, index=False)
        else:
            # fallback to CSV if parquet engine missing
            alt = os.path.splitext(path)[0] + ".csv"
            sys.stderr.write("[warn] parquet backend missing (pyarrow/fastparquet). Falling back to CSV: "
                             f"{alt}\n")
            df.to_csv(alt, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

# ---- Pipeline per station ----------------------------------------------------

def process_station_item(
    item: Dict,
    year: int,
    out_dir: str,
    fmt: str = "parquet",
    min_rows: int = 1,
) -> Optional[str]:
    station = read_station_code(item)
    assets = item.get("assets", {}) or {}
    pick = pick_hourly_historical_asset(assets)
    if not pick:
        sys.stderr.write(f"[warn] {station}: no usable asset found; skipping\n")
        return None

    _, asset = pick
    href = asset.get("href")
    if not href:
        sys.stderr.write(f"[warn] {station}: asset has no href; skipping\n")
        return None

    try:
        content = download_bytes(href)
        # Decide parser by file extension
        if href.endswith(".parquet") and parquet_supported():
            import pandas as pd  # ensure namespace
            import io as _io
            df = pd.read_parquet(_io.BytesIO(content))
            # standardize timestamp column name if present
            if "time_utc" not in df.columns:
                # try infer common time columns
                time_cols = [c for c in df.columns if re.search(r"(time|datum|date)", c, re.I)]
                if time_cols:
                    cand = time_cols[0]
                    dt_series = pd.to_datetime(df[cand], utc=True, errors="coerce")
                    df.insert(0, "time_utc", dt_series)
            df = df.dropna(subset=["time_utc"]).copy()
        else:
            df = parse_meteoswiss_csv_to_df(content)

        df_y = filter_df_to_year(df, year)
        if len(df_y) < min_rows:
            sys.stderr.write(f"[warn] {station}: after filtering year={year}, no rows; skipping\n")
            return None

        out_path = os.path.join(out_dir, f"{station}.{ 'parquet' if fmt=='parquet' else 'csv'}")
        save_df(df_y, out_path, fmt)
        return out_path
    except Exception as e:
        sys.stderr.write(f"[error] {station}: {repr(e)}\n")
        return None

# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download last N years of hourly measurements from all MeteoSwiss ground stations."
    )
    parser.add_argument("--out", required=True, help="Output directory root (year subfolders will be created)")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                        help="Output file format (default: parquet; falls back to CSV if no parquet backend)")
    parser.add_argument("--combine", action="store_true",
                        help="Also write a combined file per year of all stations")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers (default: 8)")

    # NEW: control how many years
    parser.add_argument("--years", type=int, default=1,
                        help="Number of complete past years to download, ending last year (default: 1)")
    parser.add_argument("--start-year", type=int, default=None,
                        help="Optional explicit start year (inclusive). Overrides --years if provided.")

    args = parser.parse_args()

    this_year, last_year = this_and_last_year()

    # Build list of target years
    if args.start_year is not None:
        if args.start_year > last_year:
            raise SystemExit(f"--start-year must be <= last year ({last_year})")
        years = list(range(args.start_year, last_year + 1))
    else:
        if args.years < 1:
            raise SystemExit("--years must be >= 1")
        start = last_year - (args.years - 1)
        years = list(range(start, last_year + 1))

    os.makedirs(args.out, exist_ok=True)

    # Fetch items for the collection (one per station)
    print(f"[info] Listing STAC items for collection {COLLECTION_ID} …", flush=True)
    items = list_collection_items(COLLECTION_ID)
    print(f"[info] Found {len(items)} items (stations)", flush=True)

    # Loop years
    for y in years:
        year_out = os.path.join(args.out, str(y))
        os.makedirs(year_out, exist_ok=True)
        print(f"[info] Year {y}: downloading historical assets and filtering …", flush=True)

        results = []
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(process_station_item, item, y, year_out, args.format)
                for item in items
            ]
            for fut in cf.as_completed(futures):
                path = fut.result()
                if path:
                    results.append(path)

        print(f"[info] Year {y}: wrote {len(results)} station files to {year_out}")

        if args.combine and results:
            print(f"[info] Year {y}: building combined dataset …", flush=True)
            dfs = []
            for path in results:
                try:
                    if path.endswith(".parquet"):
                        df = pd.read_parquet(path)
                    else:
                        df = pd.read_csv(path, parse_dates=["time_utc"])
                    station = os.path.splitext(os.path.basename(path))[0]
                    df.insert(1, "station", station)
                    dfs.append(df)
                except Exception as e:
                    sys.stderr.write(f"[warn] combine {y}: failed to read {path}: {repr(e)}\n")
            if dfs:
                big = pd.concat(dfs, ignore_index=True)
                big = big.sort_values(["time_utc", "station"])
                out_path = os.path.join(args.out, f"ALL_STATIONS_{y}.{ 'parquet' if args.format=='parquet' and parquet_supported() else 'csv'}")
                save_df(big, out_path, args.format)
                print(f"[info] Year {y}: combined file: {out_path}")

    print("[done]")

if __name__ == "__main__":
    pd.options.display.max_columns = 999
    pd.options.display.width = 200
    main()

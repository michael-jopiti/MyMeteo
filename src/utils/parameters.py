#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
import itertools
import pandas as pd
from string import Template

# ---------- Your parameter dictionary ----------
METEO_PARAMS: Dict[str, Dict[str, str]] = {
    "time_utc":   {"desc": "Timestamp (UTC)", "abbr": "Time"},
    "station_abbr": {"desc": "Station code", "abbr": "Station"},
    "reference_timestamp": {"desc": "Reference timestamp", "abbr": "RefTime"},

    "tre200d0": {"desc": "Air temperature 2 m above ground; daily mean", "abbr": "T2m_mean"},
    "tre200dx": {"desc": "Air temperature 2 m above ground; daily maximum", "abbr": "T2m_max"},
    "tre200dn": {"desc": "Air temperature 2 m above ground; daily minimum", "abbr": "T2m_min"},

    "tre005d0": {"desc": "Air temperature 5 cm above grass; daily mean", "abbr": "T5cm_mean"},
    "tre005dx": {"desc": "Air temperature 5 cm above grass; daily maximum", "abbr": "T5cm_max"},
    "tre005dn": {"desc": "Air temperature 5 cm above grass; daily minimum", "abbr": "T5cm_min"},

    "ure200d0": {"desc": "Relative air humidity 2 m above ground; daily mean", "abbr": "RH_mean"},
    "pva200d0": {"desc": "Vapour pressure 2 m above ground; daily mean", "abbr": "VP_mean"},

    "prestad0": {"desc": "Atmospheric pressure at barometric altitude (QFE); daily mean", "abbr": "QFE"},
    "pp0qffd0": {"desc": "Pressure reduced to sea level (QFF); daily mean", "abbr": "QFF"},
    "pp0qnhd0": {"desc": "Pressure reduced to sea level according to standard atmosphere (QNH); daily mean", "abbr": "QNH"},
    "ppz850d0": {"desc": "Geopotential height of the 850 hPa level; daily mean", "abbr": "Z850"},
    "ppz700d0": {"desc": "Geopotential height of the 700 hPa level; daily mean", "abbr": "Z700"},

    "fkl010d0": {"desc": "Wind speed scalar; daily mean in m/s", "abbr": "Wind_mean_ms"},
    "fkl010d1": {"desc": "Gust peak (one second); daily max in m/s", "abbr": "Gust1s_ms"},
    "fkl010d3": {"desc": "Gust peak (three seconds); daily max in m/s", "abbr": "Gust3s_ms"},
    "fu3010d0": {"desc": "Wind speed scalar; daily mean in km/h", "abbr": "Wind_mean_kmh"},
    "fu3010d1": {"desc": "Gust peak (one second); daily max in km/h", "abbr": "Gust1s_kmh"},
    "fu3010d3": {"desc": "Gust peak (three seconds); daily max in km/h", "abbr": "Gust3s_kmh"},

    "wcc006d0": {"desc": "Foehn index; daily value", "abbr": "FoehnIdx"},

    "rre150d0": {"desc": "Precipitation; daily total 6-6 UTC", "abbr": "Prec_6to6"},
    "rka150d0": {"desc": "Precipitation; daily total 0-0 UTC", "abbr": "Prec_0to0"},

    "htoautd0": {"desc": "Snow depth (automatic, 6 UTC)", "abbr": "SnowDepth"},

    "gre000d0": {"desc": "Global radiation; daily mean", "abbr": "GlobRad"},
    "oli000d0": {"desc": "Longwave incoming radiation; daily mean", "abbr": "LW_in"},
    "olo000d0": {"desc": "Longwave outgoing radiation; daily mean", "abbr": "LW_out"},
    "osr000d0": {"desc": "Shortwave reflected radiation; daily mean", "abbr": "SW_refl"},
    "ods000d0": {"desc": "Diffuse radiation; daily mean", "abbr": "DiffuseRad"},

    "sre000d0": {"desc": "Sunshine duration; daily total", "abbr": "SunDur"},
    "sremaxdv": {"desc": "Sunshine duration vs absolute max; daily %", "abbr": "SunRel"},

    "erefaod0": {"desc": "Reference evaporation (FAO); daily total", "abbr": "Evap_FAO"},

    "xcd000d0": {"desc": "Cooling Degree Day (CDD)", "abbr": "CDD"},
    "xno000d0": {"desc": "Heating degrees (HGT12/20)", "abbr": "HDD12/20"},
    "xno012d0": {"desc": "Heating degrees (ATD12/12)", "abbr": "HDD12/12"},

    "dkl010d0": {"desc": "Wind direction; daily mean", "abbr": "WindDir"},

    "rreetsd0": {"desc": "Hydrologic balance (R-ETS); daily total", "abbr": "HydroBal"},
}

# -------------------- Utils --------------------
def _normalize_code(s: str) -> str:
    return "" if s is None else (
        str(s)
        .replace("ﬀ", "ff").replace("\ufb00", "ff")
        .replace("ﬁ", "fi").replace("\ufb01", "fi")
        .replace("ﬂ", "fl").replace("\ufb02", "fl")
        .strip().lower()
    )

def _latex_escape(s: str) -> str:
    if s is None:
        return ""
    rep = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"
    }
    return "".join(rep.get(ch, ch) for ch in str(s))

def _choose_engine() -> str:
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception:
            raise RuntimeError("Install pyarrow or fastparquet")

def _compile_latex(tex_path: Path, out_dir: Path) -> Path:
    print("Running TeX ...")
    if shutil.which("tectonic"):
        subprocess.run(
            ["tectonic", "--keep-logs", "--keep-intermediates",
             "--outdir", str(out_dir), str(tex_path)],
            check=True
        )
    elif shutil.which("pdflatex"):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(out_dir), str(tex_path)],
            check=True
        )
    else:
        raise RuntimeError("No LaTeX engine found")
    return out_dir / f"{tex_path.stem}.pdf"

# ------------ Color palette + helpers (stable mapping) ------------
LATEX_COLOR_PREAMBLE = r"""
\usepackage{xcolor}
\definecolor{c0}{RGB}{31,119,180}   % blue
\definecolor{c1}{RGB}{214,39,40}    % red
\definecolor{c2}{RGB}{44,160,44}    % green
\definecolor{c3}{RGB}{255,127,14}   % orange
\definecolor{c4}{RGB}{148,103,189}  % purple
\definecolor{c5}{RGB}{140,86,75}    % brown
\definecolor{c6}{RGB}{23,190,207}   % cyan
\definecolor{c7}{RGB}{188,189,34}   % olive
\definecolor{c8}{RGB}{127,127,127}  % gray
\definecolor{c9}{RGB}{227,119,194}  % pink
"""

_PALETTE_SIZE = 10
def _stable_idx(name: str) -> int:
    return sum(name.encode("utf-8")) % _PALETTE_SIZE

def _station_color(station: str) -> str:
    return f"c{_stable_idx('S.' + (station or ''))}"

def _param_color(abbr: str) -> str:
    return f"c{_stable_idx('P.' + (abbr or ''))}"

# -------------------- MeteoParameters --------------------
@dataclass
class MeteoParameters:
    params: Dict[str, Dict[str, str]]

    def __post_init__(self):
        self._abbr_to_param = {v["abbr"]: k for k, v in self.params.items()}

    def get_description(self, param: str) -> Optional[str]:
        rec = self.params.get(param)
        return rec["desc"] if rec else None

    def from_abbr(self, abbr: str) -> Optional[str]:
        return self._abbr_to_param.get(abbr)

    def _collect_data(
        self, data_root: Path, station: str, code: str,
        engine: str, year: Optional[int] = None, delta: Optional[str] = None
    ) -> pd.DataFrame:
        data_root = Path(data_root).resolve()
        eng = _choose_engine() if engine == "auto" else engine
        year_dirs = [p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()]
        if year:
            year_dirs = [p for p in year_dirs if int(p.name) == year]
        if delta:
            start, end = map(int, delta.split("-"))
            year_dirs = [p for p in year_dirs if start <= int(p.name) <= end]

        frames: List[pd.DataFrame] = []
        for ydir in sorted(year_dirs):
            f = ydir / f"{station}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f, engine=eng)
            cols = {_normalize_code(c): c for c in df.columns}
            tcol, vcol = cols.get("time_utc"), cols.get(_normalize_code(code))
            if not tcol or not vcol:
                continue
            part = df[[tcol, vcol]].copy()
            part.columns = ["time_utc", "value"]
            part["time_utc"] = pd.to_datetime(part["time_utc"], utc=True, errors="coerce")
            frames.append(part.dropna())

        return pd.concat(frames) if frames else pd.DataFrame()

    # -------- Single station + single param (colored by parameter) --------
    def plot_parameter(
        self, data_root: Path, station: str, abbr: str, out: Path,
        year: Optional[int] = None, delta: Optional[str] = None, engine: str = "auto"
    ) -> Path:
        code = self.from_abbr(abbr)
        if not code:
            raise ValueError(f"Unknown parameter: {abbr}")
        df = self._collect_data(data_root, station, code, engine, year, delta)
        if df.empty:
            raise RuntimeError(f"No data for {station} {abbr}")

        csv_path = out.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df["date"] = df["time_utc"].dt.strftime("%Y-%m-%d")
        df[["date", "value"]].to_csv(csv_path, index=False)

        col = _param_color(abbr)

        tex_tpl = Template(r"""
\documentclass[tikz]{standalone}
$colors
\usepackage{pgfplots}
\usepgfplotslibrary{dateplot}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[width=16cm,height=9cm,date coordinates in=x,grid=both,
  xlabel=Date,ylabel={$ylabel},title={$title},xticklabel style={rotate=45,anchor=east}]
\addplot+[mark=none, very thick, color=$color] table [x=date, y=value, col sep=comma] {$csv};
\end{axis}
\end{tikzpicture}
\end{document}
""")
        tex_str = tex_tpl.substitute(
            colors=LATEX_COLOR_PREAMBLE,
            ylabel=_latex_escape(abbr),
            title=_latex_escape(f"{station} - {abbr}: {self.get_description(code)}"),
            csv=csv_path.name,
            color=col,
        )
        tex_path = out.with_suffix(".tex")
        tex_path.write_text(tex_str, encoding="utf-8")
        return _compile_latex(tex_path, out.parent)

    # -------- Multiple stations + one param (each station colored) --------
    def plot_multi_stations(
        self, data_root: Path, stations: List[str], abbr: str, out: Path,
        year: Optional[int] = None, delta: Optional[str] = None, engine: str = "auto"
    ) -> Path:
        code = self.from_abbr(abbr)
        if not code:
            raise ValueError(f"Unknown parameter: {abbr}")
        workdir = out.parent
        workdir.mkdir(parents=True, exist_ok=True)

        plots: List[str] = []
        for st in stations:
            df = self._collect_data(data_root, st, code, engine, year, delta)
            if df.empty:
                continue
            df["date"] = df["time_utc"].dt.strftime("%Y-%m-%d")
            csv_path = workdir / f"{st}_{abbr}.csv"
            df[["date", "value"]].to_csv(csv_path, index=False)
            col = _station_color(st)
            plots.append(
                rf"\addplot+[mark=none, very thick, color={col}] table [x=date,y=value,col sep=comma] {{{csv_path.name}}};"
                rf"\addlegendentry{{{st}}}"
            )

        tex_tpl = Template(r"""
\documentclass[a4paper,10pt]{article}
\usepackage[margin=1in]{geometry}
$colors
\usepackage{pgfplots}
\usepgfplotslibrary{dateplot}
\pgfplotsset{compat=1.18}
\begin{document}
\section*{Parameter $abbr across stations}
\begin{tikzpicture}
\begin{axis}[width=16cm,height=9cm,date coordinates in=x,grid=both,
  xlabel=Date,ylabel={$ylabel},title={$title},xticklabel style={rotate=45,anchor=east},
  legend pos=outer north east, legend cell align=left]
$plots
\end{axis}
\end{tikzpicture}
\end{document}
""")
        tex_str = tex_tpl.substitute(
            colors=LATEX_COLOR_PREAMBLE,
            abbr=_latex_escape(abbr),
            ylabel=_latex_escape(abbr),
            title=_latex_escape(f"{abbr}: {self.get_description(code)}"),
            plots="\n".join(plots),
        )
        tex_path = out.with_suffix(".tex")
        tex_path.write_text(tex_str, encoding="utf-8")
        return _compile_latex(tex_path, workdir)

    # -------- Station vs Station comparisons (2 cols × 3 rows per page) --------
    def plot_station_vs_station(
        self, data_root: Path, stations: List[str], abbrs: List[str], out: Path,
        year: Optional[int] = None, delta: Optional[str] = None, engine: str = "auto"
    ) -> Path:
        workdir = out.parent
        workdir.mkdir(parents=True, exist_ok=True)

        tile_tpl = Template(r"""
\begin{minipage}{0.48\textwidth}
\centering
\begin{tikzpicture}
\begin{axis}[date coordinates in=x,width=\linewidth,height=4.8cm,grid=both,
  title={$title},xticklabel style={rotate=45,anchor=east},
  ylabel={$y1},axis y line*=left]
\addplot+[mark=none, very thick, color=$color1] table [x=date, y=$ycol1, col sep=comma] {$csv};
\end{axis}
\begin{axis}[date coordinates in=x,width=\linewidth,height=4.8cm,grid=both,
  xticklabel style={rotate=45,anchor=east},axis y line*=right,axis x line=none,
  ylabel={$y2}]
\addplot+[mark=none, very thick, color=$color2] table [x=date, y=$ycol2, col sep=comma] {$csv};
\end{axis}
\end{tikzpicture}
\end{minipage}
""")

        tiles: List[str] = []
        for abbr in abbrs:
            code = self.from_abbr(abbr)
            if not code:
                continue
            for s1, s2 in itertools.combinations(stations, 2):
                df1 = self._collect_data(data_root, s1, code, engine, year, delta)
                df2 = self._collect_data(data_root, s2, code, engine, year, delta)
                if df1.empty or df2.empty:
                    continue

                df = pd.merge(df1, df2, on="time_utc", suffixes=(f"_{s1}", f"_{s2}"))
                df["date"] = df["time_utc"].dt.strftime("%Y-%m-%d")
                csv_path = workdir / f"{s1}_{s2}_{abbr}_compare.csv"
                df[["date", f"value_{s1}", f"value_{s2}"]].to_csv(csv_path, index=False)

                tiles.append(
                    tile_tpl.substitute(
                        title=_latex_escape(f"{s1} vs {s2} ({abbr})"),
                        y1=_latex_escape(s1),
                        y2=_latex_escape(s2),
                        ycol1=f"value_{s1}",
                        ycol2=f"value_{s2}",
                        csv=csv_path.name,
                        color1=_station_color(s1),
                        color2=_station_color(s2),
                    )
                )

        # Build pages: 6 tiles per page in 2×3 grid
        tables: List[str] = []
        for start in range(0, len(tiles), 6):
            chunk = tiles[start:start + 6]
            rows: List[str] = ["\\begin{tabular}{p{0.48\\textwidth} p{0.48\\textwidth}}"]
            for i in range(0, len(chunk), 2):
                left = chunk[i]
                if i + 1 < len(chunk):
                    right = chunk[i + 1]
                else:
                    right = "\\begin{minipage}{0.48\\textwidth}~\\end{minipage}"
                rows.append(left + " & " + right + " \\\\")
            rows.append("\\end{tabular}")
            tables.append("\n".join(rows))

        body = "\n\\clearpage\n".join(tables) if tables else "No plots."

        tex_tpl = Template(r"""
\documentclass[a4paper,10pt]{article}
\usepackage[margin=1in]{geometry}
$colors
\usepackage{pgfplots}
\usepgfplotslibrary{dateplot}
\pgfplotsset{compat=1.18}
\begin{document}
\section*{Station vs Station Comparisons}
$body
\end{document}
""")
        tex_str = tex_tpl.substitute(body=body, colors=LATEX_COLOR_PREAMBLE)
        tex_path = out.with_suffix(".tex")
        tex_path.write_text(tex_str, encoding="utf-8")
        return _compile_latex(tex_path, workdir)

    # -------- Translation table PDF (from CSVs + what Parquet actually contains) --------
    def generate_translation_pdf(
        self,
        data_root: Path,
        meta_csvs: List[Path],
        out: Path,
        engine: str = "auto",
    ) -> Path:
        """
        Scan the Parquet files under data_root to find which parameter codes exist.
        Join those codes with human-readable descriptions/units from the given CSVs.
        Produce a LaTeX PDF with a multi-page table (longtable).
        """
        data_root = Path(data_root).resolve()
        eng = _choose_engine() if engine == "auto" else engine

        # 1) Collect codes present in the parquet files
        present_codes: set[str] = set()

        # Years subfolders
        year_dirs = [p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()]
        year_dirs.sort(key=lambda p: int(p.name))

        for ydir in year_dirs:
            for f in ydir.glob("*.parquet"):
                cols: List[str] = []
                # Prefer cheap schema read with pyarrow
                try:
                    import pyarrow.parquet as pq  # type: ignore
                    pf = pq.ParquetFile(str(f))
                    cols = list(pf.schema.names)
                except Exception:
                    # Fallback: read once with pandas (heavier)
                    try:
                        df = pd.read_parquet(f, engine=eng)
                        cols = list(df.columns)
                    except Exception:
                        continue

                for c in cols:
                    n = _normalize_code(c)
                    if n and n not in {"time_utc", "station_abbr", "reference_timestamp"}:
                        present_codes.add(n)

        # 2) Load metadata from CSVs (robust column name detection)
        def _first_present(cands: List[str], cols: List[str]) -> Optional[str]:
            for c in cands:
                if c in cols:
                    return c
            return None

        meta_map: Dict[str, Dict[str, Optional[str]]] = {}  # code -> {desc, unit, short}

        for csv_path in (meta_csvs or []):
            p = Path(csv_path)
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue

            lower = {c.lower(): c for c in df.columns}
            cols_lower = list(lower.keys())

            code_col = _first_present(
                ["parameter", "code", "param", "parameter_code", "name", "key", "shortname"],
                cols_lower,
            )
            desc_col = _first_present(
                [
                    "description_en", "description", "longname_en", "long_name_en",
                    "parameter_description", "title_en", "name_en", "longname", "title", "label"
                ],
                cols_lower,
            )
            unit_col = _first_present(["unit", "unit_en", "units"], cols_lower)
            short_col = _first_present(["abbreviation", "abbr", "short", "short_name", "shortname"], cols_lower)

            if not code_col:
                continue

            for _, row in df.iterrows():
                code = _normalize_code(row[lower[code_col]])
                if not code:
                    continue
                desc = str(row[lower[desc_col]]) if desc_col and pd.notna(row[lower[desc_col]]) else None
                unit = str(row[lower[unit_col]]) if unit_col and pd.notna(row[lower[unit_col]]) else None
                short = str(row[lower[short_col]]) if short_col and pd.notna(row[lower[short_col]]) else None

                # Keep the first seen (or update if we don't have a desc yet)
                if code not in meta_map or not meta_map[code].get("desc"):
                    meta_map[code] = {"desc": desc, "unit": unit, "short": short}

        # 3) Build table rows for codes found in parquet
        rows: List[str] = []
        for code in sorted(present_codes):
            # Prefer CSV description/unit; fall back to METEO_PARAMS if available
            csv_desc = meta_map.get(code, {}).get("desc")
            csv_unit = meta_map.get(code, {}).get("unit")
            csv_short = meta_map.get(code, {}).get("short")

            mp_rec = self.params.get(code)
            mp_desc = mp_rec["desc"] if mp_rec and mp_rec.get("desc") else None
            mp_short = mp_rec["abbr"] if mp_rec and mp_rec.get("abbr") else None

            desc = csv_desc or mp_desc or ""
            short = csv_short or mp_short or ""
            unit = csv_unit or ""

            rows.append(
                f"{_latex_escape(code)} & {_latex_escape(short)} & {_latex_escape(desc)} & {_latex_escape(unit)} \\\\"
            )

        # 4) LaTeX document with longtable (multi-page)
        header = r"""
\documentclass[a4paper,10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\begin{document}
\section*{Parameter Translation (from Parquet + CSV metadata)}
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}llll@{}}
\toprule
\textbf{Code} & \textbf{Short} & \textbf{Description} & \textbf{Unit} \\
\midrule
\endfirsthead
\toprule
\textbf{Code} & \textbf{Short} & \textbf{Description} & \textbf{Unit} \\
\midrule
\endhead
\bottomrule
\endfoot
\bottomrule
\endlastfoot
"""
        footer = r"""
\end{longtable}
\end{document}
"""

        workdir = out.parent
        workdir.mkdir(parents=True, exist_ok=True)
        tex_path = out.with_suffix(".tex")

        tex_path.write_text(header + "\n".join(rows) + footer, encoding="utf-8")
        return _compile_latex(tex_path, workdir)


# -------------------- Main --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", help="Single station")
    parser.add_argument("--stations", nargs="+", help="Multiple stations")
    parser.add_argument("--parameter", help="Single parameter abbr")
    parser.add_argument("--parameters", nargs="+", help="Multiple parameters abbrs")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--year", type=int)
    parser.add_argument("--delta")
    parser.add_argument("-o", "--out", default="out/")
    parser.add_argument(
        "--traduction",
        action="store_true",
        help="Generate a PDF table translating parameter codes using CSV metadata."
    )
    parser.add_argument(
        "--meta-csvs",
        nargs="+", default=[
            "/Users/michaeljopiti/MyMeteo/.parquet_to_pdf_cache/ogd-obs_meta_parameters.csv",
            "/Users/michaeljopiti/MyMeteo/.parquet_to_pdf_cache/ogd-smn_meta_parameters.csv",
            "/Users/michaeljopiti/MyMeteo/.parquet_to_pdf_cache/ogd-smn-precip_meta_parameters.csv"
        ],
        help="Paths to metadata CSV files (e.g., ogd-obs_meta_parameters.csv ...)."
    )
    args = parser.parse_args()

    mp = MeteoParameters(METEO_PARAMS)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Translation table mode
    if args.traduction:
        default_csvs = [
            "ogd-obs_meta_parameters.csv",
            "ogd-smn_meta_parameters.csv",
            "ogd-smn-precip_meta_parameters.csv",
        ]
        meta_csvs = [Path(p) for p in (args.meta_csvs or default_csvs)]
        pdf = mp.generate_translation_pdf(
            Path(args.data_root),
            meta_csvs,
            out_dir / "parameters_translation",
            engine="auto",
        )
        print("PDF written:", pdf)
        raise SystemExit(0)

    # --- Case 1: single station + single parameter ---
    elif args.station and args.parameter:
        pdf = mp.plot_parameter(
            Path(args.data_root), args.station, args.parameter,
            out_dir / f"{args.station}_{args.parameter}",
            year=args.year, delta=args.delta
        )
        print("PDF written:", pdf)

    # --- Case 2: multiple stations + one parameter ---
    elif args.stations and args.parameter:
        pdf = mp.plot_multi_stations(
            Path(args.data_root), args.stations, args.parameter,
            out_dir / f"stations_{args.parameter}",
            year=args.year, delta=args.delta
        )
        print("PDF written:", pdf)

    # --- Case 3: multiple parameters ---
    elif args.parameters:
        if args.station:
            # Intra-station param pairs (one PDF) - (you can add a pairs function later if needed)
            # For now, call plot_parameter repeatedly or implement a pairs grid similar to earlier versions.
            # Placeholder: just plot the first parameter to keep behavior predictable.
            first = args.parameters[0]
            pdf = mp.plot_parameter(
                Path(args.data_root), args.station, first,
                out_dir / f"{args.station}_{first}",
                year=args.year, delta=args.delta
            )
            print("PDF written:", pdf)
        elif args.stations:
            # Cross-station comparisons (one global PDF)
            pdf = mp.plot_station_vs_station(
                Path(args.data_root), args.stations, args.parameters,
                out_dir / "stations_comparison",
                year=args.year, delta=args.delta
            )
            print("PDF written:", pdf)

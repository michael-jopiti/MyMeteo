#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parquet_to_pdf.py

One script, two outputs:

- <stem>_report.pdf   : data report (overview + column summary; optional preview and full table)
- <stem>_params.pdf   : parameters dictionary (official MeteoSwiss descriptions)

Behavior:
- Default (no flags): builds BOTH outputs.
- --report-only     : only the report.
- --params-only     : only the parameters PDF.

Extras:
- Merges MeteoSwiss parameter CSVs (SMN, OBS, SMN-precip) from the web or local paths.
- Robust CSV reading (cp1252/utf-8; ';' or ',').
- Uses tectonic if available; fallback to pdflatex.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from string import Template

# ----------------------------- Defaults -----------------------------

_DEFAULT_PARAM_URLS = [
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_parameters.csv",
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-obs/ogd-obs_meta_parameters.csv",
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn-precip/ogd-smn-precip_meta_parameters.csv",
]

# ----------------------------- Utils -----------------------------

def _choose_engine() -> str:
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception:
            print("ERROR: install a parquet engine: pip install pyarrow (or fastparquet)", file=sys.stderr)
            raise

def _latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    rep = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(rep.get(ch, ch) for ch in s)

def _human_bytes(n: int) -> str:
    if n is None:
        return ""
    step = 1024.0
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    for u in units:
        if abs(f) < step:
            return f"{f:.2f} {u}"
        f /= step
    return f"{f:.2f} EiB"

def _http_get(url: str, timeout: int = 30) -> bytes:
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        from urllib.request import urlopen  # type: ignore
        with urlopen(url, timeout=timeout) as resp:  # nosec B310
            return resp.read()

def _read_params_csv(path: Path) -> pd.DataFrame:
    """Try cp1252/utf-8 and ';' / ','."""
    for args in (
        {"sep": ";", "encoding": "cp1252"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "cp1252"},
        {"sep": ",", "encoding": "utf-8"},
    ):
        try:
            df = pd.read_csv(path, **args)
            df.attrs["read_args"] = args
            return df
        except Exception:
            pass
    raise RuntimeError(f"Failed to read params CSV: {path}")

def _normalize_code(code: str) -> str:
    if code is None:
        return ""
    return (str(code)
            .replace("ﬀ", "ff").replace("\ufb00", "ff")
            .replace("ﬁ", "fi").replace("\ufb01", "fi")
            .replace("ﬂ", "fl").replace("\ufb02", "fl")
            .strip().lower())

def _normalize_param_df(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    low = {c.lower().strip(): c for c in df.columns}
    code_c = (low.get("parameter_code") or low.get("parameter") or
              low.get("identifier") or list(df.columns)[0])
    desc_c = (low.get("parameter_description_en") or low.get("description_en") or
              low.get("parameter_description") or low.get("description"))
    unit_c = low.get("parameter_unit") or low.get("unit")
    intv_c = (low.get("parameter_granularity") or low.get("interval") or
              low.get("time_interval") or low.get("granularity"))

    out = pd.DataFrame({
        "parameter": df[code_c].astype(str).map(_normalize_code),
        "description": (df[desc_c].astype(str) if desc_c else pd.Series([""]*len(df))),
        "unit": (df[unit_c].astype(str) if unit_c else pd.Series([""]*len(df))),
        "interval": (df[intv_c].astype(str) if intv_c else pd.Series([""]*len(df))),
        "source": source_label,
    }).replace({np.nan: ""})
    out = out[out["parameter"] != ""].drop_duplicates(subset=["parameter"], keep="first")
    return out

def _load_params_sources(sources: Optional[List[str]], cache_dir: Path) -> List[pd.DataFrame]:
    """Load from provided paths/URLs or default URLs (cached)."""
    srcs = list(sources) if sources else list(_DEFAULT_PARAM_URLS)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: List[pd.DataFrame] = []
    for src in srcs:
        label = Path(src).name
        try:
            if src.startswith("http://") or src.startswith("https://"):
                dest = cache_dir / label
                if not dest.exists():
                    dest.write_bytes(_http_get(src))
                raw = _read_params_csv(dest)
            else:
                raw = _read_params_csv(Path(src).expanduser().resolve())
            out.append(_normalize_param_df(raw, source_label=label))
        except Exception as e:
            print(f"[warn] Skipping parameters source ({src}): {e}", file=sys.stderr)
    return out

def _merge_param_tables(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Priority merge; earlier tables win; fill blanks from later ones."""
    if not dfs:
        return pd.DataFrame(columns=["parameter", "description", "unit", "interval", "source"])
    merged: Dict[str, Dict[str, str]] = {}
    for df in dfs:
        for _, r in df.iterrows():
            code = r["parameter"]
            if code not in merged:
                merged[code] = {
                    "parameter": code,
                    "description": r.get("description", "") or "",
                    "unit": r.get("unit", "") or "",
                    "interval": r.get("interval", "") or "",
                    "source": r.get("source", "") or "",
                }
            else:
                tgt = merged[code]
                if not tgt["description"] and r.get("description"): tgt["description"] = r["description"]
                if not tgt["unit"] and r.get("unit"): tgt["unit"] = r["unit"]
                if not tgt["interval"] and r.get("interval"): tgt["interval"] = r["interval"]
    return pd.DataFrame(sorted(merged.values(), key=lambda d: d["parameter"]))

def _build_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    return {str(r["parameter"]).lower(): {
        "description": str(r.get("description", "") or ""),
        "unit": str(r.get("unit", "") or ""),
        "interval": str(r.get("interval", "") or ""),
        "source": str(r.get("source", "") or ""),
    } for _, r in df.iterrows()}

# ----------------------------- Report helpers -----------------------------

def _first_non_null(series: pd.Series) -> Optional[str]:
    for v in series:
        if pd.notna(v):
            s = str(v)
            return (s[:117] + "...") if len(s) > 120 else s
    return None

def _summarize_column(s: pd.Series) -> dict:
    name = str(s.name)
    dtype = str(s.dtype)
    n = int(s.shape[0])
    non_null = int(s.notna().sum())
    miss_pct = (1.0 - (non_null / n)) * 100.0 if n else 0.0
    try:
        nunique = int(s.nunique(dropna=True))
    except Exception:
        nunique = -1
    try:
        mem = int(s.memory_usage(deep=True))
    except Exception:
        mem = None
    ex = _first_non_null(s)

    stats = {}
    if is_numeric_dtype(s):
        ss = pd.to_numeric(s, errors="coerce")
        if ss.notna().any():
            arr = ss.to_numpy(dtype=float)
            stats = {
                "min": float(np.nanmin(arr)),
                "q25": float(np.nanpercentile(arr[~np.isnan(arr)], 25)),
                "median": float(np.nanmedian(arr)),
                "mean": float(np.nanmean(arr)),
                "q75": float(np.nanpercentile(arr[~np.isnan(arr)], 75)),
                "max": float(np.nanmax(arr)),
            }
    elif is_datetime64_any_dtype(s):
        try:
            ss = pd.to_datetime(s, errors="coerce", utc=True)
            mn, mx = ss.min(), ss.max()
            stats = {
                "min": (None if pd.isna(mn) else mn.isoformat()),
                "max": (None if pd.isna(mx) else mx.isoformat()),
            }
        except Exception:
            pass
    else:
        try:
            top = s.dropna().astype(str).value_counts().head(1)
            if len(top) > 0:
                stats = {"top": str(top.index[0]), "top_freq": int(top.iloc[0])}
        except Exception:
            pass

    return {
        "name": name,
        "dtype": dtype,
        "non_null": non_null,
        "missing_pct": miss_pct,
        "unique": nunique,
        "mem": mem,
        "example": ex,
        "stats": stats,
    }

def _schema_longtable(colsum: List[dict]) -> str:
    header = r"""
\begin{landscape}
\section*{Column summary}
\setlength{\LTpre}{6pt}
\setlength{\LTpost}{6pt}
\renewcommand{\arraystretch}{1.08}
\setlength{\tabcolsep}{2pt}
\footnotesize
\begin{longtable}{C{0.18\textwidth} C{0.10\textwidth} R{0.10\textwidth} R{0.08\textwidth} R{0.08\textwidth} C{0.10\textwidth} C{0.18\textwidth} C{0.18\textwidth}}
\toprule
\textbf{Name} & \textbf{Dtype} & \textbf{Non-null} & \textbf{Missing \%} & \textbf{Unique} & \textbf{Memory} & \textbf{Example} & \textbf{Stats}\\
\midrule
\endfirsthead
\toprule
\textbf{Name} & \textbf{Dtype} & \textbf{Non-null} & \textbf{Missing \%} & \textbf{Unique} & \textbf{Memory} & \textbf{Example} & \textbf{Stats}\\
\midrule
\endhead
\midrule
\multicolumn{8}{r}{\emph{Continued on next page}}\\
\midrule
\endfoot
\bottomrule
\endlastfoot
"""
    rows = []
    for c in colsum:
        stats = c.get("stats") or {}
        if {"min", "q25", "median", "mean", "q75", "max"} <= stats.keys():
            st = (
                f"min={stats['min']:.6g}, q25={stats['q25']:.6g}, "
                f"med={stats['median']:.6g}, mean={stats['mean']:.6g}, "
                f"q75={stats['q75']:.6g}, max={stats['max']:.6g}"
            )
        elif {"min", "max"} <= stats.keys() and isinstance(stats.get("min"), str):
            st = f"min={_latex_escape(stats['min'])}, max={_latex_escape(stats['max'])}"
        elif {"top", "top_freq"} <= stats.keys():
            st = f"top={_latex_escape(str(stats['top']))} ({stats['top_freq']})"
        else:
            st = "--"
        row = " {} & {} & {} & {:.2f} & {} & {} & {} & {} \\\\".format(
            _latex_escape(c["name"]),
            _latex_escape(c["dtype"]),
            f"{c['non_null']:,}",
            c["missing_pct"],
            ("-" if c["unique"] is None or c["unique"] < 0 else f"{c['unique']:,}"),
            ("-" if c["mem"] is None else _human_bytes(c["mem"])),
            _latex_escape("" if c["example"] is None else c["example"]),
            _latex_escape(st),
        )
        rows.append(row)
    return header + "\n".join(rows) + r"""
\end{longtable}
\end{landscape}
\clearpage
"""

def _preview_table(df: pd.DataFrame, nrows: int) -> str:
    if df.empty:
        return r"\textit{DataFrame is empty. No preview available.}"
    head = df.head(nrows).copy()
    def trunc(x):
        s = "" if pd.isna(x) else str(x)
        return s if len(s) <= 120 else (s[:117] + "...")
    try:
        head = head.map(trunc)      # pandas ≥ 2.2
    except Exception:
        head = head.applymap(trunc) # legacy
    latex_tabular = head.to_latex(index=False, escape=True)
    return r"""
\begin{landscape}
\section*{Data preview (first %d rows)}
\begingroup
\setlength{\tabcolsep}{2pt}
\footnotesize
\begin{center}
\resizebox{\textwidth}{!}{
%s
}
\end{center}
\endgroup
\end{landscape}
\clearpage
""" % (min(nrows, len(df)), latex_tabular)

def _write_full_table_longtable(df: pd.DataFrame, dest_path: Path) -> None:
    cols = list(map(str, df.columns))
    col_spec = " ".join(["l"] * len(cols))
    with dest_path.open("w", encoding="utf-8") as f:
        f.write(r"\begin{landscape}" + "\n")
        f.write(r"\section*{Full data (all rows)}" + "\n")
        f.write(r"\begingroup" + "\n")
        f.write(r"\setlength{\tabcolsep}{2pt}" + "\n")
        f.write(r"\scriptsize" + "\n")
        f.write(r"\renewcommand{\arraystretch}{1.05}" + "\n")
        f.write(fr"\begin{{longtable}}{{{col_spec}}}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(" {} \\\\".format(" & ".join(_latex_escape(c) for c in cols)) + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endfirsthead" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(" {} \\\\".format(" & ".join(_latex_escape(c) for c in cols)) + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endhead" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{" + str(len(cols)) + r"}{r}{\emph{Continued on next page}}\\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endfoot" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\endlastfoot" + "\n")
        for _, row in df.iterrows():
            cells = [("" if pd.isna(v) else _latex_escape(str(v))) for v in row.tolist()]
            f.write(" {} \\\\".format(" & ".join(cells)) + "\n")
        f.write(r"\end{longtable}" + "\n")
        f.write(r"\endgroup" + "\n")
        f.write(r"\end{landscape}" + "\n")
        f.write(r"\clearpage" + "\n")

def _tex_document_report(df: pd.DataFrame,
                         colsum: List[dict],
                         parquet_path: Path,
                         engine_used: str,
                         title: Optional[str],
                         n_preview_rows: int,
                         include_preview: bool,
                         include_full_table_basename: Optional[str],
                         dict_block: str) -> str:
    created = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_rows, n_cols = df.shape
    mem_total = int(df.memory_usage(deep=True).sum())

    schema_block = _schema_longtable(colsum)
    preview_block = (_preview_table(df, n_preview_rows) + "\n") if include_preview else ""
    full_include = f"\n\\input{{{include_full_table_basename}}}\n" if include_full_table_basename else ""

    tpl = Template(r"""
\documentclass[a4paper,10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{hyperref}
\usepackage{siunitx}
\usepackage{adjustbox}
\usepackage{pdflscape}
\usepackage{array}
\newcolumntype{L}[1]{>{\RaggedRight\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\RaggedLeft\arraybackslash}p{#1}}
\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}} % safe ragged
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}   % <-- add this
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
\newcolumntype{L}[1]{{>{\RaggedRight\arraybackslash}p{#1}}}
\newcolumntype{R}[1]{{>{\RaggedLeft\arraybackslash}p{#1}}}

\title{$safe_title}
\author{Generated by parquet\_to\_pdf.py}
\date{Generated on $created}

\begin{document}
\sloppy
\setlength{\emergencystretch}{2em}
\maketitle

\section*{Overview}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{File} & \texttt{$file_path} \\
\textbf{Rows} & $n_rows_fmt \\
\textbf{Columns} & $n_cols_fmt \\
\textbf{Memory (deep)} & $mem_total_h \\
\textbf{Engine} & $engine \\
\bottomrule
\end{tabular}

$schema_block

$dict_block

$preview_block
$full_include
\end{document}
""")
    return tpl.substitute(
        safe_title=_latex_escape(title or f"Parquet Report: {parquet_path.name}"),
        created=created,
        file_path=_latex_escape(str(parquet_path)),
        n_rows_fmt=f"{n_rows:,}",
        n_cols_fmt=f"{n_cols:,}",
        mem_total_h=_latex_escape(_human_bytes(mem_total)),
        engine=engine_used,
        schema_block=schema_block,
        dict_block=dict_block,
        preview_block=preview_block,
        full_include=full_include,
    )

# ----------------------------- Params-only PDF -----------------------------

def _tex_doc_params(columns: List[str], lookup: Dict[str, Dict[str, str]],
                    title: str, file_path: Path) -> str:
    created = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for col in columns:
        code_norm = _normalize_code(col)
        rec = lookup.get(code_norm, {})
        desc = rec.get("description", "")
        unit = rec.get("unit", "")
        interval = rec.get("interval", "")
        source = rec.get("source", "")
        if not desc and code_norm in {"time_utc", "station_abbr"}:
            desc = "Timestamp (UTC)" if code_norm == "time_utc" else "Station code"
            source = "generic"
        rows.append(" {} & {} & {} & {} & {} \\\\".format(
            _latex_escape(col),
            (_latex_escape(desc) if desc else r"\textit{(missing)}"),
            _latex_escape(unit) if unit else "-",
            _latex_escape(interval) if interval else "-",
            _latex_escape(source) if source else "-"
        ))

    tpl = Template(r"""
\documentclass[a4paper,10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}

\title{$safe_title}
\author{Generated by parquet\_to\_pdf.py}
\date{Generated on $created}

\begin{document}
\sloppy
\setlength{\emergencystretch}{2em}
\maketitle

\section*{Parameter dictionary for: \path{$file_path}}
\begingroup
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.06}
\footnotesize
\begin{longtable}{%
  >{\RaggedRight\arraybackslash}p{0.18\textwidth}%
  >{\RaggedRight\arraybackslash}p{0.50\textwidth}%
  >{\RaggedRight\arraybackslash}p{0.10\textwidth}%
  >{\RaggedRight\arraybackslash}p{0.10\textwidth}%
  >{\RaggedRight\arraybackslash}p{0.12\textwidth}%
}
\toprule
\textbf{Column code} & \textbf{Description (official)} & \textbf{Unit} & \textbf{Interval} & \textbf{Source}\\
\midrule
\endfirsthead
\toprule
\textbf{Column code} & \textbf{Description (official)} & \textbf{Unit} & \textbf{Interval} & \textbf{Source}\\
\midrule
\endhead
\midrule
\multicolumn{5}{r}{\emph{Continued on next page}}\\
\midrule
\endfoot
\bottomrule
\endlastfoot
$rows_block
\end{longtable}
\endgroup

\end{document}
""")
    return tpl.substitute(
        safe_title=_latex_escape(title),
        created=created,
        file_path=str(file_path),
        rows_block="\n".join(rows),
    )

# ----------------------------- LaTeX compile -----------------------------

def _compile_latex(tex_path: Path, out_dir: Path) -> Path:
    if shutil.which("tectonic"):
        cmd = ["tectonic", "--keep-logs", "--keep-intermediates",
               "--outdir", str(out_dir), str(tex_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            (out_dir / (tex_path.stem + ".build.log")).write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            raise RuntimeError("Tectonic compilation failed. See *.build.log")
        pdf = out_dir / (tex_path.stem + ".pdf")
        if not pdf.exists():
            raise RuntimeError("Tectonic reported success, but PDF not found.")
        return pdf
    if shutil.which("pdflatex"):
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
               "-output-directory", str(out_dir), str(tex_path)]
        for _ in range(2):
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                (out_dir / (tex_path.stem + ".build.log")).write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
                raise RuntimeError("LaTeX compilation failed. See *.build.log")
        pdf = out_dir / (tex_path.stem + ".pdf")
        if not pdf.exists():
            raise RuntimeError("pdflatex reported success, but PDF not found.")
        return pdf
    raise RuntimeError("No LaTeX engine found. Install 'tectonic' via conda-forge or add 'pdflatex' to PATH.")

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Create a data report PDF and/or a parameters dictionary PDF for a Parquet file.")
    ap.add_argument("--parquet", type=str, required=True, help="Path to the Parquet file.")
    ap.add_argument("-o", "--out-dir", type=str, default=None, help="Output directory. Default: same as input file.")
    ap.add_argument("--title", type=str, default=None, help="Custom report title.")
    ap.add_argument("--engine", type=str, choices=["pyarrow", "fastparquet", "auto"], default="auto",
                    help="Parquet engine (default: auto).")
    # report options
    ap.add_argument("--no-preview", action="store_true", help="Skip the small head() preview in the report.")
    ap.add_argument("--rows", type=int, default=15, help="Rows for the small preview table (default: 15).")
    ap.add_argument("--full-table", action="store_true", help="Include the entire DataFrame as a longtable (all rows) in the report.")
    # params source options
    ap.add_argument("--params-sources", nargs="*", default=None,
                    help="Optional list of MeteoSwiss parameter CSV sources (files or URLs). "
                         "If omitted, downloads SMN, OBS and SMN-precip into cache.")
    ap.add_argument("--params-cache-dir", type=str, default=".parquet_to_pdf_cache",
                    help="Cache directory for downloaded parameter CSVs.")
    # selection of outputs
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--report-only", action="store_true", help="Produce only the report PDF.")
    group.add_argument("--params-only", action="store_true", help="Produce only the parameters PDF.")
    args = ap.parse_args()

    in_path = Path(args.parquet).expanduser().resolve()
    if not in_path.exists():
        print(f"ERROR: File not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = _choose_engine() if args.engine == "auto" else args.engine
    # read parquet
    try:
        df = pd.read_parquet(in_path, engine=engine)
    except Exception as e1:
        if args.engine == "auto":
            alt = "fastparquet" if engine == "pyarrow" else "pyarrow"
            try:
                df = pd.read_parquet(in_path, engine=alt)
                engine = alt
            except Exception as e2:
                print(f"ERROR: failed to read parquet with both engines.\n{e1}\n{e2}", file=sys.stderr)
                sys.exit(2)
        else:
            print(f"ERROR: failed to read parquet with engine '{engine}': {e1}", file=sys.stderr)
            sys.exit(2)

    # Load & merge MeteoSwiss parameter dictionaries
    dfs = _load_params_sources(args.params_sources, Path(args.params_cache_dir))
    merged = _merge_param_tables(dfs)
    lookup = _build_lookup(merged)

    made_any = False

    # ---------------- Report PDF ----------------
    if not args.params_only:
        colsum = []
        for c in df.columns:
            try:
                colsum.append(_summarize_column(df[c]))
            except Exception:
                colsum.append({
                    "name": str(c), "dtype": str(df[c].dtype),
                    "non_null": int(df[c].notna().sum()),
                    "missing_pct": float((df[c].isna().sum() / len(df)) * 100.0) if len(df) else 0.0,
                    "unique": None, "mem": None, "example": None, "stats": {},
                })

        # build dictionary block inside report (same layout as earlier landscape dict)
        # Here we reuse the portrait dictionary columns but frame them as a landscape longtable:
        dict_rows = []
        for col in map(str, df.columns):
            code_norm = _normalize_code(col)
            rec = lookup.get(code_norm, {})
            desc = rec.get("description", "")
            unit = rec.get("unit", "")
            interval = rec.get("interval", "")
            source = rec.get("source", "")
            if not desc and code_norm in {"time_utc", "station_abbr"}:
                desc = "Timestamp (UTC)" if code_norm == "time_utc" else "Station code"
                source = "generic"
            meta = []
            if unit: meta.append(f"unit={_latex_escape(unit)}")
            if interval: meta.append(f"interval={_latex_escape(interval)}")
            if source: meta.append(_latex_escape(source))
            meta_cell = ", ".join(meta) if meta else "-"
            dict_rows.append(" {} & {} & {} \\\\".format(
                _latex_escape(col),
                (_latex_escape(desc) if desc else r"\textit{(missing)}"),
                meta_cell
            ))
        dict_block = r"""
\begin{landscape}
\section*{Parameter dictionary (official MeteoSwiss)}
\begingroup
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.06}
\footnotesize
\begin{longtable}{P{0.22\textwidth} P{0.64\textwidth} P{0.14\textwidth}}
\toprule
\textbf{Column code} & \textbf{Description (official)} & \textbf{Source / Unit / Interval}\\
\midrule
\endfirsthead
\toprule
\textbf{Column code} & \textbf{Description (official)} & \textbf{Source / Unit / Interval}\\
\midrule
\endhead
\midrule
\multicolumn{3}{r}{\emph{Continued on next page}}\\
\midrule
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(dict_rows) + r"""
\end{longtable}
\endgroup
\end{landscape}
\clearpage
"""

        include_basename = None
        if args.full_table:
            include_path = out_dir / (in_path.stem + "_fulltable.tex")
            _write_full_table_longtable(df, include_path)
            include_basename = include_path.name
            if len(df) > 50_000:
                print(f"[warn] You are including {len(df):,} rows. LaTeX compile and PDF size may be large.", file=sys.stderr)

        tex_report = _tex_document_report(
            df=df,
            colsum=colsum,
            parquet_path=in_path,
            engine_used=engine,
            title=args.title,
            n_preview_rows=args.rows,
            include_preview=not args.no_preview,
            include_full_table_basename=include_basename,
            dict_block=dict_block,
        )
        report_tex = out_dir / f"{in_path.stem}_report.tex"
        report_tex.write_text(tex_report, encoding="utf-8")
        try:
            report_pdf = _compile_latex(report_tex, out_dir)
            print(f"[OK] Report PDF written to: {report_pdf}")
            made_any = True
        except Exception as e:
            print(f"ERROR during LaTeX compilation (report): {e}", file=sys.stderr)
            print(f"LaTeX source: {report_tex}", file=sys.stderr)

    # ---------------- Params-only PDF ----------------
    if not args.report_only:
        columns = [str(c) for c in df.columns]
        title = (args.title or f"MeteoSwiss Parameters --- {in_path.name}")
        tex_params = _tex_doc_params(columns, lookup, title, in_path)
        params_tex = out_dir / f"{in_path.stem}_params.tex"
        params_tex.write_text(tex_params, encoding="utf-8")
        try:
            params_pdf = _compile_latex(params_tex, out_dir)
            print(f"[OK] Parameters PDF written to: {params_pdf}")
            made_any = True
        except Exception as e:
            print(f"ERROR during LaTeX compilation (params): {e}", file=sys.stderr)
            print(f"LaTeX source: {params_tex}", file=sys.stderr)

    if not made_any:
        sys.exit(3)

    # Tidy aux for both names
    for stem in [in_path.stem + "_report", in_path.stem + "_params", in_path.stem + "_fulltable"]:
        for ext in (".aux", ".log", ".out", ".toc"):
            p = out_dir / (stem + ext)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

if __name__ == "__main__":
    main()

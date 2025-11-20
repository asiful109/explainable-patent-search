# bootstrap_two_vs_one_by_index.py
# -----------------------------------------------------------------------------
# Bootstrap significance test by ROW INDEX:
# For each two-section Excel in --two_folder:
#   1) Parse its two section names from filename like "title-description_k....xlsx".
#   2) For each section, find the SINGLE matching one-section Excel in --one_folder
#      using the pattern "<section>-k....xlsx" (split by '-').
#   3) Read the same Excel-style row range [--start, --end] (1-based, inclusive)
#      from all three files, keeping the 'ap' column (case-insensitive).
#   4) Assume all three DataFrames have IDENTICAL row counts. Sample indices from
#      the full length n and compare mean AP:
#         mean_two >= max(mean_s1, mean_s2)
#      across --samples bootstrap resamples of size --size (with replacement).
# Prints, per two-section file, the percent of samples where the condition holds.
# Also prints the bootstrap average of the sampled mean APs and their std dev.
# Additionally:
#   - Prints % where two-section >= S1 and >= S2 separately.
#   - Saves a simplified Excel "bootstrap_results.xlsx" with percentages and avg±std.
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def read_excel_range(path: Path,
                     sheet: Optional[str | int],
                     row_start: int,
                     row_end: int,
                     cols_needed: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    """Read Excel, clamp to Excel-style 1-based inclusive [row_start, row_end],
    optionally keep only needed columns (case-insensitive), and reset positional index 0..N-1."""
    try:
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
        if isinstance(df, dict):
            df = next(iter(df.values()))
    except Exception as e:
        print(f"[skip] {path.name}: cannot read excel ({e})", file=sys.stderr)
        return None

    if df is None or len(df) == 0:
        print(f"[skip] {path.name}: empty sheet", file=sys.stderr)
        return None

    # Excel-style 1-based inclusive -> iloc slice [i0:i1)
    rs, re = (row_start, row_end) if row_start <= row_end else (row_end, row_start)
    n = len(df)
    i0 = max(rs - 1, 0)
    i1 = min(re, n)
    if i0 >= i1:
        print(f"[skip] {path.name}: empty range after clamping", file=sys.stderr)
        return None

    df = df.iloc[i0:i1].copy()

    if cols_needed:
        cmap = {str(c).lower(): c for c in df.columns}
        missing = [c for c in cols_needed if c.lower() not in cmap]
        if missing:
            print(f"[skip] {path.name}: missing columns {missing}", file=sys.stderr)
            return None
        # normalize the column names to the requested ones
        rename_map = {}
        for need in cols_needed:
            lower = need.lower()
            rename_map[cmap[lower]] = need
        df = df.rename(columns=rename_map)

    return df.reset_index(drop=True)


def parse_two_sections_from_filename(fname: str) -> Optional[Tuple[str, str]]:
    """
    Parse two sections from a two-section filename.
    Examples:
      'title-description_k51.23.xlsx'  -> ('title','description')
      'abstract-cpc_k94.18.xlsm'       -> ('abstract','cpc')
    """
    stem = Path(fname).stem
    # Split off trailing _k... if present
    name_part = stem.split('_', 1)[0]
    parts = name_part.split('-')
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0].strip(), parts[1].strip()
    return None


def find_one_section_file(one_folder: Path, target_section: str) -> Optional[Path]:
    """
    Find the SINGLE one-section Excel file for a given section.
    The expected pattern is '<section>-kXXXX.xlsx' (split by '-').
    Returns the first matching file found across supported extensions.
    """
    for ext in ("*.xlsx", "*.xlsm", "*.xls"):
        for f in one_folder.glob(ext):
            stem = f.stem  # e.g., 'title-k114.46'
            # section name is the part before the first '-'
            name_part = stem.split('-', 1)[0].strip()
            if name_part == target_section:
                return f
    return None


def bootstrap_compare_by_index(two_file: Path,
                               sheet: Optional[str | int],
                               row_start: int,
                               row_end: int,
                               one_folder: Path,
                               samples: int,
                               size: int,
                               rng: np.random.Generator):
    """Run bootstrap by ROW INDEX for a single two-section file.

    Returns:
        (
          two_filename,
          successes, valid, s1_file_name, s2_file_name,
          stats_dict   # averages, std devs, and per-section win stats
        )
    """
    pair = parse_two_sections_from_filename(two_file.name)
    if not pair:
        print(f"[skip] {two_file.name}: cannot parse two sections", file=sys.stderr)
        return (two_file.name, 0, 0, "", "", {})

    s1, s2 = pair

    # Load two-section AP column
    df_two = read_excel_range(two_file, sheet, row_start, row_end, cols_needed=["ap"])
    if df_two is None or df_two.empty:
        print(f"[skip] {two_file.name}: empty two-section data", file=sys.stderr)
        return (two_file.name, 0, 0, "", "", {})
    df_two["ap"] = pd.to_numeric(df_two["ap"], errors="coerce")

    # Find corresponding one-section files using '<section>-k...' pattern
    f1 = find_one_section_file(one_folder, s1)
    f2 = find_one_section_file(one_folder, s2)
    if f1 is None or f2 is None:
        print(f"[skip] {two_file.name}: missing one-section file(s) for '{s1}' or '{s2}'", file=sys.stderr)
        return (two_file.name, 0, 0, f1.name if f1 else "", f2.name if f2 else "", {})

    # Load one-section AP columns
    df_s1 = read_excel_range(f1, sheet, row_start, row_end, cols_needed=["ap"])
    df_s2 = read_excel_range(f2, sheet, row_start, row_end, cols_needed=["ap"])
    if df_s1 is None or df_s1.empty or df_s2 is None or df_s2.empty:
        print(f"[skip] {two_file.name}: empty one-section data for '{s1}' or '{s2}'", file=sys.stderr)
        return (two_file.name, 0, 0, f1.name if f1 else "", f2.name if f2 else "", {})

    df_s1["ap"] = pd.to_numeric(df_s1["ap"], errors="coerce")
    df_s2["ap"] = pd.to_numeric(df_s2["ap"], errors="coerce")

    # Assert identical lengths
    n_two, n_s1, n_s2 = len(df_two), len(df_s1), len(df_s2)
    if not (n_two == n_s1 == n_s2):
        print(f"[error] {two_file.name}: row counts differ (two:{n_two}, {s1}:{n_s1}, {s2}:{n_s2}). "
              f"Your assumption says they should be equal. Please fix the inputs.", file=sys.stderr)
        return (two_file.name, 0, 0, f1.name if f1 else "", f2.name if f2 else "", {})

    n = n_two
    if n <= 0:
        print(f"[skip] {two_file.name}: no rows to sample.", file=sys.stderr)
        return (two_file.name, 0, 0, f1.name if f1 else "", f2.name if f2 else "", {})

    successes = 0
    valid = 0

    # per-section comparison counters
    hits_vs_s1 = 0
    valid_vs_s1 = 0
    hits_vs_s2 = 0
    valid_vs_s2 = 0

    # Collect sampled mean APs for stats
    means_two = []
    means_s1 = []
    means_s2 = []
    means_best = []

    for _ in range(samples):
        # Sample indices with replacement from [0, n)
        idxs = rng.integers(0, n, size=size, endpoint=False)

        # Two-section mean over sampled indices
        ap_two = df_two["ap"].iloc[idxs].dropna()
        if ap_two.empty:
            continue
        mean_two = float(ap_two.mean())

        # One-section means over the SAME sampled indices
        ap1 = df_s1["ap"].iloc[idxs].dropna()
        ap2 = df_s2["ap"].iloc[idxs].dropna()
        mean_s1 = float(ap1.mean()) if not ap1.empty else float("nan")
        mean_s2 = float(ap2.mean()) if not ap2.empty else float("nan")

        # If both are NaN, skip this trial
        if np.isnan(mean_s1) and np.isnan(mean_s2):
            continue

        # Original success condition vs best single
        best_single = np.nanmax([mean_s1, mean_s2])
        valid += 1
        if mean_two >= best_single:
            successes += 1

        # Individual comparisons (skip if that single is NaN)
        if not np.isnan(mean_s1):
            valid_vs_s1 += 1
            if mean_two >= mean_s1:
                hits_vs_s1 += 1
        if not np.isnan(mean_s2):
            valid_vs_s2 += 1
            if mean_two >= mean_s2:
                hits_vs_s2 += 1

        # record for stats
        means_two.append(mean_two)
        means_s1.append(mean_s1)
        means_s2.append(mean_s2)
        means_best.append(best_single)

    # Prepare stats (use nanmean/nanstd to be robust to any NaNs in singles)
    def _avg_std(arr):
        if len(arr) == 0:
            return float("nan"), float("nan")
        a = np.array(arr, dtype=float)
        if np.sum(~np.isnan(a)) > 1:
            return float(np.nanmean(a)), float(np.nanstd(a, ddof=1))
        return float(np.nanmean(a)), float("nan")

    avg_two, std_two = _avg_std(means_two)
    avg_s1, std_s1 = _avg_std(means_s1)
    avg_s2, std_s2 = _avg_std(means_s2)
    avg_best, std_best = _avg_std(means_best)

    stats = {
        "avg_two": avg_two, "std_two": std_two,
        "avg_s1": avg_s1,   "std_s1": std_s1,
        "avg_s2": avg_s2,   "std_s2": std_s2,
        "avg_best": avg_best, "std_best": std_best,
        "trials_counted": valid,
        # per-section win stats
        "hits_vs_s1": hits_vs_s1, "valid_vs_s1": valid_vs_s1,
        "hits_vs_s2": hits_vs_s2, "valid_vs_s2": valid_vs_s2,
    }

    return (two_file.name, successes, valid, f1.name if f1 else "", f2.name if f2 else "", stats)


def main():
    p = argparse.ArgumentParser(
        description="Bootstrap by row index: percent of samples where two-section mean AP >= best of its two one-sections."
    )
    p.add_argument("--two_folder", required=True, help="Folder with two-section Excel files")
    p.add_argument("--one_folder", required=True, help="Folder with one-section Excel files")
    p.add_argument("--start", type=int, required=True, help="Excel-style row start (1-based, inclusive)")
    p.add_argument("--end", type=int, required=True, help="Excel-style row end (1-based, inclusive)")
    p.add_argument("--samples", type=int, default=1000, help="Number of bootstrap resamples X")
    p.add_argument("--size", type=int, default=2500, help="Each resample size (number of indices)")
    p.add_argument("--sheet", default=None, help="Sheet name or integer index (default: first sheet)")
    p.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility")
    args = p.parse_args()

    # Parse sheet index if provided as int
    sheet_arg = args.sheet
    if sheet_arg is not None:
        try:
            sheet_arg = int(sheet_arg)
        except ValueError:
            pass

    two_dir = Path(args.two_folder)
    one_dir = Path(args.one_folder)
    if not two_dir.is_dir() or not one_dir.is_dir():
        print("Invalid folder(s).", file=sys.stderr)
        sys.exit(1)

    # Collect two-section files
    two_files: list[Path] = []
    for ext in ("*.xlsx", "*.xlsm", "*.xls"):
        two_files.extend(two_dir.glob(ext))
    if not two_files:
        print("No two-section Excel files found.", file=sys.stderr)
        sys.exit(0)

    rng = np.random.default_rng(args.seed)

    # collect simplified results for Excel
    results = []

    print("Excel name\t% of samples (two_section ≥ best of its two singles)\tcount/valid\t[s1_file]\t[s2_file]")
    for f in sorted(two_files):
        name, hits, trials, s1_file, s2_file, stats = bootstrap_compare_by_index(
            f, sheet_arg, args.start, args.end, one_dir, args.samples, args.size, rng
        )
        pct = (hits / trials * 100.0) if trials > 0 else 0.0
        exp_name = name.split("_")[0]

        # Keep your original output line (unchanged)
        print(f"{exp_name}\t{pct:.2f}%\t({hits}/{trials})")

        if stats:
            # per-section win rates
            p_s1 = (stats["hits_vs_s1"] / stats["valid_vs_s1"] * 100.0) if stats["valid_vs_s1"] > 0 else 0.0
            p_s2 = (stats["hits_vs_s2"] / stats["valid_vs_s2"] * 100.0) if stats["valid_vs_s2"] > 0 else 0.0
            print(
                f"\tbetter_vs_S1={p_s1:.2f}%\t({stats['hits_vs_s1']}/{stats['valid_vs_s1']})"
                f"\tbetter_vs_S2={p_s2:.2f}%\t({stats['hits_vs_s2']}/{stats['valid_vs_s2']})"
            )

            def fmt(x):
                return "nan" if x != x else f"{x:.4f}"

            # averages and std devs over the bootstrap trials
            print(
                f"\tavgAP_two={fmt(stats['avg_two'])}±{fmt(stats['std_two'])}"
                f"\tavgAP_s1={fmt(stats['avg_s1'])}±{fmt(stats['std_s1'])}"
                f"\tavgAP_s2={fmt(stats['avg_s2'])}±{fmt(stats['std_s2'])}"
                f"\tavgAP_best={fmt(stats['avg_best'])}±{fmt(stats['std_best'])}"
            )

            # ---- simplified Excel record (no hit counts; avg±std in a single cell) ----
            results.append({
                "two_file": exp_name,
                "pct_two_ge_best": pct,
                "pct_ge_S1": p_s1,
                "pct_ge_S2": p_s2,
                "ap_two": f"{fmt(stats['avg_two'])}±{fmt(stats['std_two'])}",
                "ap_s1":  f"{fmt(stats['avg_s1'])}±{fmt(stats['std_s1'])}",
                "ap_s2":  f"{fmt(stats['avg_s2'])}±{fmt(stats['std_s2'])}",
                "ap_best": f"{fmt(stats['avg_best'])}±{fmt(stats['std_best'])}",
            })

    # ---- Save all results to Excel ----
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_excel("bootstrap_two_section_results.xlsx", index=False)
        print("\nSaved results to: bootstrap_results.xlsx")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()

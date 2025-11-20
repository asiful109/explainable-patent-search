# long_vs_short_bootstrap_by_index.py
# -----------------------------------------------------------------------------
# For each section in a one-section results folder where filenames look like:
#   "<section>-<section>_k<NUMBER>.xlsx"   e.g., "abstract-abstract_k24.44.xlsx"
# run a bootstrap test:
#   - LONG file = larger k within the section
#   - SHORT file = smaller k within the section
#   - For X trials, sample 'size' row indices with replacement,
#     compare mean(AP_long) vs mean(AP_short) on those exact rows.
# Prints, per section: % of samples with LONG >= SHORT, counts, and filenames.
# Additionally prints per-trial mean AP summary (avg±std) and complementary %.
# ALSO saves a simplified Excel: long_short_bootstrap_results.xlsx
#   Columns: section, k_long, k_short, pct_long_ge_short, pct_short_ge_long, ap_long, ap_short
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path
import re
import sys
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


def read_excel_range(path: Path,
                     sheet: Optional[str | int],
                     row_start: int,
                     row_end: int,
                     cols_needed: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    """
    Read Excel file and return rows in Excel-style [row_start, row_end] inclusive, 1-based.
    Keep only 'cols_needed' (case-insensitive), and reset index to 0..N-1.
    """
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

    # Excel-style to iloc slice
    rs, re_ = (row_start, row_end) if row_start <= row_end else (row_end, row_start)
    n = len(df)
    i0 = max(rs - 1, 0)
    i1 = min(re_, n)
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
        df = df.rename(columns={cmap[c.lower()]: c for c in cols_needed})

    return df.reset_index(drop=True)


def parse_section_and_k(fname: str) -> Optional[Tuple[str, float]]:
    """
    Parse section and k from a filename.

    Expected primary pattern:
      "<section>-<section>_k<NUMBER>.<ext>"
      e.g., "abstract-abstract_k24.43996.xlsx" -> ("abstract", 24.43996)

    Also supports:
      "<section>_k<NUMBER>.<ext>" or "<section>-k<NUMBER>.<ext>"
    """
    stem = Path(fname).stem  # remove extension

    # Split off the _k part
    if "_k" not in stem:
        return None
    name_part, k_part = stem.split("_k", 1)

    # Extract numeric k (consume leading digits, dot allowed)
    m = re.match(r'^([0-9]+(?:\.[0-9]+)?)', k_part)
    if not m:
        return None
    try:
        k_val = float(m.group(1))
    except ValueError:
        return None

    # Primary pattern: "<section>-<section>"
    if "-" in name_part:
        left, right = name_part.split("-", 1)
        left = left.strip()
        right = right.strip()
        # If both halves are identical, the section is that token
        if left and right and left == right:
            return left, k_val
        # Fallbacks
        if left:
            return left, k_val
        if right:
            return right, k_val

    # Fallback
    name_part = name_part.strip()
    if name_part:
        return name_part, k_val

    return None


def collect_section_files(folder: Path) -> Dict[str, List[Tuple[Path, float]]]:
    """
    Scan the folder for Excel files and group them by section.
    Returns: { section: [(path, k), ...] }
    """
    groups: Dict[str, List[Tuple[Path, float]]] = {}
    for ext in ("*.xlsx", "*.xlsm", "*.xls"):
        for f in folder.glob(ext):
            parsed = parse_section_and_k(f.name)
            if not parsed:
                continue
            section, k = parsed
            groups.setdefault(section, []).append((f, k))
    return groups


def pick_long_short(files_with_k: List[Tuple[Path, float]]) -> Optional[Tuple[Path, Path, float, float]]:
    """
    Given a list [(path, k), ...] for one section, pick:
      - long_path with max k
      - short_path with min k
    Returns (long_path, short_path, long_k, short_k)
    """
    if not files_with_k:
        return None
    files_sorted = sorted(files_with_k, key=lambda x: x[1])
    short_path, short_k = files_sorted[0]
    long_path, long_k = files_sorted[-1]
    if long_path == short_path:
        # Only one file in the section; cannot compare long vs short
        return None
    return long_path, short_path, long_k, short_k


def bootstrap_long_vs_short(long_file: Path,
                            short_file: Path,
                            sheet: Optional[str | int],
                            row_start: int,
                            row_end: int,
                            samples: int,
                            size: int,
                            rng: np.random.Generator) -> Tuple[int, int, dict]:
    """
    Bootstrap by ROW INDEX for a long-vs-short pair.

    Returns:
        (successes_long_ge_short, valid_trials, stats_dict)

    stats_dict contains:
        - 'means_long', 'means_short'  (lists of per-trial means)
    """
    df_long = read_excel_range(long_file, sheet, row_start, row_end, cols_needed=["ap"])
    df_short = read_excel_range(short_file, sheet, row_start, row_end, cols_needed=["ap"])
    if df_long is None or df_short is None or df_long.empty or df_short.empty:
        return 0, 0, {"means_long": [], "means_short": []}

    df_long["ap"] = pd.to_numeric(df_long["ap"], errors="coerce")
    df_short["ap"] = pd.to_numeric(df_short["ap"], errors="coerce")

    # Assumption: same number of rows
    n_long = len(df_long)
    n_short = len(df_short)
    if n_long != n_short:
        print(f"[error] Row counts differ: {long_file.name}({n_long}) vs {short_file.name}({n_short}). "
              f"They should be equal for index-based comparison.", file=sys.stderr)
    # Proceed only on the overlapping length
    n = min(n_long, n_short)
    if n <= 0:
        return 0, 0, {"means_long": [], "means_short": []}

    successes = 0
    valid = 0

    means_long = []
    means_short = []

    for _ in range(samples):
        idxs = rng.integers(0, n, size=size, endpoint=False)

        ap_long = df_long["ap"].iloc[idxs].dropna()
        ap_short = df_short["ap"].iloc[idxs].dropna()

        # If both empty after NaN drop, skip trial
        if ap_long.empty and ap_short.empty:
            continue

        mean_long = float(ap_long.mean()) if not ap_long.empty else float("nan")
        mean_short = float(ap_short.mean()) if not ap_short.empty else float("nan")

        # If both NaN, skip
        if np.isnan(mean_long) and np.isnan(mean_short):
            continue

        valid += 1
        if (not np.isnan(mean_long)) and (np.isnan(mean_short) or mean_long >= mean_short):
            successes += 1

        means_long.append(mean_long)
        means_short.append(mean_short)

    stats = {"means_long": means_long, "means_short": means_short}
    return successes, valid, stats


def _avg_std(arr: List[float]) -> Tuple[float, float]:
    """Return (avg, std) with NaN safety and sample std (ddof=1) when possible."""
    if not arr:
        return float("nan"), float("nan")
    a = np.array(arr, dtype=float)
    mask = ~np.isnan(a)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    if mask.sum() == 1:
        return float(np.nanmean(a)), float("nan")
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1))


def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap test: LONG (larger k) vs SHORT (smaller k) AP for one-section files "
                    "named like '<section>-<section>_k<NUMBER>.xlsx'."
    )
    ap.add_argument("--folder", required=True,
                    help="Folder with one-section Excel files (e.g., all_results/result_one_section_stat_sig_test)")
    ap.add_argument("--start", type=int, required=True,
                    help="Excel-style row start (1-based, inclusive)")
    ap.add_argument("--end", type=int, required=True,
                    help="Excel-style row end (1-based, inclusive)")
    ap.add_argument("--samples", type=int, default=1000,
                    help="Number of bootstrap resamples X")
    ap.add_argument("--size", type=int, default=2500,
                    help="Each resample size (number of row indices)")
    ap.add_argument("--sheet", default=None,
                    help="Sheet name or integer index (default: first sheet)")
    ap.add_argument("--seed", type=int, default=13,
                    help="Random seed for reproducibility")
    args = ap.parse_args()

    # Parse sheet to int if possible
    sheet_arg = args.sheet
    if sheet_arg is not None:
        try:
            sheet_arg = int(sheet_arg)
        except ValueError:
            pass

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    groups = collect_section_files(folder)
    if not groups:
        print("No matching Excel files found.", file=sys.stderr)
        sys.exit(0)

    rng = np.random.default_rng(args.seed)

    # for saving simplified results
    results = []

    print("section\t% of samples (long ≥ short)\tcount/valid\t[long_file]\t[short_file]\t[k_long]\t[k_short]")

    for section in sorted(groups.keys()):
        pair = pick_long_short(groups[section])
        if not pair:
            # Either zero or one file for this section — cannot compare
            continue
        long_path, short_path, k_long, k_short = pair

        successes, valid, stats = bootstrap_long_vs_short(
            long_path, short_path, sheet_arg, args.start, args.end, args.samples, args.size, rng
        )
        pct_long_ge_short = (successes / valid * 100.0) if valid > 0 else 0.0
        pct_short_ge_long = (100.0 - pct_long_ge_short) if valid > 0 else 0.0

        # Keep original first line
        print(f"{section}\t{pct_long_ge_short:.2f}%\t({successes}/{valid})\t"
              f"{long_path.name}\t{short_path.name}\t{k_long:.6f}\t{k_short:.6f}")

        # Complementary percentage
        print(f"\tshort ≥ long = {pct_short_ge_long:.2f}%")

        # Average mean AP ± std over the bootstrap trials
        avg_long, std_long = _avg_std(stats.get("means_long", []))
        avg_short, std_short = _avg_std(stats.get("means_short", []))

        def fmt(x: float) -> str:
            return "nan" if x != x else f"{x:.4f}"

        print(
            f"\tavgAP_long={fmt(avg_long)}±{fmt(std_long)}"
            f"\tavgAP_short={fmt(avg_short)}±{fmt(std_short)}"
        )

        # ---- Save row for Excel (simplified; percentages only, avg±std single cell) ----
        results.append({
            "section": section,
            "k_long": k_long,
            "k_short": k_short,
            "pct_long_ge_short": pct_long_ge_short,
            "pct_short_ge_long": pct_short_ge_long,
            "ap_long": f"{fmt(avg_long)}±{fmt(std_long)}",
            "ap_short": f"{fmt(avg_short)}±{fmt(std_short)}",
        })

    # ---- Write Excel file ----
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_excel("bootstrap_one_section_results.xlsx", index=False)
        print("\nSaved results to: long_short_bootstrap_results.xlsx")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()

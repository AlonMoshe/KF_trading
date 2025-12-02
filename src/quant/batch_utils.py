import os
import pandas as pd
import numpy as np

from src.data.data_utils import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    parse_filename_info, list_available_raw_files, list_available_processed_files,
    process_raw_to_processed
)
from src.labeling.label_utils import label_turning_points
from src.quant.quant_utils import add_slope_quantiles, conditional_probabilities_table
from src.quant.analysis_utils import compute_probability_curve, compute_time_to_extremum, compute_distance_curve


# =======================================================
# 2.4A) Batch RAW -> PROCESSED
# =======================================================

def process_all_raw_files(interval=None, overwrite=False, verbose=True):
    """
    Process every raw CSV in RAW_DATA_DIR into a processed CSV in PROCESSED_DATA_DIR,
    skipping ones that already exist unless overwrite=True.

    Parameters
    ----------
    interval : int or None
        If provided, only process raw files with this interval (seconds).
    overwrite : bool
        If True, reprocess and overwrite processed files.
    verbose : bool

    Returns
    -------
    list of str
        Processed file paths created/updated.
    """
    created = []
    raw_files = list_available_raw_files(symbol=None)

    for rf in raw_files:
        info = parse_filename_info(rf)
        if interval is not None and info["interval"] != interval:
            continue

        symbol = info["symbol"]
        start  = info["start_date"].isoformat()
        end    = info["end_date"].isoformat()
        inter  = info["interval"]

        proc_name = f"{symbol}_{start}_{end}_{inter}.csv"
        proc_path = os.path.join(PROCESSED_DATA_DIR, proc_name)

        if os.path.exists(proc_path) and not overwrite:
            if verbose:
                print(f"[skip] processed exists: {proc_name}")
            continue

        if verbose:
            print(f"[process] {rf} -> {proc_name}")

        df_kf = process_raw_to_processed(
            symbol=symbol,
            start=start,
            end=end,
            interval=inter,
            overwrite=overwrite
        )
        created.append(proc_path)

    return created


# =======================================================
# Helper: load processed df directly from a path
# =======================================================

def load_processed_df_from_path(file_path):
    """
    Load a processed CSV (with tz-aware datetime index saved as index_col=0).

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        # your convention is US/Eastern / local timezone
        df.index = df.index.tz_localize("US/Eastern")

    return df.sort_index()


# =======================================================
# 2.4B) Analyze ALL processed files
# =======================================================

def analyze_all_processed_files(
    symbols=None,
    date_list=None,
    interval=None,
    L=20,
    use_roll_quantiles=True,
    min_points=50,
    q_vals=(0.90, 0.95, 0.97, 0.99),
    q_curve_min=0.80,
    q_curve_max=0.99,
    q_curve_points=80,
    verbose=True
):
    """
    Run Milestone 2.2 + key 2.3 summaries on every processed file,
    optionally filtered by symbols, dates, or interval.

    Parameters
    ----------
    symbols : list[str] or None
        If None, analyze all symbols present in processed directory.
    date_list : list[str 'YYYY-MM-DD'] or None
        If provided, only analyze files whose date-range intersects this list.
        (Intersection test is inclusive.)
    interval : int or None
        If provided, analyze only files with this interval.
    L : int
        Turning point window.
    use_roll_quantiles : bool
        True => use slope_q_roll, False => slope_q_full.
    min_points : int
        Rolling ECDF warmup.
    q_vals : tuple
        Standard thresholds for the Milestone 2.2 table.
    q_curve_* : parameters for high-res probability curve summary.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        One summary row per processed file.
    """
    proc_files = list_available_processed_files(symbol=None)
    rows = []

    # normalize symbols filter
    if symbols is not None:
        symbols = [s.upper() for s in symbols]

    # normalize date_list filter
    if date_list is not None:
        date_list = set(pd.to_datetime(date_list).date)

    for pf in proc_files:
        info = parse_filename_info(pf)

        if interval is not None and info["interval"] != interval:
            continue
        if symbols is not None and info["symbol"] not in symbols:
            continue

        # date-range intersection filter
        if date_list is not None:
            start_d = info["start_date"]
            end_d   = info["end_date"]
            covered_days = set(pd.date_range(start_d, end_d, freq="D").date)
            if covered_days.isdisjoint(date_list):
                continue

        file_path = os.path.join(PROCESSED_DATA_DIR, pf)

        if verbose:
            print(f"[analyze] {pf}")

        try:
            df = load_processed_df_from_path(file_path)

            # 1) turning points
            df = label_turning_points(df, column="KF_level_adapt", L=L)

            # 2) slope quantiles
            df = add_slope_quantiles(df, slope_col="KF_slope_adapt", min_points=min_points)
            q_col = "slope_q_roll" if use_roll_quantiles else "slope_q_full"

            # 3) Milestone 2.2 table
            tab = conditional_probabilities_table(df, q_col=q_col, tp_col="TP_label", q_vals=q_vals)

            # 4) High-res probability curve + summary
            curve_df = compute_probability_curve(
                df, q_col=q_col, tp_col="TP_label",
                q_min=q_curve_min, q_max=q_curve_max, n_points=q_curve_points
            )

            # peak summaries
            peak_max = curve_df["P_max_given_high"].max()
            peak_min = curve_df["P_min_given_low"].max()

            peak_max_Q = curve_df.loc[curve_df["P_max_given_high"].idxmax(), "Threshold"] \
                         if not np.isnan(peak_max) else np.nan
            peak_min_Q = curve_df.loc[curve_df["P_min_given_low"].idxmax(), "Threshold"] \
                         if not np.isnan(peak_min) else np.nan

            # 5) Distance summaries (2.3 timing)
            df_dist = compute_time_to_extremum(df, tp_col="TP_label")
            dist_max_curve = compute_distance_curve(df_dist, "time_to_next_max", q_col=q_col, n_bins=20)
            dist_min_curve = compute_distance_curve(df_dist, "time_to_next_min", q_col=q_col, n_bins=20)

            # take median distance around high/low tails as compact stats
            # (use nearest bin center to 0.95 and 0.05)
            def _nearest_bin_median(curve, target_q):
                idx = (curve["q_bin_center"] - target_q).abs().idxmin()
                return curve.loc[idx, "median_dist"]

            med_dist_next_max_q95 = _nearest_bin_median(dist_max_curve, 0.95)
            med_dist_next_min_q05 = _nearest_bin_median(dist_min_curve, 0.05)

            # build summary row
            row = build_summary_row(
                pf, info, df, tab,
                q_col=q_col,
                peak_max=peak_max, peak_max_Q=peak_max_Q,
                peak_min=peak_min, peak_min_Q=peak_min_Q,
                med_dist_next_max_q95=med_dist_next_max_q95,
                med_dist_next_min_q05=med_dist_next_min_q05
            )
            rows.append(row)

        except Exception as e:
            if verbose:
                print(f"[error] {pf}: {e}")
            continue

    return pd.DataFrame(rows)


# =======================================================
# 2.4C) Build one summary row per processed file
# =======================================================

def build_summary_row(
    filename, info, df, tab,
    q_col,
    peak_max, peak_max_Q,
    peak_min, peak_min_Q,
    med_dist_next_max_q95,
    med_dist_next_min_q05
):
    """
    Convert Milestone 2.x results into a compact per-file summary row.
    """
    row = {
        "filename": filename,
        "symbol": info["symbol"],
        "start_date": info["start_date"].isoformat(),
        "end_date": info["end_date"].isoformat(),
        "interval": info["interval"],
        "n_bars": len(df),
        "q_col_used": q_col,

        # baseline
        "P_max_baseline": float(tab["P_max_baseline"].iloc[0]),
        "P_min_baseline": float(tab["P_min_baseline"].iloc[0]),

        # probability curve peaks
        "peak_P_max": float(peak_max) if not np.isnan(peak_max) else np.nan,
        "peak_Q_max": float(peak_max_Q) if not np.isnan(peak_max_Q) else np.nan,
        "peak_P_min": float(peak_min) if not np.isnan(peak_min) else np.nan,
        "peak_Q_min": float(peak_min_Q) if not np.isnan(peak_min_Q) else np.nan,

        # timing summaries
        "median_dist_next_max_q95": float(med_dist_next_max_q95) if not np.isnan(med_dist_next_max_q95) else np.nan,
        "median_dist_next_min_q05": float(med_dist_next_min_q05) if not np.isnan(med_dist_next_min_q05) else np.nan,
    }

    # add conditional probs + lifts for each standard threshold
    for _, r in tab.iterrows():
        Q = r["Threshold"]
        qkey = f"{Q:.2f}".replace(".", "_")
        row[f"P_max_given_high_{qkey}"] = r["P_max_given_high"]
        row[f"P_min_given_low_{qkey}"]  = r["P_min_given_low"]
        row[f"Lift_max_{qkey}"]         = r["Lift_max"]
        row[f"Lift_min_{qkey}"]         = r["Lift_min"]
        row[f"count_high_{qkey}"]       = r["count_high"]
        row[f"count_low_{qkey}"]        = r["count_low"]

    return row


# =======================================================
# 2.4D) Ledger persistence (incremental updates)
# =======================================================

LEDGER_PATH = os.path.join(PROCESSED_DATA_DIR, "analysis_ledger.csv")

def update_analysis_ledger(
    new_results_df,
    ledger_path=LEDGER_PATH,
    verbose=True
):
    """
    Update / create an analysis ledger. If a filename already exists in the
    ledger, it is not duplicated.

    Parameters
    ----------
    new_results_df : pd.DataFrame
        Output of analyze_all_processed_files(...)
    ledger_path : str
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Updated ledger.
    """
    if new_results_df is None or len(new_results_df) == 0:
        if verbose:
            print("[ledger] no new results to add.")
        if os.path.exists(ledger_path):
            return pd.read_csv(ledger_path)
        return pd.DataFrame()

    if os.path.exists(ledger_path):
        ledger = pd.read_csv(ledger_path)
        existing = set(ledger["filename"].astype(str))
        add_df = new_results_df[~new_results_df["filename"].astype(str).isin(existing)]
        updated = pd.concat([ledger, add_df], ignore_index=True)
    else:
        updated = new_results_df.copy()

    updated.to_csv(ledger_path, index=False)

    if verbose:
        print(f"[ledger] saved -> {ledger_path}")
        print(f"[ledger] rows = {len(updated)}")

    return updated

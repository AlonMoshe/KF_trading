# backtest_utils.py (or just put this in a notebook cell)

import os
import datetime as dt
import pandas as pd

# Adjust these imports to match how you normally import modules.
from src.data.data_utils import PROCESSED_DATA_DIR, parse_filename_info, select_intraday_session
from src.quant.quant_utils import add_slope_quantiles_daily
from src.signals.signal_utils import (
    add_turning_regions,
    add_slope_peaks,
    add_raw_slope_peaks,
    add_entry_signals,
    add_exit_signals,
    add_trade_ids,
    extract_trades,
    trades_to_dataframe,
)


# ------------------------------------------------------------
# 1) Discover processed files for a symbol + interval
# ------------------------------------------------------------

def discover_processed_files(symbol: str, interval: int):
    """
    Scan PROCESSED_DATA_DIR and return a list of file info dicts:

        {
          "file_path": "...",
          "symbol": "QQQ",
          "start_date": date,
          "end_date": date,
          "interval": int,
          "span_days": int,
        }
    """
    symbol = symbol.upper()
    infos = []

    for fname in os.listdir(PROCESSED_DATA_DIR):
        if not fname.endswith(".csv"):
            continue

        try:
            info = parse_filename_info(fname)
        except ValueError:
            continue  # ignore files that don't match pattern

        if info["symbol"] != symbol:
            continue
        if info["interval"] != interval:
            continue

        start_date = info["start_date"]
        end_date = info["end_date"]
        span_days = (end_date - start_date).days + 1

        infos.append({
            "file_path": os.path.join(PROCESSED_DATA_DIR, fname),
            "symbol": info["symbol"],
            "start_date": start_date,
            "end_date": end_date,
            "interval": info["interval"],
            "span_days": span_days,
        })

    return infos


# ------------------------------------------------------------
# 2) Build date -> file map (choose most specific file)
# ------------------------------------------------------------

def build_date_to_file_map(file_infos):
    """
    For overlapping files, for each date choose the file with the
    smallest span_days (most specific) that covers that date.

    Returns:
        dict[date] = "file_path"
    """
    date_to_file = {}

    for info in file_infos:
        start = info["start_date"]
        end = info["end_date"]
        span = info["span_days"]
        path = info["file_path"]

        d = start
        while d <= end:
            if d not in date_to_file:
                date_to_file[d] = (path, span)
            else:
                _, best_span = date_to_file[d]
                if span < best_span:
                    date_to_file[d] = (path, span)
            d += dt.timedelta(days=1)

    # strip span, keep only file path
    date_to_file = {d: p for d, (p, s) in date_to_file.items()}
    return date_to_file


# ------------------------------------------------------------
# 3) Run backtest for ONE day given its file
# ------------------------------------------------------------

def run_day_backtest(symbol, session_date, file_path,
                     interval=5,
                     window_max=100,
                     Q_high=0.90,
                     slope_smooth=False,
                     slope_smooth_window=5,
                     slope_peak_half_window=1,
                     slope_peak_hysteresis=0.0,   # NEW
                     cooldown=20,
                     slope_peak_min_swing=0.0, 
                     use_region_recent=True,       # NEW
                     region_recent_window=10,      # NEW
                     verbose=True):


    """
    Run the full causal trading pipeline for a single day
    using the data in `file_path` (which may contain multiple days).

    Returns a dict with summary stats or None if no data/trades.
    """
    # Normalize date
    if isinstance(session_date, str):
        session_date = dt.date.fromisoformat(session_date)

    # Load the file directly (we already know which file to use)
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    df = df.sort_index()

    # Slice intraday session for this date
    date_str = session_date.isoformat()
    df_day = select_intraday_session(df, date_str, start_time="09:30", end_time="15:59:55")

    if df_day.empty:
        if verbose:
            print(f"[WARN] {symbol} {date_str}: no intraday data in file, skipping.")
        return None

    # --------------------------------------------------------
    # Run your full causal trading pipeline on df_day
    # --------------------------------------------------------

    # 1) Daily ECDF quantiles (causal)
    df_day = add_slope_quantiles_daily(df_day, slope_col="KF_slope_adapt",
                                       min_points=50, window_max=window_max)

    # 2) Turning regions (use daily quantile column)
    df_day = add_turning_regions(df_day, q_col="slope_q_roll_daily", Q_high=Q_high)

    # 3) Causal slope peaks for entries (region-gated)
    df_day = add_slope_peaks(
        df_day,
        slope_col="KF_slope_adapt",
        smooth=slope_smooth,
        smooth_window=slope_smooth_window,
        peak_half_window=slope_peak_half_window,
        peak_hysteresis=slope_peak_hysteresis,
        peak_min_swing=slope_peak_min_swing,  
    )


    # 4) Raw causal peaks for exits
    df_day = add_raw_slope_peaks(df_day, slope_col="KF_slope_adapt")

    # 5) Entry and exit signals
    df_day = add_entry_signals(
            df_day,
            q_col="slope_q_roll_daily",
            Q_high=Q_high,
            cooldown=cooldown,        # ★ pass through
            use_region_recent=use_region_recent,            # NEW
            region_recent_window=region_recent_window       # NEW
        )

    df_day = add_exit_signals(df_day)

    # 6) Trade IDs (for plotting/debug only)
    df_day = add_trade_ids(df_day)

    # 7) Extract trades and build trades DataFrame
    trades = extract_trades(df_day)
    df_trades = trades_to_dataframe(trades)

    # --------------------------------------------------------
    # Summarize daily performance
    # --------------------------------------------------------
    if df_trades.empty:
        total_pnl = 0.0
        num_trades = 0
        avg_pnl = 0.0
        long_pnl = 0.0
        short_pnl = 0.0
        longest_trade = 0
        shortest_trade = 0
        avg_duration = 0.0
    else:
        total_pnl = float(df_trades["pnl"].sum())
        num_trades = int(len(df_trades))
        avg_pnl = float(df_trades["pnl"].mean())

        long_pnl = float(df_trades.loc[df_trades["side"] == "long", "pnl"].sum())
        short_pnl = float(df_trades.loc[df_trades["side"] == "short", "pnl"].sum())

        longest_trade = int(df_trades["holding_bars"].max())
        shortest_trade = int(df_trades["holding_bars"].min())
        avg_duration = float(df_trades["holding_bars"].mean())

    if verbose:
        print(f"{symbol} {date_str}: trades={num_trades}, PnL={total_pnl:.4f}")

    summary = {
        "date": session_date,
        "symbol": symbol.upper(),
        "num_trades": num_trades,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "longest_trade": longest_trade,
        "shortest_trade": shortest_trade,
        "avg_trade_duration": avg_duration,
    }

    return summary


# ------------------------------------------------------------
# 4) Run backtest for a DATE RANGE
# ------------------------------------------------------------

def run_backtest_range(symbol, start_date, end_date,
                        interval=5,
                        window_max=100,
                        Q_high=0.90,
                        slope_smooth=False,
                        slope_smooth_window=5,
                        slope_peak_half_window=1,
                        slope_peak_hysteresis=0.0,   # NEW
                        cooldown=20,
                        slope_peak_min_swing=0.0,
                        use_region_recent=True,        # NEW
                        region_recent_window=10,       # NEW
                        verbose=True):


    """
    Run the daily trading simulation for each date in [start_date, end_date]
    for the given symbol and interval.

    Returns:
        results_df : DataFrame with one row per trading day.
    """
    symbol = symbol.upper()

    # Normalize dates
    if isinstance(start_date, str):
        start_date = dt.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = dt.date.fromisoformat(end_date)

    # 1) Discover files
    file_infos = discover_processed_files(symbol, interval=interval)
    if not file_infos:
        print(f"[ERROR] No processed files found for {symbol} at interval {interval}.")
        return pd.DataFrame()

    # 2) Build date -> file map
    date_to_file = build_date_to_file_map(file_infos)

    # 3) Loop over each date in the requested range
    results = []
    d = start_date
    while d <= end_date:
        if d not in date_to_file:
            if verbose:
                print(f"{symbol} {d.isoformat()}: no processed file, skipping.")
        else:
            file_path = date_to_file[d]
            summary = run_day_backtest(
                symbol=symbol,
                session_date=d,
                file_path=file_path,
                interval=interval,
                window_max=window_max,
                Q_high=Q_high,
                slope_smooth=slope_smooth,
                slope_smooth_window=slope_smooth_window,
                slope_peak_half_window=slope_peak_half_window,
                slope_peak_hysteresis=slope_peak_hysteresis,   # NEW
                cooldown=cooldown,
                verbose=verbose,
                slope_peak_min_swing=slope_peak_min_swing,
                use_region_recent=use_region_recent,
                region_recent_window=region_recent_window,
            )


            if summary is not None:
                results.append(summary)
        d += dt.timedelta(days=1)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("date").reset_index(drop=True)
    return results_df


def one_day_backtest(symbol, year, month, day,
                     Q_high=0.90,
                     window_max=100,
                     cooldown=20,
                     interval=5,
                     slope_smooth=False,
                     slope_smooth_window=5,
                     slope_peak_half_window=1,
                     slope_peak_hysteresis=0.0,   # NEW
                     slope_peak_min_swing=0.0,
                     use_region_recent=True,        # NEW
                     region_recent_window=10,       # NEW
                     verbose=True):


    """
    Complete one-day simulation:
    - loads correct file for the day
    - slices intraday session
    - runs full causal pipeline
    - extracts trades
    - returns df_day, df_trades, summary_dict

    This is the ONLY function you need to call from the notebook
    (aside from plotting).
    """

    # --------------------------------------------------------
    # 1. Resolve date
    # --------------------------------------------------------
    target_date = dt.date(year, month, day)
    date_str = target_date.isoformat()

    # --------------------------------------------------------
    # 2. Identify the correct file for the day
    # --------------------------------------------------------
    infos = discover_processed_files(symbol, interval=interval)
    if not infos:
        raise RuntimeError(f"No processed files found for {symbol}")

    date_to_file = build_date_to_file_map(infos)

    if target_date not in date_to_file:
        raise RuntimeError(f"No data found for {symbol} on {target_date}")

    file_path = date_to_file[target_date]

    # --------------------------------------------------------
    # 3. Load file exactly like the backtester
    # --------------------------------------------------------
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    # --------------------------------------------------------
    # 4. Slice the intraday session
    # --------------------------------------------------------
    df_day = select_intraday_session(df, date_str,
                                     start_time="09:30",
                                     end_time="15:59:55")

    if df_day.empty:
        raise RuntimeError(f"{symbol} {date_str}: No intraday data found.")

    # --------------------------------------------------------
    # 5. FULL CAUSAL PIPELINE
    # --------------------------------------------------------

    # ECDF quantiles
    df_day = add_slope_quantiles_daily(
        df_day,
        slope_col="KF_slope_adapt",
        window_max=window_max,
    )

    # Turning regions
    df_day = add_turning_regions(
        df_day,
        q_col="slope_q_roll_daily",
        Q_high=Q_high,
    )

    # Causal slope peaks (entry)
    df_day = add_slope_peaks(
        df_day,
        slope_col="KF_slope_adapt",
        smooth=slope_smooth,
        smooth_window=slope_smooth_window,
        peak_half_window=slope_peak_half_window,
        peak_hysteresis=slope_peak_hysteresis,
        peak_min_swing=slope_peak_min_swing,

    )




    # Raw causal peaks (exit)
    df_day = add_raw_slope_peaks(df_day)

    # Entry / exit signals
    df_day = add_entry_signals(
        df_day,
        q_col="slope_q_roll_daily",
        Q_high=Q_high,
        cooldown=cooldown,
        use_region_recent=use_region_recent,
        region_recent_window=region_recent_window
    )

    df_day = add_exit_signals(df_day)

    # Assign trade IDs for plotting/debug
    df_day = add_trade_ids(df_day)

    # --------------------------------------------------------
    # 6. Extract trades + summary
    # --------------------------------------------------------
    trades = extract_trades(df_day)
    df_trades = trades_to_dataframe(trades)

    # Basic stats
    total_pnl = df_trades["pnl"].sum() if not df_trades.empty else 0.0
    num_trades = len(df_trades)
    avg_pnl = df_trades["pnl"].mean() if num_trades > 0 else 0.0

    summary = {
        "symbol": symbol,
        "date": target_date,
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "avg_pnl": avg_pnl,
    }

    if verbose:
        print(
            f"{symbol} {target_date}: "
            f"Trades={num_trades}, Total={total_pnl:.4f}, Avg={avg_pnl:.4f}"
        )

    return df_day, df_trades, summary
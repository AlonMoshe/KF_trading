import os
import pandas as pd
import datetime as dt
from src.kalman.filter_utils import clean_raw_data, apply_adaptive_kf

# -------------------------------------------------------
# 1. Directory Constants
# -------------------------------------------------------

RAW_DATA_DIR = r"C:\Users\User\OneDrive\Projects\TradingSystem\Data\DataForKF"
PROCESSED_DATA_DIR = r"C:\Users\User\OneDrive\Projects\TradingSystem\Data\ProcessedData"

# -------------------------------------------------------
# 2. Filename Parsing
# -------------------------------------------------------

def parse_filename_info(filename: str) -> dict:
    """
    Parse filenames of the form:
        SYMBOL_YYYY-MM-DD_YYYY-MM-DD_INTERVAL.csv
    Returns:
        {
            "symbol": str,
            "start_date": date,
            "end_date": date,
            "interval": int (seconds)
        }
    """
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)

    parts = name.split("_")
    if len(parts) != 4:
        raise ValueError(f"Filename {filename} does not match expected format.")

    symbol = parts[0].upper()
    start_date = dt.date.fromisoformat(parts[1])
    end_date = dt.date.fromisoformat(parts[2])
    interval = int(parts[3])  # seconds

    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval
    }

# -------------------------------------------------------
# 3. File Discovery (Flexible Matching)
# -------------------------------------------------------

def find_matching_file(symbol, start, end, interval, processed=True) -> str:
    """
    Auto-discover the file for (symbol, interval) whose date range fully
    covers the requested [start, end] interval.

    Example:
        user requests:  QQQ, 2025-11-12 to 2025-11-18, 5 seconds
        file on disk:   QQQ_2025-11-10_2025-11-20_5.csv
        => file is accepted and returned.

    Raises helpful errors if:
        - no files match
        - more than one file matches
    """

    symbol = symbol.upper()

    # normalize dates
    if isinstance(start, str):
        start = dt.date.fromisoformat(start)
    if isinstance(end, str):
        end = dt.date.fromisoformat(end)

    folder = PROCESSED_DATA_DIR if processed else RAW_DATA_DIR

    candidates = []
    for f in os.listdir(folder):
        if not f.endswith(".csv"):
            continue

        info = parse_filename_info(f)
        if info["symbol"] != symbol:
            continue
        if info["interval"] != interval:
            continue

        # Check if file covers requested [start, end]
        if info["start_date"] <= start and info["end_date"] >= end:
            candidates.append(os.path.join(folder, f))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No matching file found for symbol={symbol}, interval={interval}, "
            f"covering [{start}, {end}] in {folder}"
        )

    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple matching files found!\n" +
            "\n".join(candidates) +
            "\nPlease clean or reorganize the data directory."
        )

    return candidates[0]

# -------------------------------------------------------
# 4. Load RAW Data
# -------------------------------------------------------

def load_raw_df(symbol, start, end, interval):
    """
    Load raw IB data, auto-discovering the appropriate file.
    Parse timestamps of the form:
        YYYYMMDD HH:MM:SS US/Eastern
    Returns tz-aware DataFrame indexed by Datetime.
    """
    file_path = find_matching_file(symbol, start, end, interval, processed=False)

    df = pd.read_csv(file_path)

    # Parse Date column: remove timezone text, parse and localize
    df["Datetime"] = pd.to_datetime(
        df["Date"].str.replace(" US/Eastern", "", regex=False),
        format="%Y%m%d %H:%M:%S"
    ).dt.tz_localize("US/Eastern")

    df = df.set_index("Datetime").sort_index()

    # slice to requested date range
    start_dt = pd.Timestamp(start).tz_localize("US/Eastern")
    end_dt = pd.Timestamp(end).tz_localize("US/Eastern") + pd.Timedelta(days=1)  # include full end day

    df = df.loc[start_dt:end_dt]

    return df

# -------------------------------------------------------
# 5. Load PROCESSED (Kalman) Data
# -------------------------------------------------------

def load_processed_df(symbol, start, end, interval):
    """
    Load processed KF dataframe and slice to requested date range.
    Assumes processed CSV already has tz-aware Datetime index
    or a column that becomes the index.
    """
    file_path = find_matching_file(symbol, start, end, interval, processed=True)

    df = pd.read_csv(file_path, index_col=0)

    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")



    # Slice date range
    start_dt = pd.Timestamp(start).tz_localize("US/Eastern")
    end_dt = pd.Timestamp(end).tz_localize("US/Eastern") + pd.Timedelta(days=1)
    df = df.loc[start_dt:end_dt]

    return df

# -------------------------------------------------------
# 6. Utilities for Slicing
# -------------------------------------------------------

def select_date_range(df, start, end):
    """Simple wrapper around df.loc[start:end]."""
    return df.loc[start:end]


def select_intraday_session(df, session_date, start_time="09:30", end_time="15:59:55"):
    """
    Extract a single trading session.
    session_date: 'YYYY-MM-DD'
    start_time, end_time: HH:MM strings
    """
    session_start = f"{session_date} {start_time}"
    session_end = f"{session_date} {end_time}"
    return df.loc[session_start:session_end]

# -------------------------------------------------------
# 7. Optional Helpers
# -------------------------------------------------------

def list_available_raw_files(symbol=None):
    files = []
    for f in os.listdir(RAW_DATA_DIR):
        if f.endswith(".csv"):
            if symbol is None:
                files.append(f)
            else:
                symbol = symbol.upper()
                if parse_filename_info(f)["symbol"] == symbol:
                    files.append(f)
    return files


def list_available_processed_files(symbol=None):
    files = []
    for f in os.listdir(PROCESSED_DATA_DIR):
        if f.endswith(".csv"):
            if symbol is None:
                files.append(f)
            else:
                symbol = symbol.upper()
                if parse_filename_info(f)["symbol"] == symbol:
                    files.append(f)
    return files


# -------------------------------------------------------
# 8. Full Processing Pipeline: RAW → CLEAN → KF → SAVE
# -------------------------------------------------------


def process_raw_to_processed(symbol, start, end, interval, overwrite=False):
    """
    End-to-end pipeline that:
        1. Loads raw IB data (auto-discovery)
        2. Cleans it (Mid, Spread, EWMA var, etc.)
        3. Applies adaptive Kalman filter
        4. Saves processed result to PROCESSED_DATA_DIR
        5. Returns the processed dataframe

    Parameters
    ----------
    symbol : str
        Symbol, e.g. "QQQ". Case-insensitive.
    start : str or datetime.date
        Start date (YYYY-MM-DD).
    end : str or datetime.date
        End date (YYYY-MM-DD).
    interval : int
        Time interval in seconds (e.g. 5).
    overwrite : bool, default False
        If False, will refuse to overwrite an existing processed file.

    Returns
    -------
    pd.DataFrame
        The processed dataframe with KF columns.
    """

    symbol = symbol.upper()

    # -------------------------------------------------------
    # Load RAW dataframe
    # -------------------------------------------------------
    df_raw = load_raw_df(symbol, start, end, interval)

    # -------------------------------------------------------
    # Clean RAW dataframe
    # -------------------------------------------------------
    df_clean = clean_raw_data(df_raw, Delta=float(interval))

    # -------------------------------------------------------
    # Apply Adaptive Kalman Filter
    # -------------------------------------------------------
    df_kf = apply_adaptive_kf(df_clean, Delta=float(interval))

    # -------------------------------------------------------
    # Build output file path
    # -------------------------------------------------------
    if isinstance(start, str):
        start_str = start
    else:
        start_str = start.isoformat()

    if isinstance(end, str):
        end_str = end
    else:
        end_str = end.isoformat()

    filename = f"{symbol}_{start_str}_{end_str}_{interval}.csv"
    save_path = os.path.join(PROCESSED_DATA_DIR, filename)

    # -------------------------------------------------------
    # Check overwrite
    # -------------------------------------------------------
    if os.path.exists(save_path) and not overwrite:
        raise FileExistsError(
            f"Processed file already exists:\n{save_path}\n"
            "Use overwrite=True to replace it."
        )

    # -------------------------------------------------------
    # Save DataFrame WITH Datetime index
    # -------------------------------------------------------
    df_kf.to_csv(save_path, index=True)

    print(f"Processed data saved to:\n{save_path}")
    print(f"Shape: {df_kf.shape}")

    return df_kf

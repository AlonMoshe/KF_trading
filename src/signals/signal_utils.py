import numpy as np
import datetime as dt
import pandas as pd


def add_turning_regions(df, q_col="slope_q_roll", Q_high=0.90):
    """
    Add turning-region markers based on slope quantiles.

    region_max = True when slope quantile >= Q_high
    region_min = True when slope quantile <= (1 - Q_high)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the slope quantile column.
    q_col : str
        Column name for slope quantiles.
    Q_high : float
        Threshold for defining turning regions.

    Returns
    -------
    df : pd.DataFrame
        With added boolean columns:
            region_max
            region_min
    """
    q = df[q_col].to_numpy()

    df["region_max"] = q >= Q_high
    df["region_min"] = q <= (1 - Q_high)

    return df


def add_slope_peaks(
    df,
    slope_col="KF_slope_adapt",
    region_max_col="region_max",
    region_min_col="region_min",
    smooth=False,
    smooth_window=5,
    peak_half_window=1,
     peak_hysteresis=0.0,
     peak_min_swing=0.0
):
    """
    STRICTLY CAUSAL slope peaks/troughs for timing entries.

    Logic:
      - A local extremum of slope at index t is only known once we see s[t+1].
      - Therefore, the *signal* is placed at index t+1.
      - This function also gates peaks by turning regions.

    Adds:
        slope_peak_max  (causal, inside region_max)
        slope_peak_min  (causal, inside region_min)
    """
    df = df.copy()
    s = df[slope_col].astype(float)

    if smooth:
        s_use = s.rolling(smooth_window, min_periods=1).mean()
    else:
        s_use = s

    region_max = df[region_max_col].fillna(False).to_numpy(bool)
    region_min = df[region_min_col].fillna(False).to_numpy(bool)

    n = len(df)
    peak_max = np.zeros(n, dtype=bool)
    peak_min = np.zeros(n, dtype=bool)

    s_vals = s_use.to_numpy(dtype=float)

    # -------------------------------------------------
    # Windowed peak detection (strictly causal)
    # -------------------------------------------------
    k = int(peak_half_window)
    if k < 1:
        k = 1

    if k == 1:
        # Original 3-point logic, with optional hysteresis
        for t in range(1, n - 1):
            right = t + 1

            # raw local max at t
            if (s_vals[t] > s_vals[t - 1]) and (s_vals[t] >= s_vals[right]) and region_max[t]:

                # Hysteresis condition:
                if peak_hysteresis > 0.0 and (s_vals[t] - s_vals[right] < peak_hysteresis):
                    continue

                # Swing-size condition: peak must exceed lower neighbor by min amount
                local_min = min(s_vals[t - 1], s_vals[right])
                if peak_min_swing > 0.0 and (s_vals[t] - local_min < peak_min_swing):
                    continue

                peak_max[right] = True


            # raw local min at t
            if (s_vals[t] < s_vals[t - 1]) and (s_vals[t] <= s_vals[right]) and region_min[t]:

                # Hysteresis condition:
                if peak_hysteresis > 0.0 and (s_vals[right] - s_vals[t] < peak_hysteresis):
                    continue

                # Swing-size condition: valley must be below upper neighbor by min amount
                local_max = max(s_vals[t - 1], s_vals[right])
                if peak_min_swing > 0.0 and (local_max - s_vals[t] < peak_min_swing):
                    continue

                peak_min[right] = True



    else:
        # General windowed logic with optional hysteresis:
        #   - center index c runs from k .. n-k-1
        #   - window = [c-k, ..., c, ..., c+k]
        #   - we can only confirm after seeing s[c+k]
        #   - so the *signal* is placed at index (c+k)
        for c in range(k, n - k):
            left = c - k
            right = c + k
            window = s_vals[left:right + 1]
            val = s_vals[c]

            # Max: center is the maximum of the window and in region_max
            if region_max[c] and val == window.max():

                # Hysteresis condition:
                if peak_hysteresis > 0.0 and (val - s_vals[right] < peak_hysteresis):
                    continue

                # Swing-size condition:
                local_min = window.min()
                if peak_min_swing > 0.0 and (val - local_min < peak_min_swing):
                    continue

                peak_max[right] = True


            # Min: center is the minimum of the window and in region_min
            if region_min[c] and val == window.min():

                # Hysteresis condition:
                if peak_hysteresis > 0.0 and (s_vals[right] - val < peak_hysteresis):
                    continue

                # Swing-size condition:
                local_max = window.max()
                if peak_min_swing > 0.0 and (local_max - val < peak_min_swing):
                    continue

                peak_min[right] = True



    df["slope_peak_max"] = peak_max
    df["slope_peak_min"] = peak_min

    return df



def add_causal_slope_peaks(df, slope_col="KF_slope_adapt"):
    """
    Strictly causal slope extrema.

    A slope peak at index t is detected only when bar t+1 arrives.
    Therefore:
        raw detection at t  -> usable at t+1
        execution will occur at t+2 (handled later)

    Adds:
        slope_peak_max_causal
        slope_peak_min_causal

    IMPORTANT:
      - No look-ahead
      - Last bar cannot confirm anything (no s[t+1])
    """
    s = df[slope_col].to_numpy(dtype=float)
    n = len(s)

    peak_max = np.zeros(n, dtype=bool)
    peak_min = np.zeros(n, dtype=bool)

    # Detect peaks at t, confirm at t+1 → signal at t+1
    for t in range(1, n - 1):
        # raw extremum check at t
        is_peak_max = (s[t] > s[t-1]) and (s[t] >= s[t+1])
        is_peak_min = (s[t] < s[t-1]) and (s[t] <= s[t+1])

        if is_peak_max:
            peak_max[t + 1] = True  # usable signal at t+1
        if is_peak_min:
            peak_min[t + 1] = True

    df = df.copy()
    df["slope_peak_max_causal"] = peak_max
    df["slope_peak_min_causal"] = peak_min
    return df


def add_entry_signals( 
    df,
    q_col="slope_q_roll",
    region_max_col="region_max",
    region_min_col="region_min",
    peak_max_col="slope_peak_max",
    peak_min_col="slope_peak_min",
    Q_high=0.90,
    cooldown=20,
    last_entry_time="15:55",
    use_region_recent=True,     # NEW
    region_recent_window=10,
):
    """
    Clean entry logic (no position state here, just signals + cooldown).

    LONG entry if (raw setup):
        region_min
        region_min_recent
        slope_peak_min
        q <= 1 - Q_high

    SHORT entry if (raw setup):
        region_max
        region_max_recent
        slope_peak_max
        q >= Q_high

    Then a per-side cooldown is applied to avoid clustering.

    Causality:
      - All inputs are based only on information up to the current bar.
      - The entry signal at index i is intended to be EXECUTED at next-bar open (i+1).
      - To enforce "no new positions from 15:55:00", we drop any entry whose
        execution bar (i+1) has time >= last_entry_time (US/Eastern).

    Parameters
    ----------
    df : pd.DataFrame
        Single-day intraday dataframe (e.g. 09:30–15:59:55, US/Eastern).
    q_col : str
        Column with slope quantiles (for trading you will typically use
        'slope_q_roll_daily').
    last_entry_time : str or None
        'HH:MM' cutoff for NEW positions based on execution time (bar i+1).
        If None, no time cutoff is applied.
    """

    df = df.copy()

    # Extract relevant series
    q = df[q_col].astype(float)
    region_max = df[region_max_col].fillna(False)
    region_min = df[region_min_col].fillna(False)
    peak_max = df[peak_max_col].fillna(False)
    peak_min = df[peak_min_col].fillna(False)

    # Recency filters: region must be true at least once in last K samples
    if use_region_recent:
        region_max_recent = region_max.rolling(region_recent_window).max().astype(bool)
        region_min_recent = region_min.rolling(region_recent_window).max().astype(bool)
    else:
        # If recency disabled → treat recency as always satisfied
        region_max_recent = pd.Series(True, index=df.index)
        region_min_recent = pd.Series(True, index=df.index)


    # Probability gating
    prob_gate_max = q >= Q_high
    prob_gate_min = q <= (1 - Q_high)

    # Raw entry signals (before cooldown and time cutoff)
    entry_long_raw = (
        region_min &
        region_min_recent &
        peak_min &
        prob_gate_min
    )

    entry_short_raw = (
        region_max &
        region_max_recent &
        peak_max &
        prob_gate_max
    )

    # Apply cooldown per side
    entry_long = []
    entry_short = []

    long_cool = 0
    short_cool = 0

    for il, is_ in zip(entry_long_raw, entry_short_raw):

        # LONG side
        if long_cool > 0:
            entry_long.append(False)
            long_cool -= 1
        else:
            if il:
                entry_long.append(True)
                long_cool = cooldown
            else:
                entry_long.append(False)

        # SHORT side
        if short_cool > 0:
            entry_short.append(False)
            short_cool -= 1
        else:
            if is_:
                entry_short.append(True)
                short_cool = cooldown
            else:
                entry_short.append(False)

    entry_long = np.array(entry_long, dtype=bool)
    entry_short = np.array(entry_short, dtype=bool)

    # -------------------------------------------------
    # Time cutoff: forbid new entries whose EXECUTION
    # (next-bar open) would be at or after last_entry_time.
    # -------------------------------------------------
    if last_entry_time is not None:
        # Parse 'HH:MM'
        hh, mm = map(int, last_entry_time.split(":"))
        cutoff_time = dt.time(hour=hh, minute=mm)

        idx = df.index
        # Ensure US/Eastern for time-of-day
        if idx.tz is None:
            idx_e = idx.tz_localize("US/Eastern")
        else:
            idx_e = idx.tz_convert("US/Eastern")

        times = idx_e.time

        n = len(df)
        for i in range(n - 1):
            exec_time = times[i + 1]  # execution bar is i+1
            if exec_time >= cutoff_time:
                entry_long[i] = False
                entry_short[i] = False

        # Last bar has no execution bar → no entries
        entry_long[-1] = False
        entry_short[-1] = False

    df["entry_long"] = entry_long
    df["entry_short"] = entry_short

    return df


def add_exit_signals(
    df,
    max_adverse_bars=7,
    slope_peak_max_col="slope_peak_max_raw",
    slope_peak_min_col="slope_peak_min_raw",
):
    """
    STRICTLY CAUSAL EXIT LOGIC.

    LONG exit if (evaluated at bar i):
        1. slope_peak_max_raw[i]  (momentum turns against us)
        2. accumulated adverse bars (Mid < entry_price) >= max_adverse_bars
        3. forced flat at final bar of the day

    SHORT exit if:
        1. slope_peak_min_raw[i]
        2. accumulated adverse bars (Mid > entry_price) >= max_adverse_bars
        3. forced flat at final bar of the day

    IMPORTANT:
      - Detection happens at bar i.
      - Actual execution happens at bar i+1 open.
      - This function only marks exit signals; trade extraction computes price.
    """

    df = df.copy()

    mids = df["Mid"].to_numpy()
    peak_max = df[slope_peak_max_col].to_numpy(bool)
    peak_min = df[slope_peak_min_col].to_numpy(bool)
    entry_long = df["entry_long"].to_numpy(bool)
    entry_short = df["entry_short"].to_numpy(bool)

    n = len(df)

    exit_long = np.zeros(n, dtype=bool)
    exit_short = np.zeros(n, dtype=bool)

    # Long state
    in_long = False
    entry_price_long = None
    adverse_long = 0

    # Short state
    in_short = False
    entry_price_short = None
    adverse_short = 0

    for i in range(n):

        # -----------------------
        # LONG ENTRY TRIGGER
        # -----------------------
        if entry_long[i] and not in_long:
            in_long = True
            entry_price_long = mids[i]   # decision bar's close/mid (execution at i+1 open)
            adverse_long = 0

        # -----------------------
        # LONG EXIT LOGIC
        # -----------------------
        if in_long:

            # Adverse price action (Mid < entry price)
            if mids[i] < entry_price_long:
                adverse_long += 1

            condition_peak = peak_max[i]
            condition_adverse = (adverse_long >= max_adverse_bars)

            if condition_peak or condition_adverse:
                exit_long[i] = True
                in_long = False
                entry_price_long = None
                adverse_long = 0

        # -----------------------
        # SHORT ENTRY TRIGGER
        # -----------------------
        if entry_short[i] and not in_short:
            in_short = True
            entry_price_short = mids[i]
            adverse_short = 0

        # -----------------------
        # SHORT EXIT LOGIC
        # -----------------------
        if in_short:

            if mids[i] > entry_price_short:
                adverse_short += 1

            condition_peak = peak_min[i]
            condition_adverse = (adverse_short >= max_adverse_bars)

            if condition_peak or condition_adverse:
                exit_short[i] = True
                in_short = False
                entry_price_short = None
                adverse_short = 0

    # -----------------------
    # FORCE FLAT AT FINAL BAR
    # -----------------------
    last_i = n - 1

    if in_long:
        exit_long[last_i] = True

    if in_short:
        exit_short[last_i] = True

    df["exit_long"] = exit_long
    df["exit_short"] = exit_short

    return df



def add_level_peaks(df, level_col="KF_level_adapt"):
    """
    Detect peaks and troughs in the KF-level output.
    Returns df with:
        level_peak_max = price turning downward
        level_peak_min = price turning upward
    """
    L = df[level_col].to_numpy()

    # Price trough (start of upward turn)
    level_peak_min = (np.roll(L, 1) > L) & (L < np.roll(L, -1))

    # Price peak (start of downward turn)
    level_peak_max = (np.roll(L, 1) < L) & (L > np.roll(L, -1))

    # First/last index invalid
    level_peak_min[[0, -1]] = False
    level_peak_max[[0, -1]] = False

    df["level_peak_min"] = level_peak_min
    df["level_peak_max"] = level_peak_max

    return df


def add_raw_slope_peaks(df, slope_col="KF_slope_adapt"):
    """
    STRICTLY CAUSAL pure slope extrema for exit logic.
    No region filters, no quantile logic, no recency constraints.

    A raw extremum at index t is only known when bar t+1 arrives,
    so the signal is placed at index t+1.

    Adds:
        slope_peak_min_raw
        slope_peak_max_raw
    """
    df = df.copy()
    s = df[slope_col].to_numpy(dtype=float)
    n = len(s)

    peak_min_raw = np.zeros(n, dtype=bool)
    peak_max_raw = np.zeros(n, dtype=bool)

    for t in range(1, n - 1):
        # local min at t
        if (s[t] < s[t - 1]) and (s[t] <= s[t + 1]):
            peak_min_raw[t + 1] = True

        # local max at t
        if (s[t] > s[t - 1]) and (s[t] >= s[t + 1]):
            peak_max_raw[t + 1] = True

    df["slope_peak_min_raw"] = peak_min_raw
    df["slope_peak_max_raw"] = peak_max_raw

    return df


def add_trade_ids(
    df,
    entry_long_col="entry_long",
    exit_long_col="exit_long",
    entry_short_col="entry_short",
    exit_short_col="exit_short",
):
    """
    Assign trade IDs ONLY to real trades.
    Long and short sides independent.
    IDs alternate 1 ↔ 2 for clarity.
    """

    n = len(df)
    trade_id_long = [None] * n
    trade_id_short = [None] * n

    # -------- LONG SIDE --------
    in_long = False
    current_long_id = None
    next_long_id = 1

    for i in range(n):

        if not in_long and df[entry_long_col].iloc[i]:
            in_long = True
            current_long_id = next_long_id
            trade_id_long[i] = current_long_id
            next_long_id = 2 if next_long_id == 1 else 1

        elif in_long and df[exit_long_col].iloc[i]:
            trade_id_long[i] = current_long_id
            in_long = False
            current_long_id = None

    # -------- SHORT SIDE --------
    in_short = False
    current_short_id = None
    next_short_id = 1

    for i in range(n):

        if not in_short and df[entry_short_col].iloc[i]:
            in_short = True
            current_short_id = next_short_id
            trade_id_short[i] = current_short_id
            next_short_id = 2 if next_short_id == 1 else 1

        elif in_short and df[exit_short_col].iloc[i]:
            trade_id_short[i] = current_short_id
            in_short = False
            current_short_id = None

    df["trade_id_long"] = trade_id_long
    df["trade_id_short"] = trade_id_short

    return df

def extract_trades(df):
    """
    Extract trades from a dataframe containing:
        entry_long, exit_long,
        entry_short, exit_short

    Execution model:
        - Decisions at bar i
        - Executions at bar i+1 OPEN (strictly causal)
        - If exit happens on the last bar (no i+1), we execute at last bar CLOSE.

    Returns:
        list of dicts, each with:
            side, entry_index, exit_index,
            entry_time, exit_time,
            entry_price, exit_price, pnl
    """
    trades = []
    opens = df["Open"].to_numpy()
    closes = df["Close"].to_numpy()
    idx = df.index

    n = len(df)

    # -------- LONG SIDE --------
    in_long = False
    entry_idx = None

    for i in range(n):
        # ENTRY
        if df["entry_long"].iloc[i] and not in_long:
            if i + 1 < n:  # must have execution bar
                in_long = True
                entry_idx = i
            continue

        # EXIT
        if in_long and df["exit_long"].iloc[i]:
            exit_idx = i

            # Execution price
            if exit_idx + 1 < n:
                exit_price = float(opens[exit_idx + 1])
            else:
                # last bar -> execute at last bar CLOSE
                exit_price = float(closes[exit_idx])

            entry_price = float(opens[entry_idx + 1])
            pnl = exit_price - entry_price

            trades.append({
                "side": "long",
                "entry_index": entry_idx,
                "exit_index": exit_idx,
                "entry_time": idx[entry_idx],
                "exit_time": idx[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
            })

            in_long = False
            entry_idx = None

    # -------- SHORT SIDE --------
    in_short = False
    entry_idx = None

    for i in range(n):

        # ENTRY
        if df["entry_short"].iloc[i] and not in_short:
            if i + 1 < n:
                in_short = True
                entry_idx = i
            continue

        # EXIT
        if in_short and df["exit_short"].iloc[i]:
            exit_idx = i

            if exit_idx + 1 < n:
                exit_price = float(opens[exit_idx + 1])
            else:
                exit_price = float(closes[exit_idx])

            entry_price = float(opens[entry_idx + 1])
            pnl = entry_price - exit_price  # short PnL

            trades.append({
                "side": "short",
                "entry_index": entry_idx,
                "exit_index": exit_idx,
                "entry_time": idx[entry_idx],
                "exit_time": idx[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
            })

            in_short = False
            entry_idx = None

    return trades


def trades_to_dataframe(trades):
    """
    Convert the list of trade dictionaries from extract_trades()
    into a clean pandas DataFrame.

    Columns:
        side, entry_time, exit_time,
        entry_price, exit_price,
        pnl, holding_bars

    This is deliberately simple.
    """
    if len(trades) == 0:
        return pd.DataFrame(columns=[
            "side", "entry_time", "exit_time",
            "entry_price", "exit_price",
            "pnl", "holding_bars"
        ])

    df = pd.DataFrame(trades)

    # holding time in bars
    df["holding_bars"] = df["exit_index"] - df["entry_index"]

    # Order columns nicely
    df = df[
        [
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "pnl",
            "holding_bars"
        ]
    ]

    return df

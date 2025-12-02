import numpy as np

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
    smooth_window=5
):
    """
    Detect slope peaks/troughs for timing entries.

    slope_peak_max:
        local maxima of slope inside region_max
    slope_peak_min:
        local minima of slope inside region_min

    Parameters
    ----------
    df : pd.DataFrame
    slope_col : str
        Column with KF slope estimates.
    region_max_col : str
        Boolean column marking max anticipation region.
    region_min_col : str
        Boolean column marking min anticipation region.
    smooth : bool
        If True, use a rolling mean of slope before peak detection.
    smooth_window : int
        Window for rolling mean if smooth=True.

    Returns
    -------
    df : pd.DataFrame
        With added boolean columns:
            slope_peak_max
            slope_peak_min
    """
    s = df[slope_col].astype(float)

    if smooth:
        s_use = s.rolling(smooth_window, min_periods=1).mean()
    else:
        s_use = s

    s_prev = s_use.shift(1)
    s_next = s_use.shift(-1)

    # raw peaks/troughs
    peak_raw   = (s_use > s_prev) & (s_use > s_next)
    trough_raw = (s_use < s_prev) & (s_use < s_next)

    # gate by turning regions
    region_max = df[region_max_col].fillna(False)
    region_min = df[region_min_col].fillna(False)

    df["slope_peak_max"] = peak_raw & region_max
    df["slope_peak_min"] = trough_raw & region_min

    return df

# def add_entry_signals(
#     df,
#     q_col="slope_q_roll",
#     region_max_col="region_max",
#     region_min_col="region_min",
#     peak_max_col="slope_peak_max",
#     peak_min_col="slope_peak_min",
#     Q_high=0.90,
#     alpha=0.05,
#     K=10,
#     cooldown=20
# ):
#     """
#     Clean, original entry logic.
#     Fires immediately when:
#         LONG:
#             region_min
#             region_min_recent
#             slope_peak_min
#             prob_gate_min
#         SHORT:
#             region_max
#             region_max_recent
#             slope_peak_max
#             prob_gate_max

#     Then cooldown applies to prevent clustered entries.
#     """

#     # Extract relevant series
#     q = df[q_col].astype(float)
#     region_max = df[region_max_col].fillna(False)
#     region_min = df[region_min_col].fillna(False)
#     peak_max = df[peak_max_col].fillna(False)
#     peak_min = df[peak_min_col].fillna(False)

#     # -----------------------------------
#     # Recency filters: region must be true
#     # at least once in the last K samples
#     # -----------------------------------
#     region_min_recent = region_min.rolling(K).max().astype(bool)
#     region_max_recent = region_max.rolling(K).max().astype(bool)

#     # -----------------------------------
#     # Probability gating
#     # -----------------------------------
#     prob_gate_max = q >= Q_high
#     prob_gate_min = q <= (1 - Q_high)

#     # -----------------------------------
#     # Raw entry signals (no level constraint)
#     # -----------------------------------
#     entry_long_raw = (
#         region_min &
#         region_min_recent &
#         peak_min &
#         prob_gate_min
#     )

#     entry_short_raw = (
#         region_max &
#         region_max_recent &
#         peak_max &
#         prob_gate_max
#     )

#     # -----------------------------------
#     # Apply cooldown to avoid clustering
#     # -----------------------------------
#     entry_long = []
#     entry_short = []

#     long_cool = 0
#     short_cool = 0

#     for il, is_ in zip(entry_long_raw, entry_short_raw):

#         # LONG
#         if long_cool > 0:
#             entry_long.append(False)
#             long_cool -= 1
#         else:
#             if il:
#                 entry_long.append(True)
#                 long_cool = cooldown
#             else:
#                 entry_long.append(False)

#         # SHORT
#         if short_cool > 0:
#             entry_short.append(False)
#             short_cool -= 1
#         else:
#             if is_:
#                 entry_short.append(True)
#                 short_cool = cooldown
#             else:
#                 entry_short.append(False)

#     df["entry_long"] = entry_long
#     df["entry_short"] = entry_short

#     return df


def add_entry_signals(
    df,
    q_col="slope_q_roll",
    region_max_col="region_max",
    region_min_col="region_min",
    peak_max_col="slope_peak_max",
    peak_min_col="slope_peak_min",
    Q_high=0.90,
    K=10,
    cooldown=20,
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
    This function does NOT know whether a position is open;
    that is handled later by the trading engine (exit logic, trade_ids).
    """

    # Extract relevant series
    q = df[q_col].astype(float)
    region_max = df[region_max_col].fillna(False)
    region_min = df[region_min_col].fillna(False)
    peak_max = df[peak_max_col].fillna(False)
    peak_min = df[peak_min_col].fillna(False)

    # Recency filters: region must be true at least once in last K samples
    region_min_recent = region_min.rolling(K).max().astype(bool)
    region_max_recent = region_max.rolling(K).max().astype(bool)

    # Probability gating
    prob_gate_max = q >= Q_high
    prob_gate_min = q <= (1 - Q_high)

    # Raw entry signals
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

    df["entry_long"] = entry_long
    df["entry_short"] = entry_short

    return df



# def add_exit_signals(
#     df,
#     max_adverse_bars=5,
#     slope_peak_max_col="slope_peak_max",
#     slope_peak_min_col="slope_peak_min",
# ):
#     """
#     Exit rules (simple, robust, real-time safe):

#     LONG exit when:
#         1. slope_peak_max (slope forms a local max → momentum turning down), OR
#         2. total number of bars where Mid < entry_price reaches max_adverse_bars

#     SHORT exit when:
#         1. slope_peak_min (slope forms a local min → momentum turning up), OR
#         2. total number of bars where Mid > entry_price reaches max_adverse_bars

#     Uses only Mid for adverse pressure. No future data used.
#     """

#     mids = df["Mid"].to_numpy()
#     peak_max = df[slope_peak_max_col].to_numpy()
#     peak_min = df[slope_peak_min_col].to_numpy()

#     entry_long = df["entry_long"].to_numpy()
#     entry_short = df["entry_short"].to_numpy()

#     n = len(df)

#     exit_long = [False] * n
#     exit_short = [False] * n

#     in_long = False
#     in_short = False

#     entry_price_long = None
#     entry_price_short = None

#     adverse_count_long = 0
#     adverse_count_short = 0

#     for i in range(n):

#         # ---------------------------
#         # LONG ENTRY TRIGGER
#         # ---------------------------
#         if entry_long[i] and not in_long and not in_short:
#             in_long = True
#             entry_price_long = mids[i]
#             adverse_count_long = 0

#         # ---------------------------
#         # SHORT ENTRY TRIGGER
#         # ---------------------------
#         if entry_short[i] and not in_long and not in_short:
#             in_short = True
#             entry_price_short = mids[i]
#             adverse_count_short = 0

#         # ---------------------------
#         # LONG EXIT LOGIC
#         # ---------------------------
#         if in_long:

#             # Opposite slope extremum (momentum reversal)
#             cond_slope_reversal = df["slope_peak_max_raw"].iloc[i]


#             # Accumulated adverse movement (Mid < entry price)
#             if mids[i] < entry_price_long:
#                 adverse_count_long += 1
#             cond_adverse = (adverse_count_long >= max_adverse_bars)

#             if cond_slope_reversal or cond_adverse:
#                 exit_long[i] = True
#                 in_long = False
#                 entry_price_long = None
#                 adverse_count_long = 0

#         # ---------------------------
#         # SHORT EXIT LOGIC
#         # ---------------------------
#         if in_short:

#             # Opposite slope extremum (momentum reversal)
#             cond_slope_reversal = df["slope_peak_min_raw"].iloc[i]


#             # Accumulated adverse movement (Mid > entry price)
#             if mids[i] > entry_price_short:
#                 adverse_count_short += 1
#             cond_adverse = (adverse_count_short >= max_adverse_bars)

#             if cond_slope_reversal or cond_adverse:
#                 exit_short[i] = True
#                 in_short = False
#                 entry_price_short = None
#                 adverse_count_short = 0

#     df["exit_long"] = exit_long
#     df["exit_short"] = exit_short

#     return df

def add_exit_signals(
    df,
    max_adverse_bars=5,
    slope_peak_max_col="slope_peak_max_raw",
    slope_peak_min_col="slope_peak_min_raw",
):
    """
    Exit logic (independent for long & short):

    LONG exit if:
        slope_peak_max_raw OR
        accumulated (Mid < entry_price) >= max_adverse_bars

    SHORT exit if:
        slope_peak_min_raw OR
        accumulated (Mid > entry_price) >= max_adverse_bars
    """

    mids = df["Mid"].to_numpy()
    peak_max = df[slope_peak_max_col].to_numpy()
    peak_min = df[slope_peak_min_col].to_numpy()

    entry_long = df["entry_long"].to_numpy()
    entry_short = df["entry_short"].to_numpy()

    n = len(df)

    exit_long = [False] * n
    exit_short = [False] * n

    in_long = False
    in_short = False

    entry_price_long = None
    entry_price_short = None

    adverse_long = 0
    adverse_short = 0

    for i in range(n):

        # -----------------------
        # LONG ENTRY TRIGGER
        # -----------------------
        if entry_long[i] and not in_long:
            in_long = True
            entry_price_long = mids[i]
            adverse_long = 0

        # -----------------------
        # LONG EXIT
        # -----------------------
        if in_long:
            if mids[i] < entry_price_long:
                adverse_long += 1

            if peak_max[i] or (adverse_long >= max_adverse_bars):
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
        # SHORT EXIT
        # -----------------------
        if in_short:
            if mids[i] > entry_price_short:
                adverse_short += 1

            if peak_min[i] or (adverse_short >= max_adverse_bars):
                exit_short[i] = True
                in_short = False
                entry_price_short = None
                adverse_short = 0

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
    Pure slope extrema for exit logic.
    No region filters, no quantile logic, no recency constraints.
    """
    s = df[slope_col].to_numpy()

    peak_min_raw = (np.roll(s, 1) > s) & (s < np.roll(s, -1))
    peak_max_raw = (np.roll(s, 1) < s) & (s > np.roll(s, -1))

    peak_min_raw[[0, -1]] = False
    peak_max_raw[[0, -1]] = False

    df["slope_peak_min_raw"] = peak_min_raw
    df["slope_peak_max_raw"] = peak_max_raw

    return df

# def add_trade_ids(
#     df,
#     entry_long_col="entry_long",
#     exit_long_col="exit_long",
#     entry_short_col="entry_short",
#     exit_short_col="exit_short",
# ):
#     """
#     Correct trade ID assignment:
#     - Only assign IDs to REAL entries/exits (i.e., per-side flat→entry, entry→exit).
#     - Per side only one open position at a time.
#     - IDs alternate 1 → 2 → 1 → 2.
#     """

#     n = len(df)

#     trade_id_long = [None] * n
#     trade_id_short = [None] * n

#     # ---- LONG SIDE ----
#     in_long = False
#     current_long_id = None
#     next_long_id = 1   # alternates 1 ↔ 2

#     for i in range(n):

#         if not in_long:
#             # A real long entry happens only when flat on the long side
#             if df[entry_long_col].iloc[i]:
#                 in_long = True
#                 current_long_id = next_long_id
#                 trade_id_long[i] = current_long_id

#                 # Prepare the next ID: toggle 1↔2
#                 next_long_id = 2 if next_long_id == 1 else 1

#         else:
#             # We are in a long position
#             if df[exit_long_col].iloc[i]:
#                 trade_id_long[i] = current_long_id
#                 in_long = False
#                 current_long_id = None

#     # ---- SHORT SIDE ----
#     in_short = False
#     current_short_id = None
#     next_short_id = 1   # alternates 1 ↔ 2

#     for i in range(n):

#         if not in_short:
#             if df[entry_short_col].iloc[i]:
#                 in_short = True
#                 current_short_id = next_short_id
#                 trade_id_short[i] = current_short_id

#                 next_short_id = 2 if next_short_id == 1 else 1

#         else:
#             if df[exit_short_col].iloc[i]:
#                 trade_id_short[i] = current_short_id
#                 in_short = False
#                 current_short_id = None

#     df["trade_id_long"] = trade_id_long
#     df["trade_id_short"] = trade_id_short

#     return df

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

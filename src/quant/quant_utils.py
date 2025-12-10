import numpy as np
import pandas as pd


# -------------------------------------------------------
# 1. Full-sample ECDF (cheating / feasibility mode)
# -------------------------------------------------------

def compute_ecdf(series):
    """
    Compute the empirical CDF (ECDF) of a series.

    Parameters
    ----------
    series : array-like or pd.Series
        Input data.

    Returns
    -------
    xs : np.ndarray
        Sorted values.
    F  : np.ndarray
        ECDF values corresponding to xs (in [0,1]).
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    xs = np.sort(s)
    n = len(xs)
    if n == 0:
        raise ValueError("Series is empty after removing NaNs.")
    F = np.arange(1, n + 1) / n
    return xs, F


def ecdf_value(x, xs, F):
    """
    Evaluate ECDF at a value x using precomputed xs and F.

    ECDF(x) = proportion of samples <= x.

    Parameters
    ----------
    x : float or array-like
    xs : np.ndarray
        Sorted samples.
    F : np.ndarray
        ECDF values.

    Returns
    -------
    float or np.ndarray
        ECDF value(s).
    """
    x_arr = np.asarray(x, dtype=float)
    idx = np.searchsorted(xs, x_arr, side="right") - 1
    idx = np.clip(idx, -1, len(xs) - 1)
    out = np.where(idx == -1, 0.0, F[idx])
    return out if out.shape != () else float(out)


def compute_full_ecdf_quantiles(series):
    """
    Full-sample quantiles for every point in series using the
    ECDF computed from the entire series (cheating mode).

    Parameters
    ----------
    series : array-like or pd.Series

    Returns
    -------
    q_full : np.ndarray
        Quantile for each point in series in [0,1].
    """
    xs, F = compute_ecdf(series)
    return ecdf_value(series, xs, F)


# -------------------------------------------------------
# 2. Rolling ECDF (real-time / no cheating mode)
# -------------------------------------------------------

def compute_rolling_ecdf(series, min_points=50):
    """
    Compute rolling (online) ECDF quantiles where each quantile at time t
    uses ONLY data up to t (inclusive). This is trading-valid.

    Implementation:
        q[t] = rank(series[:t]) / t

    Parameters
    ----------
    series : array-like or pd.Series
    min_points : int, default 50
        For t < min_points, quantile is NaN (Option A).

    Returns
    -------
    q_roll : np.ndarray
        Rolling ECDF quantiles, same length as series.
    """
    s = np.asarray(series, dtype=float)
    n = len(s)
    q_roll = np.full(n, np.nan, dtype=float)

    # We'll maintain a sorted list of past values.
    # For n ~ 5k per day, this is totally fine.
    sorted_vals = []

    for t in range(n):
        x = s[t]

        if np.isnan(x):
            # keep NaN quantile; don't insert NaN into distribution
            continue

        # Insert x into sorted_vals (binary insertion)
        insert_pos = np.searchsorted(sorted_vals, x, side="right")
        sorted_vals.insert(insert_pos, x)

        if t + 1 < min_points:
            continue

        # rank = number of values <= x
        rank = insert_pos + 1
        q_roll[t] = rank / (t + 1)

    return q_roll


def compute_daily_rolling_ecdf(series, min_points=50, window_max=100):
    """
    Daily, causal ECDF quantiles with:
      - warm-up: no quantiles until at least `min_points` samples
      - window grows until `window_max`
      - then fixed-size rolling window of size `window_max`

    Parameters
    ----------
    series : array-like or pd.Series
        Input data (e.g. daily KF slope).
    min_points : int, default 50
        Minimum number of observations required before emitting quantiles.
        Until this is reached, output is NaN.
    window_max : int, default 100
        Maximum window size. After this is reached, the window becomes
        rolling: last `window_max` observations only.

    Returns
    -------
    q_roll : np.ndarray
        Quantiles in [0,1], same length as series.
    """
    s = np.asarray(series, dtype=float)
    n = len(s)
    q_roll = np.full(n, np.nan, dtype=float)

    window_vals = []

    for t in range(n):
        x = s[t]

        if np.isnan(x):
            # skip NaNs: quantile stays NaN, do not insert into window
            continue

        # Append new value
        window_vals.append(x)

        # Enforce max window size
        if len(window_vals) > window_max:
            # drop oldest
            window_vals.pop(0)

        window_size = len(window_vals)

        # Warm-up: we require at least `min_points` samples in the window
        if window_size < min_points:
            continue

        # Compute ECDF(x) = proportion of window_vals <= x
        w = np.asarray(window_vals, dtype=float)
        w_sorted = np.sort(w)
        # number of values <= x
        rank = np.searchsorted(w_sorted, x, side="right")
        q_roll[t] = rank / window_size

    return q_roll


# -------------------------------------------------------
# 3. Quantile bin utilities
# -------------------------------------------------------

def quantile_flags(q, q_values=(0.9, 0.95, 0.97, 0.99)):
    """
    Create boolean flags indicating whether quantiles are in extreme regions.

    Parameters
    ----------
    q : array-like
        Quantiles in [0,1].
    q_values : iterable
        Thresholds for extremes.

    Returns
    -------
    dict of np.ndarray
        {
          'high_0.90': q >= 0.90,
          'low_0.90' : q <= 0.10,
          ...
        }
    """
    q = np.asarray(q, dtype=float)
    flags = {}
    for v in q_values:
        flags[f"high_{v:.2f}"] = q >= v
        flags[f"low_{v:.2f}"]  = q <= (1.0 - v)
    return flags


def add_slope_quantiles(df, slope_col="KF_slope_adapt", min_points=50):
    """
    Convenience function to add both full-sample and rolling ECDF quantiles
    to a processed dataframe.

    Adds columns:
        - slope_q_full
        - slope_q_roll

    Parameters
    ----------
    df : pd.DataFrame
    slope_col : str
    min_points : int

    Returns
    -------
    pd.DataFrame
        Copy with new columns added.
    """
    if slope_col not in df.columns:
        raise ValueError(f"Column '{slope_col}' not found in dataframe.")

    out = df.copy()
    slope = out[slope_col].to_numpy(dtype=float)

    out["slope_q_full"] = compute_full_ecdf_quantiles(slope)
    out["slope_q_roll"] = compute_rolling_ecdf(slope, min_points=min_points)

    return out

def add_slope_quantiles_daily(
    df,
    slope_col="KF_slope_adapt",
    min_points=50,
    window_max=100,
):
    """
    Add daily, causal quantiles for slope using compute_daily_rolling_ecdf.

    This is intended to be used on a *single trading session* (one day),
    e.g. after select_intraday_session().

    Adds column:
        - slope_q_roll_daily   (trading-valid intraday quantile)

    Parameters
    ----------
    df : pd.DataFrame
        Single-day dataframe (already sliced to the intraday session).
    slope_col : str
        Column containing the slope (KF_slope_adapt).
    min_points : int
        Warm-up size: no quantiles before this many samples in the window.
    window_max : int
        Maximum window size; after this is reached window becomes rolling.

    Returns
    -------
    pd.DataFrame
        Copy with slope_q_roll_daily added.
    """
    if slope_col not in df.columns:
        raise ValueError(f"Column '{slope_col}' not found in dataframe.")

    out = df.copy()
    slope = out[slope_col].to_numpy(dtype=float)

    q_roll_daily = compute_daily_rolling_ecdf(
        slope,
        min_points=min_points,
        window_max=window_max,
    )

    out["slope_q_roll_daily"] = q_roll_daily

    return out


# -------------------------------------------------------
# 4. Baseline + Conditional Probabilities (Milestone 2.2)
# -------------------------------------------------------

def compute_baseline_probabilities(df, tp_col="TP_label"):
    """
    Compute unconditional baseline probabilities of turning points.

    Returns:
        {
            "P_max": P(TP_label == +1),
            "P_min": P(TP_label == -1),
            "P_extremum": P(abs(TP_label) == 1)
        }
    """
    if tp_col not in df.columns:
        raise ValueError(f"Column '{tp_col}' not found in dataframe.")

    tp = df[tp_col].to_numpy()

    n = len(tp)
    if n == 0:
        raise ValueError("Empty dataframe; cannot compute baseline probabilities.")

    P_max = np.mean(tp == 1)
    P_min = np.mean(tp == -1)
    P_ext = np.mean(np.abs(tp) == 1)

    return {
        "P_max": float(P_max),
        "P_min": float(P_min),
        "P_extremum": float(P_ext)
    }


def compute_conditional_probabilities(
    df,
    q_col="slope_q_full",
    tp_col="TP_label",
    q_vals=(0.90, 0.95, 0.97, 0.99)
):
    """
    Compute conditional probabilities of turning points given extreme slope quantiles.

    For each threshold Q in q_vals:
        High-slope region: q >= Q  -> test for maxima
        Low-slope region:  q <= 1-Q -> test for minima

    Returns a dict keyed by threshold:
        results[Q] = {
            "P_max_given_high": ...,
            "P_min_given_low": ...,
            "count_high": ...,
            "count_low": ...
        }

    Notes:
    - NaNs in q_col are ignored (especially early in rolling ECDF).
    """
    if q_col not in df.columns:
        raise ValueError(f"Column '{q_col}' not found in dataframe.")
    if tp_col not in df.columns:
        raise ValueError(f"Column '{tp_col}' not found in dataframe.")

    q = df[q_col].to_numpy(dtype=float)
    tp = df[tp_col].to_numpy(dtype=int)

    results = {}

    for Q in q_vals:
        high_mask = q >= Q
        low_mask  = q <= (1.0 - Q)

        # ignore NaNs automatically: comparisons with NaN are False

        count_high = int(np.sum(high_mask))
        count_low  = int(np.sum(low_mask))

        if count_high > 0:
            P_max_given_high = np.sum((tp == 1) & high_mask) / count_high
        else:
            P_max_given_high = np.nan

        if count_low > 0:
            P_min_given_low = np.sum((tp == -1) & low_mask) / count_low
        else:
            P_min_given_low = np.nan

        results[float(Q)] = {
            "P_max_given_high": float(P_max_given_high) if not np.isnan(P_max_given_high) else np.nan,
            "P_min_given_low": float(P_min_given_low) if not np.isnan(P_min_given_low) else np.nan,
            "count_high": count_high,
            "count_low": count_low,
        }

    return results


def conditional_probabilities_table(
    df,
    q_col="slope_q_full",
    tp_col="TP_label",
    q_vals=(0.90, 0.95, 0.97, 0.99)
):
    """
    Build a tidy pandas DataFrame summarizing:
      - baseline probabilities
      - conditional probabilities at each threshold
      - lift ratios

    Returns columns:
        Threshold,
        P_max_given_high, P_min_given_low,
        P_max_baseline,   P_min_baseline,
        Lift_max,         Lift_min,
        count_high,       count_low
    """
    base = compute_baseline_probabilities(df, tp_col=tp_col)
    cond = compute_conditional_probabilities(df, q_col=q_col, tp_col=tp_col, q_vals=q_vals)

    rows = []
    for Q in q_vals:
        r = cond[float(Q)]

        P_max_high = r["P_max_given_high"]
        P_min_low  = r["P_min_given_low"]

        P_max_base = base["P_max"]
        P_min_base = base["P_min"]

        Lift_max = (P_max_high / P_max_base) if (P_max_base > 0 and not np.isnan(P_max_high)) else np.nan
        Lift_min = (P_min_low / P_min_base) if (P_min_base > 0 and not np.isnan(P_min_low)) else np.nan

        rows.append({
            "Threshold": float(Q),
            "P_max_given_high": P_max_high,
            "P_min_given_low": P_min_low,
            "P_max_baseline": P_max_base,
            "P_min_baseline": P_min_base,
            "Lift_max": float(Lift_max) if not np.isnan(Lift_max) else np.nan,
            "Lift_min": float(Lift_min) if not np.isnan(Lift_min) else np.nan,
            "count_high": r["count_high"],
            "count_low": r["count_low"],
        })

    return pd.DataFrame(rows)

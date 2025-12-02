import numpy as np
import pandas as pd
import plotly.graph_objects as go


# -------------------------------------------------------
# 1) High-resolution probability curve
# -------------------------------------------------------

def compute_probability_curve(
    df,
    q_col="slope_q_roll",
    tp_col="TP_label",
    q_min=0.80,
    q_max=0.99,
    n_points=100,
):
    """
    Compute high-resolution conditional probability curves:
        P(max | q >= Q)
        P(min | q <= 1-Q)
    for Q in linspace(q_min, q_max, n_points).

    Returns a DataFrame with:
        Threshold, P_max_given_high, P_min_given_low,
        count_high, count_low
    """
    if q_col not in df.columns:
        raise ValueError(f"Column '{q_col}' not found.")
    if tp_col not in df.columns:
        raise ValueError(f"Column '{tp_col}' not found.")

    q = df[q_col].to_numpy(dtype=float)
    tp = df[tp_col].to_numpy(dtype=int)

    thresholds = np.linspace(q_min, q_max, n_points)
    rows = []

    for Q in thresholds:
        high_mask = q >= Q
        low_mask = q <= (1.0 - Q)

        count_high = int(np.sum(high_mask))
        count_low = int(np.sum(low_mask))

        P_max_high = (
            np.sum((tp == 1) & high_mask) / count_high
            if count_high > 0 else np.nan
        )
        P_min_low = (
            np.sum((tp == -1) & low_mask) / count_low
            if count_low > 0 else np.nan
        )

        rows.append({
            "Threshold": float(Q),
            "P_max_given_high": float(P_max_high) if not np.isnan(P_max_high) else np.nan,
            "P_min_given_low": float(P_min_low) if not np.isnan(P_min_low) else np.nan,
            "count_high": count_high,
            "count_low": count_low,
        })

    return pd.DataFrame(rows)


def plot_probability_curve(curve_df, title="P(turning point | slope quantile extreme)"):
    """
    Plot probability curves from compute_probability_curve().
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve_df["Threshold"],
        y=curve_df["P_max_given_high"],
        mode="lines",
        name="P(max | q>=Q)"
    ))

    fig.add_trace(go.Scatter(
        x=curve_df["Threshold"],
        y=curve_df["P_min_given_low"],
        mode="lines",
        name="P(min | q<=1-Q)"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Quantile threshold Q",
        yaxis_title="Conditional probability",
        template="plotly_white"
    )
    return fig


# -------------------------------------------------------
# 2) Slope distributions (overall + by TP class)
# -------------------------------------------------------

def slope_distributions(df, slope_col="KF_slope_adapt", tp_col="TP_label"):
    """
    Return slope arrays for:
        all points
        maxima points
        minima points
        non-turning points
    """
    if slope_col not in df.columns:
        raise ValueError(f"Column '{slope_col}' not found.")
    if tp_col not in df.columns:
        raise ValueError(f"Column '{tp_col}' not found.")

    slope = df[slope_col].to_numpy(dtype=float)
    tp = df[tp_col].to_numpy(dtype=int)

    return {
        "all": slope,
        "maxima": slope[tp == 1],
        "minima": slope[tp == -1],
        "non_tp": slope[tp == 0],
    }


def plot_slope_distributions(
    df,
    slope_col="KF_slope_adapt",
    tp_col="TP_label",
    nbins=80,
    title="Slope distributions"
):
    """
    Plot histograms of slope:
      - all points
      - maxima
      - minima
      - non-turning points
    """
    d = slope_distributions(df, slope_col, tp_col)

    fig = go.Figure()
    for name, arr in d.items():
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=arr,
            nbinsx=nbins,
            name=name,
            opacity=0.55,
            histnorm="probability"
        ))

    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis_title=slope_col,
        yaxis_title="Probability",
        template="plotly_white"
    )
    return fig


# -------------------------------------------------------
# 3) Time-to-next-extremum diagnostics
# -------------------------------------------------------

def compute_time_to_extremum(df, tp_col="TP_label"):
    """
    For each t, compute distance (in bars) to:
      - next local maximum
      - next local minimum

    Returns df copy with:
      time_to_next_max, time_to_next_min
    """
    if tp_col not in df.columns:
        raise ValueError(f"Column '{tp_col}' not found.")

    tp = df[tp_col].to_numpy(dtype=int)
    n = len(tp)

    next_max_dist = np.full(n, np.nan)
    next_min_dist = np.full(n, np.nan)

    max_idx = np.where(tp == 1)[0]
    min_idx = np.where(tp == -1)[0]

    # For efficiency: walk through indices
    max_ptr = 0
    min_ptr = 0

    for t in range(n):
        while max_ptr < len(max_idx) and max_idx[max_ptr] < t:
            max_ptr += 1
        while min_ptr < len(min_idx) and min_idx[min_ptr] < t:
            min_ptr += 1

        if max_ptr < len(max_idx):
            next_max_dist[t] = max_idx[max_ptr] - t
        if min_ptr < len(min_idx):
            next_min_dist[t] = min_idx[min_ptr] - t

    out = df.copy()
    out["time_to_next_max"] = next_max_dist
    out["time_to_next_min"] = next_min_dist
    return out


def compute_distance_curve(
    df,
    dist_col,
    q_col="slope_q_roll",
    n_bins=20,
    q_min=0.0,
    q_max=1.0,
):
    """
    Bin slope quantiles and compute median (and IQR) distance to next extremum.

    Returns a DataFrame with:
        q_bin_center, median_dist, p25_dist, p75_dist, count
    """
    if dist_col not in df.columns:
        raise ValueError(f"Column '{dist_col}' not found.")
    if q_col not in df.columns:
        raise ValueError(f"Column '{q_col}' not found.")

    q = df[q_col].to_numpy(dtype=float)
    d = df[dist_col].to_numpy(dtype=float)

    # drop NaNs
    mask = (~np.isnan(q)) & (~np.isnan(d))
    q = q[mask]
    d = d[mask]

    bins = np.linspace(q_min, q_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    rows = []
    for i in range(n_bins):
        m = (q >= bins[i]) & (q < bins[i+1])
        dd = d[m]
        if len(dd) == 0:
            rows.append({
                "q_bin_center": float(centers[i]),
                "median_dist": np.nan,
                "p25_dist": np.nan,
                "p75_dist": np.nan,
                "count": 0
            })
            continue

        rows.append({
            "q_bin_center": float(centers[i]),
            "median_dist": float(np.median(dd)),
            "p25_dist": float(np.percentile(dd, 25)),
            "p75_dist": float(np.percentile(dd, 75)),
            "count": int(len(dd))
        })

    return pd.DataFrame(rows)


def plot_distance_curve(curve_df, title="Distance to next extremum vs slope quantile"):
    """
    Plot median distance with IQR band.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve_df["q_bin_center"],
        y=curve_df["median_dist"],
        mode="lines+markers",
        name="median distance"
    ))

    fig.add_trace(go.Scatter(
        x=curve_df["q_bin_center"],
        y=curve_df["p75_dist"],
        mode="lines",
        name="p75",
        line=dict(width=1),
        opacity=0.4
    ))

    fig.add_trace(go.Scatter(
        x=curve_df["q_bin_center"],
        y=curve_df["p25_dist"],
        mode="lines",
        name="p25",
        line=dict(width=1),
        opacity=0.4
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Slope quantile",
        yaxis_title="Bars to next extremum",
        template="plotly_white"
    )
    return fig


# -------------------------------------------------------
# 4) Intraday stability diagnostics
# -------------------------------------------------------

def intraday_features(df, slope_col="KF_slope_adapt", q_col="slope_q_roll"):
    """
    Add time-of-day features for intraday analysis.
    Returns df copy with:
      tod_minutes (minutes since midnight)
    """
    out = df.copy()
    tod = out.index.tz_convert("US/Eastern").time
    tod_minutes = np.array([t.hour * 60 + t.minute + t.second / 60.0 for t in tod])
    out["tod_minutes"] = tod_minutes
    return out


def plot_intraday_stability(
    df,
    slope_col="KF_slope_adapt",
    q_col="slope_q_roll",
    tp_col="TP_label",
    title="Intraday stability: slope & quantile"
):
    """
    Two-panel style plot (overlaid traces):
      - slope vs time-of-day
      - quantile vs time-of-day
      - turning points highlighted
    """
    if slope_col not in df.columns:
        raise ValueError(f"Column '{slope_col}' not found.")
    if q_col not in df.columns:
        raise ValueError(f"Column '{q_col}' not found.")
    if tp_col not in df.columns:
        raise ValueError(f"Column '{tp_col}' not found.")

    dfi = intraday_features(df, slope_col, q_col)

    fig = go.Figure()

    # slope scatter
    fig.add_trace(go.Scatter(
        x=dfi["tod_minutes"],
        y=dfi[slope_col],
        mode="markers",
        name="slope",
        opacity=0.45
    ))

    # quantile scatter (scaled to slope axis? no—use secondary axis later in notebook if desired)
    fig.add_trace(go.Scatter(
        x=dfi["tod_minutes"],
        y=dfi[q_col],
        mode="markers",
        name="slope quantile",
        opacity=0.45
    ))

    # turning points overlay
    tp_max = dfi[dfi[tp_col] == 1]
    tp_min = dfi[dfi[tp_col] == -1]

    fig.add_trace(go.Scatter(
        x=tp_max["tod_minutes"],
        y=tp_max[slope_col],
        mode="markers",
        name="TP maxima",
        marker=dict(size=8),
    ))

    fig.add_trace(go.Scatter(
        x=tp_min["tod_minutes"],
        y=tp_min[slope_col],
        mode="markers",
        name="TP minima",
        marker=dict(size=8),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Minutes since midnight (US/Eastern)",
        yaxis_title="Slope / Quantile",
        template="plotly_white"
    )
    return fig


# -------------------------------------------------------
# 5) Single-day orchestration
# -------------------------------------------------------

def analyze_day(
    symbol,
    date,
    interval=5,
    L=20,
    use_roll_quantiles=True,
    min_points=50,
):
    """
    One-stop analysis for a single symbol + single date.

    Steps:
      1) load_processed_df
      2) label_turning_points
      3) add_slope_quantiles
      4) compute probability curve
      5) compute time-to-next-extremum + distance curves

    Returns:
      df_final, artifacts dict

    artifacts contains:
      - prob_curve_df
      - dist_curve_max_df
      - dist_curve_min_df
    """
    from src.data.data_utils import load_processed_df
    from src.labeling.label_utils import label_turning_points
    from src.quant.quant_utils import add_slope_quantiles

    df = load_processed_df(symbol, date, date, interval)
    df = label_turning_points(df, L=L)
    df = add_slope_quantiles(df, slope_col="KF_slope_adapt", min_points=min_points)

    q_col = "slope_q_roll" if use_roll_quantiles else "slope_q_full"

    prob_curve_df = compute_probability_curve(df, q_col=q_col, tp_col="TP_label")

    df = compute_time_to_extremum(df, tp_col="TP_label")

    dist_curve_max_df = compute_distance_curve(df, "time_to_next_max", q_col=q_col)
    dist_curve_min_df = compute_distance_curve(df, "time_to_next_min", q_col=q_col)

    artifacts = {
        "prob_curve_df": prob_curve_df,
        "dist_curve_max_df": dist_curve_max_df,
        "dist_curve_min_df": dist_curve_min_df,
        "q_col": q_col
    }

    return df, artifacts

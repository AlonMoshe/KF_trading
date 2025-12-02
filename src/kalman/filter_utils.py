import numpy as np
import pandas as pd


# -------------------------------------------------------
# 1. Cleaning raw data: Mid, Spread, EWMA variance, etc.
# -------------------------------------------------------

def clean_raw_data(df: pd.DataFrame, Delta: float, lambda_: float = 0.94) -> pd.DataFrame:
    """
    Take a raw IB dataframe with columns:
        ['Open', 'High', 'Low', 'Close', 'Volume', ... , 'Date' or Datetime index]
    and add:
        - Mid
        - Spread
        - Spread_smooth
        - var_dMid (EWMA variance of mid-price changes)
        - var_acc (acceleration variance proxy)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe indexed by Datetime (tz-aware) with OHLCV columns.
    Delta : float
        Time step in seconds (e.g. 5.0 for 5-second bars).
    lambda_ : float, default 0.94
        Decay factor for EWMA variance.

    Returns
    -------
    pd.DataFrame
        Same dataframe with additional columns.
    """
    df = df.copy()

    # Mid-price (average of open and close or high/low; we keep your choice: Open+Close)
    df["Mid"] = (df["Open"] + df["Close"]) / 2.0

    # Proxy for spread
    df["Spread"] = df["High"] - df["Low"]

    # EWMA variance of mid-price changes
    returns = df["Mid"].diff()
    df["var_dMid"] = (returns ** 2).ewm(alpha=(1 - lambda_), adjust=False).mean()

    # Acceleration variance proxy
    df["var_acc"] = df["var_dMid"] / (Delta ** 4)

    # Smoothed spread for measurement noise scaling
    df["Spread_smooth"] = df["Spread"].ewm(alpha=0.2, adjust=False).mean()

    return df


# -------------------------------------------------------
# 2. Adaptive Kalman filter (level, slope, curvature)
# -------------------------------------------------------

def _Q_ca(q_acc: float, Delta: float) -> np.ndarray:
    """
    Constant-acceleration process noise covariance for time step Delta.
    """
    d = Delta
    d2 = d ** 2
    d3 = d ** 3
    d4 = d ** 4

    return q_acc * np.array([
        [d4 / 4.0, d3 / 2.0, d2 / 2.0],
        [d3 / 2.0, d2,       d       ],
        [d2 / 2.0, d,        1.0     ]
    ], dtype=float)


def apply_adaptive_kf(df: pd.DataFrame, Delta: float) -> pd.DataFrame:
    """
    Apply the adaptive Kalman filter (local quadratic model) to a cleaned dataframe.

    Expects df to have at least:
        - 'Mid'
        - 'Spread'
        - 'Spread_smooth'
        - 'var_acc'

    Adds the columns:
        - KF_level_adapt
        - KF_slope_adapt
        - KF_curv_adapt
        - KF_level_var_adapt
        - KF_slope_var_adapt
        - KF_curv_var_adapt

    Parameters
    ----------
    df : pd.DataFrame
        Clean dataframe with required columns.
    Delta : float
        Time step in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with KF estimates and variances added.
    """
    df = df.copy()

    # --- State transition (local quadratic: level, slope, curvature) ---
    A = np.array([
        [1.0,       Delta,        0.5 * (Delta ** 2)],
        [0.0,       1.0,          Delta],
        [0.0,       0.0,          1.0]
    ], dtype=float)

    # We observe level only: Mid ≈ level
    H = np.array([[1.0, 0.0, 0.0]], dtype=float)

    # --- Basic statistics from data for scaling ---
    mid = df["Mid"].to_numpy(dtype=float)
    n = len(mid)
    if n == 0:
        raise ValueError("Dataframe is empty; cannot run Kalman filter.")

    # Spread-related stats
    half_spread_med = (df["Spread"] / 2.0).median()

    # Typical mid-price change
    dmid = np.diff(mid, prepend=mid[0])
    typical_dmid = np.median(np.abs(dmid[1:]))  # skip the first diff which is zero

    # --- Process noise scale (q_acc) ---
    q_scale = 1e-4
    q_acc = max(1e-12, float((typical_dmid / Delta) ** 2) * q_scale)

    # Adaptive scaling coefficients (tunable)
    cR = 10.0    # measurement noise scale vs spread^2
    cQ = 1e-4    # process noise scale vs var_acc

    # Floors to avoid extreme Q_t/R_t
    R_floor = (df["Spread"].median() / 2.0) ** 2 * 5.0
    Q_floor = q_acc * 0.2

    # --- Initial state and covariance ---
    x = np.zeros((3, 1), dtype=float)
    x[0, 0] = mid[0]  # initial level = first price

    P = np.diag([
        half_spread_med ** 2,
        typical_dmid ** 2,
        1.0
    ]) * 100.0

    I3 = np.eye(3, dtype=float)

    # --- Pre-allocate outputs ---
    level_est_a = np.zeros(n)
    slope_est_a = np.zeros(n)
    curv_est_a  = np.zeros(n)
    level_var_a = np.zeros(n)
    slope_var_a = np.zeros(n)
    curv_var_a  = np.zeros(n)

    gain_adapt = np.zeros(n)

    # --- Vectorized input arrays ---
    var_acc_arr = df["var_acc"].to_numpy(dtype=float)
    spread_arr  = df["Spread_smooth"].to_numpy(dtype=float)
    y_arr       = mid

    # --- Main Kalman loop ---
    for t in range(n):
        # Predict
        x = A @ x

        # Adaptive Q_t from var_acc
        q_t = var_acc_arr[t]
        Q_t = _Q_ca(max(Q_floor, cQ * q_t), Delta)

        # Adaptive R_t from smoothed spread
        spr = spread_arr[t]
        R_t = np.array([[max(R_floor, cR * (spr ** 2))]], dtype=float)

        # Covariance prediction
        P = A @ P @ A.T + Q_t

        # Measurement update
        z = np.array([[y_arr[t]]])
        S = H @ P @ H.T + R_t
        K = P @ H.T @ np.linalg.inv(S)
        innov = z - (H @ x)
        x = x + K @ innov
        P = (I3 - K @ H) @ P

        # Save results
        level_est_a[t] = x[0, 0]
        slope_est_a[t] = x[1, 0]
        curv_est_a[t]  = x[2, 0]
        level_var_a[t] = P[0, 0]
        slope_var_a[t] = P[1, 1]
        curv_var_a[t]  = P[2, 2]
        gain_adapt[t]  = K[0, 0]

    # Attach to dataframe
    df["KF_level_adapt"]      = level_est_a
    df["KF_slope_adapt"]      = slope_est_a
    df["KF_curv_adapt"]       = curv_est_a
    df["KF_level_var_adapt"]  = level_var_a
    df["KF_slope_var_adapt"]  = slope_var_a
    df["KF_curv_var_adapt"]   = curv_var_a

    # We don't return gain_adapt as a column for now, but we could if useful
    # df["KF_gain_adapt"] = gain_adapt

    return df

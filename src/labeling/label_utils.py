import numpy as np
import pandas as pd


# -------------------------------------------------------
# Turning Point Labeling
# -------------------------------------------------------

def label_turning_points(df: pd.DataFrame, column="KF_level_adapt", L=20):
    """
    Label turning points (local maxima and minima) in a price series.

    A point t is labeled a turning point if:
        p[t] is the maximum or minimum in the symmetric window [t-L, t+L]
    using the value of df[column].

    Labels:
        +1 = local maximum
        -1 = local minimum
         0 = neither

    Special handling:
        - Flat windows (all equal) → no turning point
        - Monotonic windows → naturally produce no turning point
        - Edge region (t < L or t > N-L) → label = 0

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the column with price/level data.
    column : str, default "KF_level_adapt"
        The price series to use for peak/trough detection.
    L : int
        Window size. Turning point must be max/min within ±L bars.

    Returns
    -------
    pd.DataFrame
        Same dataframe with a new column 'TP_label'.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in dataframe.")

    p = df[column].to_numpy()
    n = len(p)

    labels = np.zeros(n, dtype=int)

    # Iterate through the valid range: avoid edges
    for t in range(L, n - L):

        window = p[t - L : t + L + 1]

        # 1. Flat window? No turning point.
        if window.max() == window.min():
            continue

        # 2. Local maximum
        if p[t] == window.max():
            labels[t] = 1
            continue

        # 3. Local minimum
        if p[t] == window.min():
            labels[t] = -1
            continue

        # 4. Else: label stays 0

    df = df.copy()
    df["TP_label"] = labels

    return df

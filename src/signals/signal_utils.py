from aiohttp_client_cache import logger
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


# def add_slope_peaks(
#     df,
#     slope_col="KF_slope_adapt",
#     region_max_col="region_max",
#     region_min_col="region_min",
#     smooth=False,
#     smooth_window=5,
#     peak_half_window=1,
#     peak_hysteresis=0.0,
#     peak_min_swing=0.0,
#     logger=None
# ):
#     """
#     STRICTLY CAUSAL slope peaks/troughs for timing entries.

#     Logic:
#       - A local extremum of slope at index t is only known once we see s[t+1].
#       - Therefore, the *signal* is placed at index t+1.
#       - This function also gates peaks by turning regions.

#     Adds:
#         slope_peak_max  (causal, inside region_max)
#         slope_peak_min  (causal, inside region_min)
#     """
#     df = df.copy()
#     s = df[slope_col].astype(float)

#     if smooth:
#         s_use = s.rolling(smooth_window, min_periods=1).mean()
#     else:
#         s_use = s

#     region_max = df[region_max_col].fillna(False).to_numpy(bool)
#     region_min = df[region_min_col].fillna(False).to_numpy(bool)

#     n = len(df)
#     peak_max = np.zeros(n, dtype=bool)
#     peak_min = np.zeros(n, dtype=bool)

#     s_vals = s_use.to_numpy(dtype=float)

#     # -------------------------------------------------
#     # Windowed peak detection (strictly causal)
#     # -------------------------------------------------
#     k = int(peak_half_window)
#     if k < 1:
#         k = 1

#     if k == 1:
#         # Original 3-point logic, with optional hysteresis
#         for t in range(1, n - 1):
#             right = t + 1

#             # raw local max at t
#             if (s_vals[t] > s_vals[t - 1]) and (s_vals[t] >= s_vals[right]) and region_max[t]:

#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "PEAK MAX CANDIDATE (RAW)")
#                     logger.log(ts, f"slope_center={s_vals[t]}")
#                     logger.log(ts, f"slope_right={s_vals[right]}")
#                     logger.log(ts, f"region_max={bool(region_max[t])}")

                
#                 # Hysteresis condition:
#                 if peak_hysteresis > 0.0 and (s_vals[t] - s_vals[right] < peak_hysteresis):
#                     if logger is not None:
#                         logger.log(ts, f"=>REJECTED: hysteresis_fail (delta={s_vals[t]-s_vals[right]:.6f})")
#                     continue


#                 # Swing-size condition: peak must exceed lower neighbor by min amount
#                 local_min = min(s_vals[t - 1], s_vals[right])
#                 if peak_min_swing > 0.0 and (s_vals[t] - local_min < peak_min_swing):
#                     if logger is not None:
#                         logger.log(ts, f"=>REJECTED: swing_fail (swing={s_vals[t]-local_min:.6f})")
#                     continue


#                 peak_max[right] = True
                
#             else:
#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "=> PEAK MAX REJECTED (no promotion)")

                
#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     local_min = min(s_vals[t - 1], s_vals[right])
#                     logger.log(ts,"FILTERED PEAK MAX DETECTED")
#                     logger.log(ts,f"slope={s_vals[t]}")
#                     logger.log(ts,f"region_max={region_max[t]}")
#                     logger.log(ts,f"hysteresis_pass={(peak_hysteresis==0 or (s_vals[t]-s_vals[right]>=peak_hysteresis))}")
#                     logger.log(ts,f"swing_pass={(peak_min_swing==0 or (s_vals[t]-local_min>=peak_min_swing))}")
#                     logger.log(ts,"=>slope_peak_max=True")



#             # raw local min at t
#             if (s_vals[t] < s_vals[t - 1]) and (s_vals[t] <= s_vals[right]) and region_min[t]:

#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "PEAK MIN CANDIDATE (RAW)")
#                     logger.log(ts, f"slope_center={s_vals[t]}")
#                     logger.log(ts, f"slope_right={s_vals[right]}")
#                     logger.log(ts, f"region_min={bool(region_min[t])}")

#                 # Hysteresis condition:
#                 if peak_hysteresis > 0.0 and (s_vals[right] - s_vals[t] < peak_hysteresis):
#                     if logger is not None:
#                         logger.log(ts, f"=>REJECTED: hysteresis_fail (delta={s_vals[right]-s_vals[t]:.6f})")
#                     continue

#                 # Swing-size condition: valley must be below upper neighbor by min amount
#                 local_max = max(s_vals[t - 1], s_vals[right])
#                 if peak_min_swing > 0.0 and (local_max - s_vals[t] < peak_min_swing):
#                     if logger is not None:
#                         logger.log(ts, f"=>REJECTED: swing_fail (swing={local_max-s_vals[t]:.6f})")
#                     continue

#                 peak_min[right] = True
                
#             else:
#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "=> PEAK MIN REJECTED (no promotion)")

                
#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     local_max = max(s_vals[t - 1], s_vals[right])   # ✅ correct for k=1
#                     logger.log(ts,"FILTERED PEAK MIN DETECTED")
#                     logger.log(ts,f"slope={s_vals[t]}")
#                     logger.log(ts,f"region_min={region_min[t]}")
#                     logger.log(ts,f"hysteresis_pass={(peak_hysteresis==0 or (s_vals[right]-s_vals[t]>=peak_hysteresis))}")
#                     logger.log(ts,f"swing_pass={(peak_min_swing==0 or (local_max-s_vals[t]>=peak_min_swing))}")
#                     logger.log(ts,"=>slope_peak_min=True")




#     else:
#         # General windowed logic with optional hysteresis:
#         #   - center index c runs from k .. n-k-1
#         #   - window = [c-k, ..., c, ..., c+k]
#         #   - we can only confirm after seeing s[c+k]
#         #   - so the *signal* is placed at index (c+k)
#         for c in range(k, n - k):
#             left = c - k
#             right = c + k
#             window = s_vals[left:right + 1]
#             val = s_vals[c]
#             # Precompute local extremes for logging & swing conditions
#             window_min = window.min()
#             window_max = window.max()


#             # Max: center is the maximum of the window and in region_max
#             if region_max[c] and val == window.max():
                
#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "PEAK MAX CANDIDATE (WINDOWED)")
#                     logger.log(ts, f"slope_center={val}")
#                     logger.log(ts, f"window_min={window_min}")
#                     logger.log(ts, f"region_max={bool(region_max[c])}")


#                 # Hysteresis condition:
#                 if peak_hysteresis > 0.0 and (val - s_vals[right] < peak_hysteresis):
#                     if logger is not None:
#                         logger.log(ts, f"=>REJECTED: hysteresis_fail (delta={val-s_vals[right]:.6f})")
#                     continue

#                 # Swing-size condition:
#                 local_min = window_min
#                 if peak_min_swing > 0.0 and (val - local_min < peak_min_swing):
#                     if logger is not None:
#                         logger.log(ts, f"=>REJECTED: swing_fail (swing={val-local_min:.6f})")
#                     continue

#                 peak_max[right] = True
                
#                 # LOG WINDOWED MAX PEAK
#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
                   
#                     logger.log(ts,"FILTERED PEAK MAX DETECTED (WINDOWED)")
#                     logger.log(ts,f"slope_center={val}")
#                     logger.log(ts,f"local_min_window={local_min}")
#                     logger.log(ts,f"region_max={bool(region_max[c])}")
#                     logger.log(ts,f"hysteresis_pass={(peak_hysteresis == 0 or (val - s_vals[right] >= peak_hysteresis))}")
#                     logger.log(ts,f"swing_pass={(peak_min_swing == 0 or (val - local_min >= peak_min_swing))}")
#                     logger.log(ts,"=>slope_peak_max=True")            
            
#             else:
#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "=> PEAK MAX REJECTED (windowed)")

            
            
#             if region_min[c] and val == window.min():
                
#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "PEAK MIN CANDIDATE (WINDOWED)")
#                     logger.log(ts, f"slope_center={val}")
#                     logger.log(ts, f"window_max={window_max}")
#                     logger.log(ts, f"region_min={bool(region_min[c])}")

#                 # Hysteresis condition:
#                 if peak_hysteresis > 0.0 and (s_vals[right] - val < peak_hysteresis):
#                     if logger is not None:
#                         logger.log(ts, f"=>REJECTED: hysteresis_fail (delta={s_vals[right]-val:.6f})")
#                     continue

#                 # Swing-size condition:
#                 local_max = window_max
#                 if peak_min_swing > 0.0 and (local_max - val < peak_min_swing):
#                     if logger is not None:
#                         logger.log(ts, f"=>REJECTED: swing_fail (swing={local_max-val:.6f})")
#                     continue

#                 peak_min[right] = True

#                 # LOG WINDOWED MIN PEAK
#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")

#                     logger.log(ts,"FILTERED PEAK MIN DETECTED (WINDOWED)")
#                     logger.log(ts,f"slope_center={val}")
#                     logger.log(ts,f"local_max_window={local_max}")
#                     logger.log(ts,f"region_min={bool(region_min[c])}")
#                     logger.log(ts,f"hysteresis_pass={(peak_hysteresis == 0 or (s_vals[right] - val >= peak_hysteresis))}")
#                     logger.log(ts,f"swing_pass={(peak_min_swing == 0 or (local_max - val >= peak_min_swing))}")
#                     logger.log(ts,"=>slope_peak_min=True")
                    
#             else:
#                 if logger is not None:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "=> PEAK MIN REJECTED (windowed)")




#     df["slope_peak_max"] = peak_max
#     df["slope_peak_min"] = peak_min

#     return df

# def add_slope_peaks(
#     df,
#     slope_col="KF_slope_adapt",
#     region_max_col="region_max",
#     region_min_col="region_min",
#     smooth=False,
#     smooth_window=5,
#     peak_half_window=1,
#     peak_hysteresis=0.0,
#     peak_min_swing=0.0,
#     logger=None
# ):
#     """
#     STRICTLY CAUSAL slope peaks/troughs for timing entries.

#     A peak at index t is only confirmed once bar t+1 exists,
#     so the signal is placed at index t+1.

#     Adds:
#         slope_peak_max
#         slope_peak_min
#     """
#     df = df.copy()
#     s = df[slope_col].astype(float)

#     if smooth:
#         s_use = s.rolling(smooth_window, min_periods=1).mean()
#     else:
#         s_use = s

#     region_max = df[region_max_col].fillna(False).to_numpy(bool)
#     region_min = df[region_min_col].fillna(False).to_numpy(bool)

#     n = len(df)
#     peak_max = np.zeros(n, dtype=bool)
#     peak_min = np.zeros(n, dtype=bool)

#     s_vals = s_use.to_numpy(dtype=float)

#     k = max(1, int(peak_half_window))

#     # =========================================================
#     # k == 1 : RAW (3-point) logic
#     # =========================================================
#     if k == 1:
#         for t in range(1, n - 1):
#             right = t + 1

#             # ---------------- MAX ----------------
#             if (s_vals[t] > s_vals[t - 1]) and (s_vals[t] >= s_vals[right]) and region_max[t]:

#                 if logger:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "PEAK MAX CANDIDATE (RAW)")
#                     logger.log(ts, f"slope_center={s_vals[t]}")
#                     logger.log(ts, f"slope_right={s_vals[right]}")
#                     logger.log(ts, f"region_max={bool(region_max[t])}")

#                 # hysteresis
#                 if peak_hysteresis > 0.0 and (s_vals[t] - s_vals[right] < peak_hysteresis):
#                     if logger:
#                         logger.log(ts, f"=>REJECTED: hysteresis_fail (delta={s_vals[t]-s_vals[right]:.6f})")
#                         logger.log(ts, "=> PEAK MAX REJECTED")
#                     continue

#                 # swing
#                 local_min = min(s_vals[t - 1], s_vals[right])
#                 if peak_min_swing > 0.0 and (s_vals[t] - local_min < peak_min_swing):
#                     if logger:
#                         logger.log(ts, f"=>REJECTED: swing_fail (swing={s_vals[t]-local_min:.6f})")
#                         logger.log(ts, "=> PEAK MAX REJECTED")
#                     continue

#                 # PROMOTION
#                 peak_max[right] = True
#                 if logger:
#                     logger.log(ts, "FILTERED PEAK MAX DETECTED")
#                     logger.log(ts, "=> PEAK MAX PROMOTED")

#             # ---------------- MIN ----------------
#             if (s_vals[t] < s_vals[t - 1]) and (s_vals[t] <= s_vals[right]) and region_min[t]:

#                 if logger:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "PEAK MIN CANDIDATE (RAW)")
#                     logger.log(ts, f"slope_center={s_vals[t]}")
#                     logger.log(ts, f"slope_right={s_vals[right]}")
#                     logger.log(ts, f"region_min={bool(region_min[t])}")

#                 # hysteresis
#                 if peak_hysteresis > 0.0 and (s_vals[right] - s_vals[t] < peak_hysteresis):
#                     if logger:
#                         logger.log(ts, f"=>REJECTED: hysteresis_fail (delta={s_vals[right]-s_vals[t]:.6f})")
#                         logger.log(ts, "=> PEAK MIN REJECTED")
#                     continue

#                 # swing
#                 local_max = max(s_vals[t - 1], s_vals[right])
#                 if peak_min_swing > 0.0 and (local_max - s_vals[t] < peak_min_swing):
#                     if logger:
#                         logger.log(ts, f"=>REJECTED: swing_fail (swing={local_max-s_vals[t]:.6f})")
#                         logger.log(ts, "=> PEAK MIN REJECTED")
#                     continue

#                 # PROMOTION
#                 peak_min[right] = True
#                 if logger:
#                     logger.log(ts, "FILTERED PEAK MIN DETECTED")
#                     logger.log(ts, "=> PEAK MIN PROMOTED")

#     # =========================================================
#     # k > 1 : WINDOWED logic
#     # =========================================================
#     else:
#         for c in range(k, n - k):
#             left = c - k
#             right = c + k
#             window = s_vals[left:right + 1]
#             val = s_vals[c]

#             window_min = window.min()
#             window_max = window.max()

#             # ---------------- MAX ----------------
#             if region_max[c] and val == window_max:

#                 if logger:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "PEAK MAX CANDIDATE (WINDOWED)")
#                     logger.log(ts, f"slope_center={val}")
#                     logger.log(ts, f"region_max={bool(region_max[c])}")

#                 if peak_hysteresis > 0.0 and (val - s_vals[right] < peak_hysteresis):
#                     if logger:
#                         logger.log(ts, f"=>REJECTED: hysteresis_fail (delta={val-s_vals[right]:.6f})")
#                         logger.log(ts, "=> PEAK MAX REJECTED")
#                     continue

#                 if peak_min_swing > 0.0 and (val - window_min < peak_min_swing):
#                     if logger:
#                         logger.log(ts, f"=>REJECTED: swing_fail (swing={val-window_min:.6f})")
#                         logger.log(ts, "=> PEAK MAX REJECTED")
#                     continue

#                 peak_max[right] = True
#                 if logger:
#                     logger.log(ts, "FILTERED PEAK MAX DETECTED (WINDOWED)")
#                     logger.log(ts, "=> PEAK MAX PROMOTED")

#             # ---------------- MIN ----------------
#             if region_min[c] and val == window_min:

#                 if logger:
#                     ts = df.index[right].strftime("%H:%M:%S")
#                     logger.log(ts, "PEAK MIN CANDIDATE (WINDOWED)")
#                     logger.log(ts, f"slope_center={val}")
#                     logger.log(ts, f"region_min={bool(region_min[c])}")

#                 if peak_hysteresis > 0.0 and (s_vals[right] - val < peak_hysteresis):
#                     if logger:
#                         logger.log(ts, f"=>REJECTED: hysteresis_fail (delta={s_vals[right]-val:.6f})")
#                         logger.log(ts, "=> PEAK MIN REJECTED")
#                     continue

#                 if peak_min_swing > 0.0 and (window_max - val < peak_min_swing):
#                     if logger:
#                         logger.log(ts, f"=>REJECTED: swing_fail (swing={window_max-val:.6f})")
#                         logger.log(ts, "=> PEAK MIN REJECTED")
#                     continue

#                 peak_min[right] = True
#                 if logger:
#                     logger.log(ts, "FILTERED PEAK MIN DETECTED (WINDOWED)")
#                     logger.log(ts, "=> PEAK MIN PROMOTED")

#     df["slope_peak_max"] = peak_max
#     df["slope_peak_min"] = peak_min
#     return df

# def add_slope_peaks(
#     df,
#     slope_col="KF_slope_adapt",
#     region_max_col="region_max",
#     region_min_col="region_min",
#     smooth=False,
#     smooth_window=5,
#     peak_half_window=1,
#     peak_hysteresis=0.0,
#     peak_min_swing=0.0,
#     logger=None
# ):
#     """
#     STRICTLY CAUSAL slope peaks with FULL EXPLANATORY LOGGING.

#     Every raw extremum is:
#       - evaluated as a candidate
#       - either PROMOTED or REJECTED
#       - rejection reasons are always logged
#     """

#     df = df.copy()
#     s = df[slope_col].astype(float)

#     if smooth:
#         s_use = s.rolling(smooth_window, min_periods=1).mean()
#     else:
#         s_use = s

#     s_vals = s_use.to_numpy()
#     region_max = df[region_max_col].fillna(False).to_numpy(bool)
#     region_min = df[region_min_col].fillna(False).to_numpy(bool)

#     n = len(df)
#     peak_max = np.zeros(n, dtype=bool)
#     peak_min = np.zeros(n, dtype=bool)

#     k = max(1, int(peak_half_window))

#     # =====================================================
#     # RAW (k == 1)
#     # =====================================================
#     if k == 1:
#         for t in range(1, n - 1):
#             right = t + 1
#             ts = df.index[right].strftime("%H:%M:%S") if logger else None

#             # ---------- RAW MAX ----------
#             is_raw_max = (s_vals[t] > s_vals[t - 1]) and (s_vals[t] >= s_vals[right])

#             if is_raw_max:
#                 if logger:
#                     logger.log(ts, "RAW MAX PEAK DETECTED")
#                     logger.log(ts, f"slope_center={s_vals[t]}")

#                 reasons = []

#                 if not region_max[t]:
#                     reasons.append("region_max=False")

#                 delta = s_vals[t] - s_vals[right]
#                 if peak_hysteresis > 0.0 and delta < peak_hysteresis:
#                     reasons.append(f"hysteresis_fail (delta={delta:.6f})")

#                 local_min = min(s_vals[t - 1], s_vals[right])
#                 swing = s_vals[t] - local_min
#                 if peak_min_swing > 0.0 and swing < peak_min_swing:
#                     reasons.append(f"swing_fail (swing={swing:.6f})")

#                 if reasons:
#                     if logger:
#                         logger.log(ts, "PEAK MAX REJECTED")
#                         for r in reasons:
#                             logger.log(ts, f"  reason: {r}")
#                 else:
#                     peak_max[right] = True
#                     if logger:
#                         logger.log(ts, "PEAK MAX PROMOTED")

#             # ---------- RAW MIN ----------
#             is_raw_min = (s_vals[t] < s_vals[t - 1]) and (s_vals[t] <= s_vals[right])

#             if is_raw_min:
#                 if logger:
#                     logger.log(ts, "RAW MIN PEAK DETECTED")
#                     logger.log(ts, f"slope_center={s_vals[t]}")

#                 reasons = []

#                 if not region_min[t]:
#                     reasons.append("region_min=False")

#                 delta = s_vals[right] - s_vals[t]
#                 if peak_hysteresis > 0.0 and delta < peak_hysteresis:
#                     reasons.append(f"hysteresis_fail (delta={delta:.6f})")

#                 local_max = max(s_vals[t - 1], s_vals[right])
#                 swing = local_max - s_vals[t]
#                 if peak_min_swing > 0.0 and swing < peak_min_swing:
#                     reasons.append(f"swing_fail (swing={swing:.6f})")

#                 if reasons:
#                     if logger:
#                         logger.log(ts, "PEAK MIN REJECTED")
#                         for r in reasons:
#                             logger.log(ts, f"  reason: {r}")
#                 else:
#                     peak_min[right] = True
#                     if logger:
#                         logger.log(ts, "PEAK MIN PROMOTED")

#     # =====================================================
#     # WINDOWED (k > 1)
#     # =====================================================
#     else:
#         for c in range(k, n - k):
#             left = c - k
#             right = c + k
#             ts = df.index[right].strftime("%H:%M:%S") if logger else None

#             window = s_vals[left:right + 1]
#             val = s_vals[c]
#             window_min = window.min()
#             window_max = window.max()

#             # ---------- MAX ----------
#             if val == window_max:
#                 if logger:
#                     logger.log(ts, "RAW MAX PEAK DETECTED (WINDOWED)")
#                     logger.log(ts, f"slope_center={val}")

#                 reasons = []

#                 if not region_max[c]:
#                     reasons.append("region_max=False")

#                 delta = val - s_vals[right]
#                 if peak_hysteresis > 0.0 and delta < peak_hysteresis:
#                     reasons.append(f"hysteresis_fail (delta={delta:.6f})")

#                 swing = val - window_min
#                 if peak_min_swing > 0.0 and swing < peak_min_swing:
#                     reasons.append(f"swing_fail (swing={swing:.6f})")

#                 if reasons:
#                     if logger:
#                         logger.log(ts, "PEAK MAX REJECTED (WINDOWED)")
#                         for r in reasons:
#                             logger.log(ts, f"  reason: {r}")
#                 else:
#                     peak_max[right] = True
#                     if logger:
#                         logger.log(ts, "PEAK MAX PROMOTED (WINDOWED)")

#             # ---------- MIN ----------
#             if val == window_min:
#                 if logger:
#                     logger.log(ts, "RAW MIN PEAK DETECTED (WINDOWED)")
#                     logger.log(ts, f"slope_center={val}")

#                 reasons = []

#                 if not region_min[c]:
#                     reasons.append("region_min=False")

#                 delta = s_vals[right] - val
#                 if peak_hysteresis > 0.0 and delta < peak_hysteresis:
#                     reasons.append(f"hysteresis_fail (delta={delta:.6f})")

#                 swing = window_max - val
#                 if peak_min_swing > 0.0 and swing < peak_min_swing:
#                     reasons.append(f"swing_fail (swing={swing:.6f})")

#                 if reasons:
#                     if logger:
#                         logger.log(ts, "PEAK MIN REJECTED (WINDOWED)")
#                         for r in reasons:
#                             logger.log(ts, f"  reason: {r}")
#                 else:
#                     peak_min[right] = True
#                     if logger:
#                         logger.log(ts, "PEAK MIN PROMOTED (WINDOWED)")

#     df["slope_peak_max"] = peak_max
#     df["slope_peak_min"] = peak_min
#     return df

def add_slope_peaks(
    df,
    slope_col="KF_slope_adapt",
    region_max_col="region_max",
    region_min_col="region_min",
    smooth=False,
    smooth_window=5,
    peak_half_window=1,
    peak_hysteresis=0.0,
    peak_min_swing=0.0,
    peak_to_trough=0.0,   # NEW
    logger=None,
):
    """
    Detect slope peaks/troughs with FULL TRACEABILITY.

    Existing behavior is preserved.
    New behavior: every detected peak now ends with an explicit
    FINAL PEAK DECISION log block.
    """

    df = df.copy()
    s = df[slope_col].astype(float)
    s_raw = df[slope_col].to_numpy(dtype=float)


    if smooth:
        s_use = s.rolling(smooth_window, min_periods=1).mean()
    else:
        s_use = s

    s_vals = s_use.to_numpy()
    region_max = df[region_max_col].fillna(False).to_numpy(bool)
    region_min = df[region_min_col].fillna(False).to_numpy(bool)

    n = len(df)
    peak_max = np.zeros(n, dtype=bool)
    peak_min = np.zeros(n, dtype=bool)
    peak_max_raw_idx = np.full(n, -1, dtype=int)
    peak_min_raw_idx = np.full(n, -1, dtype=int)
    
    # Track last APPROVED windowed extremum (slope value)
    last_windowed_peak_val = None    # last MAX
    last_windowed_trough_val = None  # last MIN

    


    k = max(1, int(peak_half_window))

    # =====================================================
    # RAW PEAKS (k == 1)
    # =====================================================
    if k == 1:
        for t in range(1, n - 1):
            right = t + 1
            ts = df.index[right].strftime("%H:%M:%S") if logger else None
            peak_ts = df.index[t].strftime("%H:%M:%S")


            # ---------------- MAX ----------------
            if (s_vals[t] > s_vals[t - 1]) and (s_vals[t] >= s_vals[right]):

                if logger:
                    logger.log(ts, "RAW MAX PEAK DETECTED")
                    logger.log(ts, f"slope_center={s_vals[t]}")
                    logger.log(ts, f"peak_bar_time={peak_ts}")
                    logger.log(ts, f"evaluation_bar_time={ts}")

                reasons = []

                if not region_max[t]:
                    reasons.append("region_max=False")

                delta = s_vals[t] - s_vals[right]
                if peak_hysteresis > 0 and delta < peak_hysteresis:
                    reasons.append(f"hysteresis_fail (delta={delta:.6f})")

                local_min = min(s_vals[t - 1], s_vals[right])
                swing = s_vals[t] - local_min
                if peak_min_swing > 0 and swing < peak_min_swing:
                    reasons.append(f"min_swing_fail (swing={swing:.6f})")

                if not reasons:
                    peak_max[right] = True
                    peak_max_raw_idx[right] = t


                if logger:
                    logger.log(ts, "FINAL PEAK MAX DECISION")
                    logger.log(ts, f"promoted={not reasons}")
                    if reasons:
                        for r in reasons:
                            logger.log(ts, f"  reason: {r}")

            # ---------------- MIN ----------------
            if (s_vals[t] < s_vals[t - 1]) and (s_vals[t] <= s_vals[right]):

                if logger:
                    logger.log(ts, "RAW MIN PEAK DETECTED")
                    logger.log(ts, f"slope_center={s_vals[t]}")
                    logger.log(ts, f"peak_bar_time={peak_ts}")
                    logger.log(ts, f"evaluation_bar_time={ts}")

                reasons = []

                if not region_min[t]:
                    reasons.append("region_min=False")

                delta = s_vals[right] - s_vals[t]
                if peak_hysteresis > 0 and delta < peak_hysteresis:
                    reasons.append(f"hysteresis_fail (delta={delta:.6f})")

                local_max = max(s_vals[t - 1], s_vals[right])
                swing = local_max - s_vals[t]
                if peak_min_swing > 0 and swing < peak_min_swing:
                    reasons.append(f"min_swing_fail (swing={swing:.6f})")

                if not reasons:
                    peak_min[right] = True
                    peak_min_raw_idx[right] = t


                if logger:
                    logger.log(ts, "FINAL PEAK MIN DECISION")
                    logger.log(ts, f"promoted={not reasons}")
                    if reasons:
                        for r in reasons:
                            logger.log(ts, f"  reason: {r}")

    # =====================================================
    # WINDOWED PEAKS (k > 1)
    # =====================================================
    else:
        for c in range(k, n - k):
            left = c - k
            right = c + k
            ts = df.index[right].strftime("%H:%M:%S") if logger else None
            peak_ts = df.index[c].strftime("%H:%M:%S")


            window = s_vals[left : right + 1]
            val = s_vals[c]
            
            # raw slope extremum inside the same window
            raw_window = s_raw[left : right + 1]


            # ---------------- MAX ----------------
            if val == window.max():
                raw_ext_idx = left + np.argmax(raw_window)
                if logger:
                    logger.log(ts, "RAW MAX PEAK DETECTED (WINDOWED)")
                    logger.log(ts, f"slope_center={val}")
                    logger.log(ts, f"peak_bar_time={peak_ts}")
                    logger.log(ts, f"evaluation_bar_time={ts}")
                    
                    raw_peak_ts = df.index[raw_ext_idx].strftime("%H:%M:%S")
                    logger.log(ts, f"raw_slope_peak_bar_time={raw_peak_ts}")


                reasons = []

                if not region_max[c]:
                    reasons.append("region_max=False")

                delta = val - s_vals[right]
                if peak_hysteresis > 0 and delta < peak_hysteresis:
                    reasons.append(f"hysteresis_fail (delta={delta:.6f})")

                swing = val - window.min()
                if peak_min_swing > 0 and swing < peak_min_swing:
                    reasons.append(f"min_swing_fail (swing={swing:.6f})")

                if not reasons:

                    # ---------------------------------
                    # peak_to_trough constraint (NEW)
                    # ---------------------------------
                    if peak_to_trough > 0 and last_windowed_trough_val is not None:
                        dist = abs(val - last_windowed_trough_val)
                        if dist < peak_to_trough:
                            reasons.append(
                                f"peak_to_trough_fail (dist={dist:.6f})"
                            )

                    if not reasons:
                        peak_max[right] = True
                        peak_max_raw_idx[right] = raw_ext_idx
                        last_windowed_peak_val = val



                if logger:
                    logger.log(ts, "FINAL PEAK MAX DECISION (WINDOWED)")
                    logger.log(ts, f"promoted={not reasons}")
                    if reasons:
                        for r in reasons:
                            logger.log(ts, f"  reason: {r}")

            # ---------------- MIN ----------------
            if val == window.min():
                raw_ext_idx = left + np.argmin(raw_window)
                if logger:
                    logger.log(ts, "RAW MIN PEAK DETECTED (WINDOWED)")
                    logger.log(ts, f"slope_center={val}")
                    logger.log(ts, f"peak_bar_time={peak_ts}")
                    logger.log(ts, f"evaluation_bar_time={ts}")
                    
                    
                    raw_peak_ts = df.index[raw_ext_idx].strftime("%H:%M:%S")

                    logger.log(ts, f"raw_slope_peak_bar_time={raw_peak_ts}")


                reasons = []

                if not region_min[c]:
                    reasons.append("region_min=False")

                delta = s_vals[right] - val
                if peak_hysteresis > 0 and delta < peak_hysteresis:
                    reasons.append(f"hysteresis_fail (delta={delta:.6f})")

                swing = window.max() - val
                if peak_min_swing > 0 and swing < peak_min_swing:
                    reasons.append(f"min_swing_fail (swing={swing:.6f})")

                if not reasons:

                    # ---------------------------------
                    # peak_to_trough constraint (NEW)
                    # ---------------------------------
                    if peak_to_trough > 0 and last_windowed_peak_val is not None:
                        dist = abs(last_windowed_peak_val - val)
                        if dist < peak_to_trough:
                            reasons.append(
                                f"peak_to_trough_fail (dist={dist:.6f})"
                            )

                    if not reasons:
                        peak_min[right] = True
                        peak_min_raw_idx[right] = raw_ext_idx
                        last_windowed_trough_val = val



                if logger:
                    logger.log(ts, "FINAL PEAK MIN DECISION (WINDOWED)")
                    logger.log(ts, f"promoted={not reasons}")
                    if reasons:
                        for r in reasons:
                            logger.log(ts, f"  reason: {r}")

    df["slope_peak_max_raw_idx"] = peak_max_raw_idx
    df["slope_peak_min_raw_idx"] = peak_min_raw_idx
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
    use_region_recent=True,
    region_recent_window=10,
    logger=None,
):
    """
    ENTRY SIGNAL GENERATION (causal, no position state).

    LONG ENTRY (raw conditions):
        region_min == True
        region_min_recent == True
        slope_peak_min == True
        q <= 1 - Q_high

    SHORT ENTRY (raw conditions):
        region_max == True
        region_max_recent == True
        slope_peak_max == True
        q >= Q_high

    Then a cooldown is applied per side to prevent trade clustering.

    The signal at bar i is meant to be executed at bar (i+1).
    """

    df = df.copy()
    idx = df.index
    n = len(df)

    # -------------------------
    # Extract relevant columns
    # -------------------------
    q = df[q_col].astype(float)
    region_max = df[region_max_col].fillna(False)
    region_min = df[region_min_col].fillna(False)
    peak_max = df[peak_max_col].fillna(False)
    peak_min = df[peak_min_col].fillna(False)
    raw_idx_max = df.get("slope_peak_max_raw_idx", pd.Series(-1, index=df.index)).astype(int)
    raw_idx_min = df.get("slope_peak_min_raw_idx", pd.Series(-1, index=df.index)).astype(int)

    region_gate_max = pd.Series(False, index=df.index)
    region_gate_min = pd.Series(False, index=df.index)
    
    valid_max = raw_idx_max >= 0
    valid_min = raw_idx_min >= 0

    region_gate_max.loc[valid_max] = region_max.iloc[
        raw_idx_max[valid_max].to_numpy()
    ].to_numpy()

    region_gate_min.loc[valid_min] = region_min.iloc[
        raw_idx_min[valid_min].to_numpy()
    ].to_numpy()


    # -------------------------
    # Recency gating
    # -------------------------
    if use_region_recent:
        region_max_recent = region_gate_max.rolling(region_recent_window).max().astype(bool)
        region_min_recent = region_gate_min.rolling(region_recent_window).max().astype(bool)
    else:
        region_max_recent = pd.Series(True, index=df.index)
        region_min_recent = pd.Series(True, index=df.index)

    # -------------------------
    # Probability gating
    # -------------------------
    prob_gate_max = pd.Series(False, index=df.index)
    prob_gate_min = pd.Series(False, index=df.index)



    prob_gate_max.loc[valid_max] = q.iloc[raw_idx_max[valid_max].to_numpy()].to_numpy() >= Q_high
    prob_gate_min.loc[valid_min] = q.iloc[raw_idx_min[valid_min].to_numpy()].to_numpy() <= (1 - Q_high)

    # -------------------------
    # Raw entry conditions
    # -------------------------
    entry_long_raw = (
        region_gate_min &
        region_min_recent &
        peak_min &
        prob_gate_min
    )

    entry_short_raw = (
        region_gate_max &
        region_max_recent &
        peak_max &
        prob_gate_max
    )

    # -------------------------
    # Apply cooldown control
    # -------------------------
    entry_long = []
    entry_short = []

    long_cool = 0
    short_cool = 0

    for i in range(n):
        il = bool(entry_long_raw.iloc[i])
        is_ = bool(entry_short_raw.iloc[i])

        # ====================================================
        # LOG SHORT ENTRY EVALUATION (triggered by slope peak)
        # ====================================================
        if logger is not None and peak_max.iloc[i]:
            ts = idx[i].strftime("%H:%M:%S")
            logger.log(ts,"ENTRY SHORT EVALUATION")
            logger.log(ts,f"slope_peak_max={bool(peak_max.iloc[i])}")
            logger.log(ts,f"region_gate_max={bool(region_gate_max.iloc[i])}")
            logger.log(ts,f"region_max_recent={bool(region_max_recent.iloc[i])}")
            logger.log(ts,f"prob_gate_max={bool(prob_gate_max.iloc[i])}")
            logger.log(ts,f"cooldown={short_cool}")

        # ====================================================
        # LOG LONG ENTRY EVALUATION (triggered by slope peak)
        # ====================================================
        if logger is not None and peak_min.iloc[i]:
            ts = idx[i].strftime("%H:%M:%S")
            logger.log(ts,"ENTRY LONG EVALUATION")
            logger.log(ts,f"slope_peak_min={bool(peak_min.iloc[i])}")
            logger.log(ts,f"region_gate_min={bool(region_gate_min.iloc[i])}")
            logger.log(ts,f"region_min_recent={bool(region_min_recent.iloc[i])}")
            logger.log(ts,f"prob_gate_min={bool(prob_gate_min.iloc[i])}")
            logger.log(ts,f"cooldown={long_cool}")

        # -------------------
        # LONG side cooldown
        # -------------------
        if long_cool > 0:
            entry_long.append(False)
            long_cool -= 1

        else:
            if il:     # raw signal present
                entry_long.append(True)
                long_cool = cooldown
            else:
                entry_long.append(False)

        # -------------------
        # SHORT side cooldown
        # -------------------
        if short_cool > 0:
            entry_short.append(False)
            short_cool -= 1

        else:
            if is_:   # raw signal present
                entry_short.append(True)
                short_cool = cooldown
            else:
                entry_short.append(False)

        # ====================================================
        # SHORT RESULT LOGGING (after cooldown)
        # ====================================================
        if logger is not None and peak_max.iloc[i]:
            if entry_short[-1]:
                logger.log(ts,"=>ENTRY_SHORT=True")
            else:
                reasons = []
                if not region_gate_max.iloc[i]: reasons.append("region_gate_max=False")
                if not region_max_recent.iloc[i]: reasons.append("region_max_recent=False")
                if not prob_gate_max.iloc[i]: reasons.append("prob_gate_max=False")
                if not peak_max.iloc[i]: reasons.append("slope_peak_max=False")
                if short_cool > 0: reasons.append(f"cooldown={short_cool}")
                logger.log(ts,f"=>ENTRY_SHORT REJECTED (reason: {', '.join(reasons)})")

        # ====================================================
        # LONG RESULT LOGGING (after cooldown)
        # ====================================================
        if logger is not None and peak_min.iloc[i]:
            if entry_long[-1]:
                logger.log(ts,"=>ENTRY_LONG=True")
            else:
                reasons = []
                if not region_gate_min.iloc[i]: reasons.append("region_gate_min=False")
                if not region_min_recent.iloc[i]: reasons.append("region_min_recent=False")
                if not prob_gate_min.iloc[i]: reasons.append("prob_gate_min=False")
                if not peak_min.iloc[i]: reasons.append("slope_peak_min=False")
                if long_cool > 0: reasons.append(f"cooldown={long_cool}")
                logger.log(ts,f"=>ENTRY_LONG REJECTED (reason: {', '.join(reasons)})")

    # Convert to numpy arrays
    entry_long = np.array(entry_long, dtype=bool)
    entry_short = np.array(entry_short, dtype=bool)

    # ----------------------------------------------
    # TIME CUTOFF (execution occurs at i+1)
    # ----------------------------------------------
    if last_entry_time is not None:
        hh, mm = map(int, last_entry_time.split(":"))
        cutoff_time = dt.time(hour=hh, minute=mm)

        if idx.tz is None:
            idx_e = idx.tz_localize("US/Eastern")
        else:
            idx_e = idx.tz_convert("US/Eastern")

        times = idx_e.time

        for i in range(n - 1):
            exec_time = times[i + 1]
            if exec_time >= cutoff_time:
                entry_long[i] = False
                entry_short[i] = False

        entry_long[-1] = False
        entry_short[-1] = False

    df["entry_long"] = entry_long
    df["entry_short"] = entry_short

    return df


# def add_exit_signals(
#     df,
#     max_adverse_bars=7,
#     slope_peak_max_col="slope_peak_max_raw",
#     slope_peak_min_col="slope_peak_min_raw",
#     logger=None,  
# ):
#     """
#     STRICTLY CAUSAL EXIT LOGIC.

#     LONG exit if (evaluated at bar i):
#         1. slope_peak_max_raw[i]  (momentum turns against us)
#         2. accumulated adverse bars (Mid < entry_price) >= max_adverse_bars
#         3. forced flat at final bar of the day

#     SHORT exit if:
#         1. slope_peak_min_raw[i]
#         2. accumulated adverse bars (Mid > entry_price) >= max_adverse_bars
#         3. forced flat at final bar of the day

#     IMPORTANT:
#       - Detection happens at bar i.
#       - Actual execution happens at bar i+1 open.
#       - This function only marks exit signals; trade extraction computes price.
#     """

#     df = df.copy()

#     mids = df["Mid"].to_numpy()
#     peak_max = df[slope_peak_max_col].to_numpy(bool)
#     peak_min = df[slope_peak_min_col].to_numpy(bool)
#     entry_long = df["entry_long"].to_numpy(bool)
#     entry_short = df["entry_short"].to_numpy(bool)

#     n = len(df)

#     exit_long = np.zeros(n, dtype=bool)
#     exit_short = np.zeros(n, dtype=bool)

#     # Long state
#     in_long = False
#     entry_price_long = None
#     adverse_long = 0

#     # Short state
#     in_short = False
#     entry_price_short = None
#     adverse_short = 0

#     for i in range(n):

#         # -----------------------
#         # LONG ENTRY TRIGGER
#         # -----------------------
#         if entry_long[i] and not in_long:
#             in_long = True
#             entry_price_long = mids[i]   # decision bar's close/mid (execution at i+1 open)
#             adverse_long = 0

#         # -----------------------
#         # LONG EXIT LOGIC
#         # -----------------------
#         if in_long:

#             # Adverse price action (Mid < entry price)
#             if mids[i] < entry_price_long:
#                 adverse_long += 1

#             condition_peak = peak_max[i]
#             condition_adverse = (adverse_long >= max_adverse_bars)
            
#             if logger is not None:
#                 ts = df.index[i].strftime("%H:%M:%S")
#                 logger.log(ts,"EXIT LONG EVALUATION")
#                 logger.log(ts,f"slope_peak_max_raw={bool(peak_max[i])}")
#                 logger.log(ts,f"adverse_bars={adverse_long}")
#                 logger.log(ts,f"adverse_limit={max_adverse_bars}")


#             if condition_peak or condition_adverse:
#                 if logger is not None:
#                     logger.log(ts,"=>EXIT_LONG=True")

#                 exit_long[i] = True
#                 in_long = False
#                 entry_price_long = None
#                 adverse_long = 0
#             else:
#                 if logger is not None:
#                     logger.log(ts,"=>EXIT_LONG NOT TRIGGERED")


#         # -----------------------
#         # SHORT ENTRY TRIGGER
#         # -----------------------
#         if entry_short[i] and not in_short:
#             in_short = True
#             entry_price_short = mids[i]
#             adverse_short = 0

#         # -----------------------
#         # SHORT EXIT LOGIC
#         # -----------------------
#         if in_short:
            
#             # adverse move
#             if mids[i] > entry_price_short:
#                 adverse_short += 1

#             if logger is not None:
#                 ts = df.index[i].strftime("%H:%M:%S")
#                 logger.log(ts,"EXIT SHORT EVALUATION")
#                 logger.log(ts,f"slope_peak_min_raw={bool(peak_min[i])}")
#                 logger.log(ts,f"adverse_bars={adverse_short}")
#                 logger.log(ts,f"adverse_limit={max_adverse_bars}")

            

#             condition_peak = peak_min[i]
#             condition_adverse = (adverse_short >= max_adverse_bars)

#             # --- SHORT EXIT DECISION ---
#             if condition_peak or condition_adverse:
#                 if logger is not None:
#                     logger.log(ts,"=>EXIT_SHORT=True")

#                 exit_short[i] = True
#                 in_short = False
#                 entry_price_short = None
#                 adverse_short = 0
#             else:
#                 if logger is not None:
#                     logger.log(ts,"=>EXIT_SHORT NOT TRIGGERED")

                
        


#     # -----------------------
#     # FORCE FLAT AT FINAL BAR
#     # -----------------------
#     last_i = n - 1

#     if in_long:
#         exit_long[last_i] = True

#     if in_short:
#         exit_short[last_i] = True

#     df["exit_long"] = exit_long
#     df["exit_short"] = exit_short

#     return df

# def add_exit_signals(
#     df,
#     max_adverse_bars=7,
#     slope_peak_max_col="slope_peak_max_raw",
#     slope_peak_min_col="slope_peak_min_raw",
#     logger=None,
# ):
#     """
#     STRICTLY CAUSAL, STATE-AWARE EXIT LOGIC.

#     Exit logic is evaluated ONLY when a position of that side is open,
#     and NEVER on the same bar the position was opened.

#     LONG exit if:
#         - slope_peak_max_raw == True
#         - OR adverse bars >= max_adverse_bars

#     SHORT exit if:
#         - slope_peak_min_raw == True
#         - OR adverse bars >= max_adverse_bars
#     """

#     df = df.copy()

#     mids = df["Mid"].to_numpy()
#     peak_max = df[slope_peak_max_col].to_numpy(bool)
#     peak_min = df[slope_peak_min_col].to_numpy(bool)
#     entry_long = df["entry_long"].to_numpy(bool)
#     entry_short = df["entry_short"].to_numpy(bool)

#     n = len(df)

#     exit_long = np.zeros(n, dtype=bool)
#     exit_short = np.zeros(n, dtype=bool)

#     # -----------------------------
#     # LONG STATE
#     # -----------------------------
#     in_long = False
#     long_entry_bar = None
#     long_entry_price = None
#     adverse_long = 0

#     # -----------------------------
#     # SHORT STATE
#     # -----------------------------
#     in_short = False
#     short_entry_bar = None
#     short_entry_price = None
#     adverse_short = 0

#     for i in range(n):

#         ts = df.index[i].strftime("%H:%M:%S") if logger is not None else None

#         # =====================================================
#         # LONG ENTRY
#         # =====================================================
#         if entry_long[i] and not in_long:
#             in_long = True
#             long_entry_bar = i
#             long_entry_price = mids[i]
#             adverse_long = 0

#         # =====================================================
#         # LONG EXIT (only if long is open AND not entry bar)
#         # =====================================================
#         if in_long and i != long_entry_bar:

#             if mids[i] < long_entry_price:
#                 adverse_long += 1

#             condition_peak = peak_max[i]
#             condition_adverse = adverse_long >= max_adverse_bars

#             if logger is not None:
#                 logger.log(ts, "EXIT LONG EVALUATION")
#                 logger.log(ts, f"slope_peak_max_raw={bool(peak_max[i])}")
#                 logger.log(ts, f"adverse_bars={adverse_long}")
#                 logger.log(ts, f"adverse_limit={max_adverse_bars}")

#             if condition_peak or condition_adverse:
#                 exit_long[i] = True
#                 in_long = False
#                 long_entry_bar = None
#                 long_entry_price = None
#                 adverse_long = 0

#                 if logger is not None:
#                     logger.log(ts, "=>EXIT_LONG=True")

#             else:
#                 if logger is not None:
#                     logger.log(ts, "=>EXIT_LONG NOT TRIGGERED")

#         # =====================================================
#         # SHORT ENTRY
#         # =====================================================
#         if entry_short[i] and not in_short:
#             in_short = True
#             short_entry_bar = i
#             short_entry_price = mids[i]
#             adverse_short = 0

#         # =====================================================
#         # SHORT EXIT (only if short is open AND not entry bar)
#         # =====================================================
#         if in_short and i != short_entry_bar:

#             if mids[i] > short_entry_price:
#                 adverse_short += 1

#             condition_peak = peak_min[i]
#             condition_adverse = adverse_short >= max_adverse_bars

#             if logger is not None:
#                 logger.log(ts, "EXIT SHORT EVALUATION")
#                 logger.log(ts, f"slope_peak_min_raw={bool(peak_min[i])}")
#                 logger.log(ts, f"adverse_bars={adverse_short}")
#                 logger.log(ts, f"adverse_limit={max_adverse_bars}")

#             if condition_peak or condition_adverse:
#                 exit_short[i] = True
#                 in_short = False
#                 short_entry_bar = None
#                 short_entry_price = None
#                 adverse_short = 0

#                 if logger is not None:
#                     logger.log(ts, "=>EXIT_SHORT=True")

#             else:
#                 if logger is not None:
#                     logger.log(ts, "=>EXIT_SHORT NOT TRIGGERED")

#     # =====================================================
#     # FORCE FLAT AT FINAL BAR (NO LOGGING)
#     # =====================================================
#     last_i = n - 1

#     if in_long:
#         exit_long[last_i] = True

#     if in_short:
#         exit_short[last_i] = True

#     df["exit_long"] = exit_long
#     df["exit_short"] = exit_short

#     return df

def add_exit_signals(
    df,
    max_adverse_bars=7,
    slope_peak_max_col="slope_peak_max_raw",
    slope_peak_min_col="slope_peak_min_raw",
    logger=None,
):
    """
    STRICTLY CAUSAL, STATE-AWARE EXIT LOGIC (NO SILENT EXITS).

    Exit logic is evaluated ONLY when a position of that side is open,
    and ONLY starting from i = entry_bar + 1.

    LONG exit if:
        - slope_peak_max_raw == True
        - OR adverse bars >= max_adverse_bars
        - OR end-of-day (explicitly logged)

    SHORT exit if:
        - slope_peak_min_raw == True
        - OR adverse bars >= max_adverse_bars
        - OR end-of-day (explicitly logged)
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

    # -----------------------------
    # LONG STATE
    # -----------------------------
    in_long = False
    long_entry_bar = None
    long_entry_price = None
    adverse_long = 0

    # -----------------------------
    # SHORT STATE
    # -----------------------------
    in_short = False
    short_entry_bar = None
    short_entry_price = None
    adverse_short = 0

    for i in range(n):

        ts = df.index[i].strftime("%H:%M:%S") if logger else None

        # =====================================================
        # LONG ENTRY (state update only)
        # =====================================================
        if entry_long[i] and not in_long:
            in_long = True
            long_entry_bar = i
            long_entry_price = mids[i]
            adverse_long = 0

        # =====================================================
        # LONG EXIT (only if open AND i >= entry_bar + 1)
        # =====================================================
        if in_long and i >= long_entry_bar + 1:

            if mids[i] < long_entry_price:
                adverse_long += 1

            condition_peak = peak_max[i]
            condition_adverse = adverse_long >= max_adverse_bars

            if logger:
                logger.log(ts, "EXIT LONG EVALUATION")
                logger.log(ts, f"slope_peak_max_raw={bool(peak_max[i])}")
                logger.log(ts, f"adverse_bars={adverse_long}")
                logger.log(ts, f"adverse_limit={max_adverse_bars}")

            if condition_peak or condition_adverse:
                exit_long[i] = True
                in_long = False
                long_entry_bar = None
                long_entry_price = None
                adverse_long = 0

                if logger:
                    reason = (
                        "opposite_raw_peak"
                        if condition_peak
                        else "max_adverse_bars"
                    )
                    logger.log(ts, f"=>EXIT_LONG=True (reason={reason})")

        # =====================================================
        # SHORT ENTRY (state update only)
        # =====================================================
        if entry_short[i] and not in_short:
            in_short = True
            short_entry_bar = i
            short_entry_price = mids[i]
            adverse_short = 0

        # =====================================================
        # SHORT EXIT (only if open AND i >= entry_bar + 1)
        # =====================================================
        if in_short and i >= short_entry_bar + 1:

            if mids[i] > short_entry_price:
                adverse_short += 1

            condition_peak = peak_min[i]
            condition_adverse = adverse_short >= max_adverse_bars

            if logger:
                logger.log(ts, "EXIT SHORT EVALUATION")
                logger.log(ts, f"slope_peak_min_raw={bool(peak_min[i])}")
                logger.log(ts, f"adverse_bars={adverse_short}")
                logger.log(ts, f"adverse_limit={max_adverse_bars}")

            if condition_peak or condition_adverse:
                exit_short[i] = True
                in_short = False
                short_entry_bar = None
                short_entry_price = None
                adverse_short = 0

                if logger:
                    reason = (
                        "opposite_raw_peak"
                        if condition_peak
                        else "max_adverse_bars"
                    )
                    logger.log(ts, f"=>EXIT_SHORT=True (reason={reason})")

    # =====================================================
    # END-OF-DAY EXIT (EXPLICIT + LOGGED)
    # =====================================================
    last_i = n - 1
    ts = df.index[last_i].strftime("%H:%M:%S") if logger else None

    if in_long:
        exit_long[last_i] = True
        if logger:
            logger.log(ts, "END_OF_DAY_EXIT LONG")
            logger.log(ts, "=>EXIT_LONG=True (reason=end_of_day)")

    if in_short:
        exit_short[last_i] = True
        if logger:
            logger.log(ts, "END_OF_DAY_EXIT SHORT")
            logger.log(ts, "=>EXIT_SHORT=True (reason=end_of_day)")

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


# def add_raw_slope_peaks(df, slope_col="KF_slope_adapt", logger=None):

#     """
#     STRICTLY CAUSAL pure slope extrema for exit logic.
#     No region filters, no quantile logic, no recency constraints.

#     A raw extremum at index t is only known when bar t+1 arrives,
#     so the signal is placed at index t+1.

#     Adds:
#         slope_peak_min_raw
#         slope_peak_max_raw
#     """
#     df = df.copy()
#     s = df[slope_col].to_numpy(dtype=float)
#     n = len(s)

#     peak_min_raw = np.zeros(n, dtype=bool)
#     peak_max_raw = np.zeros(n, dtype=bool)

#     for t in range(1, n - 1):
#         # local min at t
#         if (s[t] < s[t - 1]) and (s[t] <= s[t + 1]):
#             peak_min_raw[t + 1] = True

#             if logger is not None:
#                 ts = df.index[t + 1].strftime("%H:%M:%S")
#                 logger.log(ts, "RAW MIN PEAK DETECTED")
#                 logger.log(ts, f"slope_center={s[t]}")


#         # local max at t
#         if (s[t] > s[t - 1]) and (s[t] >= s[t + 1]):
#             peak_max_raw[t + 1] = True

#             if logger is not None:
#                 ts = df.index[t + 1].strftime("%H:%M:%S")
#                 logger.log(ts, "RAW MAX PEAK DETECTED")
#                 logger.log(ts, f"slope_center={s[t]}")


#     df["slope_peak_min_raw"] = peak_min_raw
#     df["slope_peak_max_raw"] = peak_max_raw

#     return df

def add_raw_slope_peaks(
    df,
    slope_col="KF_slope_adapt",
    region_max_col="region_max",
    region_min_col="region_min",
    logger=None,
):
    """
    STRICTLY CAUSAL pure slope extrema for exit logic.
    No quantile logic, no recency constraints.

    A raw extremum at index t is only known when bar t+1 arrives,
    so the signal is placed at index t+1.

    Adds:
        slope_peak_min_raw
        slope_peak_max_raw

    LOGGING (added):
      - For every RAW peak detected, also log whether it is eligible to become
        an ENTRY candidate (based on turning-region flags).
      - This does NOT change the raw-peak detection itself.
    """
    df = df.copy()
    s = df[slope_col].to_numpy(dtype=float)
    n = len(s)

    # turning-region flags (if not present, treat as False)
    if region_max_col in df.columns:
        region_max = df[region_max_col].fillna(False).to_numpy(bool)
    else:
        region_max = np.zeros(n, dtype=bool)

    if region_min_col in df.columns:
        region_min = df[region_min_col].fillna(False).to_numpy(bool)
    else:
        region_min = np.zeros(n, dtype=bool)

    peak_min_raw = np.zeros(n, dtype=bool)
    peak_max_raw = np.zeros(n, dtype=bool)

    for t in range(1, n - 1):
        exec_i = t + 1  # signal becomes available at t+1 (causal)
        ts = df.index[exec_i].strftime("%H:%M:%S") if logger is not None else None
        peak_ts = df.index[t].strftime("%H:%M:%S")


        # -------------------------
        # RAW MIN (local trough)
        # -------------------------
        if (s[t] < s[t - 1]) and (s[t] <= s[t + 1]):
            peak_min_raw[exec_i] = True

            if logger is not None:
                logger.log(ts, "RAW MIN PEAK DETECTED")
                logger.log(ts, f"slope_center={s[t]}")
                logger.log(ts, f"peak_bar_time={peak_ts}")
                logger.log(ts, f"evaluation_bar_time={ts}")
                logger.log(ts, "RAW PEAK → ENTRY CANDIDATE CHECK")
                logger.log(ts, f"region_min_at_peak={bool(region_min[t])}")
                if region_min[t]:
                    logger.log(ts, "=> eligible_for_entry_candidate=True")
                else:
                    logger.log(ts, "=> eligible_for_entry_candidate=False")
                    logger.log(ts, "  reason: region_min=False (raw peak will NOT become a candidate)")

        # -------------------------
        # RAW MAX (local peak)
        # -------------------------
        if (s[t] > s[t - 1]) and (s[t] >= s[t + 1]):
            peak_max_raw[exec_i] = True

            if logger is not None:
                logger.log(ts, "RAW MAX PEAK DETECTED")
                logger.log(ts, f"slope_center={s[t]}")
                logger.log(ts, f"peak_bar_time={peak_ts}")
                logger.log(ts, f"evaluation_bar_time={ts}")
                logger.log(ts, "RAW PEAK → ENTRY CANDIDATE CHECK")
                logger.log(ts, f"region_max_at_peak={bool(region_max[t])}")
                if region_max[t]:
                    logger.log(ts, "=> eligible_for_entry_candidate=True")
                else:
                    logger.log(ts, "=> eligible_for_entry_candidate=False")
                    logger.log(ts, "  reason: region_max=False (raw peak will NOT become a candidate)")

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

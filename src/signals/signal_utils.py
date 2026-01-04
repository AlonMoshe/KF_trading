
from aiohttp_client_cache import logger
import numpy as np
import datetime as dt
import pandas as pd
from src.signals.trailing import DirectionalTrailing



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
    q_vals = df["slope_q_roll_daily"].to_numpy()

    n = len(df)
    peak_max = np.zeros(n, dtype=bool)
    peak_min = np.zeros(n, dtype=bool)
    peak_max_raw_idx = np.full(n, -1, dtype=int)
    peak_min_raw_idx = np.full(n, -1, dtype=int)
    
    last_pos_max_val = None   # last positive raw MAX
    last_neg_min_val = None   # last negative raw MIN


    


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
                
                # ---------------------------------
                # RAW peak_to_trough swing check
                # ---------------------------------
                raw_ext_type = "max"
                raw_ext_val = s_raw[raw_ext_idx]

                raw_swing_ok = True

                if peak_to_trough > 0 and raw_ext_val > 0 and last_neg_min_val is not None:
                    raw_swing = abs(raw_ext_val - last_neg_min_val)

                    if raw_swing < peak_to_trough:
                        raw_swing_ok = False

                    if logger:
                        logger.log(
                            ts,
                            f"PEAK_TO_TROUGH RAW CHECK "
                            f"prev_type=min, "
                            f"prev_val={last_neg_min_val:.6f}, "
                            f"curr_type=max, "
                            f"curr_val={raw_ext_val:.6f}, "
                            f"raw_swing={raw_swing:.6f}, "
                            f"threshold={peak_to_trough:.6f}, "
                            f"pass={raw_swing >= peak_to_trough}"
                        )


                     

                
                if logger:
                    logger.log(ts, "RAW MAX PEAK DETECTED (WINDOWED)")
                    logger.log(ts, f"slope_center={val}")
                    logger.log(ts, f"peak_bar_time={peak_ts}")
                    logger.log(ts, f"evaluation_bar_time={ts}")
                    logger.log(ts, f"region_max_at_center={bool(region_max[c])}")
                    logger.log(ts, f"region_max_at_raw_ext={bool(region_max[raw_ext_idx])}")
                    logger.log(ts, f"q_at_center={q_vals[c]:.4f}")
                    logger.log(ts, f"q_at_raw_ext={q_vals[raw_ext_idx]:.4f}")
                    

                    raw_peak_ts = df.index[raw_ext_idx].strftime("%H:%M:%S")
                    logger.log(ts, f"raw_slope_peak_bar_time={raw_peak_ts}")


                reasons = []

                if not region_max[raw_ext_idx]:
                    reasons.append("region_max=False (at raw_ext_idx)")

                delta = val - s_vals[right]
                if peak_hysteresis > 0 and delta < peak_hysteresis:
                    reasons.append(f"hysteresis_fail (delta={delta:.6f})")

                window_swing = val - window.min()
                if peak_min_swing > 0 and window_swing < peak_min_swing:
                    reasons.append(f"min_swing_fail (swing={window_swing:.6f})")

                if not reasons and raw_swing_ok:
                    peak_max[right] = True
                    peak_max_raw_idx[right] = raw_ext_idx
                    
                if not raw_swing_ok and peak_to_trough > 0 and raw_ext_val > 0 and last_neg_min_val is not None:
                    reasons.append(f"peak_to_trough_raw_fail (swing={raw_swing:.6f})")




                if logger:
                    logger.log(ts, "FINAL PEAK MAX DECISION (WINDOWED)")
                    logger.log(ts, f"promoted={not reasons}")
                    if reasons:
                        for r in reasons:
                            logger.log(ts, f"  reason: {r}")
                            
                if raw_ext_val > 0:
                    last_pos_max_val = raw_ext_val



            # ---------------- MIN ----------------
            if val == window.min():
                raw_ext_idx = left + np.argmin(raw_window)
                
                # ---------------------------------
                # RAW peak_to_trough swing check
                # ---------------------------------
                raw_ext_type = "min"
                raw_ext_val = s_raw[raw_ext_idx]

                raw_swing_ok = True

                if peak_to_trough > 0 and raw_ext_val < 0 and last_pos_max_val is not None:
                    raw_swing = abs(raw_ext_val - last_pos_max_val)

                    if raw_swing < peak_to_trough:
                        raw_swing_ok = False

                    if logger:
                        logger.log(
                            ts,
                            f"PEAK_TO_TROUGH RAW CHECK "
                            f"prev_type=max, "
                            f"prev_val={last_pos_max_val:.6f}, "
                            f"curr_type=min, "
                            f"curr_val={raw_ext_val:.6f}, "
                            f"raw_swing={raw_swing:.6f}, "
                            f"threshold={peak_to_trough:.6f}, "
                            f"pass={raw_swing >= peak_to_trough}"
                        )




                
                if logger:
                    logger.log(ts, "RAW MIN PEAK DETECTED (WINDOWED)")
                    logger.log(ts, f"slope_center={val}")
                    logger.log(ts, f"peak_bar_time={peak_ts}")
                    logger.log(ts, f"evaluation_bar_time={ts}")
                    logger.log(ts, f"region_min_at_center={bool(region_min[c])}")
                    logger.log(ts, f"region_min_at_raw_ext={bool(region_min[raw_ext_idx])}")
                    logger.log(ts, f"q_at_center={q_vals[c]:.4f}")
                    logger.log(ts, f"q_at_raw_ext={q_vals[raw_ext_idx]:.4f}")
                    
                    raw_peak_ts = df.index[raw_ext_idx].strftime("%H:%M:%S")

                    logger.log(ts, f"raw_slope_peak_bar_time={raw_peak_ts}")


                reasons = []

                if not region_min[raw_ext_idx]:
                    reasons.append("region_min=False (at raw_ext_idx)")


                delta = s_vals[right] - val
                if peak_hysteresis > 0 and delta < peak_hysteresis:
                    reasons.append(f"hysteresis_fail (delta={delta:.6f})")

                window_swing = window.max() - val
                if peak_min_swing > 0 and window_swing < peak_min_swing:
                    reasons.append(f"min_swing_fail (swing={window_swing:.6f})")

                if not reasons and raw_swing_ok:
                    peak_min[right] = True
                    peak_min_raw_idx[right] = raw_ext_idx
                    
                if not raw_swing_ok and peak_to_trough > 0 and raw_ext_val < 0 and last_pos_max_val is not None:
                    reasons.append(f"peak_to_trough_raw_fail (swing={raw_swing:.6f})")



                if logger:
                    logger.log(ts, "FINAL PEAK MIN DECISION (WINDOWED)")
                    logger.log(ts, f"promoted={not reasons}")
                    if reasons:
                        for r in reasons:
                            logger.log(ts, f"  reason: {r}")
                            
                if raw_ext_val < 0:
                    last_neg_min_val = raw_ext_val


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
    *,
    entry_cfg, 
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
    market_state = df.get("market_state_label", pd.Series(None, index=df.index))

    

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
    
    # --------------------------------------------------
    # Trailing engine (single instance, no overlap)
    # --------------------------------------------------
    
    entry_trailing_col = getattr(entry_cfg, "trailing_col", "KF_level_adapt")

    trailing = DirectionalTrailing(
        level_col=entry_trailing_col,
        n_consecutive=entry_cfg.n_confirm_bars,
        count_mode="with",   # ENTRY: count favorable bars
        logger=logger,
    )




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

    market_allow_long = market_state == "low"
    market_allow_short = market_state == "high"
    
    # -------------------------
    # Raw arming trailing conditions
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
        # --------------------------------------------------
        # Update trailing (if active)
        # --------------------------------------------------
        if trailing.active:
            level_value = df[entry_trailing_col].iloc[i]
            ts = idx[i].strftime("%H:%M:%S")
            trailing.update(index=i, level_value=level_value,ts=ts)

            # Fire entry when trailing confirms
            if trailing.should_execute():
                if trailing.side == "long":
                    entry_long.append(True)
                    entry_short.append(False)
                    long_cool = cooldown
                else:
                    entry_long.append(False)
                    entry_short.append(True)
                    short_cool = cooldown

                if logger is not None:
                    ts = idx[i].strftime("%H:%M:%S")
                    logger.log(
                        ts,
                        f"=>TRAIL CONFIRMED side={trailing.side} \
                            confirm_bars={trailing.n_consecutive}(entry signal emitted) "
                    )

                trailing.reset()
                continue

            # ⬅⬅⬅ THIS IS THE MISSING LINE
            entry_long.append(False)
            entry_short.append(False)
            continue


        il = bool(entry_long_raw.iloc[i])
        is_ = bool(entry_short_raw.iloc[i])
        
        # --------------------------------------------------
        # Arm trailing instead of immediate entry
        # --------------------------------------------------
        if not trailing.active:
            if il and long_cool == 0:
                ts = idx[i].strftime("%H:%M:%S")
                
                if logger is not None:
                    abs_min = df["market_abs_min"].iloc[i]
                    abs_max = df["market_abs_max"].iloc[i]
                    price   = df[entry_trailing_col].iloc[i]
                    rng     = abs_max - abs_min if pd.notna(abs_max) else np.nan

                    logger.log(
                        ts,
                        f"ENTRY ARM CHECK | side=long | "
                        f"trailing price={price:.3f} | "
                        f"abs_min={abs_min:.3f} | "
                        f"abs_max={abs_max:.3f} | "
                        f"range={rng:.3f} | "
                        f"state_value={df['market_state_value'].iloc[i]:.3f} | "
                        f"state_label={df['market_state_label'].iloc[i]}"
                    )
                    
                # ✅ MARKET STATE GATE (dominant)
                if not bool(market_allow_long.iloc[i]):
                    if logger is not None:
                        logger.log(ts, f"=>ARM_LONG REJECTED (reason: market_state={market_state.iloc[i]})")
                    entry_long.append(False)
                    entry_short.append(False)
                    continue
                                
                trailing.arm(
                    side="long",
                    index=i,
                    level_value=df[entry_trailing_col].iloc[i],
                    ts=ts,
                )
                entry_long.append(False)
                entry_short.append(False)
                continue

            if is_ and short_cool == 0:
                ts = idx[i].strftime("%H:%M:%S")
                
                if logger is not None:
                    abs_min = df["market_abs_min"].iloc[i]
                    abs_max = df["market_abs_max"].iloc[i]
                    price   = df[entry_trailing_col].iloc[i]
                    rng     = abs_max - abs_min if pd.notna(abs_max) else np.nan

                    logger.log(
                        ts,
                        f"ENTRY ARM CHECK | side=short | "
                        f"trailing price={price:.3f} | "
                        f"abs_min={abs_min:.3f} | "
                        f"abs_max={abs_max:.3f} | "
                        f"range={rng:.3f} | "
                        f"state_value={df['market_state_value'].iloc[i]:.3f} | "
                        f"state_label={df['market_state_label'].iloc[i]}"
                    )
                    
                # ✅ MARKET STATE GATE (dominant)
                if not bool(market_allow_short.iloc[i]):
                    if logger is not None:
                        logger.log(ts, f"=>ARM_SHORT REJECTED (reason: market_state={market_state.iloc[i]})")
                    entry_long.append(False)
                    entry_short.append(False)
                    continue
                    
                trailing.arm(
                    side="short",
                    index=i,
                    level_value=df[entry_trailing_col].iloc[i],
                    ts=ts,
                )
                entry_long.append(False)
                entry_short.append(False)
                continue


        # ====================================================
        # LOG SHORT ENTRY EVALUATION (triggered by slope peak)
        # ====================================================
        if logger is not None and peak_max.iloc[i]:
            ts = idx[i].strftime("%H:%M:%S")
            logger.log(ts,"ARM SHORT EVALUATION")
            logger.log(ts, f"market_state={market_state.iloc[i]}")
            logger.log(ts, f"market_allows_short={bool(market_allow_short.iloc[i])}")
            
            abs_min = df["market_abs_min"].iloc[i]
            abs_max = df["market_abs_max"].iloc[i]
            price   = df[entry_trailing_col].iloc[i]
            rng     = abs_max - abs_min if pd.notna(abs_max) else np.nan

            logger.log(
                ts,
                f"MARKET STATE SNAPSHOT | "
                f"price={price:.3f} | "
                f"abs_min={abs_min:.3f} | "
                f"abs_max={abs_max:.3f} | "
                f"range={rng:.3f} | "
                f"value={df['market_state_value'].iloc[i]:.3f} | "
                f"label={df['market_state_label'].iloc[i]}"
            )

            
            logger.log(ts,f"slope_peak_max={bool(peak_max.iloc[i])}")
            logger.log(ts,f"region_gate_max={bool(region_gate_max.iloc[i])}")
            if raw_idx_max.iloc[i] >= 0:
                logger.log(
                    ts,
                    f"q_at_raw_ext={q.iloc[raw_idx_max.iloc[i]]:.4f}"
                )
            logger.log(ts,f"region_max_recent={bool(region_max_recent.iloc[i])}")
            logger.log(ts,f"prob_gate_max={bool(prob_gate_max.iloc[i])}")
            logger.log(ts,f"cooldown={short_cool}")

        # ====================================================
        # LOG LONG ENTRY EVALUATION (triggered by slope peak)
        # ====================================================
        if logger is not None and peak_min.iloc[i]:
            ts = idx[i].strftime("%H:%M:%S")
            logger.log(ts,"ARM LONG EVALUATION")
            logger.log(ts, f"market_state={market_state.iloc[i]}")
            logger.log(ts, f"market_allows_long={bool(market_allow_long.iloc[i])}")
            
            abs_min = df["market_abs_min"].iloc[i]
            abs_max = df["market_abs_max"].iloc[i]
            price   = df[entry_trailing_col].iloc[i]
            rng     = abs_max - abs_min if pd.notna(abs_max) else np.nan

            logger.log(
                ts,
                f"MARKET STATE SNAPSHOT | "
                f"price={price:.3f} | "
                f"abs_min={abs_min:.3f} | "
                f"abs_max={abs_max:.3f} | "
                f"range={rng:.3f} | "
                f"value={df['market_state_value'].iloc[i]:.3f} | "
                f"label={df['market_state_label'].iloc[i]}"
            )


            logger.log(ts,f"slope_peak_min={bool(peak_min.iloc[i])}")
            logger.log(ts,f"region_gate_min={bool(region_gate_min.iloc[i])}")
            if raw_idx_min.iloc[i] >= 0:
                logger.log(
                    ts,
                    f"q_at_raw_ext={q.iloc[raw_idx_min.iloc[i]]:.4f}"
                )

            logger.log(ts,f"region_min_recent={bool(region_min_recent.iloc[i])}")
            logger.log(ts,f"prob_gate_min={bool(prob_gate_min.iloc[i])}")
            logger.log(ts,f"cooldown={long_cool}")

        # -------------------
        # LONG side cooldown
        # -------------------
        if long_cool > 0:
            long_cool -= 1
        


        # -------------------
        # SHORT side cooldown
        # -------------------
        if short_cool > 0:
            short_cool -= 1
        

        # Append defaults ONLY if nothing already appended
        entry_long.append(False)
        entry_short.append(False)
        ts = idx[i].strftime("%H:%M:%S")

        # ====================================================
        # SHORT RESULT LOGGING (after cooldown)
        # ====================================================
        if logger is not None and peak_max.iloc[i]:
            if entry_short[-1]:
                logger.log(ts,"=>ARM_SHORT=True")
            else:
                reasons = []
                if not region_gate_max.iloc[i]: reasons.append("region_gate_max=False")
                if not region_max_recent.iloc[i]: reasons.append("region_max_recent=False")
                if not prob_gate_max.iloc[i]: reasons.append("prob_gate_max=False")
                if not market_allow_short.iloc[i]:
                    reasons.append(f"market_state={market_state.iloc[i]}")

                if short_cool > 0: reasons.append(f"cooldown={short_cool}")
                logger.log(ts,f"=>ARM_SHORT REJECTED (reason: {', '.join(reasons)})")

        # ====================================================
        # LONG RESULT LOGGING (after cooldown)
        # ====================================================
        if logger is not None and peak_min.iloc[i]:
            if entry_long[-1]:
                logger.log(ts,"=>ARM_LONG=True")
            else:
                reasons = []
                if not region_gate_min.iloc[i]: reasons.append("region_gate_min=False")
                if not region_min_recent.iloc[i]: reasons.append("region_min_recent=False")
                if not prob_gate_min.iloc[i]: reasons.append("prob_gate_min=False")
                if not market_allow_long.iloc[i]:
                    reasons.append(f"market_state={market_state.iloc[i]}")

                if long_cool > 0: reasons.append(f"cooldown={long_cool}")
                logger.log(ts,f"=>ARM_LONG REJECTED (reason: {', '.join(reasons)})")

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

        # last bar can never execute
        entry_long[-1] = False
        entry_short[-1] = False


    df["entry_long"] = entry_long
    df["entry_short"] = entry_short

    return df


def add_exit_signals(
    df,
    *,
    exit_cfg,
    max_adverse_bars=7,
    logger=None,
):
    """
    Exit logic (current):
    - Single directional trailing exit
    - Trailing starts immediately at entry
    - Exit occurs after n_consecutive bars moving against the previous bar direction
    moving against the position

    Notes:
    - This function is stateful and strictly causal.
    - Other exit mechanisms may be added later.
    """


    df = df.copy()

    mids = df["Mid"].to_numpy()
    opens = df["Open"].to_numpy()
    
    entry_long = df["entry_long"].to_numpy(bool)
    entry_short = df["entry_short"].to_numpy(bool)

    n = len(df)

    exit_long = np.zeros(n, dtype=bool)
    exit_short = np.zeros(n, dtype=bool)

    # -----------------------------
    # LONG STATE
    # -----------------------------
    in_long = False
    exit_trailing_long = None
    long_entry_bar = None
    long_entry_price = None
    

    # -----------------------------
    # SHORT STATE
    # -----------------------------
    in_short = False
    exit_trailing_short = None
    short_entry_bar = None
    short_entry_price = None
    
    
    for i in range(n):

        ts = df.index[i].strftime("%H:%M:%S") if logger else None
        
        # =====================================================
        # MODEL A GATE: block redundant entry signals while in position
        # (This prevents trailing from "executing" again on the same side.)
        # =====================================================
        if in_long and entry_long[i]:
            # entry_long[i] = False
            if logger:
                logger.log(ts, "BLOCKED ENTRY_LONG (already in LONG position)")
                logger.log(ts, "=>ENTRY_LONG=False (reason=already_in_long)")

        if in_short and entry_short[i]:
            # entry_short[i] = False
            if logger:
                logger.log(ts, "BLOCKED ENTRY_SHORT (already in SHORT position)")
                logger.log(ts, "=>ENTRY_SHORT=False (reason=already_in_short)")


        # =====================================================
        # LONG ENTRY (state update only)
        # =====================================================
        if entry_long[i] and not in_long:
            if i+1<n:
                in_long = True
                long_entry_bar = i
                long_entry_price = opens[i+1]
                
                
                exit_trailing_long = DirectionalTrailing(
                    level_col=exit_cfg.trailing_col,
                    n_consecutive=exit_cfg.n_exit_confirm_bars,
                    count_mode="against",
                    logger=logger,
                )

                arm_i = i + 1
                arm_ts = df.index[arm_i].strftime("%H:%M:%S") if logger else None
                
                exit_trailing_long.arm(
                    side="long",
                    index=arm_i,
                    level_value=df[exit_cfg.trailing_col].iloc[arm_i],
                    ts=arm_ts,
                )
                
                if logger:
                    logger.log(
                        ts,
                        f"LONG ENTRY OPENED "
                        f"(entry_bar={df.index[long_entry_bar].time()}, "
                        f"entry_price={long_entry_price:.3f})"
                    )

        # =====================================================
        # LONG EXIT (only if open AND i >= entry_bar + 1)
        # =====================================================
        if in_long and i >= long_entry_bar + 1:

            if exit_trailing_long is not None:
                exit_trailing_long.update(
                    index=i,
                    level_value=df[exit_cfg.trailing_col].iloc[i],
                    ts=ts,
                )

                if exit_trailing_long.should_execute():
                    exit_long[i] = True
                    in_long = False
                    exit_trailing_long = None
                    long_entry_bar = None
                    long_entry_price = None

                    if logger:
                        logger.log(ts, "=>EXIT_LONG=True (reason=directional_trailing)")




        # =====================================================
        # SHORT ENTRY (state update only)
        # =====================================================
        if entry_short[i] and not in_short:
            if i+1<n:
                in_short = True
                short_entry_bar = i
                short_entry_price = opens[i+1]
                
                
                exit_trailing_short = DirectionalTrailing(
                    level_col=exit_cfg.trailing_col,
                    n_consecutive=exit_cfg.n_exit_confirm_bars,
                    count_mode="against",
                    logger=logger,
                )
                arm_i = i + 1
                arm_ts = df.index[arm_i].strftime("%H:%M:%S") if logger else None

                exit_trailing_short.arm(
                    side="short",
                    index=arm_i,
                    level_value=df[exit_cfg.trailing_col].iloc[arm_i],
                    ts=arm_ts,
                )

                
                if logger:
                    logger.log(
                    ts,
                    f"SHORT POSITION OPENED "
                    f"(entry_bar={df.index[short_entry_bar].time()}, "
                    f"entry_price={short_entry_price:.3f})"
                    )
        
        # =====================================================
        # SHORT EXIT (only if open AND i >= entry_bar + 1)
        # =====================================================
        if in_short and i >= short_entry_bar + 1:

            if exit_trailing_short is not None:
                exit_trailing_short.update(
                    index=i,
                    level_value=df[exit_cfg.trailing_col].iloc[i],
                    ts=ts,
                )

                if exit_trailing_short.should_execute():
                    exit_short[i] = True
                    in_short = False
                    exit_trailing_short = None
                    short_entry_bar = None
                    short_entry_price = None

                    if logger:
                        logger.log(ts, "=>EXIT_SHORT=True (reason=directional_trailing)")





    # =====================================================
    # END-OF-DAY EXIT (EXPLICIT + LOGGED)
    # =====================================================
    last_i = n - 1
    ts = df.index[last_i].strftime("%H:%M:%S") if logger else None

    if in_long:
        exit_long[last_i] = True
        exit_trailing_long = None

        if logger:
            logger.log(ts, "END_OF_DAY_EXIT LONG")
            logger.log(ts, "=>EXIT_LONG=True (reason=end_of_day)")

    if in_short:
        exit_short[last_i] = True
        exit_trailing_short = None

        if logger:
            logger.log(ts, "END_OF_DAY_EXIT SHORT")
            logger.log(ts, "=>EXIT_SHORT=True (reason=end_of_day)")

    # write back possibly-blocked entry signals (Model A)
    # df["entry_long"] = entry_long
    # df["entry_short"] = entry_short

    
    df["exit_long"] = exit_long
    df["exit_short"] = exit_short

    return df



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

def add_market_state(
    df,
    price_col="Mid",
    market_open_time="09:30",
    start_delay_minutes=2,
    proportion_low=0.2,
    proportion_high=0.8,
):
    """
    Causal intraday market state based on expanding day range.

    Adds:
        market_state_value : float in [0,1] or NaN
        market_state_label : 'low' | 'neutral' | 'high' | None
    """

    df = df.copy()

    price = df[price_col].to_numpy(dtype=float)
    idx = df.index

    n = len(df)

    state_value = np.full(n, np.nan)
    state_label = np.full(n, None, dtype=object)

    # -------------------------------------------------
    # Determine start index (09:30 + delay)
    # -------------------------------------------------
    hh, mm = map(int, market_open_time.split(":"))
    start_time = dt.time(hh, mm)

    if idx.tz is None:
        idx_e = idx.tz_localize("US/Eastern")
    else:
        idx_e = idx.tz_convert("US/Eastern")

    times = idx_e.time

    start_idx = None
    for i in range(n):
        if times[i] >= (
            dt.datetime.combine(dt.date.today(), start_time)
            + dt.timedelta(minutes=start_delay_minutes)
        ).time():
            start_idx = i
            break

    if start_idx is None:
        df["market_state_value"] = state_value
        df["market_state_label"] = state_label
        return df

    # -------------------------------------------------
    # Expanding intraday min / max (causal)
    # -------------------------------------------------
    abs_min = price[0]
    abs_max = price[0]
    
    abs_min_arr = np.full(n, np.nan)
    abs_max_arr = np.full(n, np.nan)

    for i in range(start_idx, n):

        p = price[i]

        abs_min = min(abs_min, p)
        abs_max = max(abs_max, p)
        
        abs_min_arr[i] = abs_min
        abs_max_arr[i] = abs_max


        market_range = abs_max - abs_min

        # Range not yet defined → no trading
        if market_range <= 0:
            continue

        v = (p - abs_min) / market_range
        state_value[i] = v

        if v <= proportion_low:
            state_label[i] = "low"
        elif v >= proportion_high:
            state_label[i] = "high"
        else:
            state_label[i] = "neutral"

    df["market_state_value"] = state_value
    df["market_state_label"] = state_label
    
    df["market_abs_min"] = abs_min_arr
    df["market_abs_max"] = abs_max_arr


    return df

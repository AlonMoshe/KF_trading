start_A = '2025-11-03 09:40:00-05:00'
end_A   = '2025-11-03 15:59:55-05:00'

df_day_A = df[(df.index >= start_A) & (df.index <= end_A)].copy()


start_B = '2025-11-04 09:40:00-05:00'
end_B   = '2025-11-04 15:59:55-05:00'

df_day_B = df[(df.index >= start_B) & (df.index <= end_B)].copy()


# --- histogram / ECDF source ---
data_A = df_day_A['KF_slope_adapt'].dropna().values
data_sorted_A = np.sort(data_A)
n_A = len(data_sorted_A)

def ecdf_A(x):
    return np.searchsorted(data_sorted_A, x, side='right') / n_A

slope_B = df_day_B['KF_slope_adapt']
valid_idx = slope_B.dropna().index

F_B = ecdf_A(slope_B.dropna().values)

df_day_B.loc[valid_idx, 'prob_low']  = (1 - F_B) * 100
df_day_B.loc[valid_idx, 'prob_high'] = F_B * 100



# -----------------------------
# 2. Add signal columns
# -----------------------------
df_day_B['signal_short'] = 0
df_day_B['signal_long']  = 0

price = df_day_B['Mid']

# -----------------------------
# 3. PARAMETERS
# -----------------------------
# Short: high tail (top)
P_enter_short = 98.0
P_exit_short  = 96.0

# Long: low tail (bottom)
P_enter_long = 98.0
P_exit_long  = 97.0

RETRACE_PCT = 0.00003     # 0.003%
COOLDOWN_BARS = 2       # 2 bars

# -----------------------------
# 4. STATE MACHINE
# -----------------------------
state = "FLAT"  # FLAT, SHORT_ZONE, LONG_ZONE, COOLDOWN
cooldown_until = -1

short_extreme_price = None
long_extreme_price  = None

for t in range(len(df_day_B)):

    p_low  = df_day_B['prob_low'].iloc[t]
    p_high = df_day_B['prob_high'].iloc[t]
    mid = price.iloc[t]

    # ---- COOLDOWN ----
    if state == "COOLDOWN":
        if t >= cooldown_until:
            state = "FLAT"
        else:
            continue

    # ==============================
    # ENTER ZONES FROM FLAT STATE
    # ==============================
    if state == "FLAT":

        # Enter SHORT zone (price top) → extreme HIGH tail prob
        if p_high >= P_enter_short:
            state = "SHORT_ZONE"
            short_extreme_price = mid
            continue

        # Enter LONG zone (price bottom) → extreme LOW tail prob
        if p_low >= P_enter_long:
            state = "LONG_ZONE"
            long_extreme_price = mid
            continue

    # ==============================
    # SHORT ZONE (look for top and retrace DOWN)
    # ==============================
    if state == "SHORT_ZONE":

        # Update swing HIGH while probability is still extreme enough
        if p_high >= P_exit_short:
            if mid > short_extreme_price:
                short_extreme_price = mid

        # Conditions for exit + retrace confirmation
        prob_exit = (p_high < P_exit_short)
        retrace = (short_extreme_price - mid) / short_extreme_price >= RETRACE_PCT

        if prob_exit and retrace:
            # SHORT ENTRY
            df_day_B.at[df_day_B.index[t], 'signal_short'] = 1

            # Cooldown
            state = "COOLDOWN"
            cooldown_until = t + COOLDOWN_BARS
            short_extreme_price = None
            continue

        # If probability leaves zone but retrace not reached → abandon
        if prob_exit and not retrace:
            state = "FLAT"
            short_extreme_price = None
            continue

    # ==============================
    # LONG ZONE (look for bottom and retrace UP)
    # ==============================
    if state == "LONG_ZONE":

        # Update swing LOW while probability is still extreme
        if p_low >= P_exit_long:
            if mid < long_extreme_price:
                long_extreme_price = mid

        # Exit conditions
        prob_exit = (p_low < P_exit_long)
        retrace = (mid - long_extreme_price) / long_extreme_price >= RETRACE_PCT

        if prob_exit and retrace:
            # LONG ENTRY
            df_day_B.at[df_day_B.index[t], 'signal_long'] = 1

            # Cooldown
            state = "COOLDOWN"
            cooldown_until = t + COOLDOWN_BARS
            long_extreme_price = None
            continue

        # If exit zone but no retrace → abandon
        if prob_exit and not retrace:
            state = "FLAT"
            long_extreme_price = None
            continue
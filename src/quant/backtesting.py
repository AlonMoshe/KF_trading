# backtest_utils.py (or just put this in a notebook cell)

import os
import datetime as dt
import pandas as pd
from collections import OrderedDict
from aiohttp_client_cache import logger
from src.config.strategy_config import StrategyConfig



# Adjust these imports to match how you normally import modules.
from src.data.data_utils import PROCESSED_DATA_DIR, parse_filename_info, select_intraday_session
from src.quant.quant_utils import add_slope_quantiles_daily
from src.signals.signal_utils import (
    add_turning_regions,
    add_slope_peaks,
    add_raw_slope_peaks,
    add_entry_signals,
    add_exit_signals,
    add_trade_ids,
    extract_trades,
    trades_to_dataframe,
    add_market_state,
)


# ------------------------------------------------------------
# 1) Discover processed files for a symbol + interval
# ------------------------------------------------------------

def discover_processed_files(symbol: str, interval: int):
    """
    Scan PROCESSED_DATA_DIR and return a list of file info dicts:

        {
          "file_path": "...",
          "symbol": "QQQ",
          "start_date": date,
          "end_date": date,
          "interval": int,
          "span_days": int,
        }
    """
    symbol = symbol.upper()
    infos = []

    for fname in os.listdir(PROCESSED_DATA_DIR):
        if not fname.endswith(".csv"):
            continue

        try:
            info = parse_filename_info(fname)
        except ValueError:
            continue  # ignore files that don't match pattern

        if info["symbol"] != symbol:
            continue
        if info["interval"] != interval:
            continue

        start_date = info["start_date"]
        end_date = info["end_date"]
        span_days = (end_date - start_date).days + 1

        infos.append({
            "file_path": os.path.join(PROCESSED_DATA_DIR, fname),
            "symbol": info["symbol"],
            "start_date": start_date,
            "end_date": end_date,
            "interval": info["interval"],
            "span_days": span_days,
        })

    return infos


# ------------------------------------------------------------
# 2) Build date -> file map (choose most specific file)
# ------------------------------------------------------------

def build_date_to_file_map(file_infos):
    """
    For overlapping files, for each date choose the file with the
    smallest span_days (most specific) that covers that date.

    Returns:
        dict[date] = "file_path"
    """
    date_to_file = {}

    for info in file_infos:
        start = info["start_date"]
        end = info["end_date"]
        span = info["span_days"]
        path = info["file_path"]

        d = start
        while d <= end:
            if d not in date_to_file:
                date_to_file[d] = (path, span)
            else:
                _, best_span = date_to_file[d]
                if span < best_span:
                    date_to_file[d] = (path, span)
            d += dt.timedelta(days=1)

    # strip span, keep only file path
    date_to_file = {d: p for d, (p, s) in date_to_file.items()}
    return date_to_file




def one_day_backtest(
    symbol, year, month, day,
    *,
    cfg: StrategyConfig,
    log: bool = True,
    verbose: bool = True,
):

    """
    Complete one-day simulation:
    - loads correct file for the day
    - slices intraday session
    - runs full causal pipeline
    - extracts trades
    - returns df_day, df_trades, summary_dict

    This is the ONLY function you need to call from the notebook
    (aside from plotting).
    """
    
    if cfg is None:
        raise ValueError("one_day_backtest requires a StrategyConfig (cfg)")

    # --------------------------------------------------------
    # 1. Resolve date
    # --------------------------------------------------------
    target_date = dt.date(year, month, day)
    date_str = target_date.isoformat()          # for data/session logic
    date_str_for_log = date_str                 # same value, explicit name


    # Build logger for this run (one file per test)
    logger = None
    if log:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"BT_{symbol}_{date_str_for_log}_{ts}.log"
        # You can put logs into e.g. "logs" folder if you want:
        log_path = os.path.join("logs", filename)
        logger = BacktestLogger(log_path)
        
    if logger is not None:
        logger.log("00:00:00","[CONFIG] "+"=" * 50)
        logger.log("00:00:00","[CONFIG] "+"BACKTEST CONFIGURATION")
        logger.log("00:00:00","[CONFIG] "+"=" * 50)

        logger.log("00:00:00","[CONFIG] "+f"symbol = {symbol}")
        logger.log("00:00:00","[CONFIG] "+f"date = {date_str_for_log}")
        logger.log("00:00:00","[CONFIG] "+"")

        logger.log("00:00:00","[CONFIG] "+f"Q_high = {cfg.Q_high}")
        logger.log("00:00:00","[CONFIG] "+f"window_max = {cfg.window_max}")
        logger.log("00:00:00","[CONFIG] "+f"cooldown = {cfg.entry.cooldown}")
        logger.log("00:00:00","[CONFIG] "+f"interval = {cfg.interval}")
        logger.log("00:00:00","[CONFIG] "+"")

        logger.log("00:00:00","[CONFIG] "+f"slope_smooth = {cfg.slope.smooth}")
        logger.log("00:00:00","[CONFIG] "+f"slope_smooth_window = {cfg.slope.smooth_window}")
        logger.log("00:00:00","[CONFIG] "+f"slope_peak_half_window = {cfg.slope.peak_half_window}")
        logger.log("00:00:00","[CONFIG] "+f"slope_peak_hysteresis = {cfg.slope.peak_hysteresis}")
        logger.log("00:00:00","[CONFIG] "+f"slope_peak_min_swing = {cfg.slope.peak_min_swing}")
        logger.log("00:00:00","[CONFIG] "+"")

        logger.log("00:00:00","[CONFIG] "+f"use_region_recent = {cfg.entry.use_region_recent}")
        logger.log("00:00:00","[CONFIG] "+f"region_recent_window = {cfg.entry.region_recent_window}")
        logger.log("00:00:00","[CONFIG] "+f"peak_to_trough = {cfg.slope.peak_to_trough}")
        logger.log("00:00:00","[CONFIG] "+"")

        logger.log("00:00:00","[CONFIG] "+"=" * 50)
        logger.log("00:00:00","[CONFIG] "+"")



    # --------------------------------------------------------
    # 2. Identify the correct file for the day
    # --------------------------------------------------------
    
    infos = discover_processed_files(
    symbol,
    interval=cfg.interval,  # interval is strategy-level config
)

    if not infos:
        raise RuntimeError(f"No processed files found for {symbol}")

    date_to_file = build_date_to_file_map(infos)

    if target_date not in date_to_file:
        raise RuntimeError(f"No data found for {symbol} on {target_date}")

    file_path = date_to_file[target_date]

    # --------------------------------------------------------
    # 3. Load file exactly like the backtester
    # --------------------------------------------------------
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    # --------------------------------------------------------
    # 4. Slice the intraday session
    # --------------------------------------------------------
    df_day = select_intraday_session(df, date_str,
                                     start_time="09:30",
                                     end_time="15:59:55")

    if df_day.empty:
        raise RuntimeError(f"{symbol} {date_str}: No intraday data found.")
    
    # --------------------------------------------------------
    # 4.5. ADD MARKET STATE
    # --------------------------------------------------------

    df_day = add_market_state(
        df_day,
        price_col="Mid",
        proportion_low=cfg.market.proportion_low,
        proportion_high=cfg.market.proportion_high,
    )    
    
    
    
    
    # --------------------------------------------------------
    # 5. FULL CAUSAL PIPELINE
    # --------------------------------------------------------

    # ECDF quantiles
    df_day = add_slope_quantiles_daily(
        df_day,
        slope_col="KF_slope_adapt",
        window_max=cfg.window_max,
    )

    # Turning regions
    df_day = add_turning_regions(
        df_day,
        q_col="slope_q_roll_daily",
        Q_high=cfg.Q_high,
    )

    # Causal slope peaks (entry)
    df_day = add_slope_peaks(
        df_day,
        slope_col="KF_slope_adapt",
        smooth=cfg.slope.smooth,
        smooth_window=cfg.slope.smooth_window,
        peak_half_window=cfg.slope.peak_half_window,
        peak_hysteresis=cfg.slope.peak_hysteresis,
        peak_min_swing=cfg.slope.peak_min_swing,
        peak_to_trough=cfg.slope.peak_to_trough,   # NEW
        logger=logger,    

    )




    # Raw causal peaks (exit)
    df_day = add_raw_slope_peaks(df_day, logger=logger)


    # Entry / exit signals
    df_day = add_entry_signals(
        df_day,
        entry_cfg=cfg.entry,    # ← NEW
        q_col="slope_q_roll_daily",
        Q_high=cfg.Q_high,
        cooldown=cfg.entry.cooldown,
        use_region_recent=cfg.entry.use_region_recent,
        region_recent_window=cfg.entry.region_recent_window,
        logger=logger,    
    )

    df_day = add_exit_signals(
            df_day,
            exit_cfg=cfg.exit,
            max_adverse_bars=cfg.exit.max_adverse_bars,
            logger=logger,
            )


    # Assign trade IDs for plotting/debug
    df_day = add_trade_ids(df_day)
    



    # --------------------------------------------------------
    # 6. Extract trades + summary
    # --------------------------------------------------------
    trades = extract_trades(df_day)
    df_trades = trades_to_dataframe(trades)

    # Basic stats
    total_pnl = df_trades["pnl"].sum() if not df_trades.empty else 0.0
    num_trades = len(df_trades)
    avg_pnl = df_trades["pnl"].mean() if num_trades > 0 else 0.0

    summary = {
        "symbol": symbol,
        "date": target_date,
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "avg_pnl": avg_pnl,
    }

    if verbose:
        print(
            f"{symbol} {target_date}: "
            f"Trades={num_trades}, Total={total_pnl:.4f}, Avg={avg_pnl:.4f}"
        )
        
    if logger is not None:
        
        logger.log("23:59:59","[END] Backtest finished.")
        logger.save()


    return df_day, df_trades, summary




class BacktestLogger:
    """
    Time-bucketed backtest logger.
    Groups all log messages by timestamp.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.blocks = OrderedDict()  # ts -> list[str]

    def log(self, ts: str, msg: str):
        """
        Log a message under a given timestamp.
        """
        if ts not in self.blocks:
            self.blocks[ts] = []
        self.blocks[ts].append(msg)

    def save(self):
        """
        Write logs to disk in strict chronological order.
        """
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        # Flatten (ts, msg) pairs
        events = []
        for ts, msgs in self.blocks.items():
            for m in msgs:
                events.append((ts, m))

        # Sort chronologically by timestamp
        events.sort(key=lambda x: x[0])

        lines = []
        for ts, msg in events:
            lines.append(f"[{ts}] {msg}")

        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))




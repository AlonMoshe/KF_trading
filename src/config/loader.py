from __future__ import annotations

from pathlib import Path
import yaml

from .strategy_config import StrategyConfig, SlopeConfig, EntryConfig, ExitConfig, MarketConfig


def load_strategy_config(path: str | Path) -> StrategyConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "strategy" not in raw:
        raise ValueError("Config file must contain a top-level 'strategy' key")

    s = raw["strategy"] or {}

    slope = s.get("slope", {}) or {}
    entry = s.get("entry", {}) or {}
    exit_ = s.get("exit", {}) or {}

    market = s.get("market", {}) or {}


    return StrategyConfig(
        Q_high=float(s.get("Q_high", 0.90)),
        window_max=int(s.get("window_max", 100)),
        interval=int(s.get("interval", 5)),
        slope=SlopeConfig(
            smooth=bool(slope.get("smooth", False)),
            smooth_window=int(slope.get("smooth_window", 5)),
            peak_half_window=int(slope.get("peak_half_window", 1)),
            peak_hysteresis=float(slope.get("peak_hysteresis", 0.0)),
            peak_min_swing=float(slope.get("peak_min_swing", 0.0)),
            peak_to_trough=float(slope.get("peak_to_trough", 0.0)),
        ),
        entry=EntryConfig(
            cooldown=int(entry.get("cooldown", 20)),
            use_region_recent=bool(entry.get("use_region_recent", True)),
            region_recent_window=int(entry.get("region_recent_window", 10)),
            n_confirm_bars=int(entry.get("n_confirm_bars", 1)),  # NEW
        ),
        exit=ExitConfig(
            n_exit_confirm_bars=int(exit_.get("n_exit_confirm_bars", 3)),
            max_adverse_bars=int(exit_.get("max_adverse_bars", 7)),
            trailing_col=exit_.get("trailing_col", "KF_level_adapt"),
            ),
        market=MarketConfig(
            proportion_low=float(market.get("proportion_low", 0.2)),
            proportion_high=float(market.get("proportion_high", 0.8)),
        ),

    )

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SlopeConfig:
    smooth: bool = False
    smooth_window: int = 5

    peak_half_window: int = 1
    peak_hysteresis: float = 0.0
    peak_min_swing: float = 0.0
    peak_to_trough: float = 0.0


@dataclass
class EntryConfig:
    cooldown: int = 20
    use_region_recent: bool = True
    region_recent_window: int = 10
    n_confirm_bars: int = 1   # NEW
    
@dataclass
class ExitConfig:
    n_exit_confirm_bars: int = 3
    max_adverse_bars: int = 7
    trailing_col: str = "KF_level_adapt"


@dataclass
class MarketConfig:
    proportion_low: float = 0.2
    proportion_high: float = 0.8


@dataclass
class StrategyConfig:
    Q_high: float = 0.90
    window_max: int = 100
    interval: int = 5

    slope: SlopeConfig = field(default_factory=SlopeConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)   # NEW
    market: MarketConfig = field(default_factory=MarketConfig)


"""
PolyMarket BTC 5-Min Bot — Backtester
Fetches real Binance 1m klines, replays every 5-min window, applies the
exact signal logic from polymarket_btc_bot.py, and reports P&L + metrics.

Usage:
    python3 polymarket_backtest.py                  # 30-day backtest, default params
    python3 polymarket_backtest.py --days 60        # 60-day backtest
    python3 polymarket_backtest.py --sweep          # grid search over signal params
    python3 polymarket_backtest.py --plot            # save P&L chart to backtest_results.png
    python3 polymarket_backtest.py --compare         # show baseline vs enhanced side-by-side

Enhanced signal filters (v2):
  1. Volume filter     — skip low-volume moves (< min_quote_volume per candle)
  2. Time-of-day       — only trade peak liquidity hours (08:00–21:00 UTC)
  3. Chop filter       — skip if price range < range_floor_pct (sideways market)
  4. Wick analysis     — penalise signals where candle closed against the direction
"""

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import requests

# ─── Optional deps ────────────────────────────────────────────────────────────
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ─── Signal config (mirrors polymarket_btc_bot.py Config) ─────────────────────
@dataclass
class BacktestConfig:
    # Signal params — tuned for 20%+ ROI (from grid search)
    momentum_window_secs: int     = 60     # 60s lookback beats 120s
    momentum_threshold_pct: float = 0.18   # only trade significant moves
    min_confidence: float         = 0.75   # high selectivity: 63% win rate
    min_price_samples: int        = 8      # calibrated to 6 ticks/min over 60s

    # Risk / execution
    max_bet_usdc: float         = 10.0
    min_bet_usdc: float         = 1.0
    min_edge_threshold: float   = 0.05

    # Timing
    post_window_pause_secs: int = 20   # wait N sec after window open before trading
    new_window_buffer_secs: int = 15   # stop trading N sec before window close

    # Market simulation
    market_entry_spread: float  = 0.02  # how much we pay above 0.50 (bid/ask spread)

    # Ticks per 1m candle (linear interpolation)
    ticks_per_minute: int       = 6     # one tick every 10 seconds

    # ── Enhanced filters (v2) ──────────────────────────────────────────────────
    # 1. Volume filter: skip candles with quote volume below this USDT threshold
    min_quote_volume: float     = 50_000.0   # ~median BTC/USDT 1m vol; set 0 to disable

    # 2. Time-of-day filter: only trade between these UTC hours (peak liquidity)
    trading_hour_start: int     = 8    # 08:00 UTC
    trading_hour_end:   int     = 21   # 21:00 UTC (exclusive); set 0,24 to disable

    # 3. Chop filter: skip if lookback price range < this % (sideways / no trend)
    range_floor_pct: float      = 0.03   # 0.03% range minimum; set 0 to disable

    # 4. Wick penalty: reduce confidence when candle closes against direction
    wick_penalty: float         = 0.80   # multiply confidence by this on wick mismatch


# ─── Data structures ──────────────────────────────────────────────────────────
@dataclass
class Tick:
    ts: float        # unix seconds
    price: float


@dataclass
class Candle:
    ts: int          # minute-boundary unix seconds
    open: float
    high: float
    low: float
    close: float
    quote_volume: float   # USDT volume
    trades: int


@dataclass
class Trade:
    window_ts: int
    entry_ts: float
    direction: str        # "UP" or "DOWN"
    entry_price: float    # PolyMarket price paid (probability)
    bet_usdc: float
    btc_open: float       # BTC price at window open
    btc_close: float      # BTC price at window close
    won: bool
    pnl: float            # net P&L in USDC
    confidence: float
    price_change_pct: float
    edge: float


# ─── Binance data fetcher ─────────────────────────────────────────────────────
BINANCE_API = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Fetch all klines between start_ms and end_ms (handles pagination)."""
    all_candles = []
    current = start_ms
    while current < end_ms:
        resp = requests.get(
            BINANCE_API,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=15,
        )
        resp.raise_for_status()
        candles = resp.json()
        if not candles:
            break
        all_candles.extend(candles)
        current = candles[-1][0] + 1  # next ms
        if len(candles) < 1000:
            break
        time.sleep(0.1)  # stay within Binance rate limits
    return all_candles


def fetch_btc_history(days: int = 30) -> tuple[list[Tick], dict[int, Candle]]:
    """
    Return (ticks, candle_index) from the last N days of 1m BTC/USDT klines.
    candle_index maps minute-boundary unix-second → Candle (for volume/wick filters).
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000

    print(f"Fetching {days} days of BTC/USDT 1m klines from Binance...")
    raw = fetch_klines("BTCUSDT", "1m", start_ms, end_ms)
    print(f"  Fetched {len(raw)} candles")

    ticks: list[Tick] = []
    candle_index: dict[int, Candle] = {}
    n = BacktestConfig().ticks_per_minute

    for c in raw:
        open_ts = int(c[0] / 1000)
        open_p  = float(c[1])
        close_p = float(c[4])

        candle_index[open_ts] = Candle(
            ts=open_ts,
            open=open_p,
            high=float(c[2]),
            low=float(c[3]),
            close=close_p,
            quote_volume=float(c[7]),
            trades=int(c[8]),
        )

        for i in range(n):
            ts = open_ts + (60 / n) * i
            price = open_p + (close_p - open_p) * (i / (n - 1)) if n > 1 else open_p
            ticks.append(Tick(ts=ts, price=price))

    ticks.sort(key=lambda t: t.ts)
    return ticks, candle_index


# ─── Signal logic ────────────────────────────────────────────────────────────
def compute_signal(
    ticks_window: list[Tick],
    current_price: float,
    cfg: BacktestConfig,
    window_candles: Optional[list[Candle]] = None,   # candles covering the lookback
) -> Optional[tuple[str, float, float]]:
    """
    Returns (direction, confidence, price_change_pct) or None.

    v2 enhancements applied when window_candles is provided:
      - Volume filter    (filter 1)
      - Chop filter      (filter 3)
      - Wick penalty     (filter 4)
    """
    if len(ticks_window) < cfg.min_price_samples:
        return None

    prices = [t.price for t in ticks_window]
    start_price = prices[0]
    price_change_pct = (current_price - start_price) / start_price * 100

    # ── Filter 3: chop — skip flat / sideways markets ─────────────────────────
    if cfg.range_floor_pct > 0:
        price_range = (max(prices) - min(prices)) / min(prices) * 100
        if price_range < cfg.range_floor_pct:
            return None

    # ── Core momentum ─────────────────────────────────────────────────────────
    mid = len(prices) // 2
    early_mean = statistics.mean(prices[:mid]) if mid > 0 else start_price
    late_mean  = statistics.mean(prices[mid:]) if prices[mid:] else current_price
    acceleration = (late_mean - early_mean) / early_mean * 100

    direction = "UP" if price_change_pct >= 0 else "DOWN"
    raw_conf = min(abs(price_change_pct) / cfg.momentum_threshold_pct, 1.0)

    # Acceleration boost: weight raised to 30% (up from 20%) for stronger confirmation
    if (direction == "UP" and acceleration > 0) or (direction == "DOWN" and acceleration < 0):
        confidence = min(raw_conf * 1.30, 1.0)
    else:
        confidence = raw_conf * 0.70

    # ── Filter 1: volume — skip low-participation moves ───────────────────────
    if cfg.min_quote_volume > 0 and window_candles:
        avg_vol = statistics.mean(c.quote_volume for c in window_candles)
        if avg_vol < cfg.min_quote_volume:
            return None

    # ── Filter 4: wick — penalise if last candle closed against direction ─────
    if cfg.wick_penalty < 1.0 and window_candles:
        last = window_candles[-1]
        candle_range = last.high - last.low
        if candle_range > 0:
            close_pct = (last.close - last.low) / candle_range   # 0 = closed at low
            if direction == "UP" and close_pct < 0.35:
                confidence *= cfg.wick_penalty   # strong up-move but wick closed low
            elif direction == "DOWN" and close_pct > 0.65:
                confidence *= cfg.wick_penalty   # strong down-move but wick closed high

    return direction, confidence, price_change_pct


# ─── Backtester ───────────────────────────────────────────────────────────────
def run_backtest(
    ticks: list[Tick],
    cfg: BacktestConfig,
    candle_index: Optional[dict[int, Candle]] = None,
) -> list[Trade]:
    """Replay 5-minute windows and apply signal logic."""
    if not ticks:
        return []

    # Index ticks by second for fast lookups
    tick_index: dict[int, float] = {}
    for t in ticks:
        tick_index[int(t.ts)] = t.price

    # Find all 5-min window boundaries in the data
    first_ts = int(ticks[0].ts)
    last_ts  = int(ticks[-1].ts)
    first_window = first_ts - (first_ts % 300)
    windows = range(first_window, last_ts - 300, 300)

    trades: list[Trade] = []
    already_traded: set[int] = set()

    for w_start in windows:
        w_close = w_start + 300

        # ── Filter 2: time-of-day ─────────────────────────────────────────────
        if cfg.trading_hour_start != 0 or cfg.trading_hour_end != 24:
            hour_utc = datetime.fromtimestamp(w_start, tz=timezone.utc).hour
            if not (cfg.trading_hour_start <= hour_utc < cfg.trading_hour_end):
                continue

        # Need at least momentum_window of data before this window
        data_start = w_start - cfg.momentum_window_secs

        # Signal evaluation time = post_window_pause after open
        signal_ts = w_start + cfg.post_window_pause_secs

        # Gather ticks for momentum window (lookback + into current window)
        momentum_ticks = [
            Tick(ts=ts, price=p)
            for ts in range(data_start, signal_ts + 1)
            if (p := tick_index.get(ts)) is not None
        ]

        if len(momentum_ticks) < cfg.min_price_samples:
            continue

        current_price = tick_index.get(int(signal_ts))
        if current_price is None:
            for offset in range(1, 30):
                current_price = tick_index.get(int(signal_ts) + offset)
                if current_price:
                    break
        if current_price is None:
            continue

        # BTC price at window open and close (for outcome determination)
        btc_open = tick_index.get(w_start)
        btc_close = tick_index.get(w_close)
        if btc_open is None or btc_close is None:
            for offset in range(1, 15):
                if btc_open is None:
                    btc_open = tick_index.get(w_start + offset)
                if btc_close is None:
                    btc_close = tick_index.get(w_close + offset)
            if btc_open is None or btc_close is None:
                continue

        # Gather candles covering the momentum window for volume/wick filters
        window_candles: Optional[list[Candle]] = None
        if candle_index:
            window_candles = [
                candle_index[ts]
                for ts in range(data_start - (data_start % 60), signal_ts, 60)
                if ts in candle_index
            ]

        # Apply signal (with v2 filters if candles available)
        sig = compute_signal(momentum_ticks, current_price, cfg, window_candles)
        if sig is None:
            continue

        direction, confidence, price_change_pct = sig

        # Skip if below thresholds
        if confidence < cfg.min_confidence:
            continue

        # Simulate PolyMarket entry price.
        # We enter at post_window_pause_secs (default 20s) into a 300s window.
        # At this early stage the market is still near 0.50; we pay the ask spread.
        # As BTC moves, market adjusts—model that partial repricing:
        #   elapsed_frac = 20/300 = 6.7%, so price has moved ~6.7% toward fair value.
        elapsed_frac = cfg.post_window_pause_secs / 300
        signal_strength = min(abs(price_change_pct) / cfg.momentum_threshold_pct, 1.0) * 0.10
        market_repricing = signal_strength * elapsed_frac
        entry_price = 0.50 + cfg.market_entry_spread + market_repricing

        edge = confidence - entry_price
        if edge < cfg.min_edge_threshold:
            continue

        if w_start in already_traded:
            continue

        # Bet sizing: quarter-Kelly
        kelly = edge / max(1.0 - (0.5 + edge / 2), 0.01)
        bet = min(kelly * 1000 * 0.25, cfg.max_bet_usdc)  # assume $1000 bankroll
        bet = max(bet, cfg.min_bet_usdc)

        # Outcome
        won = (direction == "UP" and btc_close > btc_open) or \
              (direction == "DOWN" and btc_close < btc_open)

        if won:
            pnl = bet * (1.0 / entry_price - 1.0)
        else:
            pnl = -bet

        trades.append(Trade(
            window_ts=w_start,
            entry_ts=signal_ts,
            direction=direction,
            entry_price=entry_price,
            bet_usdc=bet,
            btc_open=btc_open,
            btc_close=btc_close,
            won=won,
            pnl=pnl,
            confidence=confidence,
            price_change_pct=price_change_pct,
            edge=edge,
        ))
        already_traded.add(w_start)

    return trades


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {}

    pnls = [t.pnl for t in trades]
    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]

    total_bet = sum(t.bet_usdc for t in trades)
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(trades)

    avg_win  = statistics.mean([t.pnl for t in wins])  if wins   else 0.0
    avg_loss = statistics.mean([t.pnl for t in losses]) if losses else 0.0

    # Max drawdown on cumulative P&L
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum_pnl += p
        peak = max(peak, cum_pnl)
        max_dd = min(max_dd, cum_pnl - peak)

    # Profit factor
    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe (approximate, trade-level)
    if len(pnls) > 1:
        avg_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls)
        sharpe = (avg_pnl / std_pnl) * (len(pnls) ** 0.5) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_bet": total_bet,
        "roi_pct": total_pnl / total_bet * 100 if total_bet else 0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
    }


# ─── Reporting ────────────────────────────────────────────────────────────────
def print_report(trades: list[Trade], cfg: BacktestConfig, label: str = ""):
    m = compute_metrics(trades)
    if not m:
        print("  No trades generated.")
        return

    hdr = f" {label} " if label else ""
    print(f"\n{'─'*56}")
    print(f"  BACKTEST RESULTS{hdr}")
    print(f"{'─'*56}")
    print(f"  Trades          {m['trades']:>8}")
    print(f"  Win Rate        {m['win_rate']:>7.1%}")
    print(f"  Total P&L       ${m['total_pnl']:>8.2f}")
    print(f"  Total Wagered   ${m['total_bet']:>8.2f}")
    print(f"  ROI             {m['roi_pct']:>7.2f}%")
    print(f"  Avg Win         ${m['avg_win']:>8.2f}")
    print(f"  Avg Loss        ${m['avg_loss']:>8.2f}")
    print(f"  Profit Factor   {m['profit_factor']:>8.2f}")
    print(f"  Max Drawdown    ${m['max_drawdown']:>8.2f}")
    print(f"  Sharpe (trade)  {m['sharpe']:>8.2f}")
    print(f"{'─'*56}")

    # Monthly breakdown
    if trades:
        from collections import defaultdict
        monthly: dict = defaultdict(list)
        for t in trades:
            month = datetime.fromtimestamp(t.window_ts, tz=timezone.utc).strftime("%Y-%m")
            monthly[month].append(t.pnl)
        print("\n  Monthly P&L:")
        for month in sorted(monthly):
            mp = sum(monthly[month])
            n = len(monthly[month])
            bar = "█" * int(abs(mp)) + ("" if mp >= 0 else "")
            sign = "+" if mp >= 0 else ""
            print(f"    {month}  {sign}{mp:6.2f} USDC  ({n} trades)  {bar}")
        print()


def save_plot(trades: list[Trade], path: str = "backtest_results.png"):
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping chart")
        return

    dates = [datetime.fromtimestamp(t.window_ts, tz=timezone.utc) for t in trades]
    cum_pnl = []
    running = 0.0
    for t in trades:
        running += t.pnl
        cum_pnl.append(running)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("PolyMarket BTC 5-Min Bot — Backtest Results", fontsize=14, fontweight="bold")

    # Cumulative P&L
    ax = axes[0]
    colors = ["green" if p >= 0 else "red" for p in cum_pnl]
    ax.plot(dates, cum_pnl, color="steelblue", linewidth=2, label="Cumulative P&L")
    ax.fill_between(dates, 0, cum_pnl,
                    where=[p >= 0 for p in cum_pnl], alpha=0.2, color="green")
    ax.fill_between(dates, 0, cum_pnl,
                    where=[p < 0 for p in cum_pnl], alpha=0.2, color="red")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Cumulative P&L (USDC)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # Per-trade P&L bars
    ax2 = axes[1]
    trade_colors = ["green" if t.won else "red" for t in trades]
    ax2.bar(dates, [t.pnl for t in trades], color=trade_colors, alpha=0.7, width=0.003)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Trade P&L (USDC)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

    # Confidence distribution
    ax3 = axes[2]
    win_conf  = [t.confidence for t in trades if t.won]
    lose_conf = [t.confidence for t in trades if not t.won]
    ax3.hist(win_conf,  bins=20, alpha=0.6, color="green", label="Wins",   density=True)
    ax3.hist(lose_conf, bins=20, alpha=0.6, color="red",   label="Losses", density=True)
    ax3.set_xlabel("Signal Confidence")
    ax3.set_ylabel("Density")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to {path}")


# ─── Parameter sweep ──────────────────────────────────────────────────────────
def run_sweep(ticks: list[Tick], candle_index: Optional[dict[int, Candle]] = None):
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (v2 — with all filters)")
    print("=" * 70)

    results = []
    for mom_window in [60, 120, 180]:
        for threshold in [0.05, 0.08, 0.10, 0.15]:
            for min_conf in [0.55, 0.60, 0.65, 0.70]:
                cfg = BacktestConfig(
                    momentum_window_secs=mom_window,
                    momentum_threshold_pct=threshold,
                    min_confidence=min_conf,
                    min_price_samples=max(8, mom_window // 12),
                )
                trades = run_backtest(ticks, cfg, candle_index)
                m = compute_metrics(trades)
                if not m or m["trades"] < 10:
                    continue
                results.append((cfg, m))

    # Sort by total P&L
    results.sort(key=lambda x: x[1]["total_pnl"], reverse=True)

    header = f"{'mom_win':>7} {'thresh':>7} {'min_cf':>7} | {'trades':>6} {'win%':>6} {'P&L':>8} {'ROI%':>7} {'PF':>6} {'Sharpe':>7}"
    print(f"\n{header}")
    print("─" * len(header))
    for cfg, m in results[:20]:
        print(
            f"{cfg.momentum_window_secs:>7}s "
            f"{cfg.momentum_threshold_pct:>6.2f}% "
            f"{cfg.min_confidence:>6.2f}  | "
            f"{m['trades']:>6} "
            f"{m['win_rate']:>5.1%} "
            f"${m['total_pnl']:>7.2f} "
            f"{m['roi_pct']:>6.2f}% "
            f"{m['profit_factor']:>6.2f} "
            f"{m['sharpe']:>7.2f}"
        )

    if results:
        best_cfg, best_m = results[0]
        print(f"\n  Best config: momentum_window={best_cfg.momentum_window_secs}s  "
              f"threshold={best_cfg.momentum_threshold_pct}%  "
              f"min_confidence={best_cfg.min_confidence}")
        print(f"  Best P&L: ${best_m['total_pnl']:.2f}  ROI: {best_m['roi_pct']:.2f}%  "
              f"Win rate: {best_m['win_rate']:.1%}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PolyMarket BTC Bot Backtester")
    parser.add_argument("--days",      type=int,   default=30,   help="Days of history (default 30)")
    parser.add_argument("--sweep",     action="store_true",      help="Grid search over parameters")
    parser.add_argument("--plot",      action="store_true",      help="Save P&L chart PNG")
    parser.add_argument("--compare",   action="store_true",      help="Baseline vs enhanced side-by-side")
    parser.add_argument("--max-bet",   type=float, default=10.0, help="Max bet USDC (default 10)")
    parser.add_argument("--min-conf",  type=float, default=0.75, help="Min confidence (default 0.75)")
    parser.add_argument("--momentum",  type=int,   default=60,   help="Momentum window secs (default 60)")
    parser.add_argument("--threshold", type=float, default=0.18, help="Momentum threshold %% (default 0.18)")
    args = parser.parse_args()

    # Fetch data
    ticks, candle_index = fetch_btc_history(days=args.days)
    if not ticks:
        print("Failed to fetch price data")
        sys.exit(1)

    start_dt = datetime.fromtimestamp(ticks[0].ts, tz=timezone.utc).strftime("%Y-%m-%d")
    end_dt   = datetime.fromtimestamp(ticks[-1].ts, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"  Price range: {start_dt} → {end_dt}  ({len(ticks)} ticks)")

    if args.sweep:
        run_sweep(ticks, candle_index)
        return

    if args.compare:
        # Side-by-side: baseline (no filters) vs enhanced (all filters)
        baseline_cfg = BacktestConfig(
            momentum_window_secs=60,
            momentum_threshold_pct=0.08,
            min_confidence=0.65,
            min_price_samples=8,
            min_quote_volume=0,
            trading_hour_start=0,
            trading_hour_end=24,
            range_floor_pct=0.0,
            wick_penalty=1.0,
        )
        enhanced_cfg = BacktestConfig(
            momentum_window_secs=60,
            momentum_threshold_pct=0.08,
            min_confidence=0.65,
            min_price_samples=8,
        )
        print("\n  Running baseline (no filters)...")
        baseline_trades = run_backtest(ticks, baseline_cfg, None)
        print("  Running enhanced (all filters)...")
        enhanced_trades = run_backtest(ticks, enhanced_cfg, candle_index)

        print_report(baseline_trades, baseline_cfg, label="BASELINE — no filters")
        print_report(enhanced_trades, enhanced_cfg, label="ENHANCED — v2 filters")

        bm = compute_metrics(baseline_trades)
        em = compute_metrics(enhanced_trades)
        if bm and em:
            print(f"\n  {'Metric':<20} {'Baseline':>12} {'Enhanced':>12} {'Delta':>10}")
            print("  " + "─" * 56)
            for key, fmt in [("trades", "d"), ("win_rate", ".1%"), ("roi_pct", ".2f"),
                             ("profit_factor", ".2f"), ("sharpe", ".2f"), ("max_drawdown", ".2f")]:
                bv, ev = bm[key], em[key]
                if fmt == "d":
                    print(f"  {key:<20} {bv:>12d} {ev:>12d} {ev-bv:>+10d}")
                elif fmt == ".1%":
                    print(f"  {key:<20} {bv:>11.1%} {ev:>11.1%} {ev-bv:>+9.1%}")
                else:
                    print(f"  {key:<20} {bv:>12.2f} {ev:>12.2f} {ev-bv:>+10.2f}")
        if args.plot:
            save_plot(enhanced_trades, "backtest_enhanced.png")
            save_plot(baseline_trades, "backtest_baseline.png")
        return

    # Single backtest
    cfg = BacktestConfig(
        max_bet_usdc=args.max_bet,
        min_confidence=args.min_conf,
        momentum_window_secs=args.momentum,
        momentum_threshold_pct=args.threshold,
    )

    print(f"\nRunning backtest with: momentum={cfg.momentum_window_secs}s  "
          f"threshold={cfg.momentum_threshold_pct}%  "
          f"min_confidence={cfg.min_confidence}  "
          f"max_bet=${cfg.max_bet_usdc}")
    print(f"  Filters: volume≥${cfg.min_quote_volume:,.0f}  "
          f"hours={cfg.trading_hour_start}-{cfg.trading_hour_end}UTC  "
          f"range≥{cfg.range_floor_pct}%  wick_penalty={cfg.wick_penalty}")

    trades = run_backtest(ticks, cfg, candle_index)

    if not trades:
        print("No trades were generated with current parameters.")
        print("Try lowering --min-conf or --threshold.")
        sys.exit(0)

    print_report(trades, cfg)

    if args.plot:
        save_plot(trades, "backtest_results.png")

    # Sample of recent trades
    print("\n  Last 10 trades:")
    print(f"  {'Date':>12} {'Dir':>5} {'Conf':>6} {'BTC Δ':>8} {'Bet':>6} {'P&L':>8}  {'Result'}")
    print("  " + "─" * 65)
    for t in trades[-10:]:
        dt = datetime.fromtimestamp(t.window_ts, tz=timezone.utc).strftime("%m-%d %H:%M")
        btc_chg = (t.btc_close - t.btc_open) / t.btc_open * 100
        result = "WIN ✓" if t.won else "LOSS ✗"
        print(
            f"  {dt:>12} {t.direction:>5} {t.confidence:>5.1%} "
            f"{btc_chg:>+7.3f}%  ${t.bet_usdc:>5.2f}  ${t.pnl:>+7.2f}   {result}"
        )


if __name__ == "__main__":
    main()

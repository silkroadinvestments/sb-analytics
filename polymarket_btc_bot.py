"""
PolyMarket BTC 5-Minute Prediction Trading Bot
Trades BTC up/down markets on PolyMarket using momentum signals from Binance price feed.
"""

import os
import time
import asyncio
import logging
import json
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import threading

import requests
import websocket
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, MarketOrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("polymarket_bot.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # PolyMarket credentials
    private_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_PRIVATE_KEY", ""))
    api_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_SECRET", ""))
    api_passphrase: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_PASSPHRASE", ""))
    wallet_address: str = field(default_factory=lambda: os.getenv("POLYMARKET_WALLET_ADDRESS", ""))

    # PolyMarket endpoints
    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137  # Polygon

    # Risk management
    max_bet_usdc: float = field(default_factory=lambda: float(os.getenv("MAX_BET_USDC", "10.0")))
    min_bet_usdc: float = 1.0
    max_open_positions: int = 3
    min_edge_threshold: float = 0.05   # min edge (our prob - market price) to place bet
    min_confidence: float = 0.60       # min signal confidence to act

    # Signal parameters
    momentum_window_secs: int = 120    # lookback window for momentum signal
    min_price_samples: int = 20        # min samples before generating signal
    momentum_threshold_pct: float = 0.15  # % move to consider significant

    # Timing
    market_refresh_secs: int = 30      # how often to refresh market data
    new_window_buffer_secs: int = 15   # seconds before close to stop trading window
    post_window_pause_secs: int = 20   # seconds after new window opens before trading

    # Binance websocket
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"

    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "true").lower() == "true")

    def validate(self):
        if not self.dry_run:
            assert self.private_key, "POLYMARKET_PRIVATE_KEY required"
            assert self.wallet_address, "POLYMARKET_WALLET_ADDRESS required"


# ---------------------------------------------------------------------------
# BTC Price Feed (Binance WebSocket)
# ---------------------------------------------------------------------------
class BTCPriceFeed:
    """Streams BTC/USDT trade prices from Binance and keeps a rolling history."""

    def __init__(self, config: Config):
        self.config = config
        self._prices: deque = deque(maxlen=500)  # (timestamp, price)
        self._latest_price: Optional[float] = None
        self._lock = threading.Lock()
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_ws, daemon=True)
        self._thread.start()
        log.info("BTC price feed started")

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()

    def _run_ws(self):
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.config.binance_ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                log.error(f"WebSocket error: {e}")
            if self._running:
                log.info("Reconnecting price feed in 5s...")
                time.sleep(5)

    def _on_message(self, ws, message):
        data = json.loads(message)
        price = float(data["p"])
        ts = time.time()
        with self._lock:
            self._latest_price = price
            self._prices.append((ts, price))

    def _on_error(self, ws, error):
        log.error(f"WebSocket error: {error}")

    def _on_close(self, ws, code, msg):
        log.info(f"WebSocket closed: {code} {msg}")

    @property
    def latest_price(self) -> Optional[float]:
        with self._lock:
            return self._latest_price

    def get_prices_since(self, seconds_ago: float) -> list:
        """Return list of (ts, price) tuples within the last N seconds."""
        cutoff = time.time() - seconds_ago
        with self._lock:
            return [(ts, p) for ts, p in self._prices if ts >= cutoff]

    def ready(self) -> bool:
        with self._lock:
            return len(self._prices) >= self.config.min_price_samples


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    direction: str        # "UP" or "DOWN"
    confidence: float     # 0.0 – 1.0
    price_change_pct: float
    current_price: float


class SignalGenerator:
    """Generates UP/DOWN signals from BTC price momentum."""

    def __init__(self, config: Config, price_feed: BTCPriceFeed):
        self.config = config
        self.feed = price_feed

    def generate(self) -> Optional[Signal]:
        if not self.feed.ready():
            log.debug("Not enough price samples yet")
            return None

        current = self.feed.latest_price
        if current is None:
            return None

        recent = self.feed.get_prices_since(self.config.momentum_window_secs)
        if len(recent) < self.config.min_price_samples:
            return None

        prices = [p for _, p in recent]
        start_price = prices[0]
        price_change_pct = (current - start_price) / start_price * 100

        # Momentum signal: weighted toward recent half of window
        mid = len(prices) // 2
        early_mean = statistics.mean(prices[:mid]) if mid > 0 else start_price
        late_mean = statistics.mean(prices[mid:]) if prices[mid:] else current
        acceleration = (late_mean - early_mean) / early_mean * 100

        # Directional signal
        if price_change_pct > 0:
            direction = "UP"
            raw_confidence = min(abs(price_change_pct) / self.config.momentum_threshold_pct, 1.0)
        else:
            direction = "DOWN"
            raw_confidence = min(abs(price_change_pct) / self.config.momentum_threshold_pct, 1.0)

        # Boost confidence if acceleration agrees with direction
        if (direction == "UP" and acceleration > 0) or (direction == "DOWN" and acceleration < 0):
            confidence = min(raw_confidence * 1.2, 1.0)
        else:
            confidence = raw_confidence * 0.8

        return Signal(
            direction=direction,
            confidence=confidence,
            price_change_pct=price_change_pct,
            current_price=current,
        )


# ---------------------------------------------------------------------------
# PolyMarket Client
# ---------------------------------------------------------------------------
@dataclass
class BTCMarket:
    event_slug: str
    yes_token_id: str
    no_token_id: str
    yes_price: float   # current YES ask price
    no_price: float    # current NO ask price
    window_ts: int     # unix timestamp of window start
    window_close: int  # unix timestamp of window close


class PolyMarketClient:
    """Thin wrapper around py-clob-client for BTC 5-min markets."""

    def __init__(self, config: Config):
        self.config = config
        self._client: Optional[ClobClient] = None

    def connect(self):
        if self.config.dry_run:
            log.info("[DRY RUN] Skipping PolyMarket client connection")
            return

        self._client = ClobClient(
            self.config.clob_host,
            key=self.config.private_key,
            chain_id=self.config.chain_id,
            signature_type=1,
            funder=self.config.wallet_address,
        )

        if self.config.api_key:
            creds = ApiCreds(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                api_passphrase=self.config.api_passphrase,
            )
            self._client.set_api_creds(creds)
        else:
            log.info("No API keys found, deriving from private key...")
            creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(creds)
            log.info(f"Generated API key: {creds.api_key}")
            log.info("Save these to your .env file to avoid re-deriving each run")
            log.info(f"  POLYMARKET_API_KEY={creds.api_key}")
            log.info(f"  POLYMARKET_API_SECRET={creds.api_secret}")
            log.info(f"  POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")

        log.info("Connected to PolyMarket CLOB")

    def get_current_btc_market(self) -> Optional[BTCMarket]:
        """Fetch the active BTC 5-minute market window."""
        now = int(time.time())
        window_ts = now - (now % 300)

        # Try current and next window
        for ts in [window_ts, window_ts + 300]:
            market = self._fetch_market_by_window(ts)
            if market:
                return market

        # Fallback: search Gamma API for active BTC markets
        return self._search_active_btc_market()

    def _fetch_market_by_window(self, window_ts: int) -> Optional[BTCMarket]:
        slug = f"btc-updown-5m-{window_ts}"
        try:
            resp = requests.get(
                f"{self.config.gamma_host}/events",
                params={"slug": slug, "active": "true"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data:
                return None

            event = data[0] if isinstance(data, list) else data
            return self._parse_event(event, window_ts)

        except Exception as e:
            log.debug(f"Market not found for window {window_ts}: {e}")
            return None

    def _search_active_btc_market(self) -> Optional[BTCMarket]:
        """Search for any active BTC 5-min market via Gamma API."""
        try:
            resp = requests.get(
                f"{self.config.gamma_host}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "tag_slug": "crypto",
                    "limit": 50,
                },
                timeout=10,
            )
            resp.raise_for_status()
            events = resp.json()

            for event in events:
                slug = event.get("slug", "")
                if "btc-updown-5m" in slug or "btc-up-down-5m" in slug:
                    window_ts = int(slug.split("-")[-1]) if slug.split("-")[-1].isdigit() else 0
                    market = self._parse_event(event, window_ts)
                    if market:
                        log.info(f"Found BTC market via search: {slug}")
                        return market

        except Exception as e:
            log.error(f"Failed to search for BTC markets: {e}")

        return None

    def _parse_event(self, event: dict, window_ts: int) -> Optional[BTCMarket]:
        """Parse a Gamma API event into a BTCMarket object."""
        try:
            markets = event.get("markets", [])
            if not markets:
                return None

            yes_token_id = no_token_id = None
            yes_price = no_price = 0.5

            for m in markets:
                outcomes = m.get("outcomes", "[]")
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                outcome_prices = m.get("outcomePrices", "[]")
                if isinstance(outcome_prices, str):
                    outcome_prices = json.loads(outcome_prices)
                clob_token_ids = m.get("clobTokenIds", "[]")
                if isinstance(clob_token_ids, str):
                    clob_token_ids = json.loads(clob_token_ids)

                for i, outcome in enumerate(outcomes):
                    price = float(outcome_prices[i]) if i < len(outcome_prices) else 0.5
                    token_id = clob_token_ids[i] if i < len(clob_token_ids) else None

                    if outcome.upper() in ("YES", "UP", "HIGHER"):
                        yes_token_id = token_id
                        yes_price = price
                    elif outcome.upper() in ("NO", "DOWN", "LOWER"):
                        no_token_id = token_id
                        no_price = price

            if not yes_token_id or not no_token_id:
                return None

            return BTCMarket(
                event_slug=event.get("slug", ""),
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                yes_price=yes_price,
                no_price=no_price,
                window_ts=window_ts,
                window_close=window_ts + 300,
            )
        except Exception as e:
            log.error(f"Failed to parse event: {e}")
            return None

    def get_token_price(self, token_id: str) -> Optional[float]:
        """Get current best ask price for a token."""
        if self.config.dry_run or not self._client:
            return None
        try:
            return float(self._client.get_price(token_id, side="BUY"))
        except Exception as e:
            log.error(f"Failed to get price for {token_id}: {e}")
            return None

    def place_market_order(self, token_id: str, usdc_amount: float, side: str = BUY) -> Optional[dict]:
        """Place a market order for a YES or NO token."""
        if self.config.dry_run:
            log.info(f"[DRY RUN] Would place {side} order: {usdc_amount} USDC on token {token_id[:16]}...")
            return {"dry_run": True, "token_id": token_id, "amount": usdc_amount}

        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=usdc_amount,
            )
            signed = self._client.create_market_order(order_args)
            result = self._client.post_order(signed, orderType="FOK")
            log.info(f"Order placed: {result}")
            return result
        except Exception as e:
            log.error(f"Order failed: {e}")
            return None

    def get_balance(self) -> float:
        """Return USDC balance."""
        if self.config.dry_run or not self._client:
            return 1000.0  # simulated balance
        try:
            balance = self._client.get_balance()
            return float(balance)
        except Exception as e:
            log.error(f"Failed to get balance: {e}")
            return 0.0


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------
class RiskManager:
    """Sizes bets and enforces risk limits."""

    def __init__(self, config: Config):
        self.config = config
        self.open_positions: list = []

    def can_trade(self) -> bool:
        return len(self.open_positions) < self.config.max_open_positions

    def size_bet(self, confidence: float, edge: float, balance: float) -> float:
        """Kelly-inspired bet sizing capped at max_bet_usdc."""
        if edge <= 0 or confidence < self.config.min_confidence:
            return 0.0

        # Simplified Kelly: f = edge / (1 - probability_of_loss)
        kelly_fraction = edge / max(1.0 - (0.5 + edge / 2), 0.01)
        kelly_bet = balance * kelly_fraction * 0.25  # quarter-Kelly for safety

        bet = min(kelly_bet, self.config.max_bet_usdc)
        bet = max(bet, self.config.min_bet_usdc)
        return round(bet, 2)

    def register_position(self, market: BTCMarket, direction: str, bet_usdc: float):
        self.open_positions.append({
            "window_ts": market.window_ts,
            "window_close": market.window_close,
            "direction": direction,
            "bet_usdc": bet_usdc,
            "opened_at": time.time(),
        })

    def expire_closed(self):
        """Remove positions whose market windows have closed."""
        now = time.time()
        before = len(self.open_positions)
        self.open_positions = [p for p in self.open_positions if p["window_close"] > now]
        removed = before - len(self.open_positions)
        if removed:
            log.info(f"Cleared {removed} expired position(s)")

    def already_in_window(self, window_ts: int) -> bool:
        return any(p["window_ts"] == window_ts for p in self.open_positions)


# ---------------------------------------------------------------------------
# Main Trading Bot
# ---------------------------------------------------------------------------
class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.price_feed = BTCPriceFeed(config)
        self.signal_gen = SignalGenerator(config, self.price_feed)
        self.poly = PolyMarketClient(config)
        self.risk = RiskManager(config)
        self._current_market: Optional[BTCMarket] = None
        self._last_market_refresh: float = 0
        self._running = False
        self._trade_count = 0
        self._pnl_estimate = 0.0

    def start(self):
        log.info("=" * 60)
        log.info("PolyMarket BTC 5-Min Trading Bot")
        log.info(f"Mode: {'DRY RUN' if self.config.dry_run else '*** LIVE TRADING ***'}")
        log.info(f"Max bet: ${self.config.max_bet_usdc} USDC")
        log.info(f"Min edge: {self.config.min_edge_threshold:.0%}")
        log.info(f"Min confidence: {self.config.min_confidence:.0%}")
        log.info("=" * 60)

        self.config.validate()
        self.price_feed.start()
        self.poly.connect()

        # Wait for price feed to warm up
        log.info("Warming up price feed...")
        for _ in range(30):
            if self.price_feed.ready():
                break
            time.sleep(1)
        log.info(f"Price feed ready. BTC price: ${self.price_feed.latest_price:,.2f}")

        self._running = True
        self._run_loop()

    def stop(self):
        self._running = False
        self.price_feed.stop()
        log.info("Bot stopped")

    def _run_loop(self):
        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                log.info("Interrupted by user")
                self.stop()
                break
            except Exception as e:
                log.error(f"Tick error: {e}", exc_info=True)
            time.sleep(5)

    def _tick(self):
        now = time.time()

        # Expire old positions
        self.risk.expire_closed()

        # Refresh market data periodically
        if now - self._last_market_refresh > self.config.market_refresh_secs:
            self._current_market = self.poly.get_current_btc_market()
            self._last_market_refresh = now
            if self._current_market:
                secs_left = self._current_market.window_close - now
                log.info(
                    f"Market: {self._current_market.event_slug} | "
                    f"YES={self._current_market.yes_price:.2f} "
                    f"NO={self._current_market.no_price:.2f} | "
                    f"{secs_left:.0f}s left | "
                    f"BTC=${self.price_feed.latest_price:,.2f}"
                )
            else:
                log.warning("No active BTC 5-min market found")
                return

        if not self._current_market:
            return

        secs_left = self._current_market.window_close - now

        # Don't trade near window close or just after open
        if secs_left < self.config.new_window_buffer_secs:
            log.debug(f"Too close to window close ({secs_left:.0f}s), skipping")
            return

        window_age = now - self._current_market.window_ts
        if window_age < self.config.post_window_pause_secs:
            log.debug(f"Window too fresh ({window_age:.0f}s old), waiting...")
            return

        # Already in this window?
        if self.risk.already_in_window(self._current_market.window_ts):
            return

        # Check risk limits
        if not self.risk.can_trade():
            log.debug("Max open positions reached")
            return

        # Generate signal
        signal = self.signal_gen.generate()
        if not signal:
            log.debug("No signal yet")
            return

        if signal.confidence < self.config.min_confidence:
            log.debug(f"Signal confidence too low: {signal.confidence:.2%}")
            return

        # Calculate edge
        if signal.direction == "UP":
            token_id = self._current_market.yes_token_id
            market_price = self._current_market.yes_price
            our_prob = signal.confidence
        else:
            token_id = self._current_market.no_token_id
            market_price = self._current_market.no_price
            our_prob = signal.confidence

        edge = our_prob - market_price

        if edge < self.config.min_edge_threshold:
            log.debug(
                f"Insufficient edge: direction={signal.direction} "
                f"our_prob={our_prob:.2%} market={market_price:.2%} edge={edge:.2%}"
            )
            return

        # Size the bet
        balance = self.poly.get_balance()
        bet_usdc = self.risk.size_bet(signal.confidence, edge, balance)
        if bet_usdc < self.config.min_bet_usdc:
            log.debug(f"Bet size too small: ${bet_usdc}")
            return

        # Execute
        log.info(
            f"TRADE SIGNAL | direction={signal.direction} | "
            f"confidence={signal.confidence:.2%} | "
            f"price_chg={signal.price_change_pct:+.3f}% | "
            f"edge={edge:.2%} | bet=${bet_usdc}"
        )

        result = self.poly.place_market_order(token_id, bet_usdc)
        if result:
            self.risk.register_position(self._current_market, signal.direction, bet_usdc)
            self._trade_count += 1
            self._print_status(signal, bet_usdc, edge)

    def _print_status(self, signal: Signal, bet_usdc: float, edge: float):
        log.info(
            f"[Trade #{self._trade_count}] "
            f"BTC={signal.direction} @ ${signal.current_price:,.2f} | "
            f"Bet=${bet_usdc} | Edge={edge:.2%} | "
            f"Open positions: {len(self.risk.open_positions)}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    config = Config()

    print("\nPolyMarket BTC 5-Min Bot")
    print(f"DRY_RUN={config.dry_run}")
    print(f"MAX_BET_USDC={config.max_bet_usdc}")
    print()

    bot = TradingBot(config)
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        bot.stop()


if __name__ == "__main__":
    main()

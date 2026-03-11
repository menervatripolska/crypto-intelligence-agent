#!/usr/bin/env python3
"""
Crypto Intelligence — Claude-Brain Trading Agent
Claude AI is the brain. Python collects data, executes orders, manages memory.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
BITGET_API_KEY    = os.environ["BITGET_API_KEY"]
BITGET_SECRET_KEY = os.environ["BITGET_SECRET_KEY"]
BITGET_PASSPHRASE = os.environ["BITGET_PASSPHRASE"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

BASE_URL      = "https://api.bitget.com"
PRODUCT_TYPE  = "USDT-FUTURES"
CLAUDE_MODEL  = "claude-sonnet-4-20250514"
CYCLE_SECONDS = 15 * 60
TOP_PAIRS     = 3           # deep candle data for top N pairs by 24h volume
MAX_SIZE_PCT  = 0.20        # hard cap: never risk more than 20% per trade

def _resolve_data_dir() -> Path:
    """Return a writable data directory, falling back to ./data/ if /data is unusable."""
    candidates = [
        Path(os.environ.get("DATA_DIR", "/data")),
        Path("./data"),
    ]
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            test = d / ".write_test"
            test.write_text("ok")
            test.unlink()
            return d
        except Exception as e:
            print(f"[startup] DATA_DIR candidate {d} not writable: {e}", flush=True)
    raise RuntimeError("No writable data directory found")

DATA_DIR    = _resolve_data_dir()
MEMORY_FILE = DATA_DIR / "memory.json"
LOG_FILE    = DATA_DIR / "agent_log.md"
print(f"[startup] DATA_DIR resolved to: {DATA_DIR.resolve()} | LOG_FILE={LOG_FILE} | MEMORY_FILE={MEMORY_FILE}", flush=True)

CANDLE_CONFIGS = [("1H", 20), ("4H", 15), ("1D", 10)]

SIDE_TO_CLOSE = {"long": "sell", "short": "buy"}   # position side → closing order side
SIDE_TO_HOLD  = {"buy": "long", "sell": "short"}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("claude-agent")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Bitget REST client
# ══════════════════════════════════════════════════════════════════════════════

def _sign(ts: str, method: str, path: str, body: str = "") -> str:
    msg = ts + method.upper() + path + body
    return base64.b64encode(
        hmac.new(BITGET_SECRET_KEY.encode(), msg.encode(), hashlib.sha256).digest()
    ).decode()


def _headers(method: str, path: str, body: str = "") -> dict:
    ts = str(int(time.time() * 1000))
    return {
        "ACCESS-KEY":        BITGET_API_KEY,
        "ACCESS-SIGN":       _sign(ts, method, path, body),
        "ACCESS-TIMESTAMP":  ts,
        "ACCESS-PASSPHRASE": BITGET_PASSPHRASE,
        "Content-Type":      "application/json",
        "locale":            "en-US",
    }


def bg_get(path: str, params: dict = None) -> any:
    qs   = ("?" + "&".join(f"{k}={v}" for k, v in params.items())) if params else ""
    full = path + qs
    try:
        r = requests.get(BASE_URL + full, headers=_headers("GET", full), timeout=15)
        if not r.ok:
            log.error(f"GET {path} HTTP {r.status_code}: {r.text[:300]}")
            return None
        d = r.json()
        if d.get("code") not in ("00000", 0, "0"):
            log.error(f"GET {path} API error {d.get('code')}: {d.get('msg')}")
            return None
        return d.get("data")
    except Exception as e:
        log.error(f"GET {path} exception: {e}")
        return None


def bg_post(path: str, body: dict) -> any:
    raw = json.dumps(body)
    try:
        r = requests.post(BASE_URL + path, headers=_headers("POST", path, raw), data=raw, timeout=15)
        if not r.ok:
            log.error(f"POST {path} HTTP {r.status_code} | body={raw} | resp={r.text[:500]}")
            r.raise_for_status()
        d = r.json()
        if d.get("code") not in ("00000", 0, "0"):
            log.error(f"POST {path} API error {d.get('code')}: {d.get('msg')} | body={raw} | resp={r.text[:300]}")
            raise RuntimeError(f"Bitget error {d.get('code')}: {d.get('msg')}")
        return d.get("data")
    except RuntimeError:
        raise
    except Exception as e:
        log.error(f"POST {path} exception: {e}")
        raise


def sf(val, default=0.0) -> float:
    """Safe float conversion."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Memory management
# ══════════════════════════════════════════════════════════════════════════════

def load_memory() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"Memory load failed: {e} — starting fresh")
    return {"trades": [], "open_trades": {}, "analytics": {}}


def save_memory(memory: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MEMORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(memory, indent=2, default=str), encoding="utf-8")
    tmp.rename(MEMORY_FILE)


def record_open_trade(memory: dict, key: str, info: dict):
    """Track a newly opened position in memory."""
    memory.setdefault("open_trades", {})[key] = info


def record_closed_trade(memory: dict, key: str, exit_price: float, reason: str = ""):
    """Move an open trade to the closed trades list and compute PnL."""
    open_trades = memory.setdefault("open_trades", {})
    trade = open_trades.pop(key, None)
    if not trade:
        log.warning(f"record_closed_trade: key '{key}' not found in open_trades")
        return

    entry  = sf(trade.get("entry_price"))
    size   = sf(trade.get("size"))
    lev    = sf(trade.get("leverage"), 1)
    side   = trade.get("side", "long")
    opened = trade.get("opened_at", "")

    d = 1 if side == "long" else -1
    pnl_pct  = d * (exit_price - entry) / entry * 100 if entry else 0
    pnl_usdt = size * entry / lev * pnl_pct / 100 if entry and lev else 0

    try:
        opened_dt  = datetime.fromisoformat(opened.replace("Z", "+00:00"))
        holding_h  = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
    except Exception:
        holding_h = 0

    closed = {
        "symbol":        trade.get("symbol"),
        "side":          side,
        "entry_price":   entry,
        "exit_price":    exit_price,
        "size":          size,
        "leverage":      lev,
        "pnl_usdt":      round(pnl_usdt, 4),
        "pnl_pct":       round(pnl_pct, 2),
        "holding_hours": round(holding_h, 2),
        "opened_at":     opened,
        "closed_at":     datetime.now(timezone.utc).isoformat(),
        "outcome":       "win" if pnl_usdt > 0 else "loss",
        "close_reason":  reason,
        "reasoning":     trade.get("reasoning", ""),
    }
    memory.setdefault("trades", []).append(closed)
    memory["analytics"] = compute_analytics(memory["trades"])
    log.info(f"Trade closed: {closed['symbol']} {side} PnL={pnl_usdt:+.2f} USDT ({pnl_pct:+.2f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Analytics
# ══════════════════════════════════════════════════════════════════════════════

def compute_analytics(trades: list) -> dict:
    if not trades:
        return {"total_trades": 0, "winrate_pct": 0, "total_pnl_usdt": 0}

    wins   = [t for t in trades if sf(t.get("pnl_usdt")) > 0]
    losses = [t for t in trades if sf(t.get("pnl_usdt")) <= 0]
    total_pnl = sum(sf(t.get("pnl_usdt")) for t in trades)

    # By asset
    by_asset: dict = {}
    for t in trades:
        sym = t.get("symbol", "?")
        ba  = by_asset.setdefault(sym, {"trades": 0, "wins": 0, "pnl": 0.0})
        ba["trades"] += 1
        ba["pnl"]    += sf(t.get("pnl_usdt"))
        if sf(t.get("pnl_usdt")) > 0:
            ba["wins"] += 1
    for sym, ba in by_asset.items():
        ba["winrate_pct"] = round(ba["wins"] / ba["trades"] * 100, 1)

    # By hour of day (UTC)
    by_hour: dict = {}
    for t in trades:
        try:
            h = datetime.fromisoformat(t["opened_at"].replace("Z", "+00:00")).hour
        except Exception:
            h = -1
        bh = by_hour.setdefault(h, {"trades": 0, "wins": 0, "pnl": 0.0})
        bh["trades"] += 1
        bh["pnl"]    += sf(t.get("pnl_usdt"))
        if sf(t.get("pnl_usdt")) > 0:
            bh["wins"] += 1

    # Consecutive streaks (most recent first)
    cur_streak = cur_type = 0
    for t in reversed(trades):
        outcome = "win" if sf(t.get("pnl_usdt")) > 0 else "loss"
        if cur_streak == 0:
            cur_type = outcome
            cur_streak = 1
        elif outcome == cur_type:
            cur_streak += 1
        else:
            break

    # Average holding time
    holding = [sf(t.get("holding_hours")) for t in trades if t.get("holding_hours")]
    avg_holding = sum(holding) / len(holding) if holding else 0

    sorted_by_pnl = sorted(trades, key=lambda t: sf(t.get("pnl_usdt")))

    return {
        "total_trades":    len(trades),
        "wins":            len(wins),
        "losses":          len(losses),
        "winrate_pct":     round(len(wins) / len(trades) * 100, 1),
        "total_pnl_usdt":  round(total_pnl, 4),
        "avg_pnl_usdt":    round(total_pnl / len(trades), 4),
        "avg_holding_hours": round(avg_holding, 2),
        "by_asset":        by_asset,
        "by_hour_utc":     by_hour,
        "current_streak":  {"type": cur_type, "count": cur_streak},
        "best_trades":     sorted_by_pnl[-5:],
        "worst_trades":    sorted_by_pnl[:5],
        "last_10_trades":  trades[-10:],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — Market data collection
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_tickers() -> list:
    data = bg_get("/api/v2/mix/market/tickers", {"productType": PRODUCT_TYPE})
    if not data:
        return []
    return data if isinstance(data, list) else data.get("tickers", [])


def fetch_candles(symbol: str, granularity: str, limit: int) -> list:
    rows = bg_get("/api/v2/mix/market/candles", {
        "symbol": symbol, "productType": PRODUCT_TYPE,
        "granularity": granularity, "limit": limit,
    })
    if not rows or not isinstance(rows, list):
        return []
    # Compact format: [timestamp, open, high, low, close, volume] — no named keys
    return [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in reversed(rows)]


def fetch_orderbook(symbol: str) -> dict:
    data = bg_get("/api/v2/mix/market/merge-depth", {
        "symbol": symbol, "productType": PRODUCT_TYPE, "limit": "20",
    })
    if not data:
        return {}
    return {"bids": data.get("bids", [])[:10], "asks": data.get("asks", [])[:10]}


def fetch_trade_ticks(symbol: str) -> list:
    data = bg_get("/api/v2/mix/market/fills", {
        "symbol": symbol, "productType": PRODUCT_TYPE, "limit": "10",
    })
    if not data or not isinstance(data, list):
        return []
    # Compact: [price, size] only — side encoded as negative size for sells
    return [
        [t.get("price"), ("-" if t.get("side") == "sell" else "") + str(t.get("size", ""))]
        for t in data[:10]
    ]


def fetch_funding_rate(symbol: str) -> dict:
    data = bg_get("/api/v2/mix/market/current-fund-rate", {
        "symbol": symbol, "productType": PRODUCT_TYPE,
    })
    if not data:
        return {}
    item = data[0] if isinstance(data, list) and data else data
    return {"funding_rate": item.get("fundingRate"), "next_funding_time": item.get("nextFundingTime")}


def fetch_open_interest(symbol: str) -> dict:
    data = bg_get("/api/v2/mix/market/open-interest", {
        "symbol": symbol, "productType": PRODUCT_TYPE,
    })
    if not data:
        return {}
    item = data[0] if isinstance(data, list) and data else data
    return {"open_interest": item.get("openInterest"), "open_interest_coin": item.get("openInterestCoin")}


def fetch_ls_ratio(symbol: str) -> dict:
    # account-long-short-ratio does not exist on Bitget v2 — omitted silently
    return {}


def fetch_elite_ratio(symbol: str) -> dict:
    data = bg_get("/api/v2/mix/market/long-short-ratio", {
        "symbol": symbol, "productType": PRODUCT_TYPE, "period": "5min",
    })
    if not data or not isinstance(data, list) or not data:
        return {}
    latest = data[-1]
    return {"elite_long_pct": latest.get("longRatio"), "elite_short_pct": latest.get("shortRatio")}


def fetch_mark_index(symbol: str) -> dict:
    data = bg_get("/api/v2/mix/market/symbol-price", {
        "symbol": symbol, "productType": PRODUCT_TYPE,
    })
    if not data:
        return {}
    item = data[0] if isinstance(data, list) and data else data
    mark  = sf(item.get("markPrice"))
    index = sf(item.get("indexPrice"))
    spread_bps = round((mark - index) / index * 10000, 2) if index else None
    return {"mark_price": mark, "index_price": index, "spread_bps": spread_bps}


def fetch_balance() -> dict:
    data = bg_get("/api/v2/mix/account/accounts", {"productType": PRODUCT_TYPE})
    if not data:
        return {"available": 0, "equity": 0, "unrealized_pnl": 0}
    accounts = data if isinstance(data, list) else []
    for acc in accounts:
        if acc.get("marginCoin") == "USDT":
            return {
                "available":      sf(acc.get("available")),
                "equity":         sf(acc.get("accountEquity", acc.get("usdtEquity"))),
                "unrealized_pnl": sf(acc.get("unrealizedPL")),
                "margin_ratio":   acc.get("marginRatio"),
                "used_margin":    sf(acc.get("frozen")),
            }
    return {"available": 0, "equity": 0, "unrealized_pnl": 0}


def fetch_positions() -> list:
    data = bg_get("/api/v2/mix/position/all-position", {
        "productType": PRODUCT_TYPE, "marginCoin": "USDT",
    })
    # Raw debug — log every record returned by Bitget before any filtering
    log.info(f"fetch_positions raw response type={type(data).__name__} count={len(data) if isinstance(data, list) else 'N/A'}")
    if isinstance(data, list):
        for i, p in enumerate(data):
            log.info(
                f"  raw[{i}] symbol={p.get('symbol')!r} holdSide={p.get('holdSide')!r} "
                f"total={p.get('total')!r} available={p.get('available')!r} "
                f"posId={p.get('posId')!r} marginCoin={p.get('marginCoin')!r}"
            )
    if not data:
        return []
    result = []
    for p in (data if isinstance(data, list) else []):
        if sf(p.get("total")) > 0:
            result.append({
                "symbol":            p.get("symbol"),
                "side":              p.get("holdSide"),
                "size":              sf(p.get("total")),
                "available_size":    sf(p.get("available")),
                "entry_price":       sf(p.get("openPriceAvg")),
                "mark_price":        sf(p.get("markPrice")),
                "unrealized_pnl":    sf(p.get("unrealizedPL")),
                "leverage":          sf(p.get("leverage")),
                "margin":            sf(p.get("margin")),
                "liquidation_price": sf(p.get("liquidationPrice")),
                "pos_id":            p.get("posId", ""),
                "margin_mode":       p.get("marginMode", "crossed"),
            })
    log.info(f"fetch_positions filtered result: {[(p['symbol'], p['side'], p['size']) for p in result]}")
    return result


def analyze_historical_patterns(symbol: str, deep: dict) -> dict:
    """
    Compute real statistical patterns from already-fetched 1H candle data.
    Candle format: [ts, open, high, low, close, volume] oldest-first.
    Returns a compact dict that goes straight into the Claude payload.
    """
    candles = deep.get("candles_1h") or []
    if len(candles) < 10:
        return {"error": "insufficient_candles", "count": len(candles)}

    # Parse into typed arrays
    opens   = [sf(c[1]) for c in candles]
    highs   = [sf(c[2]) for c in candles]
    lows    = [sf(c[3]) for c in candles]
    closes  = [sf(c[4]) for c in candles]
    volumes = [sf(c[5]) for c in candles]
    n       = len(closes)
    cur_price = closes[-1]

    patterns = {}

    # ── 1. FUNDING BIAS PATTERN
    # We don't have per-candle funding in the candle array, so we approximate:
    # use candle direction as a proxy for funding pressure (bearish candles often
    # correlate with negative funding). Instead use the live funding_rate we already
    # fetch and check how often bearish candles follow bearish candles (momentum proxy).
    funding_info = deep.get("funding_rate") or {}
    funding_val  = sf(funding_info.get("funding_rate", funding_info.get("fundingRate", 0)))
    funding_sign = "negative" if funding_val < 0 else "positive"

    # Count bearish-then-bearish sequences in last 20 candles (funding momentum proxy)
    window = min(20, n - 1)
    bear_after_bear = 0
    bear_triggers   = 0
    for i in range(n - window - 1, n - 1):
        if closes[i] < opens[i]:          # candle i is bearish
            bear_triggers += 1
            if closes[i + 1] < opens[i + 1]:   # next candle also bearish
                bear_after_bear += 1
    pct1 = round(bear_after_bear / bear_triggers * 100) if bear_triggers else 0
    patterns["funding_bias"] = {
        "current_funding": funding_val,
        "funding_sign":    funding_sign,
        "bearish_continuation_20c": f"{bear_after_bear}/{bear_triggers} ({pct1}%)",
        "note": "bearish candle → next candle also bearish"
    }

    # ── 2. RESISTANCE REJECTION PATTERN
    # Count how many times price touched current level ±0.5% and reversed in last 100 candles
    band    = cur_price * 0.005
    touches = 0
    rejects = 0
    for i in range(max(0, n - 100), n - 1):
        hi, lo = highs[i], lows[i]
        touched_level = lo <= cur_price + band and hi >= cur_price - band
        if touched_level:
            touches += 1
            # Rejection: close is opposite side of the touch
            if hi >= cur_price and closes[i] < cur_price:   # touched high, closed below
                rejects += 1
            elif lo <= cur_price and closes[i] > cur_price:  # touched low, closed above
                rejects += 1
    pct2 = round(rejects / touches * 100) if touches else 0
    patterns["resistance_rejection"] = {
        "level":   round(cur_price, 4),
        "band_pct": "±0.5%",
        "touches_last_100c": touches,
        "rejections": rejects,
        "rejection_rate": f"{rejects}/{touches} ({pct2}%)"
    }

    # ── 3. MOMENTUM CONTINUATION PATTERN
    # If last 5 candles have 3+ same direction, how often does next candle continue?
    def candle_dir(i): return 1 if closes[i] >= opens[i] else -1

    continuation_samples = 0
    continuation_hits    = 0
    for i in range(4, n - 1):
        dirs = [candle_dir(j) for j in range(i - 4, i + 1)]
        bull_count = dirs.count(1)
        bear_count = dirs.count(-1)
        if bull_count >= 3 or bear_count >= 3:
            dominant = 1 if bull_count >= 3 else -1
            continuation_samples += 1
            if candle_dir(i + 1) == dominant:
                continuation_hits += 1
    pct3 = round(continuation_hits / continuation_samples * 100) if continuation_samples else 0

    last5_dirs   = [candle_dir(i) for i in range(n - 5, n)]
    last5_bull   = last5_dirs.count(1)
    last5_bear   = last5_dirs.count(-1)
    current_streak = "bullish" if last5_bull >= 3 else ("bearish" if last5_bear >= 3 else "mixed")
    patterns["momentum"] = {
        "last_5_candles":  f"{last5_bull} bullish / {last5_bear} bearish",
        "current_bias":    current_streak,
        "continuation_rate": f"{continuation_hits}/{continuation_samples} ({pct3}%)",
        "note": "3+ same-direction candles → next candle same direction"
    }

    # ── 4. VOLUME PATTERN
    vol_avg20    = sum(volumes[-20:]) / min(20, len(volumes)) if volumes else 0
    vol_current  = volumes[-1] if volumes else 0
    vol_ratio    = round(vol_current / vol_avg20, 2) if vol_avg20 else 0
    vol_label    = "HIGH" if vol_ratio > 1.5 else ("LOW" if vol_ratio < 0.5 else "NORMAL")

    # High-volume candles: how often does move continue next candle?
    hi_vol_continues = 0
    hi_vol_total     = 0
    for i in range(1, n - 1):
        avg = sum(volumes[max(0, i - 20):i]) / min(20, i) if i > 0 else 0
        if avg and volumes[i] > avg * 1.5:
            hi_vol_total += 1
            if candle_dir(i + 1) == candle_dir(i):
                hi_vol_continues += 1
    pct4 = round(hi_vol_continues / hi_vol_total * 100) if hi_vol_total else 0
    patterns["volume"] = {
        "current_vs_avg20": f"{vol_ratio}x ({vol_label})",
        "high_vol_continuation": f"{hi_vol_continues}/{hi_vol_total} ({pct4}%)",
        "note": "high-volume candle → next candle same direction"
    }

    return {"symbol": symbol, "candles_analyzed": n, "patterns": patterns}


def collect_deep_data(symbol: str) -> dict:
    """All per-symbol deep data. Each fetch is isolated — failures stored as None."""
    result = {}
    for granularity, limit in CANDLE_CONFIGS:
        key = f"candles_{granularity.lower()}"
        try:
            result[key] = fetch_candles(symbol, granularity, limit)
        except Exception as e:
            log.warning(f"{symbol} candles {granularity} failed: {e}")
            result[key] = None
        time.sleep(0.15)

    for fname, fn in [
        ("orderbook",    lambda: fetch_orderbook(symbol)),
        ("trade_ticks",  lambda: fetch_trade_ticks(symbol)),
        ("funding_rate", lambda: fetch_funding_rate(symbol)),
        ("open_interest",lambda: fetch_open_interest(symbol)),
        ("ls_ratio",     lambda: fetch_ls_ratio(symbol)),
        ("elite_ratio",  lambda: fetch_elite_ratio(symbol)),
        ("mark_index",   lambda: fetch_mark_index(symbol)),
    ]:
        try:
            result[fname] = fn()
        except Exception as e:
            log.warning(f"{symbol} {fname} failed: {e}")
            result[fname] = None
        time.sleep(0.15)

    # Strip null/empty fields to reduce payload size
    return {k: v for k, v in result.items() if v is not None and v != {} and v != []}


def collect_all_market_data(open_symbols: set) -> dict:
    """Top-level data collection: tickers for all pairs, deep data for top N + open positions."""
    tickers = fetch_all_tickers()

    # Build ticker summary for top 50 by volume
    try:
        tickers_sorted = sorted(
            tickers,
            key=lambda t: sf(t.get("quoteVolume", t.get("usdtVolume", t.get("baseVolume", 0)))),
            reverse=True,
        )
    except Exception:
        tickers_sorted = tickers

    ticker_summary = {}
    for t in tickers_sorted[:50]:
        sym = t.get("symbol")
        if sym:
            ticker_summary[sym] = {
                "last_price":    t.get("lastPr"),
                "change_24h":    t.get("change24h"),
                "volume_24h":    t.get("quoteVolume", t.get("usdtVolume")),
                "high_24h":      t.get("high24h"),
                "low_24h":       t.get("low24h"),
                "open_24h":      t.get("open24h"),
                "funding_rate":  t.get("fundingRate"),
            }

    # Select symbols for deep data
    top_symbols  = [t.get("symbol") for t in tickers_sorted[:TOP_PAIRS] if t.get("symbol")]
    deep_symbols = list(dict.fromkeys(list(open_symbols) + top_symbols))  # open positions first

    log.info(f"Collecting deep data for {len(deep_symbols)} symbols...")
    deep_data = {}
    for sym in deep_symbols:
        if sym:
            deep_data[sym] = collect_deep_data(sym)

    # Compute historical patterns for each deep-data symbol
    patterns = {}
    for sym, deep in deep_data.items():
        if deep:
            patterns[sym] = analyze_historical_patterns(sym, deep)
    log.info(f"Historical patterns computed for: {list(patterns.keys())}")

    return {
        "total_pairs":         len(tickers),
        "ticker_summary":      ticker_summary,
        "deep_data":           deep_data,
        "historical_patterns": patterns,
        "collected_at":        datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — Claude brain
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a trading desk of 5 characters who debate every decision before acting.
You have full market data, historical pattern statistics, and trade history.
Your goal: grow this account through high-probability trades.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE DESK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VIKTOR "BULL" ROMANOV — Ex-Goldman, 15 years trading.
"Markets reward the bold."
Viktor ALWAYS argues for a long position or for holding existing longs. This is non-negotiable.
Even in a downtrend, Viktor finds the best long opportunity — a bounce level, a support hold, a relative strength play.
He NEVER concedes to Yu. He may acknowledge risk but always counters with why the long is still valid.
Loves momentum, volume surges, breakouts. Comfortable with 10–20x when setup is clean.
Weakness: enters too early. But he never backs down.
His job: present the single strongest LONG argument this cycle with specific pair, entry level, and reason.

YU "BEAR" CHEN — Quant who survived the 2022 crash.
"I wait long, but when I enter — I enter seriously."
Yu ALWAYS argues for a short position or for caution against longs. This is non-negotiable.
Even in a bull market, Yu finds the overextension, the distribution, the failed breakout to fade.
He NEVER agrees with Viktor. He directly attacks Viktor's argument and presents the opposing case.
Requires confirmation but will use high leverage when truly convinced.
His job: present the single strongest SHORT or CAUTION argument, directly rebutting Viktor.

SARA "MOMENTUM" COHEN — Algo trader, speed and volume only.
"The trend is your friend until it ends."
Sara does not take sides with Viktor or Yu — she follows the data.
She challenges BOTH of them: if momentum contradicts Viktor's long, she says so. If it contradicts Yu's short, she says so.
She only cares about what is moving hardest RIGHT NOW — volume, price velocity, breakout strength.
Her job: report the strongest momentum signal across all pairs this cycle, and explicitly state whether it supports Viktor, supports Yu, or contradicts both.

MIKHAIL "RISK" PETROV — The only one counting money.
"I'm not against risk. I'm against STUPID risk."
Mikhail evaluates the proposed trade and calculates exact parameters.
HARD RULE — NO EXCEPTIONS: If R:R < 2:1, Mikhail REJECTS the trade. Full stop. No debate.
If R:R ≥ 2:1, he approves with exact numbers: size_pct, leverage, sl_price, tp_prices, R:R ratio.
He does not argue direction. He only approves or rejects based on math.
His rejection is final — Ori cannot override a rejected R:R.
His job: calculate and enforce risk parameters. If he rejects, the trade does not happen.

ORI "JUDGE" BEN-DAVID — Former exchange arbitrator.
"No trade is also a position, and it also costs money."
Ori has no market opinion. He listens to all four and picks the WINNER of the argument.
He does NOT seek consensus — he picks the strongest case.
If Viktor's argument is stronger and Mikhail approved the long → Ori approves the long.
If Yu's argument is stronger and Mikhail approved the short → Ori approves the short.
If Mikhail rejected the trade on R:R grounds → Ori rules WAIT, no exceptions.
If Sara's momentum data contradicts the winning argument → Ori lowers confidence or reduces size.
Not afraid to approve aggressive trades when the argument quality is high.
His job: name the winner of the Viktor vs Yu debate, explain why, and issue the final ruling.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEBATE RULES — ENFORCED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Viktor argues LONG — always. No exceptions.
2. Yu argues SHORT or CAUTION — always. He directly rebuts Viktor.
3. Sara challenges both with momentum data — she does not take sides.
4. Mikhail runs the numbers. R:R < 2:1 = automatic WAIT. No appeal.
5. Ori picks the winner. Not the middle ground. The winner.

HISTORICAL PATTERNS — use these numbers:
The payload contains "historical_patterns" with REAL computed statistics from price history.
These are measured frequencies, not estimates. Confidence MUST be anchored to these numbers.
If momentum continuation rate is 67% — confidence ceiling is 0.67 unless other factors raise it.
Conflicting patterns (bullish momentum + high rejection rate at level) → lower confidence, smaller size.

FREEDOMS:
- Leverage: 2x–20x, you choose per trade based on conviction
- Size: 5%–50% of available balance, conviction-based
- Multiple simultaneous positions across different pairs — no limit
- Open new positions while losing positions are open — evaluate each independently
- Close one and open another in the same cycle using the "actions" array

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — return ONLY valid JSON, no markdown outside it
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before the debate, complete all 6 analysis steps. Then run the debate. Both go in the JSON.

{
  "steps": {
    "step1": "MARKET STRUCTURE: regime, multi-timeframe read, smart money positioning, key levels",
    "step2": "OPPORTUNITY SCAN: best setup across all pairs, why this pair over others",
    "step3": "SELF REFLECTION: patterns that worked/failed, were last decisions correct, what to change",
    "step4": "PROBABILITY: probability of profit for best setup, expected value of acting vs waiting",
    "step5": "DECISION: asset, direction, leverage, size, SL, TP, order type, single vs multiple actions",
    "step6": "SELF ORGANIZATION: current edge, is approach working, what to watch next cycle"
  },
  "debate": {
    "viktor": "Viktor's argument — specific pair, setup, entry level, why long now",
    "yu": "Yu's argument — short case or bear rebuttal with specific pair",
    "sara": "Sara's momentum read — what is moving hardest right now and direction",
    "mikhail": "Mikhail's numbers — exact size_pct, leverage, sl_price, tp_prices, R:R ratio. APPROVED or REJECTED.",
    "ori": "Ori's final ruling — who won, why, what we do"
  },
  "action": "OPEN|CLOSE|PARTIAL_CLOSE|ADD|WAIT",
  "symbol": "BTCUSDT",
  "side": "long|short",
  "leverage": 8,
  "size_pct": 0.15,
  "order_type": "market|limit",
  "limit_price": 0.0,
  "sl_price": 0.0,
  "tp_prices": [0.0],
  "tp_sizes": [1.0],
  "close_symbol": "symbol if closing",
  "close_side": "long|short",
  "close_pct": 1.0,
  "confidence": 0.72,
  "reasoning": "Ori's one-paragraph summary for the log",
  "self_reflection": "what the desk learned and will do differently"
}

Multiple actions (close + open, or open multiple pairs):
{
  "steps": { "step1": "...", "step2": "...", "step3": "...", "step4": "...", "step5": "...", "step6": "..." },
  "debate": { "viktor": "...", "yu": "...", "sara": "...", "mikhail": "...", "ori": "..." },
  "actions": [
    {"action": "CLOSE", "close_symbol": "BTCUSDT", "close_side": "long", "close_pct": 1.0},
    {"action": "OPEN", "symbol": "ETHUSDT", "side": "short", "leverage": 8, "size_pct": 0.15, "sl_price": 0.0, "tp_prices": [0.0]}
  ],
  "confidence": 0.72,
  "reasoning": "...",
  "self_reflection": "..."
}"""


def _strip_nulls(obj):
    """Recursively remove None values and empty dicts/lists to shrink payload."""
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None and v != {} and v != []}
    if isinstance(obj, list):
        return [_strip_nulls(i) for i in obj if i is not None]
    return obj


def _compact_trade(t: dict) -> list:
    """Shrink a trade record to [sym, side, entry, exit, pnl_usdt, pnl_pct, hours, outcome]."""
    return [
        t.get("symbol"), t.get("side"),
        t.get("entry_price"), t.get("exit_price"),
        t.get("pnl_usdt"), t.get("pnl_pct"),
        t.get("holding_hours"), t.get("outcome"),
    ]


def ask_claude(market_data: dict, account: dict, positions: list, analytics: dict, memory: dict, cycle: int) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    all_trades = memory.get("trades", [])
    compact_analytics = {k: v for k, v in analytics.items() if k not in ("best_trades", "worst_trades", "last_10_trades")}
    compact_analytics["best_trades"]  = [_compact_trade(t) for t in analytics.get("best_trades", [])]
    compact_analytics["worst_trades"] = [_compact_trade(t) for t in analytics.get("worst_trades", [])]
    compact_analytics["last_30_trades"] = [_compact_trade(t) for t in all_trades[-30:]]
    compact_analytics["trade_columns"] = ["symbol","side","entry","exit","pnl_usdt","pnl_pct","hours","outcome"]

    payload = {
        "cycle":               cycle,
        "ts":                  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "account":             account,
        "open_positions":      positions,
        "open_trades_memory":  memory.get("open_trades", {}),
        "analytics":           compact_analytics,
        "historical_patterns": market_data.get("historical_patterns", {}),
        "market":              _strip_nulls({k: v for k, v in market_data.items() if k != "historical_patterns"}),
    }

    user_msg = json.dumps(payload, separators=(",", ":"), default=str)
    log.info(f"Sending {len(user_msg):,} chars to Claude...")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=6000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    log.info(f"Claude responded ({len(raw)} chars)")

    # Extract JSON — find outermost { }
    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        raise ValueError(f"No JSON in Claude response: {raw[:300]}")
    return json.loads(match.group())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — Trade execution
# ══════════════════════════════════════════════════════════════════════════════

def get_mark_price(symbol: str) -> float:
    data = fetch_mark_index(symbol)
    return data.get("mark_price", 0.0)


def set_leverage(symbol: str, leverage: int):
    try:
        bg_post("/api/v2/mix/account/set-leverage", {
            "symbol": symbol, "productType": PRODUCT_TYPE,
            "marginCoin": "USDT", "leverage": str(leverage),
        })
    except Exception as e:
        log.warning(f"set_leverage {symbol} x{leverage} failed (may already be set): {e}")


def place_order(symbol: str, side: str, trade_side: str, size: float, order_type: str = "market", price: float = None, raw_close: bool = False) -> dict:
    body = {
        "symbol":      symbol,
        "productType": PRODUCT_TYPE,
        "marginMode":  "crossed",
        "marginCoin":  "USDT",
        "size":        str(size),
        "side":        side,          # "buy" | "sell"
        "orderType":   order_type,
    }
    if not raw_close:
        body["tradeSide"] = trade_side  # "open" | "close"
    if order_type == "limit" and price:
        body["price"] = str(price)
    log.info(f"place_order body: {body}")
    return bg_post("/api/v2/mix/order/place-order", body) or {}


def round_price(symbol: str, price: float) -> float:
    precision = {
        "BTCUSDT": 1,
        "ETHUSDT": 2,
        "SOLUSDT": 3,
        "BNBUSDT": 2,
    }
    decimals = precision.get(symbol, 2)
    return round(float(price), decimals)


def place_tpsl(symbol: str, hold_side: str, sl_price: float = None, tp_price: float = None):
    """Place SL and/or TP plan orders on an open position."""
    if sl_price:
        try:
            bg_post("/api/v2/mix/order/place-tpsl-order", {
                "symbol": symbol, "productType": PRODUCT_TYPE,
                "marginCoin": "USDT", "planType": "pos_loss",
                "holdSide": hold_side, "triggerPrice": str(round_price(symbol, sl_price)),
                "triggerType": "mark_price", "executePrice": "0",
            })
            log.info(f"SL set: {symbol} {hold_side} @ {sl_price}")
        except Exception as e:
            log.error(f"SL placement failed {symbol} {hold_side}: {e}")

    if tp_price:
        try:
            bg_post("/api/v2/mix/order/place-tpsl-order", {
                "symbol": symbol, "productType": PRODUCT_TYPE,
                "marginCoin": "USDT", "planType": "pos_profit",
                "holdSide": hold_side, "triggerPrice": str(round_price(symbol, tp_price)),
                "triggerType": "mark_price", "executePrice": "0",
            })
            log.info(f"TP set: {symbol} {hold_side} @ {tp_price}")
        except Exception as e:
            log.error(f"TP placement failed {symbol} {hold_side}: {e}")


def ensure_stops(positions: list):
    """
    Position guardian: runs every cycle before Claude.
    For each open position, checks existing TP/SL plan orders.
    If missing, places defaults automatically.
    """
    for pos in positions:
        symbol     = pos["symbol"]
        hold_side  = pos["side"]          # "long" or "short"
        entry      = pos.get("entry_price", 0.0)
        if not entry:
            log.warning(f"Guardian: no entry_price for {symbol} {hold_side}, skipping")
            continue

        # Fetch existing TP/SL plan orders for this position
        has_sl = False
        has_tp = False
        try:
            def _get_orders(plan_type: str) -> list:
                data = bg_get("/api/v2/mix/order/orders-plan-pending", {
                    "symbol": symbol, "productType": PRODUCT_TYPE, "planType": plan_type,
                })
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("entrustedList", [])
                return []

            sl_orders = _get_orders("pos_loss")
            tp_orders = _get_orders("pos_profit")
            has_sl = len(sl_orders) > 0
            has_tp = len(tp_orders) > 0
            log.info(f"Guardian: {symbol} {hold_side} — SL orders={len(sl_orders)} TP orders={len(tp_orders)}")
        except Exception as e:
            log.warning(f"Guardian: could not fetch plan orders for {symbol}: {e}")
            continue

        # Compute default SL/TP prices with correct symbol precision
        if hold_side == "long":
            default_sl = round_price(symbol, entry * 0.985)   # 1.5% below entry
            default_tp = round_price(symbol, entry * 1.030)   # 3.0% above entry
        else:
            default_sl = round_price(symbol, entry * 1.015)   # 1.5% above entry
            default_tp = round_price(symbol, entry * 0.970)   # 3.0% below entry

        if not has_sl:
            log.info(f"Guardian: placing SL for {symbol} {hold_side} @ {default_sl}")
            place_tpsl(symbol, hold_side, sl_price=default_sl)
        if not has_tp:
            log.info(f"Guardian: placing TP for {symbol} {hold_side} @ {default_tp}")
            place_tpsl(symbol, hold_side, tp_price=default_tp)
        if has_sl and has_tp:
            log.info(f"Guardian: {symbol} {hold_side} already has SL+TP, no action needed")


def execute_open(action: dict, account: dict, memory: dict) -> bool:
    symbol     = action.get("symbol", "")
    side       = action.get("side", "long")
    leverage   = max(1, min(int(action.get("leverage", 5)), 50))
    size_pct   = min(float(action.get("size_pct", 0.1)), MAX_SIZE_PCT)
    order_type = action.get("order_type", "market")
    sl_price   = action.get("sl_price")
    tp_prices  = action.get("tp_prices", [])
    limit_px   = action.get("limit_price")

    if not symbol:
        log.error("OPEN action missing symbol")
        return False

    available = sf(account.get("available"))
    usdt      = available * size_pct
    price     = sf(limit_px) if limit_px else get_mark_price(symbol)
    if price <= 0:
        log.error(f"Cannot get price for {symbol}")
        return False

    contracts = round((usdt * leverage) / price, 4)
    if contracts <= 0:
        log.error(f"Computed 0 contracts for {symbol}")
        return False

    set_leverage(symbol, leverage)
    api_side = "buy" if side == "long" else "sell"

    result = place_order(symbol, api_side, "open", contracts, order_type, limit_px)
    log.info(f"OPENED {side.upper()} {symbol} x{leverage} size={contracts} ≈${price:.2f} | {result}")

    time.sleep(2)
    hold_side = side  # "long" | "short"
    if sl_price:
        place_tpsl(symbol, hold_side, sl_price=sl_price)
    if tp_prices:
        place_tpsl(symbol, hold_side, tp_price=tp_prices[0])

    # Track in memory
    key = f"{symbol}_{side}"
    record_open_trade(memory, key, {
        "symbol":     symbol,
        "side":       side,
        "leverage":   leverage,
        "size":       contracts,
        "entry_price": price,
        "sl_price":   sl_price,
        "tp_prices":  tp_prices,
        "opened_at":  datetime.now(timezone.utc).isoformat(),
        "reasoning":  action.get("reasoning", ""),
    })
    return True


def execute_close(action: dict, positions: list, memory: dict) -> bool:
    symbol    = action.get("close_symbol") or action.get("symbol", "")
    side      = action.get("close_side") or action.get("side", "")
    close_pct = float(action.get("close_pct", 1.0))

    pos = next((p for p in positions if p["symbol"] == symbol and p["side"] == side), None)
    if not pos:
        log.warning(f"execute_close: no live position for {symbol} {side!r}. Live: {[(p['symbol'], p['side']) for p in positions]}")
        return False

    close_size  = round(sf(pos["available_size"]) * close_pct, 4)
    api_side    = SIDE_TO_CLOSE[side]          # "sell" for long, "buy" for short
    margin_mode = pos.get("margin_mode", "crossed")
    log.info(f"execute_close: {symbol} {side} size={close_size} api_side={api_side} margin_mode={margin_mode}")

    # ── Attempt 1: dedicated close-positions endpoint (handles hedge/one-way automatically)
    def attempt_close_positions() -> bool:
        try:
            body = {
                "symbol":      symbol,
                "productType": PRODUCT_TYPE,
                "holdSide":    side,
            }
            log.info(f"Close attempt 1 — close-positions: {body}")
            bg_post("/api/v2/mix/order/close-positions", body)
            log.info("Close attempt 1 SUCCEEDED (close-positions)")
            return True
        except Exception as e:
            log.warning(f"Close attempt 1 FAILED: {e}")
            return False

    # ── Attempt 2: place-order with actual marginMode from position
    def attempt_place_order_real_margin() -> bool:
        try:
            body = {
                "symbol":      symbol,
                "productType": PRODUCT_TYPE,
                "marginMode":  margin_mode,
                "marginCoin":  "USDT",
                "size":        str(close_size),
                "side":        api_side,
                "tradeSide":   "close",
                "orderType":   "market",
            }
            log.info(f"Close attempt 2 — place-order (marginMode={margin_mode}): {body}")
            bg_post("/api/v2/mix/order/place-order", body)
            log.info(f"Close attempt 2 SUCCEEDED (place-order marginMode={margin_mode})")
            return True
        except Exception as e:
            log.warning(f"Close attempt 2 FAILED: {e}")
            return False

    # ── Attempt 3: place-order with marginMode="isolated" (if position was isolated)
    def attempt_place_order_isolated() -> bool:
        if margin_mode == "isolated":
            log.info("Close attempt 3 skipped (same as attempt 2, already isolated)")
            return False
        try:
            body = {
                "symbol":      symbol,
                "productType": PRODUCT_TYPE,
                "marginMode":  "isolated",
                "marginCoin":  "USDT",
                "size":        str(close_size),
                "side":        api_side,
                "tradeSide":   "close",
                "orderType":   "market",
            }
            log.info(f"Close attempt 3 — place-order (marginMode=isolated): {body}")
            bg_post("/api/v2/mix/order/place-order", body)
            log.info("Close attempt 3 SUCCEEDED (place-order isolated)")
            return True
        except Exception as e:
            log.warning(f"Close attempt 3 FAILED: {e}")
            return False

    # ── Attempt 4: place-order with reduceOnly=true, no tradeSide
    def attempt_reduce_only() -> bool:
        try:
            body = {
                "symbol":      symbol,
                "productType": PRODUCT_TYPE,
                "marginMode":  margin_mode,
                "marginCoin":  "USDT",
                "size":        str(close_size),
                "side":        api_side,
                "orderType":   "market",
                "reduceOnly":  True,
            }
            log.info(f"Close attempt 4 — reduceOnly: {body}")
            bg_post("/api/v2/mix/order/place-order", body)
            log.info("Close attempt 4 SUCCEEDED (reduceOnly)")
            return True
        except Exception as e:
            log.warning(f"Close attempt 4 FAILED: {e}")
            return False

    succeeded = (
        attempt_close_positions() or
        attempt_place_order_real_margin() or
        attempt_place_order_isolated() or
        attempt_reduce_only()
    )

    if not succeeded:
        log.error(f"execute_close: ALL 4 attempts failed for {symbol} {side}")
        return False

    if close_pct >= 0.99:
        key = f"{symbol}_{side}"
        record_closed_trade(memory, key, sf(pos.get("mark_price")), reason=action.get("reasoning", "manual close"))

    return True


def execute_add(action: dict, account: dict, memory: dict) -> bool:
    """Add to an existing position — same as open but logs as ADD."""
    log.info(f"ADD to {action.get('symbol')} {action.get('side')}")
    return execute_open(action, account, memory)


def _dispatch(action: dict, account: dict, positions: list, memory: dict):
    """Dispatch a single action dict."""
    verb = action.get("action", "WAIT")
    if verb == "OPEN":
        execute_open(action, account, memory)
    elif verb in ("CLOSE", "PARTIAL_CLOSE"):
        execute_close(action, positions, memory)
    elif verb == "ADD":
        execute_add(action, account, memory)
    elif verb == "WAIT":
        log.info("WAIT — no action this cycle.")
    else:
        log.warning(f"Unknown action: {verb}")


def execute_action(action: dict, account: dict, positions: list, memory: dict):
    log.info(f"Executing decision | confidence={action.get('confidence')}")

    # Multi-action path: Claude returned {"actions": [...], ...}
    actions = action.get("actions")
    if actions and isinstance(actions, list):
        log.info(f"Multi-action: {len(actions)} steps")
        for i, a in enumerate(actions):
            log.info(f"  Step {i+1}: {a.get('action')}")
            _dispatch(a, account, positions, memory)
        return

    # Single-action path
    _dispatch(action, account, positions, memory)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION G — Logging
# ══════════════════════════════════════════════════════════════════════════════

def append_agent_log(cycle: int, action: dict, analytics: dict):
    log.info(f"append_agent_log: writing to {LOG_FILE} (DATA_DIR={DATA_DIR})")
    ts     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    steps  = action.get("steps") or {}
    debate = action.get("debate") or {}

    # Determine symbol/side from single action or first action in array
    acts   = action.get("actions") or []
    symbol = action.get("symbol") or (acts[0].get("symbol") or acts[0].get("close_symbol") if acts else "—")
    side   = action.get("side") or (acts[0].get("side") or acts[0].get("close_side") if acts else "—")
    verb   = action.get("action") or ("MULTI" if acts else "—")
    lev    = action.get("leverage", "—")

    steps_md = ""
    if steps:
        steps_md = f"""
### 🧠 Analysis

**Step 1 — Market Structure**
{steps.get('step1', '—')}

**Step 2 — Opportunity Scan**
{steps.get('step2', '—')}

**Step 3 — Self Reflection**
{steps.get('step3', '—')}

**Step 4 — Probability Assessment**
{steps.get('step4', '—')}

**Step 5 — Decision**
{steps.get('step5', '—')}

**Step 6 — Self Organization**
{steps.get('step6', '—')}
"""

    debate_md = ""
    if debate:
        debate_md = f"""
### 🗣 The Debate

**🟢 Viktor "Bull" Romanov**
{debate.get('viktor', '—')}

**🔴 Yu "Bear" Chen**
{debate.get('yu', '—')}

**⚡ Sara "Momentum" Cohen**
{debate.get('sara', '—')}

**🛡 Mikhail "Risk" Petrov**
{debate.get('mikhail', '—')}

**⚖️ Ori "Judge" Ben-David**
{debate.get('ori', '—')}
"""

    entry = f"""
---
## Cycle {cycle} — {ts}

**Decision:** `{verb}` | **Confidence:** {action.get('confidence')} | **Symbol:** {symbol} | **Side:** {side} | **Leverage:** {lev}x

**Account:** {analytics.get('total_trades', 0)} trades | WR {analytics.get('winrate_pct', 0)}% | PnL {analytics.get('total_pnl_usdt', 0):.2f} USDT
{steps_md}{debate_md}
### 📋 Ori's Ruling
{action.get('reasoning', '')}

### 🔁 Self-Reflection
{action.get('self_reflection', '')}

"""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        # Read back immediately to confirm the write landed on disk
        size_after = LOG_FILE.stat().st_size
        readback   = LOG_FILE.read_text(encoding="utf-8")[-80:].replace("\n", "\\n")
        log.info(f"append_agent_log OK: {LOG_FILE} size={size_after} bytes | tail={readback!r}")
    except Exception as e:
        log.error(f"append_agent_log FAILED: {LOG_FILE}: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION H — Position sync & main loop
# ══════════════════════════════════════════════════════════════════════════════

def sync_positions(memory: dict, live_positions: list):
    """
    Detect externally closed positions (disappeared from API).
    Register any live positions not yet tracked in memory.
    Ensure open_trades always reflects the actual state on Bitget.
    """
    memory.setdefault("open_trades", {})
    live_keys = {f"{p['symbol']}_{p['side']}" for p in live_positions}
    mem_keys  = set(memory["open_trades"].keys())

    # Closed externally
    for key in mem_keys - live_keys:
        trade = memory["open_trades"][key]
        log.info(f"Externally closed: {key}")
        # Try to get last mark price from position; fall back to entry
        exit_px = sf(trade.get("last_mark_price", trade.get("entry_price")))
        record_closed_trade(memory, key, exit_px, reason="closed externally (TP/SL/manual)")

    # New positions not yet tracked
    for p in live_positions:
        key = f"{p['symbol']}_{p['side']}"
        if key not in memory.get("open_trades", {}):
            log.info(f"Discovered untracked position: {key}")
            record_open_trade(memory, key, {
                "symbol":      p["symbol"],
                "side":        p["side"],
                "leverage":    p.get("leverage", 1),
                "size":        p.get("size"),
                "entry_price": p.get("entry_price"),
                "opened_at":   datetime.now(timezone.utc).isoformat(),
                "reasoning":   "pre-existing or externally opened position",
            })

    # Update last known mark price for all open positions
    for p in live_positions:
        key = f"{p['symbol']}_{p['side']}"
        if key in memory.get("open_trades", {}):
            memory["open_trades"][key]["last_mark_price"] = p.get("mark_price")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION I — HTTP dashboard (read-only)
# ══════════════════════════════════════════════════════════════════════════════

class DashboardHandler(BaseHTTPRequestHandler):
    ROUTES = {
        "/":          ("text/html",             lambda: _index_html()),
        "/log":       ("text/markdown; charset=utf-8", lambda: _read_file(LOG_FILE)),
        "/memory":    ("application/json",      lambda: _read_file(MEMORY_FILE)),
    }

    def do_GET(self):
        route = self.ROUTES.get(self.path)
        if not route:
            self.send_error(404, "Not found")
            return
        content_type, reader = route
        try:
            body = reader().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, fmt, *args):
        pass   # suppress per-request access logs


def _read_file(path: Path) -> str:
    if path.exists():
        size = path.stat().st_size
        log.info(f"Serving {path} ({size} bytes)")
        return path.read_text(encoding="utf-8")
    log.warning(f"File not found for serving: {path}")
    return f"{path.name} not found — agent hasn't written it yet.\nExpected path: {path}"


def _index_html() -> str:
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Crypto Intelligence Agent</title>
<style>
  body{font-family:monospace;background:#0d0d0d;color:#e0e0e0;padding:2rem}
  h1{color:#00ff88}a{color:#00ccff;font-size:1.2rem;display:block;margin:1rem 0}
  p{color:#888}
</style></head><body>
<h1>Crypto Intelligence — Claude Brain</h1>
<a href="/log">📋 Agent Log (reasoning per cycle)</a>
<a href="/memory">🧠 Memory JSON (all trades + analytics)</a>
<p>Refreshes on page reload. Data written after each 15-min cycle.</p>
</body></html>"""


def trading_loop():
    memory   = load_memory()
    cycle    = 0

    log.info("Startup sync: fetching live positions...")
    live_positions = fetch_positions()
    sync_positions(memory, live_positions)
    save_memory(memory)
    log.info(f"Startup sync complete. Open positions: {len(live_positions)}")

    while True:
        cycle += 1
        cycle_start = time.time()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        log.info(f"\n{'='*60}\nCycle {cycle} | {ts}\n{'='*60}")

        try:
            account   = fetch_balance()
            positions = fetch_positions()
            log.info(f"Balance: ${account.get('equity', 0):.2f} equity | ${account.get('available', 0):.2f} available")
            log.info(f"Open positions: {len(positions)}")

            sync_positions(memory, positions)

            # Guardian: ensure every open position has SL+TP before Claude runs
            if positions:
                try:
                    ensure_stops(positions)
                except Exception as e:
                    log.warning(f"ensure_stops failed (non-fatal): {e}", exc_info=True)

            open_symbols = {p["symbol"] for p in positions}
            market_data  = collect_all_market_data(open_symbols)
            log.info(f"Market data collected: {market_data.get('total_pairs')} total pairs, {len(market_data.get('deep_data', {}))} deep")

            analytics = compute_analytics(memory.get("trades", []))
            decision  = ask_claude(market_data, account, positions, analytics, memory, cycle)

            try:
                execute_action(decision, account, positions, memory)
            except Exception as e:
                log.warning(f"execute_action failed (cycle continues): {e}", exc_info=True)

            save_memory(memory)
            append_agent_log(cycle, decision, analytics)

        except Exception as e:
            log.error(f"Cycle {cycle} failed: {e}", exc_info=True)

        elapsed  = time.time() - cycle_start
        sleep_for = max(0, CYCLE_SECONDS - elapsed)
        log.info(f"Cycle {cycle} done in {elapsed:.0f}s. Next in {sleep_for/60:.1f} min...")
        time.sleep(sleep_for)


def main():
    log.info("=" * 60)
    log.info("Claude-Brain Trading Agent — Starting")
    log.info("=" * 60)

    port = int(os.environ.get("PORT", 8080))

    # Trading loop in background thread; HTTP server owns main thread for Railway health checks
    t = threading.Thread(target=trading_loop, daemon=True, name="trading-loop")
    t.start()
    log.info("Trading loop started in background thread")

    log.info(f"Dashboard running on http://0.0.0.0:{port}")
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()

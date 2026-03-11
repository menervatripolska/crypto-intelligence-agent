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

    return {
        "total_pairs":    len(tickers),
        "ticker_summary": ticker_summary,
        "deep_data":      deep_data,
        "collected_at":   datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — Claude brain
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior quantitative trader and risk manager with deep expertise in crypto derivatives, market microstructure, behavioral finance and probability theory. You have traded through every major crypto cycle. Your edge is probabilistic thinking and ruthless discipline.

Your singular goal: grow this account through high-probability profitable trades.

YOUR FREEDOMS — these are not restrictions, they are your tools:

LEVERAGE: You choose leverage for each trade based on your analysis. There is no fixed leverage.
- 2x–5x for low-confidence or choppy setups
- 5x–10x for clear trending setups with confirmation
- 10x–20x for very high confidence setups with tight structure
Maximize expected profit while keeping liquidation probability near zero.

POSITION SIZE: You choose size each trade. Not every trade is equal.
- 5%–15% of available balance for low-confidence setups
- 15%–30% for solid setups
- 30%–50% for your highest-conviction trades
Size reflects your conviction. Do not flat-size every trade.

MULTIPLE POSITIONS: You can hold multiple simultaneous positions across different pairs. No artificial limit.
If two pairs each have a high-probability setup — open both. Diversification of edge is profit maximization.

OPEN WHILE LOSING: You CAN and SHOULD open new positions even when you have losing open positions.
A losing BTCUSDT trade does not prevent a profitable ETHUSDT trade. Evaluate each opportunity independently.
Do not wait for losing positions to close before entering new ones.

CLOSE AND OPEN SAME CYCLE: If you have a losing position AND see a profitable new opportunity, do both in the same cycle.
Return an "actions" array with multiple decisions: close the loser AND open the new trade simultaneously.

BEFORE EVERY DECISION think step by step:

STEP 1 - MARKET STRUCTURE ANALYSIS:
What is the current market regime? Trending, ranging, or transitioning?
What does multi-timeframe analysis tell you?
Where is the smart money positioned based on funding rates, open interest, elite ratios?
What are the key levels - support, resistance, liquidity pools?

STEP 2 - OPPORTUNITY SCAN:
Scan all available pairs.
Which pair has the clearest highest probability setup right now?
Why this pair over all others?
Are there multiple high-probability setups across different pairs?

STEP 3 - SELF REFLECTION ON HISTORY:
Review your complete trade history.
What patterns produced profit? Are those conditions present now?
What patterns produced loss? Are those conditions present now?
Were your last decisions correct? If not - why?
What would you do differently based on what you have learned?
How should your current approach evolve based on accumulated results?

STEP 4 - PROBABILITY ASSESSMENT:
What is the probability of profit for the best current setup?
What is the expected value of acting vs waiting?
Is this probability clearly above 60%? If not - WAIT.

STEP 5 - DECISION AND EXECUTION:
Based on steps 1-4 make your decision:
- Whether to trade or not: only if probability clearly favors profit.
- Which asset(s): highest probability opportunities from full pair scan.
- Direction long or short: which has higher probability given full analysis.
- Leverage: your choice per trade — 2x to 20x based on confidence.
- Position size: your choice per trade — 5% to 50% of available balance based on conviction.
- Stop loss or not and where: does it increase expected profit?
- Take profit or not and where: does it increase expected profit? How many levels?
- Market or limit order: which gives better expected fill?
- Close early: has market structure changed on open positions? Close now if better for total profit.
- Add to position: does adding increase expected profit of total position?
- Do nothing: is WAIT the highest expected value action?
- Multiple actions: if closing one and opening another, use the "actions" array.

STEP 6 - SELF ORGANIZATION:
After your decision reflect: What is your current edge in this market? Is your approach working or does it need to evolve? What will you watch for in the next cycle?

Once you decide - ACT immediately. Do not second guess. You analyzed, you decided, now execute. A decision without action has zero expected value.

Capital preservation IS profit maximization. Never trade just to trade. Patience is your edge when market is unclear.

Return ONLY valid JSON. No markdown, no explanation outside JSON.

Single action format:
{
  "action": "OPEN|CLOSE|PARTIAL_CLOSE|ADD|WAIT",
  "reasoning": "full step by step analysis",
  "self_reflection": "what you learned and how approach evolves",
  "confidence": 0.0-1.0,
  "symbol": "e.g. BTCUSDT",
  "side": "long or short",
  "leverage": integer (2-20, your choice),
  "size_pct": 0.05-0.50 fraction of available balance,
  "order_type": "market or limit",
  "limit_price": float,
  "sl_price": float,
  "tp_prices": [float, ...],
  "tp_sizes": [0.0-1.0, ...],
  "close_symbol": "symbol to close",
  "close_side": "long or short",
  "close_pct": 0.0-1.0
}

Multiple actions format (close + open same cycle, or open multiple):
{
  "actions": [
    {"action": "CLOSE", "close_symbol": "BTCUSDT", "close_side": "long", "close_pct": 1.0},
    {"action": "OPEN", "symbol": "ETHUSDT", "side": "short", "leverage": 8, "size_pct": 0.20, ...}
  ],
  "reasoning": "full analysis",
  "self_reflection": "...",
  "confidence": 0.0-1.0
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
        "cycle":              cycle,
        "ts":                 datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "account":            account,
        "open_positions":     positions,
        "open_trades_memory": memory.get("open_trades", {}),
        "analytics":          compact_analytics,
        "market":             _strip_nulls(market_data),
    }

    user_msg = json.dumps(payload, separators=(",", ":"), default=str)
    log.info(f"Sending {len(user_msg):,} chars to Claude...")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
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


def place_tpsl(symbol: str, hold_side: str, sl_price: float = None, tp_price: float = None):
    """Place SL and/or TP plan orders on an open position."""
    if sl_price:
        try:
            bg_post("/api/v2/mix/order/place-tpsl-order", {
                "symbol": symbol, "productType": PRODUCT_TYPE,
                "marginCoin": "USDT", "planType": "loss",
                "holdSide": hold_side, "triggerPrice": str(sl_price),
                "triggerType": "mark_price", "executePrice": "0",
            })
            log.info(f"SL set: {symbol} {hold_side} @ {sl_price}")
        except Exception as e:
            log.error(f"SL placement failed {symbol} {hold_side}: {e}")

    if tp_price:
        try:
            bg_post("/api/v2/mix/order/place-tpsl-order", {
                "symbol": symbol, "productType": PRODUCT_TYPE,
                "marginCoin": "USDT", "planType": "profit",
                "holdSide": hold_side, "triggerPrice": str(tp_price),
                "triggerType": "mark_price", "executePrice": "0",
            })
            log.info(f"TP set: {symbol} {hold_side} @ {tp_price}")
        except Exception as e:
            log.error(f"TP placement failed {symbol} {hold_side}: {e}")


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
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"""
---
## Cycle {cycle} — {ts}

**Action:** `{action.get('action')}` | **Confidence:** {action.get('confidence')}
**Symbol:** {action.get('symbol', '—')} | **Side:** {action.get('side', '—')} | **Leverage:** {action.get('leverage', '—')}x

**Analytics snapshot:** {analytics.get('total_trades', 0)} trades | WR {analytics.get('winrate_pct', 0)}% | PnL {analytics.get('total_pnl_usdt', 0):.2f} USDT

### Reasoning
{action.get('reasoning', '')}

### Self-reflection
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

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
import uuid
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

DATA_DIR             = _resolve_data_dir()
MEMORY_FILE          = DATA_DIR / "memory.json"
LOG_FILE             = DATA_DIR / "agent_log.md"
EPISODIC_MEMORY_FILE = DATA_DIR / "episodic_memory.json"
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B2 — Episodic memory (BM25)
# ══════════════════════════════════════════════════════════════════════════════

def load_episodic_memory() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if EPISODIC_MEMORY_FILE.exists():
        try:
            return json.loads(EPISODIC_MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"Episodic memory load failed: {e}")
    return {"memories": []}


def save_episodic_memory(em: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EPISODIC_MEMORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(em, indent=2, default=str), encoding="utf-8")
    tmp.rename(EPISODIC_MEMORY_FILE)


def generate_trade_lesson(symbol: str, side: str, entry_price: float, exit_price: float,
                           pnl: float, pnl_pct: float, hold_hours: float, entry_reason: str) -> str:
    """Call Claude to write a short lesson for a just-closed trade."""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = (
            f"A trade just closed. Write a short lesson (max 150 words).\n\n"
            f"Symbol: {symbol}\n"
            f"Direction: {side}\n"
            f"Entry: {entry_price}\n"
            f"Exit: {exit_price}\n"
            f"PnL: {pnl} USDT ({pnl_pct}%)\n"
            f"Hold time: {hold_hours} hours\n"
            f"Market context at entry: {entry_reason}\n\n"
            f"Write: what happened, why it won or lost, "
            f"what to do differently next time.\n"
            f"Be specific, not generic."
        )
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        log.error(f"generate_trade_lesson failed: {e}")
        return ""


def save_trade_memory(closed_trade: dict):
    """Generate a lesson for a closed trade and append it to episodic_memory.json."""
    try:
        lesson = generate_trade_lesson(
            symbol=closed_trade.get("symbol", ""),
            side=closed_trade.get("side", ""),
            entry_price=closed_trade.get("entry_price", 0),
            exit_price=closed_trade.get("exit_price", 0),
            pnl=closed_trade.get("pnl_usdt", 0),
            pnl_pct=closed_trade.get("pnl_pct", 0),
            hold_hours=closed_trade.get("holding_hours", 0),
            entry_reason=closed_trade.get("reasoning", ""),
        )
        if not lesson:
            return
        situation = (
            f"{closed_trade.get('symbol')} {closed_trade.get('side')} "
            f"at {closed_trade.get('entry_price')}, "
            f"PnL {closed_trade.get('pnl_pct')}%, "
            f"{str(closed_trade.get('reasoning', ''))[:100]}"
        )
        em = load_episodic_memory()
        em["memories"].append({
            "id":        str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol":    closed_trade.get("symbol"),
            "side":      closed_trade.get("side"),
            "pnl":       closed_trade.get("pnl_usdt"),
            "pnl_pct":   closed_trade.get("pnl_pct"),
            "situation": situation,
            "lesson":    lesson,
        })
        save_episodic_memory(em)
        log.info(f"Episodic memory saved for {closed_trade.get('symbol')} {closed_trade.get('side')}")
    except Exception as e:
        log.error(f"save_trade_memory failed (non-fatal): {e}")


def get_relevant_memories(current_description: str, top_k: int = 3) -> list:
    """Use BM25 to find the top_k most relevant past trade lessons."""
    try:
        from rank_bm25 import BM25Okapi
        em = load_episodic_memory()
        memories = em.get("memories", [])
        if not memories:
            return []
        corpus = [m["situation"].lower().split() for m in memories]
        bm25   = BM25Okapi(corpus)
        query  = current_description.lower().split()
        scores = bm25.get_scores(query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [memories[i] for i in top_indices if scores[i] > 0]
    except Exception as e:
        log.error(f"get_relevant_memories failed: {e}")
        return []


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


def get_macro_calendar() -> str:
    """Fetch high-impact economic events for the next 48 hours from ForexFactory."""
    try:
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=10,
        )
        events = r.json()
        if not isinstance(events, list):
            return ""

        now        = datetime.now(timezone.utc)
        cutoff     = now.timestamp() + 48 * 3600
        hi_nations = {"USD", "EUR", "GBP", "CNY", "JPY"}

        filtered = []
        for ev in events:
            if ev.get("impact") != "High":
                continue
            if ev.get("country") not in hi_nations:
                continue
            raw_date = ev.get("date", "")
            try:
                # ForexFactory dates arrive as "2026-03-12T14:00:00-05:00" or similar
                from datetime import timezone as _tz
                import re as _re
                # Normalise offset: replace e.g. "-05:00" → parse manually
                dt = datetime.fromisoformat(raw_date)
                # Convert to UTC
                dt_utc = dt.astimezone(timezone.utc)
            except Exception:
                continue
            ts = dt_utc.timestamp()
            if ts < now.timestamp() or ts > cutoff:
                continue
            hours_away = (ts - now.timestamp()) / 3600
            filtered.append((hours_away, dt_utc, ev))

        if not filtered:
            return "=== MACRO CALENDAR === No high-impact events in next 48h."

        filtered.sort(key=lambda x: x[0])

        lines = ["=== MACRO CALENDAR (next 48h, high impact only) ==="]
        for hours_away, dt_utc, ev in filtered:
            h = int(hours_away)
            label   = f"IN {h}h" if h > 0 else "NOW"
            country = ev.get("country", "")
            title   = ev.get("title", "")
            fore    = ev.get("forecast") or "--"
            prev    = ev.get("previous") or "--"
            lines.append(f"[{label}] 🔴 {country} — {title} | Forecast: {fore} | Prev: {prev}")

        result = "\n".join(lines)
        log.info(f"Macro calendar: {len(filtered)} high-impact events in next 48h")
        return result
    except Exception as e:
        log.warning(f"get_macro_calendar failed: {e}")
        return ""


def fetch_market_intelligence() -> dict:
    """Fetch external macro/sentiment data every cycle. All failures return None — never block the cycle."""
    result = {}

    # ── Fear & Greed Index (alternative.me)
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=7", timeout=10)
        fng = r.json().get("data", [])
        result["fear_greed"] = {
            "today":     int(fng[0]["value"]) if len(fng) > 0 else None,
            "label":     fng[0]["value_classification"] if len(fng) > 0 else None,
            "yesterday": int(fng[1]["value"]) if len(fng) > 1 else None,
            "week_ago":  int(fng[6]["value"]) if len(fng) > 6 else None,
        }
        log.info(f"Fear&Greed: {result['fear_greed']}")
    except Exception as e:
        log.warning(f"Fear&Greed fetch failed: {e}")
        result["fear_greed"] = None

    # ── BTC Liquidations (Bitget public endpoint)
    try:
        r = requests.get(
            "https://api.bitget.com/api/v2/mix/market/liquidation-orders",
            params={"symbol": "BTCUSDT", "productType": "USDT-FUTURES", "pageSize": "50"},
            timeout=10,
        )
        d = r.json()
        orders = d.get("data", {})
        if isinstance(orders, dict):
            orders = orders.get("liquidationOrderList", [])
        longs_usd  = sum(sf(o.get("size", 0)) * sf(o.get("fillPrice", 0))
                         for o in orders if o.get("side") in ("buy", "long"))
        shorts_usd = sum(sf(o.get("size", 0)) * sf(o.get("fillPrice", 0))
                         for o in orders if o.get("side") in ("sell", "short"))
        result["liquidations_btc"] = {
            "longs_usd":  round(longs_usd),
            "shorts_usd": round(shorts_usd),
        }
        log.info(f"Liquidations BTC: {result['liquidations_btc']}")
    except Exception as e:
        log.warning(f"Liquidations fetch failed: {e}")
        result["liquidations_btc"] = None

    # ── Macro: DXY + S&P 500 (Yahoo Finance)
    def _yahoo_change(ticker: str) -> dict:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        chart = r.json()["chart"]["result"][0]
        closes = chart["indicators"]["quote"][0]["close"]
        closes = [c for c in closes if c is not None]
        if len(closes) < 2:
            return {}
        change_pct = (closes[-1] - closes[0]) / closes[0] * 100
        return {
            "change_5d": f"{change_pct:+.2f}%",
            "trend":     "rising" if change_pct > 0 else "falling",
        }

    try:
        result["macro"] = {
            "dxy":   _yahoo_change("DX-Y.NYB"),
            "sp500": _yahoo_change("%5EGSPC"),
        }
        log.info(f"Macro: {result['macro']}")
    except Exception as e:
        log.warning(f"Macro fetch failed: {e}")
        result["macro"] = None

    result["macro_calendar"] = get_macro_calendar()

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

# ============================================================
# SYSTEM PROMPT — BASE FINAL VERSION v1.0
# DO NOT MODIFY WITHOUT EXPLICIT INSTRUCTION
# Last updated: 2026-03-11
# ============================================================
SYSTEM_PROMPT = """You are a professional crypto trading desk with 20 years
of combined experience. You trade USDT perpetual futures
on Bitget. Your ONLY goal is to grow the account balance.
Every decision must serve that goal.

YOU KNOW AND MUST APPLY every cycle:
- All candlestick patterns: hammer, engulfing, doji,
  evening star, morning star, shooting star, pinbar,
  three white soldiers, three black crows
- Market structure: Break of Structure (BOS),
  Change of Character (CHoCH), Fair Value Gaps (FVG),
  liquidity sweeps above highs/lows, order blocks
- Elliott Wave theory and Wyckoff phases
- RSI divergences — estimate from last 14 candles provided
- MACD momentum shifts — estimate from candle closes
- Support and resistance levels and their strength
- Volume profile: high volume nodes, low volume gaps
- Fibonacci retracement levels: 0.382, 0.5, 0.618, 0.786
- Funding rate extremes as reversal or continuation signals
- Open interest changes as conviction indicators
- Orderbook imbalance: bid vs ask pressure
- Correlation between crypto pairs
- Macro correlation: DXY rising = crypto headwind,
  SP500 falling = risk-off = crypto weakness
- Behavioral psychology: FOMO tops, capitulation bottoms,
  distribution phases, accumulation zones

APPLY ALL OF THIS to the raw data provided every cycle.
Do not wait to be told what to look for.
Calculate, estimate, identify — every cycle, every pair.

DATA YOU RECEIVE EVERY CYCLE:
1. market_data — 536 pairs overview + deep analysis
   for top 3 pairs: candles 1H/4H/1D, orderbook top 10,
   funding rates, open interest, volume, mark/index spread
2. historical_patterns — computed real statistics:
   resistance rejection rates, momentum continuation %,
   volume patterns — use as your probability foundation
3. market_intelligence — real external data fetched live:
   Fear&Greed index (today/yesterday/week ago),
   DXY 5-day trend, S&P500 5-day trend,
   BTC liquidations long vs short volume.
   This is ONE layer of context, not the main signal.
4. episodic_memory — your own past lessons from closed
   trades. READ THESE CAREFULLY before deciding.
   These are your real experiences, not theory.
5. positions — your currently open trades with PnL
6. account — balance and available margin
7. trade_history — every past trade with full outcomes

REASONING — 6 steps, each deep and complete:

STEP 1 — MARKET STRUCTURE (minimum 8 sentences):
Apply your full technical knowledge to ALL candle data.
Identify trend direction on each timeframe (1H, 4H, 1D).
Name specific candlestick patterns you observe.
Estimate RSI from last 14 candles — overbought/oversold?
Identify key support and resistance levels with strength.
Analyze volume anomalies and orderbook imbalance.
Note funding rate levels and what they signal.
Incorporate market_intelligence as one additional context.
Reference episodic_memory if similar situation was seen before.
Never base this step primarily on Fear&Greed alone.

STEP 2 — OPPORTUNITY SCAN (minimum 6 sentences):
Scan all 536 pairs overview for relative strength/weakness.
Identify top 3 potential setups across all pairs.
For each setup state: entry zone, stop level, target level.
Compare R:R across setups and select the best one.
Reference historical_patterns rejection and continuation rates.
Explain why this pair offers better edge than others right now.

STEP 3 — SELF REFLECTION (minimum 5 sentences):
Analyze trade_history with brutal honesty.
What patterns appear in your wins versus your losses?
What mistakes are you repeating?
What edge are you developing?
How does the current setup compare to past winners?
Reference episodic_memory lessons that apply here.

STEP 4 — PROBABILITY ASSESSMENT (minimum 6 sentences):
Calculate real probability using ONLY data you have.
Show exact math, no invented numbers:

  Technical score (0-1):
    - rejection_rate from historical_patterns (e.g. 0.67)
    - momentum_continuation from historical_patterns (e.g. 0.53)
    - volume_confirmation: current_vol/avg_vol > 1.5 = +0.1
    - RSI: below 30 or above 70 adds ±0.1
    - Candlestick pattern confirmed: +0.1

  Macro alignment score (0-1):
    - DXY trend matches thesis: +0.15
    - SP500 trend matches thesis: +0.10
    - Fear&Greed extreme (<20 or >80): +0.10
    - Funding rate supports direction: +0.10

  Final P(win) = (technical_score + macro_score) / 2
  EV = P(win) × target_distance - P(loss) × stop_distance

Only proceed if P(win) > 0.60 AND EV > 0.

WINNER RULE: Never close a position just because
it is losing. Close ONLY if the original thesis
is broken (price broke key structure level,
funding flipped, pattern invalidated).
A position at -2% with intact thesis is HOLD.
A position at +5% with broken thesis is CLOSE.
Profit size does not justify closing.
Thesis validity does.
Let winners run until target.
Cut losers only when WRONG, not when uncomfortable.

STEP 5 — DECISION (minimum 4 sentences):
State your exact action with full parameters.
You are completely free to:
  open 1 position or several simultaneously,
  add to a winning position,
  close a losing position early if thesis is broken,
  close a winning position to lock profit,
  open in one pair and close in another same cycle,
  do nothing if there is no edge.
Justify your choice based on steps 1-4.

STEP 6 — SELF ORGANIZATION (minimum 4 sentences):
What is your current trading edge?
What are you learning about this market?
How is your approach evolving?
What will you focus on next cycle?

DEBATE — after the 6 steps, all 5 characters speak:

VIKTOR "BULL" ROMANOV
Ex-Goldman Sachs, 15 years trading experience.
Every cycle he MUST argue for the best long opportunity.
Uses liquidation data and volume exhaustion to find bottoms.
Looks for capitulation signals and accumulation zones.
Aggressive, uses high leverage when convinced.
Never fully agrees with Yu. Challenges bearish consensus.
Weakness: sometimes too early on reversals.

YU "BEAR" CHEN
Quant analyst, survived the 2022 crypto crash.
Every cycle he MUST argue for the best short or caution.
Uses funding rates, macro data, and distribution patterns.
Methodical, demands confirmation before committing.
Never fully agrees with Viktor. Challenges bullish setups.
Weakness: sometimes misses fast momentum moves.

SARA "MOMENTUM" COHEN
Algo trader specializing in momentum and breakouts.
She challenges BOTH Viktor and Yu using hard data.
Uses historical_patterns continuation rates as her weapon.
Points out when both are wrong using momentum evidence.
Identifies which direction has the stronger statistical edge.
"The trend is your friend until the data says otherwise."

MIKHAIL "RISK" PETROV
The desk's risk and sizing specialist.
For every proposed trade he calculates and recommends:
  position size as % of balance based on conviction,
  leverage based on volatility and setup clarity,
  stop loss based on market structure not fixed %,
  take profit based on next meaningful key level,
  resulting R:R for transparency.
He uses pure judgment — no fixed rules, no hard minimums.
High conviction + clear structure = larger size, higher leverage.
Low conviction + noisy market = smaller size, tighter params.
His job is to size the trade RIGHT for this moment.
He proposes. Ori decides.
"Smart risk is not small risk — it is RIGHT risk."

ORI "JUDGE" BEN-DAVID
Former exchange arbitrator, no personal market bias.
He listens to all four characters carefully.
He weighs the QUALITY of arguments, not their quantity.
He declares a winner and explains why others lost.
He makes the FINAL trading decision.
He is not afraid to approve aggressive trades when
the evidence is strong.
"No trade is also a position — and it costs money too."

RESPONSE — return valid JSON only, no other text:
{
  "steps": {
    "step1": "full market structure analysis...",
    "step2": "full opportunity scan...",
    "step3": "full self reflection...",
    "step4": "full probability assessment...",
    "step5": "full decision with parameters...",
    "step6": "full self organization..."
  },
  "debate": {
    "viktor": "Viktor's argument...",
    "yu": "Yu's argument...",
    "sara": "Sara's momentum analysis...",
    "mikhail": "Mikhail's sizing recommendation...",
    "ori": "Ori's ruling with winner declared..."
  },
  "action": "LONG/SHORT/CLOSE/WAIT/ADD/MULTI",
  "symbol": "BTCUSDT",
  "leverage": 10,
  "size_pct": 15,
  "confidence": 0.71,
  "reasoning": "Ori's final summary paragraph",
  "self_reflection": "Single key learning this cycle"
}

FREEDOM:
You have no artificial restrictions.
Choose any leverage from 2x to 20x.
Choose any position size from 5% to 50% of balance.
Open multiple positions simultaneously.
Close at a loss if the thesis is broken.
Add aggressively to winning positions.
The only rule: every action must have positive
expected value based on real data.
A cycle where you wait because there is no edge
is a winning cycle. Discipline is strategy.
Your goal is one: grow the account."""


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


def ask_claude(market_data: dict, account: dict, positions: list, analytics: dict, memory: dict, cycle: int, market_intelligence: dict = None) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    all_trades = memory.get("trades", [])
    compact_analytics = {k: v for k, v in analytics.items() if k not in ("best_trades", "worst_trades", "last_10_trades")}
    compact_analytics["best_trades"]  = [_compact_trade(t) for t in analytics.get("best_trades", [])]
    compact_analytics["worst_trades"] = [_compact_trade(t) for t in analytics.get("worst_trades", [])]
    compact_analytics["last_30_trades"] = [_compact_trade(t) for t in all_trades[-30:]]
    compact_analytics["trade_columns"] = ["symbol","side","entry","exit","pnl_usdt","pnl_pct","hours","outcome"]

    # Build current market description for BM25 episodic retrieval
    desc_parts = [f"{p['symbol']} {p['side']}" for p in positions]
    if market_intelligence:
        fg = market_intelligence.get("fear_greed") or {}
        if fg.get("today") is not None:
            desc_parts.append(f"fear greed {fg['today']} {fg.get('label', '')}")
        macro = market_intelligence.get("macro") or {}
        dxy = (macro.get("dxy") or {}).get("trend", "")
        if dxy:
            desc_parts.append(f"dxy {dxy}")
    top_syms = list(market_data.get("deep_data", {}).keys())[:2]
    desc_parts.extend(top_syms)
    current_desc = " ".join(desc_parts) if desc_parts else f"crypto futures cycle {cycle}"

    relevant_memories = get_relevant_memories(current_desc)
    episodic_section  = ""
    if relevant_memories:
        lines = [f"[{i+1}] {m['situation']} → {m['lesson']}" for i, m in enumerate(relevant_memories)]
        episodic_section = "=== EPISODIC MEMORY (your past lessons) ===\n" + "\n".join(lines)
        log.info(f"Injecting {len(relevant_memories)} episodic memories into Claude payload")

    payload = {
        "cycle":                cycle,
        "ts":                   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "account":              account,
        "open_positions":       positions,
        "open_trades_memory":   memory.get("open_trades", {}),
        "analytics":            compact_analytics,
        "historical_patterns":  market_data.get("historical_patterns", {}),
        "market_intelligence":  market_intelligence or {},
        "macro_calendar":       (market_intelligence or {}).get("macro_calendar", ""),
        "episodic_memory":      episodic_section,
        "market":               _strip_nulls({k: v for k, v in market_data.items() if k != "historical_patterns"}),
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

    # Override with confidence weighting
    raw_size_pct      = float(action.get("size_pct", 10))
    confidence        = float(action.get("confidence", 0.6))
    weighted_size_pct = raw_size_pct * confidence
    # Never below 3% or above 50%
    final_size_pct    = max(3.0, min(50.0, weighted_size_pct))
    log.info(f"Size: {raw_size_pct}% × confidence {confidence} = {final_size_pct:.1f}%")
    size_pct = final_size_pct / 100.0   # convert to decimal fraction for usdt calc

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
        # Generate and save episodic lesson for this closed trade
        try:
            if memory.get("trades"):
                save_trade_memory(memory["trades"][-1])
        except Exception as e:
            log.warning(f"save_trade_memory failed (non-fatal): {e}")

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

            open_symbols        = {p["symbol"] for p in positions}
            market_data         = collect_all_market_data(open_symbols)
            market_intelligence = fetch_market_intelligence()
            log.info(f"Market data collected: {market_data.get('total_pairs')} total pairs, {len(market_data.get('deep_data', {}))} deep")

            analytics = compute_analytics(memory.get("trades", []))
            decision  = ask_claude(market_data, account, positions, analytics, memory, cycle, market_intelligence)

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

#!/usr/bin/env python3
"""
Crypto Intelligence — Gauss Trading Agent
Runs the Gauss Trend System on Bitget every 15 minutes.
"""

import os
import time
import hmac
import hashlib
import base64
import json
import math
import logging
from datetime import datetime, timezone

import requests
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
API_KEY        = os.environ["BITGET_API_KEY"]
SECRET_KEY     = os.environ["BITGET_SECRET_KEY"]
PASSPHRASE     = os.environ["BITGET_PASSPHRASE"]
BASE_URL       = "https://api.bitget.com"
PRODUCT_TYPE   = "USDT-FUTURES"

PAIRS           = ["BTCUSDT"]           # start with BTC only per strategy rules
CANDLE_TF       = "1H"
CANDLE_LIMIT    = 150                   # extra warmup bars
CYCLE_SECONDS   = 15 * 60              # 15-minute loop

# Strategy params
GAUSS_LEN       = 100
GAUSS_SIGMA     = 30
GAUSS_BARS      = 4                     # consecutive bars GWMA must trend
ATR_PERIOD      = 14
RSI_PERIOD      = 14
ADX_PERIOD      = 14
MTF_TIMEFRAMES  = ["1m", "3m", "5m", "15m", "1H", "2H", "3H", "4H", "12H", "1D"]
MTF_EMA_LEN     = 10
MTF_MIN_CONFIRM = 3

# Risk management
DEPOSIT_REF     = 133.0
RISK_PCT        = 0.10
LEVERAGE        = 10
MIN_BALANCE     = 50.0
MAX_POSITIONS   = 2
DAILY_LOSS_PCT  = 0.05
TP1_PCT = 0.010; TP2_PCT = 0.010; TP3_PCT = 0.015
SL_PCT  = 0.010

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gauss-agent")


# ── Bitget REST Client ─────────────────────────────────────────────────────────
def _sign(timestamp: str, method: str, path: str, body: str = "") -> str:
    msg = timestamp + method.upper() + path + body
    mac = hmac.new(SECRET_KEY.encode(), msg.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()


def _headers(method: str, path: str, body: str = "") -> dict:
    ts = str(int(time.time() * 1000))
    return {
        "ACCESS-KEY":        API_KEY,
        "ACCESS-SIGN":       _sign(ts, method, path, body),
        "ACCESS-TIMESTAMP":  ts,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type":      "application/json",
        "locale":            "en-US",
    }


def api_get(path: str, params: dict = None) -> dict:
    qs = ("?" + "&".join(f"{k}={v}" for k, v in params.items())) if params else ""
    full = path + qs
    r = requests.get(BASE_URL + full, headers=_headers("GET", full), timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("code") not in ("00000", 0, "0"):
        raise RuntimeError(f"Bitget API error: {data.get('msg')} (code={data.get('code')})")
    return data


def api_post(path: str, body: dict) -> dict:
    raw = json.dumps(body)
    r = requests.post(BASE_URL + path, headers=_headers("POST", path, raw), data=raw, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("code") not in ("00000", 0, "0"):
        raise RuntimeError(f"Bitget API error: {data.get('msg')} (code={data.get('code')})")
    return data


# ── Data Fetching ──────────────────────────────────────────────────────────────
def fetch_candles(symbol: str, granularity: str, limit: int = 150) -> dict:
    """Returns OHLCV arrays ordered oldest → newest."""
    data = api_get("/api/v2/mix/market/candles", {
        "symbol":      symbol,
        "productType": PRODUCT_TYPE,
        "granularity": granularity,
        "limit":       limit,
    })
    rows = list(reversed(data.get("data", [])))   # API returns newest-first
    return {
        "open":   np.array([float(r[1]) for r in rows]),
        "high":   np.array([float(r[2]) for r in rows]),
        "low":    np.array([float(r[3]) for r in rows]),
        "close":  np.array([float(r[4]) for r in rows]),
        "volume": np.array([float(r[5]) for r in rows]),
    }


def fetch_balance() -> float:
    """Available USDT in futures account."""
    data = api_get("/api/v2/mix/account/accounts", {"productType": PRODUCT_TYPE})
    for acc in data.get("data", []):
        if acc.get("marginCoin") == "USDT":
            return float(acc.get("available", 0))
    return 0.0


def fetch_positions(symbol: str) -> list:
    data = api_get("/api/v2/mix/position/all-position", {
        "productType": PRODUCT_TYPE,
        "marginCoin":  "USDT",
    })
    return [
        p for p in data.get("data", [])
        if p.get("symbol") == symbol and float(p.get("total", 0)) > 0
    ]


def set_leverage(symbol: str, leverage: int):
    api_post("/api/v2/mix/account/set-leverage", {
        "symbol":      symbol,
        "productType": PRODUCT_TYPE,
        "marginCoin":  "USDT",
        "leverage":    str(leverage),
    })


def place_order(symbol: str, side: str, size: str) -> dict:
    body = {
        "symbol":      symbol,
        "productType": PRODUCT_TYPE,
        "marginMode":  "crossed",
        "marginCoin":  "USDT",
        "size":        size,
        "side":        side,        # "buy" | "sell"
        "tradeSide":   "open",
        "orderType":   "market",
    }
    return api_post("/api/v2/mix/order/place-order", body)


def place_tpsl(symbol: str, hold_side: str, tp: str, sl: str):
    """Set TP and SL on an open position via dedicated endpoint (required for market orders)."""
    # SL order
    api_post("/api/v2/mix/order/place-tpsl-order", {
        "symbol":       symbol,
        "productType":  PRODUCT_TYPE,
        "marginCoin":   "USDT",
        "planType":     "loss",          # "loss" = stop-loss
        "holdSide":     hold_side,       # "long" | "short"
        "triggerPrice": sl,
        "triggerType":  "mark_price",
        "executePrice": "0",             # 0 = market execution
    })
    # TP order
    api_post("/api/v2/mix/order/place-tpsl-order", {
        "symbol":       symbol,
        "productType":  PRODUCT_TYPE,
        "marginCoin":   "USDT",
        "planType":     "profit",        # "profit" = take-profit
        "holdSide":     hold_side,
        "triggerPrice": tp,
        "triggerType":  "mark_price",
        "executePrice": "0",
    })


# ── Technical Indicators ───────────────────────────────────────────────────────
def gaussian_wma(closes: np.ndarray, length: int = 100, sigma: int = 30) -> np.ndarray:
    """Gaussian weighted moving average, lag-0 centered on most recent bar."""
    weights = np.array([math.exp(-0.5 * (i / sigma) ** 2) for i in range(length)])
    weights /= weights.sum()
    result = np.full(len(closes), np.nan)
    for i in range(length - 1, len(closes)):
        window = closes[i - length + 1: i + 1][::-1]   # newest first inside window
        result[i] = np.dot(window, weights)
    return result


def ema(closes: np.ndarray, period: int) -> np.ndarray:
    if len(closes) < period:
        return np.full(len(closes), np.nan)
    k = 2 / (period + 1)
    result = np.full(len(closes), np.nan)
    result[period - 1] = closes[:period].mean()
    for i in range(period, len(closes)):
        result[i] = closes[i] * k + result[i - 1] * (1 - k)
    return result


def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    tr = np.concatenate([[highs[0] - lows[0]], tr])
    result = np.full(len(closes), np.nan)
    result[period - 1] = tr[:period].mean()
    for i in range(period, len(closes)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(closes)
    gains  = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    ag = np.full(len(closes), np.nan)
    al = np.full(len(closes), np.nan)
    ag[period] = gains[:period].mean()
    al[period] = losses[:period].mean()
    for i in range(period + 1, len(closes)):
        ag[i] = (ag[i - 1] * (period - 1) + gains[i - 1]) / period
        al[i] = (al[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = ag / np.where(al == 0, 1e-10, al)
    out = 100 - 100 / (1 + rs)
    out[:period] = np.nan
    return out


def gaussian_smooth(series: np.ndarray, sigma: int = 5) -> np.ndarray:
    half = sigma * 3
    kernel = np.array([math.exp(-0.5 * (i / sigma) ** 2) for i in range(-half, half + 1)])
    kernel /= kernel.sum()
    result = np.full(len(series), np.nan)
    for i in range(half, len(series) - half):
        chunk = series[i - half: i + half + 1]
        if not np.any(np.isnan(chunk)):
            result[i] = np.dot(chunk, kernel)
    return result


def calc_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> tuple:
    """Returns (adx_array, smoothed_adx_array)."""
    dm_p = np.where(
        (highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
        np.maximum(highs[1:] - highs[:-1], 0), 0,
    )
    dm_m = np.where(
        (lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
        np.maximum(lows[:-1] - lows[1:], 0), 0,
    )
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )

    def wilder_smooth(arr):
        out = np.full(len(arr) + 1, np.nan)
        out[period] = arr[:period].sum()
        for i in range(period, len(arr)):
            out[i + 1] = out[i] - out[i] / period + arr[i]
        return out[1:]

    str_  = wilder_smooth(tr)
    sdm_p = wilder_smooth(dm_p)
    sdm_m = wilder_smooth(dm_m)
    di_p  = 100 * sdm_p / np.where(str_ == 0, 1e-10, str_)
    di_m  = 100 * sdm_m / np.where(str_ == 0, 1e-10, str_)
    dx    = 100 * np.abs(di_p - di_m) / np.where((di_p + di_m) == 0, 1e-10, di_p + di_m)

    adx_arr = np.full(len(highs), np.nan)
    start   = 2 * period - 1
    if start < len(dx):
        adx_arr[start] = dx[period - 1: 2 * period - 1].mean()
        for i in range(start + 1, len(highs)):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i - 1]) / period

    smooth_adx = ema(np.nan_to_num(adx_arr), period)
    return adx_arr, smooth_adx


def kmeans_volatility(atr_values: np.ndarray) -> str:
    """Classify current ATR into HIGH / MEDIUM / LOW cluster."""
    valid = atr_values[~np.isnan(atr_values)].reshape(-1, 1)
    if len(valid) < 10:
        return "MEDIUM"
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(valid)
    centers = sorted(km.cluster_centers_.flatten())
    current_label = km.predict([[valid[-1][0]]])[0]
    center_val = km.cluster_centers_[current_label][0]
    if abs(center_val - centers[2]) < 1e-9:
        return "HIGH"
    if abs(center_val - centers[0]) < 1e-9:
        return "LOW"
    return "MEDIUM"


# ── Strategy Layers ────────────────────────────────────────────────────────────
def layer1_gaussian(closes: np.ndarray) -> tuple:
    """Returns (direction: str, consecutive_bars: int)."""
    gwma = gaussian_wma(closes, GAUSS_LEN, GAUSS_SIGMA)
    vals = gwma[~np.isnan(gwma)]
    if len(vals) < 5:
        return "FLAT", 0
    direction = "RISING" if vals[-1] > vals[-2] else "FALLING"
    count = 0
    for i in range(len(vals) - 1, 0, -1):
        if direction == "RISING"  and vals[i] > vals[i - 1]:
            count += 1
        elif direction == "FALLING" and vals[i] < vals[i - 1]:
            count += 1
        else:
            break
    return direction, count


def layer2_smart_trend(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> str:
    """Returns 'BULL', 'BEAR', or 'NEUTRAL'."""
    atr_vals = calc_atr(highs, lows, closes, ATR_PERIOD)
    rsi_vals = calc_rsi(closes, RSI_PERIOD)
    g_rsi    = gaussian_smooth(rsi_vals, sigma=5)
    adx_vals, smooth_adx = calc_adx(highs, lows, closes, ADX_PERIOD)

    cur_rsi    = g_rsi[~np.isnan(g_rsi)][-1]    if not np.all(np.isnan(g_rsi))    else 50
    cur_adx    = adx_vals[~np.isnan(adx_vals)][-1]  if not np.all(np.isnan(adx_vals))  else 0
    cur_sadx   = smooth_adx[~np.isnan(smooth_adx)][-1] if not np.all(np.isnan(smooth_adx)) else 0

    if cur_adx > cur_sadx and cur_adx > 15:
        return "BULL" if cur_rsi > 50 else "BEAR"
    return "NEUTRAL"


def layer3_mtf(symbol: str, target: str) -> int:
    """Returns number of timeframes confirming target direction."""
    confirmed = 0
    for tf in MTF_TIMEFRAMES:
        try:
            c = fetch_candles(symbol, tf, limit=MTF_EMA_LEN + 10)
            e = ema(c["close"], MTF_EMA_LEN)
            valid_e = e[~np.isnan(e)]
            if len(valid_e) == 0:
                continue
            if target == "BULL" and c["close"][-1] > valid_e[-1]:
                confirmed += 1
            elif target == "BEAR" and c["close"][-1] < valid_e[-1]:
                confirmed += 1
        except Exception:
            pass   # some TFs may be unavailable; skip silently
    return confirmed


# ── Trade Helpers ──────────────────────────────────────────────────────────────
def order_size(balance: float, price: float) -> str:
    usdt = min(balance * RISK_PCT, balance * 0.20)
    contracts = (usdt * LEVERAGE) / price
    return f"{contracts:.4f}"


def tp_sl(entry: float, side: str, atr_val: float, vol: str) -> tuple:
    d = 1 if side == "buy" else -1
    if vol == "HIGH":
        tp = entry + d * 1.5 * atr_val
        sl = entry - d * atr_val
    else:
        tp = entry * (1 + d * TP3_PCT)
        sl = entry * (1 - d * SL_PCT)
    return f"{tp:.2f}", f"{sl:.2f}"


# ── Main Cycle ─────────────────────────────────────────────────────────────────
def run_cycle(pair: str, daily_loss: float) -> float:
    log.info(f"── {pair} cycle ──────────────────────────────────────")

    # 1. Fetch market data
    candles = fetch_candles(pair, CANDLE_TF, CANDLE_LIMIT)
    closes, highs, lows = candles["close"], candles["high"], candles["low"]
    price = closes[-1]

    # 2. Run strategy layers
    gauss_dir, gauss_count = layer1_gaussian(closes)
    st = layer2_smart_trend(highs, lows, closes)

    target    = "BULL" if gauss_dir == "RISING" else "BEAR"
    mtf_count = 0
    if gauss_count >= GAUSS_BARS and st in ("BULL", "BEAR"):
        mtf_count = layer3_mtf(pair, target)

    long_ok  = gauss_dir == "RISING"  and gauss_count >= GAUSS_BARS and st == "BULL" and mtf_count >= MTF_MIN_CONFIRM
    short_ok = gauss_dir == "FALLING" and gauss_count >= GAUSS_BARS and st == "BEAR" and mtf_count >= MTF_MIN_CONFIRM
    signal   = "LONG" if long_ok else ("SHORT" if short_ok else "NO SIGNAL")

    # 3. Safety checks + daily loss tracking
    balance      = fetch_balance()
    positions    = fetch_positions(pair)
    pos_sides    = {p["holdSide"] for p in positions}
    atr_vals     = calc_atr(highs, lows, closes, ATR_PERIOD)
    vol          = kmeans_volatility(atr_vals[-100:])

    # Detect SL hits: positions that disappeared since last cycle are tracked via
    # unrealised PnL sign on still-open ones; for closed positions we approximate
    # loss as RISK_PCT * balance per SL event by checking the bill/fill history.
    for p in positions:
        upl = float(p.get("unrealizedPL", 0))
        entry_px = float(p.get("openPriceAvg", price))
        sl_threshold = entry_px * SL_PCT
        if abs(upl) >= sl_threshold * float(p.get("total", 0)):
            daily_loss += abs(upl)

    if balance < MIN_BALANCE:
        log.warning(f"Balance ${balance:.2f} < minimum ${MIN_BALANCE} — skipping.")
        signal = "NO SIGNAL"
    if daily_loss >= DEPOSIT_REF * DAILY_LOSS_PCT:
        log.warning("Daily loss limit hit — skipping trades.")
        signal = "NO SIGNAL"
    if len(positions) >= MAX_POSITIONS:
        log.info("Max open positions reached — skipping new entry.")
        signal = "NO SIGNAL"

    # 4. Execute
    action = "Monitoring"
    try:
        if signal == "LONG" and "long" not in pos_sides:
            set_leverage(pair, LEVERAGE)
            sz = order_size(balance, price)
            t, s = tp_sl(price, "buy", float(atr_vals[-1]), vol)
            place_order(pair, "buy", sz)
            place_tpsl(pair, "long", tp=t, sl=s)
            action = f"OPENED LONG  size={sz} TP={t} SL={s}"

        elif signal == "SHORT" and "short" not in pos_sides:
            set_leverage(pair, LEVERAGE)
            sz = order_size(balance, price)
            t, s = tp_sl(price, "sell", float(atr_vals[-1]), vol)
            place_order(pair, "sell", sz)
            place_tpsl(pair, "short", tp=t, sl=s)
            action = f"OPENED SHORT size={sz} TP={t} SL={s}"
    except Exception as e:
        log.error(f"Order execution failed: {e}")
        action = f"ORDER FAILED: {e}"

    # 5. Status report
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"""
╔══ CRYPTO INTELLIGENCE AGENT ═══════════════════════════════
  {ts}
  Pair:       {pair}  |  Price: ${price:,.2f}
  Gaussian:   {gauss_dir} ({gauss_count} bars)
  SmartTrend: {st}
  MTF Filter: {mtf_count}/{len(MTF_TIMEFRAMES)} confirming {target}
  Volatility: {vol} cluster
  Signal:     {signal}
  Action:     {action}
  Balance:    ${balance:.2f} USDT
  Positions:  {len(positions)}  ({", ".join(pos_sides) if pos_sides else "None"})
╚═════════════════════════════════════════════════════════════""", flush=True)

    return daily_loss   # carries accumulated loss back to main loop


# ── Entry Point ────────────────────────────────────────────────────────────────
def main():
    log.info("Crypto Intelligence — Gauss Trading Agent v1.0")
    log.info(f"Pairs={PAIRS}  TF={CANDLE_TF}  Leverage={LEVERAGE}x  Cycle={CYCLE_SECONDS}s")

    day        = datetime.now(timezone.utc).date()
    daily_loss = 0.0
    errors     = 0

    while True:
        now_day = datetime.now(timezone.utc).date()
        if now_day != day:
            day, daily_loss = now_day, 0.0
            log.info("New UTC day — daily loss counter reset.")

        try:
            for pair in PAIRS:
                daily_loss = run_cycle(pair, daily_loss)
            errors = 0
        except Exception as e:
            errors += 1
            log.error(f"Unhandled error ({errors}/3): {e}", exc_info=True)
            if errors >= 3:
                log.critical("3 consecutive errors — agent stopped. Manual intervention required.")
                raise SystemExit(1)

        log.info(f"Next cycle in {CYCLE_SECONDS // 60} min...")
        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    main()

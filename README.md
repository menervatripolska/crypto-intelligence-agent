# Crypto Intelligence — Gauss Trading Agent

AI-powered crypto trading agent using Gaussian filter trend strategy on Bitget.

## Stack
- **AI**: Claude Code (Anthropic)
- **Exchange**: Bitget via MCP
- **Strategy**: Gauss Trend System (Gaussian filter + SmartTrend + K-Means volatility + MTF filter)

## Setup

### 1. Connect Bitget MCP
```bash
claude mcp add --transport stdio \
  --env BITGET_API_KEY=your-key \
  --env BITGET_SECRET_KEY=your-secret \
  --env BITGET_PASSPHRASE=your-passphrase \
  bitget -- npx -y bitget-mcp-server --modules all
```

### 2. Verify connection
```bash
claude mcp list
```

### 3. Run agent
```bash
cd crypto-intelligence-agent
claude
```

Claude Code will automatically read `CLAUDE.md` as system instructions.

## Usage

| Command | Action |
|---------|--------|
| `START` | Begin monitoring and trading |
| `STOP` | Halt all activity |
| `STATUS` | Report current state |
| `CLOSE ALL` | Close all open positions |

## Strategy Overview

**Gauss Trend System** — 3-layer confirmation:
1. **Gaussian Filter** — noise-free trend detection (length=100, sigma=30)
2. **SmartTrend** — RSI + ADX + dynamic volatility bands with K-Means clustering
3. **MTF Filter** — EMA trend check across 10 timeframes

**Risk**: 10% per trade | SL: 1% | TP1: 1% (30%) | TP2: 1% (30%) | TP3: 1.5% (40%)

## Account

- Subaccount: `Claude_ag` (virtual subaccount on Bitget)
- Trading: BTCUSDT futures perpetual (primary)

---
*Crypto Intelligence © 2026*

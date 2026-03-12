# :robot: Crypto Intelligence Agent

**Autonomous AI trading agent. Claude Sonnet + Bitget USDT Futures. Running live with real money.**

Built by [@qasqyrka](https://instagram.com/qasqyrka)

---

## What makes it different

Most crypto bots send alerts. This one trades.

- **Actually executes trades autonomously** — opens, closes, scales, sets SL/TP, all without human input
- **5-character internal debate per cycle** — Viktor (bull), Yu (bear), Sara (momentum), Mikhail (risk/sizing), Ori (judge) argue before every decision
- **Learns from own trade history** — BM25 episodic memory retrieves relevant past lessons before each cycle
- **Peak PnL tracking** — tracks unrealized high-water mark per position so Claude knows when profits are fading
- **Full reasoning log public at `/log`** — every cycle's 6-step analysis + debate is readable in real time
- **Open source** — deploy your own instance, use your own keys, own your agent

---

## How it works

Every 15 minutes, `trading_loop()` runs a full cycle:

1. **Data collection** — fetches all USDT-FUTURES tickers (~536 pairs), then deep data (1H/4H/1D candles, orderbook, funding rate, open interest, elite long/short ratio, mark/index spread, recent trades) for top 3 pairs by volume + any open positions
2. **Market intelligence** — Fear & Greed index (7-day history), DXY and S&P 500 5-day trends via Yahoo Finance, BTC liquidation volumes, macro economic calendar (next 48h high-impact events from ForexFactory)
3. **Historical pattern analysis** — `analyze_historical_patterns()` computes funding bias, resistance rejection rates, momentum continuation %, and volume patterns from raw candle data
4. **Episodic memory retrieval** — `get_relevant_memories()` uses BM25 to find top 3 most relevant past trade lessons
5. **Claude Sonnet receives ~27k chars** — `ask_claude()` sends all of the above as a single JSON payload with the system prompt
6. **6-step reasoning + 5-character debate** — Claude returns structured JSON with market structure analysis, opportunity scan, self-reflection, probability assessment, decision, and self-organization, followed by the full debate
7. **Execution** — `execute_action()` dispatches OPEN/CLOSE/ADD/WAIT via Bitget REST API with HMAC authentication
8. **Guardian** — `ensure_stops()` runs before Claude every cycle, checking that every open position has SL and TP plan orders on Bitget; places defaults if missing
9. **Position sync** — `sync_positions()` detects externally closed positions (TP/SL triggered on exchange) and generates episodic lessons for them

---

## Active skills

| Skill | What it does |
|-------|-------------|
| **BM25 Episodic Memory** | After each closed trade, Claude writes a lesson. On each new cycle, BM25 retrieves the most relevant past lessons and injects them into the prompt. (`generate_trade_lesson()`, `get_relevant_memories()`) |
| **Confidence-Weighted Sizing** | Claude outputs `size_pct` and `confidence`. Actual size = `size_pct * confidence`, clamped 3%-50%. (`execute_open()`) |
| **Macro Calendar** | Fetches high-impact economic events (FOMC, CPI, NFP) for next 48h from ForexFactory. Filters USD/EUR/GBP/CNY/JPY only. (`get_macro_calendar()`) |
| **Fear & Greed + DXY + S&P 500** | Live sentiment and macro correlation data fetched every cycle. (`fetch_market_intelligence()`) |
| **Peak PnL Tracking** | Tracks unrealized PnL high-water mark per position. Claude sees current drawdown from peak to decide when to take profits. (`sync_positions()`, `_format_open_positions()`) |
| **Guardian** | Every cycle, checks all open positions for existing SL/TP orders on Bitget. Places defaults (1.5% SL, 3% TP) if missing. (`ensure_stops()`) |

---

## Requirements

- Python 3.11+
- Bitget account with a USDT Futures subaccount and API keys
- Anthropic API key (Claude Sonnet)
- Railway account (or any server with persistent storage)

---

## Quick start

```bash
git clone https://github.com/menervatripolska/crypto-intelligence-agent.git
cd crypto-intelligence-agent
pip install -r requirements.txt
```

Create `.env` in the project root:

```
BITGET_API_KEY=your_api_key
BITGET_SECRET_KEY=your_secret_key
BITGET_PASSPHRASE=your_passphrase
ANTHROPIC_API_KEY=your_anthropic_key
```

Run:

```bash
python agent.py
```

The agent starts the trading loop in a background thread and serves the dashboard on port 8080.

---

## Deploy to Railway

1. Fork this repo on GitHub
2. Go to [railway.app](https://railway.app) and create a new project
3. Select **Deploy from GitHub repo** and connect your fork
4. Add environment variables in the Railway dashboard:
   - `BITGET_API_KEY`
   - `BITGET_SECRET_KEY`
   - `BITGET_PASSPHRASE`
   - `ANTHROPIC_API_KEY`
5. Add a **Volume** mounted at `/data` — this is where memory, episodic memory, and logs persist across deploys
6. Deploy. Railway detects the `Procfile` (`web: python agent.py`) automatically

The agent will start trading on the next 15-minute cycle.

---

## Dashboard endpoints

| Endpoint | Content-Type | Description |
|----------|-------------|-------------|
| `/` | `text/html` | Index page with links |
| `/log` | `text/markdown` | Full reasoning log — every cycle's 6-step analysis, debate, and ruling |
| `/log/text` | `text/plain` | Same log as plain text, for integrations |
| `/memory` | `application/json` | `memory.json` — all trades, open positions, analytics, peak PnL data |

---

## Security

- **API key permissions**: on Bitget, enable **Read** and **Trade** only. Never enable Withdraw or Transfer.
- **Keys in env vars only** — all 4 keys are loaded via `os.environ[]`, never hardcoded in source
- **`.gitignore` protects `.env`** from accidental commits
- **Each user deploys their own instance** with their own API keys. No shared infrastructure, no shared keys.
- **If a key is compromised**: immediately delete the API key on Bitget, create a new one, update your environment variables, and redeploy. The agent will pick up the new keys on restart.

---

## Cost

| Component | Estimate |
|-----------|----------|
| Railway | ~$10/mo |
| Anthropic API (Claude Sonnet, ~96 cycles/day) | ~$80-130/mo |
| **Total** | **~$90-140/mo** |

---

## Community

If you deploy your own agent, open a GitHub Discussion and share your GitHub or Instagram — I'm building a network of agent runners and will personally reach out.

---

## Disclaimer

This software trades cryptocurrency futures with real money. Futures trading involves substantial risk of loss. Past performance does not guarantee future results. You are solely responsible for your own trading decisions and any losses incurred. The author is not a financial advisor. Use at your own risk.

---

Follow the build: [@qasqyrka](https://instagram.com/qasqyrka)

Built while GetClaw sends alerts. This one actually trades.

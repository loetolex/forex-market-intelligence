# Forex Market Intelligence

Research-to-paper-trading backend for the Forex Market Intelligence project.

## Current status

- Primary market data: Tiingo FX
- Secondary market data: Twelve Data
- Macro context: FRED / ECB
- Research source: existing Colab master notebook
- Deployment target: Railway
- Broker candidate: Interactive Brokers
- Runtime mode: PAPER/DEMO only
- Live trading: **DISABLED**
- Order placement: **DISABLED** in the current deployment
- Timeframe source: Tiingo 1H REAL_DATA
- Derived timeframes: 4H and 1D calculated only from complete closed 1H candles
- Canonical FX daily boundary: **17:00 America/New_York** (DST-aware)
- Tiingo research history window: **1095 calendar days** by default

The application is intentionally fail-closed. Missing data, stale data, invalid data,
model failure, or risk failure produces `NO TRADE` / `HOLD` rather than an order.

## Architecture

```text
Tiingo FX 1H REAL_DATA
    |
    v
Closed-Candle Validation
    |
    +----> Calculated 4H
    |
    +----> Calculated 1D (17:00 America/New_York boundary)
    |
    v
Feature Engine
    |
    v
Forecast Model
    |
    v
Strict Validation / Admission
    |
    v
Signal Engine
    |
    v
Risk Engine
    |
    v
Execution Approval
    |
    v
IBKR Paper / Read-Only
```

Twelve Data remains available as a secondary provider for diagnostics and cross-checks.
A provider disagreement never authorizes trading.

The 1H source is used to calculate both 4H and 1D candles so all timeframes use the same
verified clock and candle lineage. DST transition sessions are handled with the canonical
New York 17:00 boundary instead of assuming every FX day is exactly 24 UTC hours.

The Colab notebook remains the research/reproducibility environment. The deployable
application is the runtime environment.

## Watchlist

The default portfolio watchlist contains 12 pairs:

- EUR/USD
- GBP/USD
- USD/JPY
- USD/CHF
- AUD/USD
- USD/CAD
- NZD/USD
- EUR/GBP
- EUR/JPY
- GBP/JPY
- AUD/JPY
- NZD/JPY

Any individual supported Tiingo FX pair can also be queried through `/market/{symbol}`.
Unsupported or unavailable provider data fails closed as `DATA UNAVAILABLE`.

## Environment variables

Required for market data:

- `TIINGO_API_TOKEN`

Secondary market-data diagnostics:

- `TWELVE_DATA_API_KEY`

Optional at this stage:

- `FRED_API_KEY`
- `IBKR_HOST`
- `IBKR_PORT`
- `IBKR_CLIENT_ID`

Safety defaults:

- `TRADING_MODE=PAPER`
- `LIVE_TRADING_ENABLED=false`
- `ORDER_PLACEMENT_ENABLED=false`

Never commit API keys or broker credentials.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export TIINGO_API_TOKEN="..."
export TWELVE_DATA_API_KEY="..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health endpoint:

```text
GET /health
```

Provider diagnostic:

```text
GET /provider/inspect/EURUSD?interval=1h&outputsize=10
```

Provider cross-check:

```text
GET /provider/crosscheck/EURUSD?interval=1h&outputsize=10
```

Market snapshot:

```text
GET /market/EURUSD?interval=1h
```

Portfolio snapshot (12 default pairs):

```text
GET /portfolio
```

`/portfolio` starts the lightweight, single-worker scanner and returns promptly.
It never runs the deep `/market/{symbol}` research cycle for every pair. Poll
these read-only endpoints while it is running; neither starts another scan:

```text
GET /portfolio/status
GET /portfolio/results
```

Completed results are cached in process until `GET /portfolio?refresh=true`.
The scanner uses Twelve Data 15m bars (deriving 30m), Tiingo 1h bars
(deriving 4h), and Tiingo native 1D bars. Its `portfolio-fast-hgb-v1` output
is labelled `SCANNER_ONLY`; it is not a research-admitted model and cannot
authorize execution.

## Railway

Use the existing Railway `forex-api` service connected to this repository.

Start command:

```text
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set secrets in Railway Variables. Do not commit credentials.

## Deployment synchronization

This repository is synchronized with the existing Railway `forex-api` service only.
Do not create a second production service for the same API.

## Important

This deployment is not a live trading system.

It intentionally contains no live broker credentials, no live order path, and no automatic
real-money execution. Paper/demo promotion requires separate testing and explicit approval.

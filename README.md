# Forex Market Intelligence

Research-to-paper-trading backend for the Forex Market Intelligence project.

## Current status

- Primary market data: Twelve Data
- Macro context: FRED / ECB
- Research source: existing Colab master notebook
- Deployment target: Railway
- Broker candidate: Interactive Brokers
- Runtime mode: PAPER/DEMO only
- Live trading: **DISABLED**
- Order placement: **NOT IMPLEMENTED in this first deployment package**

The application is intentionally fail-closed. Missing data, stale data, invalid data,
model failure, or risk failure produces `NO TRADE` / `HOLD` rather than an order.

## Architecture

```text
Twelve Data
    |
    v
Market Data Validation
    |
    v
Feature Engine
    |
    v
Forecast Model
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
IBKR Paper (read-only in this first deployment)
```

The Colab notebook remains the research/reproducibility environment. The deployable
application is the runtime environment.

## Environment variables

Required for market data:

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

export TWELVE_DATA_API_KEY="..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health endpoint:

```text
GET /health
```

Market snapshot:

```text
GET /market/EURUSD?interval=1h
```

## Railway

Create a Railway service from this repository.

Start command:

```text
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set secrets in Railway Variables. Do not commit credentials.

## Important

This deployment is not a live trading system.

It intentionally contains no live broker credentials, no live order path, and no automatic
real-money execution. Paper/demo promotion requires separate testing and explicit approval.
